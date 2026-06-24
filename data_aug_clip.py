"""CLIP-guided single-source data augmentation for domain generalization.

Idea
----
Given a *single* source domain, we synthesize augmented copies of every source
image whose appearance has been shifted towards the styles we expect to see in
unseen target domains -- *without ever looking at target images*. The shift is
driven entirely by CLIP text prompts.

Method (following the user's specification)
-------------------------------------------
1. A source prompt ``p^s`` and target prompts ``p^t_j`` built from domain style
   and class label, e.g. ``"photo style image of dog"`` and
   ``"sketch style image of dog"``. ``V`` is the CLIP image encoder, ``T`` the
   text encoder.
2. For every source image ``x`` we want a SINGLE augmentation ``A`` whose
   semantic shift jointly reflects the differences between ``p^s`` and every
   ``p^t_j`` (one augmented image per source image, not one per prompt).
3. We embed the prompts ``q^s = T(p^s)`` and ``q^t_j = T(p^t_j)`` and form, for
   each prompt, a *target image embedding*

       z*_j = z + (q^t_j - q^s) / ||q^t_j - q^s||_2 ,    where z = V(x).

   (We work in the unit-norm CLIP space, so ``z`` is L2-normalized and a tunable
   ``SHIFT_SCALE`` scales the unit text-difference direction.)
4. We optimize a single additive augmentation ``A`` (in pixel space) so that the
   embedding of the augmented image ``z# = V(x + A)`` is simultaneously close
   (cosine distance) to all target embeddings ``z*_j``. The loss SUMS the risk
   over all prompts j, plus an L1 embedding-preservation term:

       L = sum_j  D(z*_j, z#)  +  lambda * || z# - z ||_1 ,
       D(a, b) = 1 - <a, b> / (||a|| ||b||)   (cosine distance).

   The L1 term keeps the augmented embedding near its initial value, preserving
   image content. Only source-domain images are used.
5. We train a classifier on the CLIP-augmented source images (only) and compare
   its domain-generalization accuracy on the *other* domains against a baseline
   classifier trained on the pure original source data.

This file is self-contained and reuses the repo's data pipeline
(``data.data_helper``), model (``models.resnet.resnet18``) and helpers. It does
not modify any existing code.

Requires the OpenAI CLIP package: ``pip install git+https://github.com/openai/CLIP.git``
"""

import copy
import hashlib
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are only saved to disk
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from data import data_helper
from data.JigsawLoader import _dataset_info, JigsawTestNewDataset
from utils.util import fix_all_seed
from train import get_args


# ---------------------------------------------------------------------------
# Configuration (edit here; this script intentionally avoids extra CLI flags so
# it stays drop-in compatible with ``train.get_args``).
# ---------------------------------------------------------------------------

# Domains available per task.
TASK_DOMAINS = {
    'PACS': ["art_painting", "sketch"],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'HOME': ['art', 'clip', 'product', 'real'],
}

# The single source domain we augment and train on.
SOURCE_DOMAIN = 'sketch'

# CLIP backbone used both as image encoder V and text encoder T.
CLIP_MODEL_NAME = "ViT-B/32"

# Per-domain names plugged into PROMPT_TEMPLATE as {domain}.
STYLE_PHRASES = {
    # PACS
    'photo': 'photo',
    'art_painting': 'art painting',
    'cartoon': 'cartoon',
    'sketch': 'sketch',
    # Office-Home
    'art': 'art',
    'clip': 'clipart',
    'product': 'product',
    'real': 'real-world',
    # VLCS
    'CALTECH': 'CALTECH',
    'LABELME': 'LABELME',
    'PASCAL': 'PASCAL',
    'SUN': 'SUN',
}
PROMPT_TEMPLATE = "{domain} style image of {label}"

# --- Augmentation-optimization hyper-parameters ---
N_OPT_STEPS = 30        # gradient steps per image-batch per prompt
AUG_LR = 0.02           # Adam lr for the additive perturbation A_j
LAMBDA_REG = 0.3        # weight on the L1 embedding-preservation term
SHIFT_SCALE = 1.0       # magnitude of the unit text-difference shift
MAX_AUG_PIXEL = 0.5    # clamp |A_j| to this (in [0,1] pixel units) to keep content
GEN_BATCH_SIZE = 32     # batch size while generating augmentations

# Cap the number of source images that get augmented (None = all). Useful for
# quick experiments; augmentation is the expensive part of the pipeline.
LIMIT_SOURCE_IMAGES = None

# Re-use cached augmentations on disk when the config matches.
CACHE_DIR = 'clip_aug_cache'
OUTPUT_DIR = 'clip_aug_figures'

# CLIP's image preprocessing normalization (kept separate from ImageNet stats
# used to train the ResNet classifier).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# ImageNet stats used by the repo's ResNet classifier.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# CLIP helpers (image encoder V, text encoder T).
# ---------------------------------------------------------------------------

def load_clip(device):
    """Load CLIP in fp32 (stable for gradient-based image optimization)."""
    try:
        import clip
    except ImportError as e:
        raise ImportError(
            "The OpenAI CLIP package is required. Install with:\n"
            "    pip install git+https://github.com/openai/CLIP.git"
        ) from e
    model, _ = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    input_res = model.visual.input_resolution
    return clip, model, input_res


def _normalize(t, mean, std):
    mean = torch.tensor(mean, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


def clip_encode_image(model, x01, input_res):
    """Encode a batch of [0,1] images with V (resizing + CLIP normalization).

    Gradients flow through ``x01`` (CLIP weights are frozen), so this can be used
    inside the augmentation optimization loop.
    """
    if x01.shape[-1] != input_res or x01.shape[-2] != input_res:
        x01 = F.interpolate(x01, size=(input_res, input_res), mode='bicubic',
                            align_corners=False)
    x = _normalize(x01.clamp(0, 1), CLIP_MEAN, CLIP_STD)
    return model.encode_image(x)


def format_prompt(domain, label_name):
    """Build a CLIP text prompt for ``domain`` and object class ``label_name``."""
    domain_phrase = STYLE_PHRASES.get(domain, domain.replace('_', ' '))
    return PROMPT_TEMPLATE.format(domain=domain_phrase, label=label_name)


def get_target_domains(task, source_domain):
    """Return target domain names (all task domains except the source)."""
    return [d for d in TASK_DOMAINS[task] if d != source_domain]


def precompute_text_directions(clip_mod, model, source_domain, target_domains,
                                 class_names, device):
    """Precompute unit text-difference directions for every (target, class) pair.

    For class ``c`` and target domain ``j``:

        d_{j,c} = normalize( T(p^t_{j,c}) - T(p^s_c) )

    where ``p^s_c = format_prompt(source, label_c)`` and
    ``p^t_{j,c} = format_prompt(target_j, label_c)``.

    Returns a tensor of shape ``(M, C, D)`` where M = len(target_domains),
    C = len(class_names).
    """
    c = len(class_names)
    src_prompts = [format_prompt(source_domain, class_names[i]) for i in range(c)]
    tgt_prompts = []
    for td in target_domains:
        for i in range(c):
            tgt_prompts.append(format_prompt(td, class_names[i]))

    with torch.no_grad():
        q_s = model.encode_text(clip_mod.tokenize(src_prompts).to(device)).float()
        q_t = model.encode_text(clip_mod.tokenize(tgt_prompts).to(device)).float()
        q_t = q_t.view(len(target_domains), c, -1)
        d = q_t - q_s.unsqueeze(0)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return d


def batch_text_directions(directions_mc, labels):
    """Index precomputed ``(M, C, D)`` directions by batch labels ``(N,)``.

    Returns ``(M, N, D)``.
    """
    return directions_mc[:, labels.long(), :]


# ---------------------------------------------------------------------------
# Step 4: per-image augmentation optimization.
# ---------------------------------------------------------------------------

def cosine_distance(a, b):
    """1 - cosine similarity, computed row-wise."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def optimize_augmentation(model, input_res, x01, directions, device):
    """Find ONE augmentation ``A`` per image so its CLIP embedding is jointly
    close to every prompt's shifted target embedding.

    ``directions`` has shape ``(M, N, D)``: per-image text shift directions
    (one direction per target prompt and per batch image, derived from that
    image's class label).

    Returns a tensor of shape ``(N, 3, H, W)`` of augmented [0,1] images.
    """
    with torch.no_grad():
        z = clip_encode_image(model, x01, input_res)            # (N, D)
        z_hat = F.normalize(z, dim=-1)                          # (N, D)
        # Per-prompt, per-image shifted target embeddings: (M, N, D), fixed.
        z_star = F.normalize(
            z_hat.unsqueeze(0) + SHIFT_SCALE * directions, dim=-1)

    a = torch.zeros_like(x01, requires_grad=True)               # one A per image
    opt = torch.optim.Adam([a], lr=AUG_LR)
    for _ in range(N_OPT_STEPS):
        opt.zero_grad()
        x_aug = (x01 + a).clamp(0, 1)
        z_sharp = clip_encode_image(model, x_aug, input_res)
        z_sharp_n = F.normalize(z_sharp, dim=-1)                # (N, D)

        # Sum the cosine-distance risk over all prompts j (mean over batch).
        sim = cosine_distance(z_star, z_sharp_n.unsqueeze(0))   # (M, N)
        sim_loss = sim.sum(dim=0).mean()
        # L1 embedding-preservation term: keep z# near the initial z.
        reg = (z_sharp_n - z_hat).abs().sum(dim=-1).mean()
        loss = sim_loss + LAMBDA_REG * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-MAX_AUG_PIXEL, MAX_AUG_PIXEL)

    with torch.no_grad():
        x_aug = (x01 + a).clamp(0, 1)
    return x_aug.detach().cpu()                                 # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Step 3: raw [0,1] source loader + augmented-dataset generation.
# ---------------------------------------------------------------------------

def _raw_transform(args):
    return transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # -> [0,1], no normalization
    ])


def get_raw_source_dataset(args, domain):
    """Deterministic dataset of (image[0,1], 0, label) for the source domain.

    Works for both the HuggingFace PACS backend and the local-file backend, and
    returns the *training* split for the given domain.
    """
    transform = _raw_transform(args)
    if data_helper.use_hf_backend(args):
        split, idx_by_domain = data_helper._load_hf_pacs()
        idx = data_helper._hf_domain_indices(idx_by_domain, domain)
        train_idx, _ = data_helper._hf_split_train_val(idx, args.val_size, args.seed)
        return data_helper.HFImageDataset(split, train_idx, transform)

    join = os.path.join
    base = os.path.join(os.path.dirname(data_helper.__file__), 'correct_txt_lists')
    if args.task == 'PACS':
        names, labels = _dataset_info(join(base, '%s_train_kfold.txt' % domain))
    elif args.task == 'VLCS':
        names, labels = _dataset_info(join(base, '%s_train.txt' % domain))
    elif args.task == 'HOME':
        names, labels = _dataset_info(join(base, '%s_full.txt' % domain))
    else:
        raise NotImplementedError("Unsupported task: %s" % args.task)
    return JigsawTestNewDataset(args, names, labels, img_transformer=transform,
                                patches=False, jig_classes=30)


def _config_signature(args, source, target_domains, class_names):
    raw = "|".join([
        "v3-label-prompts",
        PROMPT_TEMPLATE, CLIP_MODEL_NAME, args.task, source, str(args.image_size),
        str(SHIFT_SCALE), str(N_OPT_STEPS), str(AUG_LR), str(LAMBDA_REG),
        str(MAX_AUG_PIXEL), str(LIMIT_SOURCE_IMAGES),
        "::".join(target_domains), "::".join(class_names),
    ])
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_augmented_source(args, source, device):
    """Build / load the augmented source set.

    Returns ``(orig_u8, labels, aug_u8, target_names)`` where:
      * ``orig_u8``  : (N, 3, H, W) uint8 original source images,
      * ``labels``   : (N,) long labels,
      * ``aug_u8``   : (N, 3, H, W) uint8 CLIP-augmented images (one per source
        image; prompts include both domain style and class label),
      * ``target_names`` : list of M target domain names.
    Augmentations are cached on disk keyed by the config signature.
    """
    class_names = get_class_names(args)
    if class_names is None:
        class_names = [str(i) for i in range(args.n_classes)]
    target_domains = get_target_domains(args.task, source)

    os.makedirs(CACHE_DIR, exist_ok=True)
    sig = _config_signature(args, source, target_domains, class_names)
    cache_path = os.path.join(CACHE_DIR, "aug_%s_%s.pt" % (source, sig))
    if os.path.exists(cache_path):
        print("Loading cached augmentations: %s" % cache_path)
        blob = torch.load(cache_path, map_location='cpu')
        return blob['orig'], blob['labels'], blob['aug'], blob['names']

    print("Prompt template: %s" % PROMPT_TEMPLATE)
    print("Example source p^s : %s" % format_prompt(source, class_names[0]))
    for d in target_domains:
        print("Example target [%s] : %s" % (d, format_prompt(d, class_names[0])))

    clip_mod, model, input_res = load_clip(device)
    directions_mc = precompute_text_directions(
        clip_mod, model, source, target_domains, class_names, device)

    dataset = get_raw_source_dataset(args, source)
    if LIMIT_SOURCE_IMAGES is not None and len(dataset) > LIMIT_SOURCE_IMAGES:
        idx = torch.randperm(len(dataset))[:LIMIT_SOURCE_IMAGES].tolist()
        dataset = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=GEN_BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    orig_chunks, label_chunks, aug_chunks = [], [], []
    n_done = 0
    n_total = len(dataset)
    for (data, _, class_l) in loader:
        x01 = data.to(device)
        labels_dev = class_l.to(device)
        directions = batch_text_directions(directions_mc, labels_dev)
        aug = optimize_augmentation(model, input_res, x01, directions, device)
        orig_chunks.append((x01.detach().cpu() * 255).round().to(torch.uint8))
        label_chunks.append(class_l.clone())
        aug_chunks.append((aug * 255).round().to(torch.uint8))
        n_done += x01.size(0)
        print("  augmented %d/%d images" % (n_done, n_total))

    orig_u8 = torch.cat(orig_chunks, dim=0)                 # (N,3,H,W)
    labels = torch.cat(label_chunks, dim=0)                 # (N,)
    aug_u8 = torch.cat(aug_chunks, dim=0)                   # (N,3,H,W)

    blob = {'orig': orig_u8, 'labels': labels, 'aug': aug_u8, 'names': target_domains}
    torch.save(blob, cache_path)
    print("Saved augmentations to %s" % cache_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return orig_u8, labels, aug_u8, target_domains


# ---------------------------------------------------------------------------
# Step 5: classifier training + target-domain evaluation.
# ---------------------------------------------------------------------------

def make_resnet(args, device):
    from models.resnet import resnet18
    return resnet18(classes=args.n_classes).to(device)


def standard_augment(x01, args):
    """Generic geometric augmentation (zoom/crop, translate, flip) on [0,1]."""
    n, c, h, w = x01.shape
    device = x01.device
    scale = torch.empty(n, device=device).uniform_(args.min_scale, args.max_scale)
    flip = torch.where(torch.rand(n, device=device) < args.random_horiz_flip,
                       -torch.ones(n, device=device), torch.ones(n, device=device))
    tx = (torch.rand(n, device=device) * 2 - 1) * (1 - scale)
    ty = (torch.rand(n, device=device) * 2 - 1) * (1 - scale)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = scale * flip
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x01.size(), align_corners=False)
    return F.grid_sample(x01, grid, align_corners=False, padding_mode='reflection')


def train_classifier(name, images_u8, labels, args, device, epochs):
    """Train a fresh ResNet-18 on uint8 [0..255] images with plain CE.

    Online: cast to [0,1] -> geometric augment -> ImageNet-normalize -> forward.
    """
    model = make_resnet(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                nesterov=True, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(images_u8, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    print("  [%s] training on %d images for %d epochs" % (name, len(ds), epochs))
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device).float() / 255.0
            yb = yb.to(device)
            xb = standard_augment(xb, args)
            xb = _normalize(xb, IMAGENET_MEAN, IMAGENET_STD)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        print("    [%s] epoch %2d/%d  loss = %.4f"
              % (name, epoch + 1, epochs, running / max(len(loader), 1)))
    return model


def get_eval_loader(args, domain):
    """Normalized (ImageNet) test loader for a domain, via the repo pipeline."""
    saved = args.target
    try:
        args.target = [domain]
        loader = data_helper.get_val_dataloader(args, patches=False)
    finally:
        args.target = saved
    return loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ((data, _, class_l), _, _) in loader:
            data, class_l = data.to(device), class_l.to(device)
            logits = model(data, train=False)[0]
            _, pred = logits.max(dim=1)
            correct += torch.sum(pred == class_l.data).item()
            total += class_l.size(0)
    return float(correct) / max(total, 1)


def evaluate_pair(model_aug, model_base, loader, device):
    """Evaluate both models on the same loader in a single pass.

    Returns ``(acc_aug, acc_base, aug_win, base_win)`` where:
      * ``aug_win``  : samples the augmented model got right but baseline wrong,
      * ``base_win`` : samples the baseline got right but augmented wrong.
    Each sample is a tuple ``(img_norm_cpu, true_label, pred_aug, pred_base)``;
    ``img_norm_cpu`` is the ImageNet-normalized input tensor.
    """
    model_aug.eval()
    model_base.eval()
    correct_a = correct_b = total = 0
    aug_win, base_win = [], []
    with torch.no_grad():
        for ((data, _, class_l), _, _) in loader:
            data, class_l = data.to(device), class_l.to(device)
            pred_a = model_aug(data, train=False)[0].argmax(dim=1)
            pred_b = model_base(data, train=False)[0].argmax(dim=1)
            ca = pred_a == class_l
            cb = pred_b == class_l
            correct_a += ca.sum().item()
            correct_b += cb.sum().item()
            total += class_l.size(0)
            for i in range(data.size(0)):
                if ca[i] and not cb[i]:
                    aug_win.append((data[i].cpu(), int(class_l[i]),
                                    int(pred_a[i]), int(pred_b[i])))
                elif cb[i] and not ca[i]:
                    base_win.append((data[i].cpu(), int(class_l[i]),
                                     int(pred_a[i]), int(pred_b[i])))
    return (float(correct_a) / max(total, 1), float(correct_b) / max(total, 1),
            aug_win, base_win)


def get_class_names(args):
    """Best-effort list of class names; falls back to None (use indices)."""
    try:
        if data_helper.use_hf_backend(args):
            split, _ = data_helper._load_hf_pacs()
            feat = split.features.get('label')
            if hasattr(feat, 'names'):
                return list(feat.names)
    except Exception:
        pass
    if args.task == 'PACS':
        # flwrlabs/pacs alphabetical label order.
        return ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    return None


def _denorm_to_uint8(img_norm):
    """Undo ImageNet normalization -> uint8 [0,255] image for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img_norm * std + mean).clamp(0, 1)
    return (x * 255).round().to(torch.uint8)


# ---------------------------------------------------------------------------
# Visualization helpers.
# ---------------------------------------------------------------------------

def save_aug_grid(orig_u8, aug_u8, target_names, out_path, n_show=6):
    """Save a grid: rows = sample images, cols = [original, CLIP-augmented]."""
    n_show = min(n_show, orig_u8.size(0))
    fig, axes = plt.subplots(n_show, 2, figsize=(4.0, 2.0 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    aug_title = "augmented (-> %s)" % ", ".join(target_names)
    for r in range(n_show):
        axes[r, 0].imshow(orig_u8[r].permute(1, 2, 0).numpy())
        axes[r, 0].set_axis_off()
        axes[r, 1].imshow(aug_u8[r].permute(1, 2, 0).numpy())
        axes[r, 1].set_axis_off()
        if r == 0:
            axes[r, 0].set_title("original", fontsize=9)
            axes[r, 1].set_title(aug_title, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print("Saved augmentation preview: %s" % out_path)


def save_disagreement_grid(samples, class_names, title, out_path, cols=6):
    """Save a grid of ALL disagreement samples with true / aug / base labels.

    ``samples`` is a list of ``(img_norm, true_label, pred_aug, pred_base)``.
    Every sample is displayed (no subsampling); the grid grows as many rows as
    needed to fit ``len(samples)`` images.
    """
    if len(samples) == 0:
        print("  (no samples for: %s)" % title)
        return

    def name(idx):
        if class_names is not None and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)

    n = len(samples)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.3 * rows),
                             squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        ax.set_axis_off()
        if k < n:
            img, true_l, pa, pb = samples[k]
            ax.imshow(_denorm_to_uint8(img).permute(1, 2, 0).numpy())
            ax.set_title("true: %s\naug: %s | base: %s"
                         % (name(true_l), name(pa), name(pb)), fontsize=8)
    fig.suptitle("%s  (all %d)" % (title, n), fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=140)
    plt.close()
    print("  Saved %d samples: %s" % (n, out_path))


def plot_accuracy_comparison(source, results_aug, results_base, out_path):
    domains = list(results_aug.keys())
    x = range(len(domains))
    width = 0.38
    plt.figure(figsize=(max(7, 1.6 * len(domains)), 5))
    plt.bar([i - width / 2 for i in x], [100 * results_aug[d] for d in domains],
            width, label='CLIP-augmented')
    plt.bar([i + width / 2 for i in x], [100 * results_base[d] for d in domains],
            width, label='Baseline (orig only)')
    plt.xticks(list(x), [d.capitalize() for d in domains])
    plt.ylabel('Target accuracy (%)')
    plt.title('Single-source DG from %s' % source.capitalize())
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved comparison figure: %s" % out_path)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def setup_args():
    args = get_args()
    if args.task == 'PACS':
        args.n_classes = 7
    elif args.task == 'VLCS':
        args.n_classes = 5
    elif args.task == 'HOME':
        args.n_classes = 65
    else:
        raise NotImplementedError("Unsupported task: %s" % args.task)

    domains = TASK_DOMAINS[args.task]
    source = SOURCE_DOMAIN if SOURCE_DOMAIN in domains else domains[-1]
    args.source = [source]
    args.target = [d for d in domains if d != source]
    return args, domains, source


def main():
    args, domains, source = setup_args()
    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    targets = [d for d in domains if d != source]
    print("Task: %s | Source: %s | Targets: %s" % (args.task, source, targets))

    # --- Steps 1-4: generate CLIP-guided augmentations of the source domain. ---
    print("\n==== Generating CLIP-guided augmentations for source '%s' ====" % source)
    orig_u8, labels, aug_u8, target_names = generate_augmented_source(args, source, device)
    print("Original images: %s | Augmented: %s (loss summed over %d prompts: %s)"
          % (tuple(orig_u8.shape), tuple(aug_u8.shape), len(target_names),
             ", ".join(target_names)))

    save_aug_grid(orig_u8, aug_u8, target_names,
                  os.path.join(OUTPUT_DIR, "aug_preview_%s.png" % source))

    # Build training tensors.
    #   Augmented model: one CLIP-augmented image per source image (no originals).
    #   Baseline model : original only.
    aug_images = aug_u8
    aug_all_labels = labels

    # --- Step 5: train both classifiers. ---
    print("\n==== Training CLIP-augmented classifier ====")
    model_aug = train_classifier("augmented", aug_images, aug_all_labels,
                                 args, device, args.epochs)
    print("\n==== Training baseline classifier (pure source) ====")
    model_base = train_classifier("baseline", orig_u8, labels,
                                  args, device, args.epochs)

    # --- Evaluate on every other domain (and source test for reference). ---
    print("\n==== Evaluating domain-generalization performance ====")
    class_names = get_class_names(args)
    results_aug, results_base = {}, {}
    for d in targets:
        loader = get_eval_loader(args, d)
        a, b, aug_win, base_win = evaluate_pair(model_aug, model_base, loader, device)
        results_aug[d], results_base[d] = a, b
        print("  target %-14s  augmented = %.2f%%   baseline = %.2f%%   (%+.2f)"
              % (d, 100 * a, 100 * b, 100 * (a - b)))
        print("    disagreements: aug-correct/base-wrong = %d, base-correct/aug-wrong = %d"
              % (len(aug_win), len(base_win)))
        save_disagreement_grid(
            aug_win, class_names,
            "%s: augmented CORRECT, baseline WRONG" % d.capitalize(),
            os.path.join(OUTPUT_DIR, "disagree_aug_correct_%s_to_%s.png" % (source, d)))
        save_disagreement_grid(
            base_win, class_names,
            "%s: baseline CORRECT, augmented WRONG" % d.capitalize(),
            os.path.join(OUTPUT_DIR, "disagree_base_correct_%s_to_%s.png" % (source, d)))

    plot_accuracy_comparison(source, results_aug, results_base,
                             os.path.join(OUTPUT_DIR, "dg_comparison_%s.png" % source))

    # --- Summary. ---
    avg_aug = sum(results_aug.values()) / len(results_aug)
    avg_base = sum(results_base.values()) / len(results_base)
    print("\n==== Summary (single-source DG from %s) ====" % source)
    print("%-16s %12s %12s   %s" % ("target", "augmented", "baseline", "winner"))
    for d in targets:
        winner = "augmented" if results_aug[d] > results_base[d] else "baseline"
        print("%-16s %11.2f%% %11.2f%%   %s"
              % (d, 100 * results_aug[d], 100 * results_base[d], winner))
    print("%-16s %11.2f%% %11.2f%%   %s"
          % ("AVERAGE", 100 * avg_aug, 100 * avg_base,
             "augmented" if avg_aug > avg_base else "baseline"))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
