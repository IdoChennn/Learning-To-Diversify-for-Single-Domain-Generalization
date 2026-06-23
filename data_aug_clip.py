"""CLIP-guided single-source data augmentation for domain generalization.

Idea
----
Given a *single* source domain, we synthesize augmented copies of every source
image whose appearance has been shifted towards the styles we expect to see in
unseen target domains -- *without ever looking at target images*. The shift is
driven entirely by CLIP text prompts.

Method (following the user's specification)
-------------------------------------------
1. A generic source prompt ``p^s`` (e.g. "an image in the style of a photo") and
   a set of ``M`` target prompts ``P^t = {p^t_j}`` describing variations expected
   in different target domains (e.g. sketch / cartoon / art styles). ``V`` is the
   CLIP image encoder, ``T`` the CLIP text encoder.
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
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'HOME': ['art', 'clip', 'product', 'real'],
}

# The single source domain we augment and train on.
SOURCE_DOMAIN = 'photo'

# CLIP backbone used both as image encoder V and text encoder T.
CLIP_MODEL_NAME = "ViT-B/32"

# Per-domain *style phrases* plugged into PROMPT_TEMPLATE to build p^s / p^t_j.
STYLE_PHRASES = {
    # PACS
    'photo': 'a photo',
    'art_painting': 'an art painting',
    'cartoon': 'a cartoon',
    'sketch': 'a sketch',
    # Office-Home
    'art': 'an artistic painting',
    'clip': 'a clipart image',
    'product': 'a product image on a white background',
    'real': 'a real-world photo',
    # VLCS domains are datasets rather than visual styles; we fall back to a
    # neutral phrasing so the direction is at least well-defined.
    'CALTECH': 'a clean object-centric photo',
    'LABELME': 'a cluttered scene photo',
    'PASCAL': 'a natural scene photo',
    'SUN': 'a wide-angle scene photo',
}
PROMPT_TEMPLATE = "an image in the style of {}"

# Optional extra target prompts (generic nuisance variations) appended to the
# per-domain style prompts. Set to [] to disable.
EXTRA_TARGET_PROMPTS = []

# --- Augmentation-optimization hyper-parameters ---
N_OPT_STEPS = 30        # gradient steps per image-batch per prompt
AUG_LR = 0.02           # Adam lr for the additive perturbation A_j
LAMBDA_REG = 0.2        # weight on the L1 embedding-preservation term
SHIFT_SCALE = 1.0       # magnitude of the unit text-difference shift
MAX_AUG_PIXEL = 0.25    # clamp |A_j| to this (in [0,1] pixel units) to keep content
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


def build_prompts(task, source_domain):
    """Return ``(source_prompt, target_prompts, target_names)``.

    Target prompts = style prompts for every *other* domain of the task, plus any
    EXTRA_TARGET_PROMPTS.
    """
    def style_prompt(domain):
        phrase = STYLE_PHRASES.get(domain, domain.replace('_', ' '))
        return PROMPT_TEMPLATE.format(phrase)

    source_prompt = style_prompt(source_domain)
    target_prompts, target_names = [], []
    for d in TASK_DOMAINS[task]:
        if d == source_domain:
            continue
        target_prompts.append(style_prompt(d))
        target_names.append(d)
    for i, p in enumerate(EXTRA_TARGET_PROMPTS):
        target_prompts.append(p)
        target_names.append("extra%d" % i)
    return source_prompt, target_prompts, target_names


def text_directions(clip_mod, model, source_prompt, target_prompts, device):
    """Compute unit text-difference directions  d_j = (q^t_j - q^s)/||.||_2.

    Returns a tensor of shape ``(M, D)``.
    """
    tokens = clip_mod.tokenize([source_prompt] + list(target_prompts)).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float()
    q_s = feats[0:1]                 # (1, D)
    q_t = feats[1:]                  # (M, D)
    d = q_t - q_s                    # (M, D)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return d


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

    A single additive perturbation ``A`` is optimized for each source image. Its
    augmented embedding ``z# = V(x + A)`` is pulled towards all ``M`` target
    embeddings ``z*_j`` at once via a loss summed over the prompts:

        L = sum_j  D(z*_j, z#)  +  lambda * || z# - z ||_1 .

    Returns a tensor of shape ``(N, 3, H, W)`` of augmented [0,1] images (one per
    source image, detached on CPU).
    """
    with torch.no_grad():
        z = clip_encode_image(model, x01, input_res)            # (N, D)
        z_hat = F.normalize(z, dim=-1)                          # (N, D)
        # Per-prompt shifted target embeddings: (M, N, D), fixed.
        z_star = F.normalize(
            z_hat.unsqueeze(0) + SHIFT_SCALE * directions.unsqueeze(1), dim=-1)

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


def _config_signature(args, source, target_prompts):
    raw = "|".join([
        "v2-unified",  # single augmented image per source, loss summed over prompts
        CLIP_MODEL_NAME, args.task, source, str(args.image_size),
        str(SHIFT_SCALE), str(N_OPT_STEPS), str(AUG_LR), str(LAMBDA_REG),
        str(MAX_AUG_PIXEL), str(LIMIT_SOURCE_IMAGES), "::".join(target_prompts),
    ])
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_augmented_source(args, source, device):
    """Build / load the augmented source set.

    Returns ``(orig_u8, labels, aug_u8, target_names)`` where:
      * ``orig_u8``  : (N, 3, H, W) uint8 original source images,
      * ``labels``   : (N,) long labels,
      * ``aug_u8``   : (N, 3, H, W) uint8 CLIP-augmented images (one per source
        image; its embedding jointly targets all M prompts),
      * ``target_names`` : list of M prompt names.
    Augmentations are cached on disk keyed by the config signature.
    """
    source_prompt, target_prompts, target_names = build_prompts(args.task, source)

    os.makedirs(CACHE_DIR, exist_ok=True)
    sig = _config_signature(args, source, target_prompts)
    cache_path = os.path.join(CACHE_DIR, "aug_%s_%s.pt" % (source, sig))
    if os.path.exists(cache_path):
        print("Loading cached augmentations: %s" % cache_path)
        blob = torch.load(cache_path, map_location='cpu')
        return blob['orig'], blob['labels'], blob['aug'], blob['names']

    print("Source prompt p^s : %s" % source_prompt)
    for n, p in zip(target_names, target_prompts):
        print("Target prompt [%s] : %s" % (n, p))

    clip_mod, model, input_res = load_clip(device)
    directions = text_directions(clip_mod, model, source_prompt, target_prompts, device)

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
        aug = optimize_augmentation(model, input_res, x01, directions, device)  # (B,3,H,W)
        orig_chunks.append((x01.detach().cpu() * 255).round().to(torch.uint8))
        label_chunks.append(class_l.clone())
        aug_chunks.append((aug * 255).round().to(torch.uint8))
        n_done += x01.size(0)
        print("  augmented %d/%d images" % (n_done, n_total))

    orig_u8 = torch.cat(orig_chunks, dim=0)                 # (N,3,H,W)
    labels = torch.cat(label_chunks, dim=0)                 # (N,)
    aug_u8 = torch.cat(aug_chunks, dim=0)                   # (N,3,H,W)

    blob = {'orig': orig_u8, 'labels': labels, 'aug': aug_u8, 'names': target_names}
    torch.save(blob, cache_path)
    print("Saved augmentations to %s" % cache_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return orig_u8, labels, aug_u8, target_names


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
    results_aug, results_base = {}, {}
    for d in targets:
        loader = get_eval_loader(args, d)
        a = evaluate(model_aug, loader, device)
        b = evaluate(model_base, loader, device)
        results_aug[d], results_base[d] = a, b
        print("  target %-14s  augmented = %.2f%%   baseline = %.2f%%   (%+.2f)"
              % (d, 100 * a, 100 * b, 100 * (a - b)))

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
