"""Combined comparison: ONE benign baseline vs TWO backdoored models.

This file unifies ``updates_amount.py`` (target-domain poisoning) and
``updates_amount_reverse.py`` (source-domain poisoning). As noted, the *benign*
model in both scripts is identical -- it is trained on the clean source domain
and then adapted to the clean target domain. The only thing that differed
between the two scripts was *how* the malicious model was built. We therefore
keep a single benign baseline and compare it against BOTH poisoning strategies
on the same figure (three curves per panel).

Three models
------------
1. benign            : clean source  ->  clean target.
2. malicious-source  : POISONED source (BadNets)  ->  clean target.
                       (the ``updates_amount_reverse.py`` scenario)
3. malicious-target  : clean source  ->  POISONED target (BadNets).
                       (the ``updates_amount.py`` scenario)

Shared initialisation
----------------------
* The clean source model (``theta_clean``) and the poisoned source model
  (``theta_psrc``) are trained together from an *identical* initialisation on
  the source domain, consuming the same aligned batch each step (the only
  per-position difference is the trigger patch + relabel). ``theta_clean`` is
  the source weight for both the benign and the malicious-target model;
  ``theta_psrc`` is the source weight for the malicious-source model.

Target-domain training
----------------------
* All three models consume the SAME batch positions each step over the target
  domain via one aligned dataset. The benign and malicious-source models see the
  identical clean view (img_b, label_b); the malicious-target model sees the
  trigger-stamped/relabelled view (img_m, label_m) at poisoned positions.
  Consequently:
    - benign vs malicious-target differ only by the target trigger,
    - benign vs malicious-source differ only by the source initialisation.

Per epoch we record, for each model, how far its parameters move relative to its
OWN source weights:
    * step-wise update length  : sum over the epoch of ||theta_t+1 - theta_t||_2
    * epoch displacement        : ||theta_epoch_end - theta_epoch_start||_2
    * drift from source weights : ||theta_epoch_end - theta_S||_2

Self-contained; reuses the repo's data pipeline (``data.data_helper``), model
(``models.resnet.resnet18``) and ``train.get_args``. Does not modify existing
code.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are only saved to disk
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data import data_helper
from data.JigsawLoader import _dataset_info, JigsawTestNewDataset
from utils.util import fix_all_seed
from train import get_args


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

TASK_DOMAINS = {
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'HOME': ['art', 'clip', 'product', 'real'],
}

# Source domain used to pre-train the shared weights.
SOURCE_DOMAIN = 'photo'
# Target domains to transfer to. None = every task domain except the source;
# otherwise a list of domain names to restrict to.
TARGET_DOMAINS = None

# When True, also fine-tune on the SOURCE domain itself (source == target) as a
# same-domain reference point, giving one extra results figure for it.
INCLUDE_SOURCE_AS_TARGET = True

# Training length (epochs). Fall back to args.epochs if these are None.
SOURCE_EPOCHS = 20
TARGET_EPOCHS = 20

# --- BadNets poisoning hyper-parameters ---
SOURCE_POISON_RATE = 0.1    # fraction of SOURCE-train images poisoned (mal-source)
TARGET_POISON_RATE = 0.25   # fraction of TARGET-train images poisoned (mal-target)
TARGET_LABEL = 0            # label all poisoned samples are forced to
TRIGGER_SIZE = 24           # side length (pixels) of the square trigger patch
TRIGGER_VALUE = 1.0         # patch pixel value in [0,1] space (white)
TRIGGER_POS = 'br'          # corner: 'br','bl','tr','tl'

# Cap the number of training images per phase (None = all). Useful for quick runs.
LIMIT_TRAIN_IMAGES = None
# Number of images used to measure clean acc / attack success rate per domain.
EVAL_LIMIT = 1000

# Match the benign label distribution to the malicious one (within each
# poisoning phase) so the ONLY difference is the trigger patch. See the original
# scripts' docstrings for the rationale.
FAIR_DISTRIBUTION = True

OUTPUT_DIR = 'updates_figures'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Raw [0,1] per-domain datasets + helpers.
# ---------------------------------------------------------------------------

def _raw_transform(args):
    return transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # -> [0,1]
    ])


def get_raw_train_dataset(args, domain):
    """Deterministic dataset of (image[0,1], 0, label) for ``domain``'s train split."""
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


class IndexedDataset(Dataset):
    """Wrap a (img, 0, label) dataset to yield (img, label, index)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _, label = self.base[i]
        return img, int(label), int(i)


def _maybe_limit(dataset, limit, seed):
    if limit is not None and len(dataset) > limit:
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(dataset), generator=g)[:limit].tolist()
        return torch.utils.data.Subset(dataset, idx)
    return dataset


def _normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# BadNets poisoning + fair (distribution-matched) aligned dataset.
# ---------------------------------------------------------------------------

def _trigger_slice(h, w):
    s = TRIGGER_SIZE
    if TRIGGER_POS == 'br':
        return slice(h - s, h), slice(w - s, w)
    if TRIGGER_POS == 'bl':
        return slice(h - s, h), slice(0, s)
    if TRIGGER_POS == 'tr':
        return slice(0, s), slice(w - s, w)
    if TRIGGER_POS == 'tl':
        return slice(0, s), slice(0, s)
    raise ValueError("Unknown TRIGGER_POS: %s" % TRIGGER_POS)


def stamp_trigger(x01):
    """Stamp the BadNets trigger patch onto a batch of [0,1] images (in place)."""
    h, w = x01.shape[-2], x01.shape[-1]
    ys, xs = _trigger_slice(h, w)
    x01[:, :, ys, xs] = TRIGGER_VALUE
    return x01


def build_poison_set(n, rate, seed, salt=0):
    g = torch.Generator().manual_seed(int(seed) + 12345 + int(salt))
    n_poison = int(round(n * rate))
    idx = torch.randperm(n, generator=g)[:n_poison].tolist()
    return set(int(i) for i in idx), n_poison


def _dataset_labels(dataset):
    """Cheaply extract 0-indexed labels for every position of a raw dataset."""
    if isinstance(dataset, torch.utils.data.Subset):
        inner = _dataset_labels(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    if isinstance(dataset, data_helper.HFImageDataset):
        col = dataset.hf_split['label']  # column access: no image decode
        return [int(col[i]) for i in dataset.indices]
    if hasattr(dataset, 'labels'):
        if getattr(dataset, 'task', None) == 'PACS':
            return [int(l) - 1 for l in dataset.labels]
        return [int(l) for l in dataset.labels]
    raise TypeError("Cannot extract labels from dataset of type %s" % type(dataset))


class FairCompareDataset(Dataset):
    """Aligned benign / malicious views for a fair update-magnitude comparison.

    For each position ``p`` returns ``(img_b01, label_b, img_m01, label_m,
    poison_flag)`` where the malicious view has the trigger stamped + label
    ``TARGET_LABEL`` iff poisoned, and (when ``fair``) the benign view replaces
    flipped-poison positions with a randomly duplicated clean target-class image
    so both views share an identical per-position label sequence.
    """

    def __init__(self, base, labels, poison_set, target_label, fair, seed):
        self.base = base
        self.labels = labels
        self.n = len(base)
        self.poison_set = poison_set
        self.target_label = target_label
        self.flipped_set = set(i for i in poison_set if labels[i] != target_label)
        self.target_pool = [i for i in range(self.n) if labels[i] == target_label]
        self.fair = bool(fair) and len(self.flipped_set) > 0

        self.dup_index = {}
        if self.fair:
            if len(self.target_pool) == 0:
                print("  [warn] no target-class samples to duplicate; "
                      "falling back to non-fair benign training.")
                self.fair = False
            else:
                g = torch.Generator().manual_seed(int(seed) + 999)
                flipped_sorted = sorted(self.flipped_set)
                choice = torch.randint(len(self.target_pool),
                                       (len(flipped_sorted),), generator=g).tolist()
                for k, p in enumerate(flipped_sorted):
                    self.dup_index[p] = self.target_pool[choice[k]]

    def __len__(self):
        return self.n

    def benign_labels(self):
        out = list(self.labels)
        if self.fair:
            for p in self.flipped_set:
                out[p] = self.target_label
        return out

    def malicious_labels(self):
        out = list(self.labels)
        for p in self.poison_set:
            out[p] = self.target_label
        return out

    def _stamp(self, img):
        ys, xs = _trigger_slice(img.shape[-2], img.shape[-1])
        out = img.clone()
        out[:, ys, xs] = TRIGGER_VALUE
        return out

    def __getitem__(self, p):
        img, _, lab = self.base[p]
        poison = p in self.poison_set

        if poison:
            img_m = self._stamp(img)
            lab_m = self.target_label
        else:
            img_m = img
            lab_m = lab

        if self.fair and p in self.flipped_set:
            img_b, _, _ = self.base[self.dup_index[p]]
            lab_b = self.target_label
        else:
            img_b = img
            lab_b = lab

        return img_b, int(lab_b), img_m, int(lab_m), int(poison)


# ---------------------------------------------------------------------------
# Model + training utilities.
# ---------------------------------------------------------------------------

def make_resnet(args, device):
    from models.resnet import resnet18
    return resnet18(classes=args.n_classes).to(device)


def make_optimizer(model, args):
    return torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                           nesterov=True, momentum=0.9, weight_decay=0.0005)


def flat_params(model):
    """Flatten all trainable parameters into a single 1-D tensor (detached)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_source_compare(args, device, epochs):
    """Train a clean source model (theta_clean) and a poisoned source model
    (theta_psrc) from an identical initialization on the SOURCE domain.

    Both consume the same aligned batch each step; the only per-position
    difference is the trigger patch on the poisoned model's images. Returns
    ``(clean_state, psrc_state)``.
    """
    fix_all_seed(args.seed)
    clean = make_resnet(args, device)
    psrc = make_resnet(args, device)
    init_state = copy.deepcopy(clean.state_dict())
    psrc.load_state_dict(init_state)  # identical starting weights

    opt_c = make_optimizer(clean, args)
    opt_p = make_optimizer(psrc, args)
    sch_c = torch.optim.lr_scheduler.StepLR(opt_c, step_size=max(1, int(epochs * 0.8)))
    sch_p = torch.optim.lr_scheduler.StepLR(opt_p, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, SOURCE_DOMAIN)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    n = len(base)
    labels_all = _dataset_labels(base)
    poison_set, n_poison = build_poison_set(n, SOURCE_POISON_RATE, args.seed, salt=0)
    ds = FairCompareDataset(base, labels_all, poison_set, TARGET_LABEL,
                            FAIR_DISTRIBUTION, args.seed)

    n_flipped = len(ds.flipped_set)
    print("  [source] %d images | poison %d (%.1f%%) -> label %d | flipped %d "
          "| target-class pool %d"
          % (n, n_poison, 100.0 * n_poison / max(n, 1), TARGET_LABEL,
             n_flipped, len(ds.target_pool)))
    if ds.fair:
        identical = (ds.benign_labels() == ds.malicious_labels())
        print("  [fair] clean-source drops %d flipped samples and adds %d "
              "duplicated clean target-class images; per-position labels "
              "identical: %s" % (n_flipped, n_flipped, identical))
    else:
        print("  [fair] DISABLED: clean-source trains on all-clean data "
              "(label distributions differ from poisoned-source).")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    print("  [source] training clean + poisoned source models for %d epochs" % epochs)
    for epoch in range(epochs):
        clean.train()
        psrc.train()
        loss_c_sum = loss_p_sum = 0.0
        for img_b, lab_b, img_m, lab_m, _ in loader:
            img_b = img_b.to(device); lab_b = lab_b.to(device)
            img_m = img_m.to(device); lab_m = lab_m.to(device)

            opt_c.zero_grad()
            logits_c, _ = clean(_normalize(img_b))
            loss_c = criterion(logits_c, lab_b)
            loss_c.backward()
            opt_c.step()
            loss_c_sum += loss_c.item()

            opt_p.zero_grad()
            logits_p, _ = psrc(_normalize(img_m))
            loss_p = criterion(logits_p, lab_m)
            loss_p.backward()
            opt_p.step()
            loss_p_sum += loss_p.item()
        sch_c.step()
        sch_p.step()
        nb = max(len(loader), 1)
        print("    [source] epoch %2d/%d  clean_loss = %.4f  poisoned_loss = %.4f"
              % (epoch + 1, epochs, loss_c_sum / nb, loss_p_sum / nb))

    return (copy.deepcopy(clean.state_dict()),
            copy.deepcopy(psrc.state_dict()))


def train_target_compare(args, device, clean_state, psrc_state, epochs,
                         target_domain):
    """Fine-tune three models on ``target_domain`` from their source weights.

    * benign           : init ``clean_state``, trains on the CLEAN target view.
    * malicious-source : init ``psrc_state`` (backdoored source), trains on the
      SAME clean target view as benign.
    * malicious-target : init ``clean_state``, trains on the trigger-stamped /
      relabelled target view (BadNets on the target domain).

    All three consume the SAME batch positions each step. Per epoch we record
    each model's parameter movement relative to its OWN source weights.
    """
    benign = make_resnet(args, device)
    mal_source = make_resnet(args, device)
    mal_target = make_resnet(args, device)
    benign.load_state_dict(clean_state)
    mal_source.load_state_dict(psrc_state)
    mal_target.load_state_dict(clean_state)

    opt_b = make_optimizer(benign, args)
    opt_s = make_optimizer(mal_source, args)
    opt_t = make_optimizer(mal_target, args)
    sch_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=max(1, int(epochs * 0.8)))
    sch_s = torch.optim.lr_scheduler.StepLR(opt_s, step_size=max(1, int(epochs * 0.8)))
    sch_t = torch.optim.lr_scheduler.StepLR(opt_t, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, target_domain)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    n = len(base)
    labels_all = _dataset_labels(base)
    poison_set, n_poison = build_poison_set(n, TARGET_POISON_RATE, args.seed, salt=777)
    ds = FairCompareDataset(base, labels_all, poison_set, TARGET_LABEL,
                            FAIR_DISTRIBUTION, args.seed)

    n_flipped = len(ds.flipped_set)
    print("  [target] %d images | poison %d (%.1f%%) -> label %d | flipped %d "
          "| target-class pool %d"
          % (n, n_poison, 100.0 * n_poison / max(n, 1), TARGET_LABEL,
             n_flipped, len(ds.target_pool)))
    if ds.fair:
        identical = (ds.benign_labels() == ds.malicious_labels())
        print("  [fair] benign/mal-source use the clean view; mal-target uses "
              "the poisoned view; per-position clean/poison labels identical "
              "after matching: %s" % identical)
    else:
        print("  [fair] DISABLED: benign/mal-source train on all-clean data "
              "(label distributions differ from mal-target).")

    # One loader feeds all three models the SAME positions each step.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    theta_S_b = flat_params(benign).clone()       # clean source weights
    theta_S_s = flat_params(mal_source).clone()   # poisoned source weights
    theta_S_t = flat_params(mal_target).clone()   # == clean source weights
    history = {
        'epoch': [],
        'benign_step': [], 'msrc_step': [], 'mtgt_step': [],
        'benign_disp': [], 'msrc_disp': [], 'mtgt_disp': [],
        'benign_drift': [], 'msrc_drift': [], 'mtgt_drift': [],
        'benign_loss': [], 'msrc_loss': [], 'mtgt_loss': [],
    }

    for epoch in range(epochs):
        benign.train()
        mal_source.train()
        mal_target.train()

        start_b = flat_params(benign).clone()
        start_s = flat_params(mal_source).clone()
        start_t = flat_params(mal_target).clone()
        prev_b = start_b.clone()
        prev_s = start_s.clone()
        prev_t = start_t.clone()
        step_b = step_s = step_t = 0.0
        loss_b_sum = loss_s_sum = loss_t_sum = 0.0

        for img_b, lab_b, img_m, lab_m, _ in loader:
            img_b = img_b.to(device); lab_b = lab_b.to(device)
            img_m = img_m.to(device); lab_m = lab_m.to(device)
            x_clean = _normalize(img_b)

            # --- benign: clean source init, clean target view ---
            opt_b.zero_grad()
            logits_b, _ = benign(x_clean)
            loss_b = criterion(logits_b, lab_b)
            loss_b.backward()
            opt_b.step()
            loss_b_sum += loss_b.item()

            # --- malicious-source: poisoned source init, SAME clean target view ---
            opt_s.zero_grad()
            logits_s, _ = mal_source(x_clean)
            loss_s = criterion(logits_s, lab_b)
            loss_s.backward()
            opt_s.step()
            loss_s_sum += loss_s.item()

            # --- malicious-target: clean source init, poisoned target view ---
            opt_t.zero_grad()
            logits_t, _ = mal_target(_normalize(img_m))
            loss_t = criterion(logits_t, lab_m)
            loss_t.backward()
            opt_t.step()
            loss_t_sum += loss_t.item()

            cur_b = flat_params(benign)
            cur_s = flat_params(mal_source)
            cur_t = flat_params(mal_target)
            step_b += torch.norm(cur_b - prev_b).item()
            step_s += torch.norm(cur_s - prev_s).item()
            step_t += torch.norm(cur_t - prev_t).item()
            prev_b = cur_b.clone()
            prev_s = cur_s.clone()
            prev_t = cur_t.clone()

        sch_b.step()
        sch_s.step()
        sch_t.step()

        end_b = flat_params(benign)
        end_s = flat_params(mal_source)
        end_t = flat_params(mal_target)
        nb = max(len(loader), 1)

        history['epoch'].append(epoch + 1)
        history['benign_step'].append(step_b)
        history['msrc_step'].append(step_s)
        history['mtgt_step'].append(step_t)
        history['benign_disp'].append(torch.norm(end_b - start_b).item())
        history['msrc_disp'].append(torch.norm(end_s - start_s).item())
        history['mtgt_disp'].append(torch.norm(end_t - start_t).item())
        history['benign_drift'].append(torch.norm(end_b - theta_S_b).item())
        history['msrc_drift'].append(torch.norm(end_s - theta_S_s).item())
        history['mtgt_drift'].append(torch.norm(end_t - theta_S_t).item())
        history['benign_loss'].append(loss_b_sum / nb)
        history['msrc_loss'].append(loss_s_sum / nb)
        history['mtgt_loss'].append(loss_t_sum / nb)

        print("    epoch %2d/%d | step-len  b=%.3f m-src=%.3f m-tgt=%.3f "
              "| drift  b=%.3f m-src=%.3f m-tgt=%.3f"
              % (epoch + 1, epochs,
                 history['benign_step'][-1], history['msrc_step'][-1],
                 history['mtgt_step'][-1],
                 history['benign_drift'][-1], history['msrc_drift'][-1],
                 history['mtgt_drift'][-1]))

    return benign, mal_source, mal_target, history


# ---------------------------------------------------------------------------
# Evaluation (clean accuracy + attack-success-rate on a domain).
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_models(benign, mal_source, mal_target, args, device, domain):
    benign.eval()
    mal_source.eval()
    mal_target.eval()
    base = get_raw_train_dataset(args, domain)
    base = _maybe_limit(base, EVAL_LIMIT, args.seed + 7)
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    correct_b = correct_s = correct_t = total = 0
    asr_s_hit = asr_t_hit = asr_total = 0
    for x01, labels, _ in loader:
        x01 = x01.to(device)
        labels = labels.to(device)
        xn = _normalize(x01)

        pb = benign(xn, train=False)[0].argmax(1)
        ps = mal_source(xn, train=False)[0].argmax(1)
        pt = mal_target(xn, train=False)[0].argmax(1)
        correct_b += (pb == labels).sum().item()
        correct_s += (ps == labels).sum().item()
        correct_t += (pt == labels).sum().item()
        total += labels.size(0)

        # Attack success rate: stamp the trigger on non-target-label samples and
        # check whether each backdoored model flips them to TARGET_LABEL.
        keep = labels != TARGET_LABEL
        if keep.any():
            xt = _normalize(stamp_trigger(x01[keep].clone()))
            pred_s = mal_source(xt, train=False)[0].argmax(1)
            pred_t = mal_target(xt, train=False)[0].argmax(1)
            asr_s_hit += (pred_s == TARGET_LABEL).sum().item()
            asr_t_hit += (pred_t == TARGET_LABEL).sum().item()
            asr_total += keep.sum().item()

    return {
        'benign_clean_acc': correct_b / max(total, 1),
        'msrc_clean_acc': correct_s / max(total, 1),
        'mtgt_clean_acc': correct_t / max(total, 1),
        'msrc_asr': asr_s_hit / max(asr_total, 1),
        'mtgt_asr': asr_t_hit / max(asr_total, 1),
    }


# ---------------------------------------------------------------------------
# Plotting (three curves per panel).
# ---------------------------------------------------------------------------

_SERIES = [
    ('benign', 'benign (clean->clean)', 'o-', 'tab:green'),
    ('msrc', 'malicious-source (poisoned src->clean tgt)', 's--', 'tab:purple'),
    ('mtgt', 'malicious-target (clean src->poisoned tgt)', '^:', 'tab:red'),
]


def plot_history(history, out_path, src, tgt):
    e = history['epoch']
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    for key, label, style, color in _SERIES:
        ax.plot(e, history['%s_drift' % key], style, color=color, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('Parameter Updates Magnitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


def plot_all_targets(histories, src, out_path):
    """Overlay every target domain's per-epoch curves for all three models.

    Color = target domain; line style = model (benign solid, malicious-source
    dashed, malicious-target dotted).
    """
    targets = list(histories.keys())
    cmap = plt.get_cmap('tab10')
    colors = {t: cmap(i % 10) for i, t in enumerate(targets)}
    model_styles = [('benign', '-'), ('msrc', '--'), ('mtgt', ':')]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    for t in targets:
        c = colors[t]
        h = histories[t]
        for key, ls in model_styles:
            ax.plot(h['epoch'], h['%s_drift' % key], ls, color=c)
    ax.set_xlabel('epoch')
    ax.set_ylabel('Parameter Updates Magnitude')
    ax.grid(True, alpha=0.3)

    # Two legends: one for target-domain colors, one for model line styles.
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=colors[t], lw=2, label=t) for t in targets]
    style_handles = [
        Line2D([0], [0], color='k', lw=2, ls='-', label='Benign Model'),
        Line2D([0], [0], color='k', lw=2, ls='--', label='malicious-source'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='malicious-target'),
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=8, title='target', loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=8, title='model', loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved combined figure: %s" % out_path)


def plot_target_summary(histories, src, out_path):
    """Bar chart: final drift from source weights per target for all three models."""
    targets = list(histories.keys())
    tot_b = [histories[t]['benign_drift'][-1] for t in targets]
    tot_s = [histories[t]['msrc_drift'][-1] for t in targets]
    tot_t = [histories[t]['mtgt_drift'][-1] for t in targets]
    x = range(len(targets))
    width = 0.27
    plt.figure(figsize=(max(8, 2.2 * len(targets)), 5))
    plt.bar([i - width for i in x], tot_b, width, label='Benign Model',
            color='tab:green')
    plt.bar([i for i in x], tot_s, width, label='Malicious Model (Source Backdoored)',
            color='tab:purple')
    plt.bar([i + width for i in x], tot_t, width, label='Malicious Model (Target Backdoored)',
            color='tab:red')
    for i in x:
        plt.text(i - width, tot_b[i], 'x%.2f' % (tot_b[i] / max(tot_b[i], 1e-8)),
                 ha='center', va='bottom', fontsize=7)
        plt.text(i, tot_s[i], 'x%.2f' % (tot_s[i] / max(tot_b[i], 1e-8)),
                 ha='center', va='bottom', fontsize=7)
        plt.text(i + width, tot_t[i], 'x%.2f' % (tot_t[i] / max(tot_b[i], 1e-8)),
                 ha='center', va='bottom', fontsize=7)
    plt.xticks(list(x), [t.capitalize() for t in targets])
    plt.ylabel('Parameter Updates Magnitude')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved summary figure: %s" % out_path)


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
    src = SOURCE_DOMAIN if SOURCE_DOMAIN in domains else domains[0]
    if TARGET_DOMAINS is not None:
        targets = [d for d in TARGET_DOMAINS if d in domains and d != src]
    else:
        targets = [d for d in domains if d != src]
    # Optionally prepend the source domain as a same-domain (source == target)
    # reference point so we get one extra results figure for it.
    if INCLUDE_SOURCE_AS_TARGET and src not in targets:
        targets = [src] + targets
    if not targets:
        raise ValueError(
            "No valid target domains for source '%s' in task %s. "
            "Check SOURCE_DOMAIN / TARGET_DOMAINS (available: %s)."
            % (src, args.task, domains))
    args.source = [src]
    args.target = [targets[0]]
    return args, src, targets


def main():
    args, src, targets = setup_args()
    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    src_epochs = SOURCE_EPOCHS if SOURCE_EPOCHS is not None else args.epochs
    tgt_epochs = TARGET_EPOCHS if TARGET_EPOCHS is not None else args.epochs

    print("Task: %s | Source: %s | Targets: %s"
          % (args.task, src, ", ".join(targets)))
    print("Source epochs: %d | Target epochs: %d | src poison %.2f | tgt poison %.2f "
          "| target label: %d"
          % (src_epochs, tgt_epochs, SOURCE_POISON_RATE, TARGET_POISON_RATE,
             TARGET_LABEL))

    # --- Step 1: clean source model + poisoned source model (shared init). ---
    print("\n==== Step 1: source training (clean vs poisoned '%s') ====" % src)
    clean_state, psrc_state = train_source_compare(args, device, src_epochs)

    print("\n  -- source-domain eval ('%s') --" % src)
    eb = make_resnet(args, device); eb.load_state_dict(clean_state)
    es = make_resnet(args, device); es.load_state_dict(psrc_state)
    src_metrics = evaluate_models(eb, eb, es, args, device, src)
    print("    clean-source clean acc    : %.2f%%" % (100 * src_metrics['benign_clean_acc']))
    print("    poisoned-source clean acc : %.2f%%" % (100 * src_metrics['mtgt_clean_acc']))
    print("    poisoned-source ASR       : %.2f%%" % (100 * src_metrics['mtgt_asr']))
    del eb, es
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 2: per-target three-model comparison. ---
    histories = {}
    metrics_by_target = {}
    for tgt in targets:
        args.target = [tgt]
        print("\n==== Target '%s': benign vs malicious-source vs malicious-target ====" % tgt)
        benign, mal_source, mal_target, history = train_target_compare(
            args, device, clean_state, psrc_state, tgt_epochs, tgt)
        histories[tgt] = history

        plot_history(history, os.path.join(
            OUTPUT_DIR, "final_param_updates_%s_to_%s.png" % (src, tgt)), src, tgt)

        print("  -- sanity eval on '%s' --" % tgt)
        metrics = evaluate_models(benign, mal_source, mal_target, args, device, tgt)
        metrics_by_target[tgt] = metrics
        print("    benign clean acc          : %.2f%%" % (100 * metrics['benign_clean_acc']))
        print("    mal-source clean acc      : %.2f%%" % (100 * metrics['msrc_clean_acc']))
        print("    mal-target clean acc      : %.2f%%" % (100 * metrics['mtgt_clean_acc']))
        print("    mal-source ASR (retained) : %.2f%%" % (100 * metrics['msrc_asr']))
        print("    mal-target ASR            : %.2f%%" % (100 * metrics['mtgt_asr']))

        del benign, mal_source, mal_target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 3: combined cross-domain figures. ---
    plot_all_targets(histories, src, os.path.join(
        OUTPUT_DIR, "final_param_updates_%s_all_targets.png" % src))
    plot_target_summary(histories, src, os.path.join(
        OUTPUT_DIR, "final_param_updates_%s_summary.png" % src))

    # --- Cross-domain summary table. ---
    print("\n==== Cross-domain summary (source %s) ====" % src)
    print("%-14s %12s %12s %12s %10s %10s %10s %9s %9s"
          % ("target", "tot_benign", "tot_msrc", "tot_mtgt",
             "b_acc", "msrc_acc", "mtgt_acc", "msrc_ASR", "mtgt_ASR"))
    for tgt in targets:
        h = histories[tgt]
        tb = sum(h['benign_step'])
        ts = sum(h['msrc_step'])
        tt = sum(h['mtgt_step'])
        m = metrics_by_target[tgt]
        print("%-14s %12.4f %12.4f %12.4f %9.2f%% %9.2f%% %9.2f%% %8.2f%% %8.2f%%"
              % (tgt, tb, ts, tt,
                 100 * m['benign_clean_acc'], 100 * m['msrc_clean_acc'],
                 100 * m['mtgt_clean_acc'], 100 * m['msrc_asr'],
                 100 * m['mtgt_asr']))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
