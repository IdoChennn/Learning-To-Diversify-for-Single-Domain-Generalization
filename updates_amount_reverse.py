"""Reverse scenario: poison the SOURCE domain, then transfer to CLEAN targets.

This is the mirror image of ``updates_amount.py``. There, the source domain was
clean and the malicious model was poisoned during *target* training. Here:

1. During SOURCE-domain training we train TWO models from an identical
   initialization:
     * benign    -> clean source samples,
     * malicious -> BadNets-poisoned source samples (a trigger patch on a
       fraction of images, their labels overwritten with ``TARGET_LABEL``).
   As in ``updates_amount.py`` the benign model's label distribution is matched
   to the malicious one (flipped-poison positions are dropped and replaced by
   duplicated clean target-class images), so the ONLY difference between the two
   source models is the trigger patch.

2. Both source-trained models are then fine-tuned on each *clean* target domain
   (no poisoning during target training). Both consume the same clean batches.

3. During the target-domain training we record, per epoch, how much each model's
   parameters move relative to its OWN source weights:
     * step-wise update length  : sum over the epoch of ||theta_t+1 - theta_t||_2
     * epoch displacement        : ||theta_epoch_end - theta_epoch_start||_2
     * drift from source weights : ||theta_epoch_end - theta_S||_2
   This reveals whether a backdoored initialization moves differently than a
   clean one when adapting to an unseen clean domain.

Self-contained; reuses the repo's data pipeline (``data.data_helper``), model
(``models.resnet.resnet18``) and ``train.get_args``. Does not modify any
existing code.
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

# Source domain (POISONED for the malicious model) used to pre-train weights.
SOURCE_DOMAIN = 'photo'
# Clean target domains to transfer to. None = every task domain except the
# source; otherwise a list of domain names to restrict to.
TARGET_DOMAINS = None

# Training length (epochs). Fall back to args.epochs if these are None.
SOURCE_EPOCHS = 20
TARGET_EPOCHS = 20

# --- BadNets poisoning hyper-parameters (applied to the SOURCE domain) ---
POISON_RATE = 0.1     # fraction of source-train images that get the trigger
TARGET_LABEL = 0      # label all poisoned samples are forced to
TRIGGER_SIZE = 24     # side length (pixels) of the square trigger patch
TRIGGER_VALUE = 1.0   # patch pixel value in [0,1] space (white)
TRIGGER_POS = 'br'    # corner: 'br','bl','tr','tl'

# Cap the number of training images per phase (None = all). Useful for quick runs.
LIMIT_TRAIN_IMAGES = None
# Number of images used to measure clean acc / attack success rate per domain.
EVAL_LIMIT = 1000

# Match the benign source model's label distribution to the malicious one so the
# ONLY source-training difference is the trigger patch (see module docstring).
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


def build_poison_set(n, rate, seed):
    g = torch.Generator().manual_seed(int(seed) + 12345)
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
    """Aligned benign / malicious views for fair source-domain training.

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
    """Train benign (clean) and malicious (poisoned) models on the SOURCE domain
    from an identical initialization; return both state dicts.

    Both models consume the same aligned batch each step (same labels, same
    order); the only per-position difference is the trigger patch on the
    malicious model's poisoned images.
    """
    fix_all_seed(args.seed)
    benign = make_resnet(args, device)
    malicious = make_resnet(args, device)
    init_state = copy.deepcopy(benign.state_dict())
    malicious.load_state_dict(init_state)  # identical starting weights

    opt_b = make_optimizer(benign, args)
    opt_m = make_optimizer(malicious, args)
    sch_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=max(1, int(epochs * 0.8)))
    sch_m = torch.optim.lr_scheduler.StepLR(opt_m, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, SOURCE_DOMAIN)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    n = len(base)
    labels_all = _dataset_labels(base)
    poison_set, n_poison = build_poison_set(n, POISON_RATE, args.seed)
    ds = FairCompareDataset(base, labels_all, poison_set, TARGET_LABEL,
                            FAIR_DISTRIBUTION, args.seed)

    n_flipped = len(ds.flipped_set)
    print("  [source] %d images | poison %d (%.1f%%) -> label %d | flipped %d "
          "| target-class pool %d"
          % (n, n_poison, 100.0 * n_poison / max(n, 1), TARGET_LABEL,
             n_flipped, len(ds.target_pool)))
    if ds.fair:
        identical = (ds.benign_labels() == ds.malicious_labels())
        print("  [fair] benign drops %d flipped samples and adds %d duplicated "
              "clean target-class images; per-position labels identical: %s"
              % (n_flipped, n_flipped, identical))
    else:
        print("  [fair] DISABLED: benign trains on all-clean data "
              "(label distributions differ from malicious).")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    print("  [source] training benign + malicious for %d epochs" % epochs)
    for epoch in range(epochs):
        benign.train()
        malicious.train()
        loss_b_sum = loss_m_sum = 0.0
        for img_b, lab_b, img_m, lab_m, _ in loader:
            img_b = img_b.to(device); lab_b = lab_b.to(device)
            img_m = img_m.to(device); lab_m = lab_m.to(device)

            opt_b.zero_grad()
            logits_b, _ = benign(_normalize(img_b))
            loss_b = criterion(logits_b, lab_b)
            loss_b.backward()
            opt_b.step()
            loss_b_sum += loss_b.item()

            opt_m.zero_grad()
            logits_m, _ = malicious(_normalize(img_m))
            loss_m = criterion(logits_m, lab_m)
            loss_m.backward()
            opt_m.step()
            loss_m_sum += loss_m.item()
        sch_b.step()
        sch_m.step()
        nb = max(len(loader), 1)
        print("    [source] epoch %2d/%d  benign_loss = %.4f  malicious_loss = %.4f"
              % (epoch + 1, epochs, loss_b_sum / nb, loss_m_sum / nb))

    return (copy.deepcopy(benign.state_dict()),
            copy.deepcopy(malicious.state_dict()))


def train_target_compare(args, device, benign_state, malicious_state, epochs,
                         target_domain):
    """Fine-tune the benign and malicious source models on a CLEAN target domain.

    Both models start from their respective source weights and consume the SAME
    clean batches each step (no poisoning here). Per epoch we record each model's
    parameter movement relative to its OWN source weights theta_S.
    """
    benign = make_resnet(args, device)
    malicious = make_resnet(args, device)
    benign.load_state_dict(benign_state)
    malicious.load_state_dict(malicious_state)

    opt_b = make_optimizer(benign, args)
    opt_m = make_optimizer(malicious, args)
    sch_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=max(1, int(epochs * 0.8)))
    sch_m = torch.optim.lr_scheduler.StepLR(opt_m, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, target_domain)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    print("  [target %s] %d clean images" % (target_domain, len(base)))

    # One loader feeds both models the SAME clean positions each step.
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    theta_S_b = flat_params(benign).clone()      # benign source weights
    theta_S_m = flat_params(malicious).clone()   # malicious (backdoored) source weights
    history = {
        'epoch': [],
        'benign_step': [], 'malicious_step': [],
        'benign_disp': [], 'malicious_disp': [],
        'benign_drift': [], 'malicious_drift': [],
        'benign_loss': [], 'malicious_loss': [],
    }

    for epoch in range(epochs):
        benign.train()
        malicious.train()

        start_b = flat_params(benign).clone()
        start_m = flat_params(malicious).clone()
        prev_b = start_b.clone()
        prev_m = start_m.clone()
        step_len_b = step_len_m = 0.0
        loss_b_sum = loss_m_sum = 0.0

        for x01, labels, _ in loader:
            x01 = x01.to(device)
            labels = labels.to(device)
            x = _normalize(x01)

            # --- benign model on the clean batch ---
            opt_b.zero_grad()
            logits_b, _ = benign(x)
            loss_b = criterion(logits_b, labels)
            loss_b.backward()
            opt_b.step()
            loss_b_sum += loss_b.item()

            # --- malicious model on the SAME clean batch ---
            opt_m.zero_grad()
            logits_m, _ = malicious(x)
            loss_m = criterion(logits_m, labels)
            loss_m.backward()
            opt_m.step()
            loss_m_sum += loss_m.item()

            cur_b = flat_params(benign)
            cur_m = flat_params(malicious)
            step_len_b += torch.norm(cur_b - prev_b).item()
            step_len_m += torch.norm(cur_m - prev_m).item()
            prev_b = cur_b.clone()
            prev_m = cur_m.clone()

        sch_b.step()
        sch_m.step()

        end_b = flat_params(benign)
        end_m = flat_params(malicious)
        nb = max(len(loader), 1)

        history['epoch'].append(epoch + 1)
        history['benign_step'].append(step_len_b)
        history['malicious_step'].append(step_len_m)
        history['benign_disp'].append(torch.norm(end_b - start_b).item())
        history['malicious_disp'].append(torch.norm(end_m - start_m).item())
        history['benign_drift'].append(torch.norm(end_b - theta_S_b).item())
        history['malicious_drift'].append(torch.norm(end_m - theta_S_m).item())
        history['benign_loss'].append(loss_b_sum / nb)
        history['malicious_loss'].append(loss_m_sum / nb)

        print("    epoch %2d/%d | step-len  benign=%.3f malicious=%.3f "
              "(x%.2f) | disp b=%.3f m=%.3f | drift b=%.3f m=%.3f"
              % (epoch + 1, epochs,
                 history['benign_step'][-1], history['malicious_step'][-1],
                 history['malicious_step'][-1] / max(history['benign_step'][-1], 1e-8),
                 history['benign_disp'][-1], history['malicious_disp'][-1],
                 history['benign_drift'][-1], history['malicious_drift'][-1]))

    return benign, malicious, history


# ---------------------------------------------------------------------------
# Evaluation (clean accuracy + attack-success-rate retention on a domain).
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_models(benign, malicious, args, device, domain):
    benign.eval()
    malicious.eval()
    base = get_raw_train_dataset(args, domain)
    base = _maybe_limit(base, EVAL_LIMIT, args.seed + 7)
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    correct_b = correct_m = total = 0
    asr_hit = asr_total = 0
    for x01, labels, _ in loader:
        x01 = x01.to(device)
        labels = labels.to(device)

        pb = benign(_normalize(x01), train=False)[0].argmax(1)
        pm = malicious(_normalize(x01), train=False)[0].argmax(1)
        correct_b += (pb == labels).sum().item()
        correct_m += (pm == labels).sum().item()
        total += labels.size(0)

        keep = labels != TARGET_LABEL
        if keep.any():
            xt = stamp_trigger(x01[keep].clone())
            pred = malicious(_normalize(xt), train=False)[0].argmax(1)
            asr_hit += (pred == TARGET_LABEL).sum().item()
            asr_total += keep.sum().item()

    return {
        'benign_clean_acc': correct_b / max(total, 1),
        'malicious_clean_acc': correct_m / max(total, 1),
        'attack_success_rate': asr_hit / max(asr_total, 1),
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------

def plot_history(history, out_path, src, tgt):
    e = history['epoch']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(e, history['benign_step'], 'o-', label='benign (clean source)')
    axes[0].plot(e, history['malicious_step'], 's-', label='malicious (poisoned source)')
    axes[0].set_title('Per-epoch step-wise update length\n(sum of ||theta_{t+1}-theta_t||)')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('L2 update length')

    axes[1].plot(e, history['benign_disp'], 'o-', label='benign (clean source)')
    axes[1].plot(e, history['malicious_disp'], 's-', label='malicious (poisoned source)')
    axes[1].set_title('Per-epoch displacement\n||theta_end - theta_start||')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('L2 displacement')

    axes[2].plot(e, history['benign_drift'], 'o-', label='benign (clean source)')
    axes[2].plot(e, history['malicious_drift'], 's-', label='malicious (poisoned source)')
    axes[2].set_title('Drift from source weights\n||theta_end - theta_S||')
    axes[2].set_xlabel('epoch'); axes[2].set_ylabel('L2 drift')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle('Clean vs poisoned-source model updates on clean target: %s -> %s'
                 % (src, tgt), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


def plot_all_targets(histories, src, out_path):
    """Overlay every target domain's per-epoch curves (benign vs malicious)."""
    targets = list(histories.keys())
    cmap = plt.get_cmap('tab10')
    colors = {t: cmap(i % 10) for i, t in enumerate(targets)}
    panels = [
        ('benign_step', 'malicious_step',
         'Per-epoch step-wise update length\n(sum of ||theta_{t+1}-theta_t||)',
         'L2 update length'),
        ('benign_disp', 'malicious_disp',
         'Per-epoch displacement\n||theta_end - theta_start||', 'L2 displacement'),
        ('benign_drift', 'malicious_drift',
         'Drift from source weights\n||theta_end - theta_S||', 'L2 drift'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (bk, mk, title, ylabel) in zip(axes, panels):
        for t in targets:
            h = histories[t]
            c = colors[t]
            ax.plot(h['epoch'], h[bk], '-', color=c, label='%s (benign)' % t)
            ax.plot(h['epoch'], h[mk], '--', color=c, label='%s (malicious)' % t)
        ax.set_title(title)
        ax.set_xlabel('epoch'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle('Clean vs poisoned-source model updates on clean targets '
                 'from source %s (solid=benign, dashed=malicious)' % src, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved combined figure: %s" % out_path)


def plot_target_summary(histories, src, out_path):
    """Bar chart: total step-wise update length per target (benign vs malicious)."""
    targets = list(histories.keys())
    tot_b = [sum(histories[t]['benign_step']) for t in targets]
    tot_m = [sum(histories[t]['malicious_step']) for t in targets]
    x = range(len(targets))
    width = 0.38
    plt.figure(figsize=(max(7, 1.8 * len(targets)), 5))
    plt.bar([i - width / 2 for i in x], tot_b, width, label='benign (clean source)')
    plt.bar([i + width / 2 for i in x], tot_m, width, label='malicious (poisoned source)')
    for i, t in enumerate(targets):
        ratio = tot_m[i] / max(tot_b[i], 1e-8)
        plt.text(i, max(tot_b[i], tot_m[i]), 'x%.2f' % ratio,
                 ha='center', va='bottom', fontsize=9)
    plt.xticks(list(x), [t.capitalize() for t in targets])
    plt.ylabel('Total step-wise update length (summed over epochs)')
    plt.title('Total parameter movement on clean targets from source %s'
              % src.capitalize())
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

    print("Task: %s | Source (poisoned): %s | Clean targets: %s"
          % (args.task, src, ", ".join(targets)))
    print("Source epochs: %d | Target epochs: %d | poison rate: %.2f | target label: %d"
          % (src_epochs, tgt_epochs, POISON_RATE, TARGET_LABEL))

    # --- Step 1: train benign + malicious source models. ---
    print("\n==== Step 1: source training (benign clean vs malicious poisoned '%s') ====" % src)
    benign_state, malicious_state = train_source_compare(args, device, src_epochs)

    print("\n  -- source-domain eval ('%s') --" % src)
    sb = make_resnet(args, device); sb.load_state_dict(benign_state)
    sm = make_resnet(args, device); sm.load_state_dict(malicious_state)
    src_metrics = evaluate_models(sb, sm, args, device, src)
    print("    benign clean acc    : %.2f%%" % (100 * src_metrics['benign_clean_acc']))
    print("    malicious clean acc : %.2f%%" % (100 * src_metrics['malicious_clean_acc']))
    print("    attack success rate : %.2f%%" % (100 * src_metrics['attack_success_rate']))
    del sb, sm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Steps 2-3: fine-tune both models on each clean target domain. ---
    histories = {}
    metrics_by_target = {}
    for tgt in targets:
        args.target = [tgt]
        print("\n==== Clean target '%s': benign vs malicious-source model ====" % tgt)
        benign, malicious, history = train_target_compare(
            args, device, benign_state, malicious_state, tgt_epochs, tgt)
        histories[tgt] = history

        plot_history(history, os.path.join(
            OUTPUT_DIR, "reverse_param_updates_%s_to_%s.png" % (src, tgt)), src, tgt)

        print("  -- eval on '%s' (ASR = backdoor retention after clean fine-tuning) --" % tgt)
        metrics = evaluate_models(benign, malicious, args, device, tgt)
        metrics_by_target[tgt] = metrics
        print("    benign clean acc    : %.2f%%" % (100 * metrics['benign_clean_acc']))
        print("    malicious clean acc : %.2f%%" % (100 * metrics['malicious_clean_acc']))
        print("    attack success rate : %.2f%%" % (100 * metrics['attack_success_rate']))

        del benign, malicious
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 4: combined cross-domain figures. ---
    plot_all_targets(histories, src, os.path.join(
        OUTPUT_DIR, "reverse_param_updates_%s_all_targets.png" % src))
    plot_target_summary(histories, src, os.path.join(
        OUTPUT_DIR, "reverse_param_updates_%s_summary.png" % src))

    # --- Cross-domain summary table. ---
    print("\n==== Cross-domain summary (poisoned source %s -> clean targets) ====" % src)
    print("%-14s %14s %14s %10s %12s %12s %10s"
          % ("target", "tot_benign", "tot_malic", "ratio",
             "benign_acc", "malic_acc", "ASR"))
    for tgt in targets:
        h = histories[tgt]
        tot_b = sum(h['benign_step'])
        tot_m = sum(h['malicious_step'])
        m = metrics_by_target[tgt]
        print("%-14s %14.4f %14.4f %10.3f %11.2f%% %11.2f%% %9.2f%%"
              % (tgt, tot_b, tot_m, tot_m / max(tot_b, 1e-8),
                 100 * m['benign_clean_acc'], 100 * m['malicious_clean_acc'],
                 100 * m['attack_success_rate']))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
