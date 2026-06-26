"""Same-domain parameter-update study on CIFAR-10 / CIFAR-100.

Unlike the PACS scripts there is NO domain shift here: phase-1 (pre-training)
and phase-2 (the "transfer" we measure) both use the same CIFAR dataset. We
compare the amount of parameter movement during phase-2 in two directions:

  * 'clean_to_backdoor' (forward): pre-train ONE clean model, then continue
    training a benign copy on clean data vs a malicious copy on BadNets-poisoned
    data (both starting from the same clean weights). Mirrors ``updates_amount.py``.

  * 'poison_to_clean' (reverse): pre-train a benign (clean) and a malicious
    (poisoned) model from an identical init, then fine-tune BOTH on clean data.
    Mirrors ``updates_amount_reverse.py``.

Poisoning (BadNets) and the fair label-distribution matching are identical to
the other scripts: a fixed trigger patch is stamped on a fraction of images and
their labels overwritten with ``TARGET_LABEL``; the benign side drops the
flipped-poison positions and replaces them 1-for-1 with duplicated clean
target-class images, so both sides see an identical per-position label sequence.

Per epoch (phase-2) we record, for each model, relative to its OWN pre-trained
weights theta_S:
  * step-wise update length : sum over the epoch of ||theta_t+1 - theta_t||_2
  * epoch displacement       : ||theta_epoch_end - theta_epoch_start||_2
  * drift from theta_S       : ||theta_epoch_end - theta_S||_2

Reuses the repo's model (``models.resnet.resnet18``, ImageNet-pretrained, so
CIFAR images are resized to ``image_size``) and ``train.get_args``. Does not
modify any existing code. CIFAR is downloaded/cached via torchvision.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are only saved to disk
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.util import fix_all_seed
from train import get_args


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

# 'CIFAR10' or 'CIFAR100'.
DATASET = 'CIFAR10'
CIFAR_ROOT = './data_cifar'

# Which scenarios to run (one figure each).
SCENARIOS = ['clean_to_backdoor', 'poison_to_clean']

# Training length (epochs). Fall back to args.epochs if these are None.
PHASE1_EPOCHS = 15   # pre-training
PHASE2_EPOCHS = 15   # the "transfer" whose updates we measure

# --- BadNets poisoning hyper-parameters ---
POISON_RATE = 0.1     # fraction of images that get the trigger
TARGET_LABEL = 0      # label all poisoned samples are forced to
TRIGGER_SIZE = 24     # side length (pixels, at image_size) of the trigger patch
TRIGGER_VALUE = 1.0   # patch pixel value in [0,1] space (white)
TRIGGER_POS = 'br'    # corner: 'br','bl','tr','tl'

# Match the benign model's label distribution to the malicious one (fairness).
FAIR_DISTRIBUTION = True

# Cap the number of training / eval images (None = all). Useful for quick runs.
LIMIT_TRAIN_IMAGES = None
EVAL_LIMIT = 2000

OUTPUT_DIR = 'updates_figures'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# CIFAR raw [0,1] dataset + helpers.
# ---------------------------------------------------------------------------

def _n_classes():
    return 100 if DATASET == 'CIFAR100' else 10


class RawCIFAR(Dataset):
    """CIFAR-10/100 returning (image[0,1] resized to ``image_size``, 0, label).

    The 0 mirrors the repo's (image, jigsaw_label, class_label) convention so
    ``IndexedDataset`` / ``FairCompareDataset`` work unchanged.
    """

    def __init__(self, train, image_size, root=CIFAR_ROOT):
        cls = torchvision.datasets.CIFAR100 if DATASET == 'CIFAR100' \
            else torchvision.datasets.CIFAR10
        self.ds = cls(root=root, train=train, download=True)
        self.labels = [int(t) for t in self.ds.targets]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> [0,1]
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]  # PIL image, int
        return self.transform(img), 0, int(label)


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
    """0-indexed labels for every position (handles ``Subset`` wrapping)."""
    if isinstance(dataset, torch.utils.data.Subset):
        inner = _dataset_labels(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    if hasattr(dataset, 'labels'):
        return [int(l) for l in dataset.labels]
    raise TypeError("Cannot extract labels from dataset of type %s" % type(dataset))


class FairCompareDataset(Dataset):
    """Aligned benign / malicious views.

    Returns ``(img_b01, label_b, img_m01, label_m, poison_flag)``. The malicious
    view has the trigger stamped + label ``TARGET_LABEL`` iff poisoned; when
    ``fair`` the benign view replaces flipped-poison positions with a randomly
    duplicated clean target-class image so both views share an identical
    per-position label sequence.

    An empty ``poison_set`` yields identical clean views for both sides (used by
    the reverse scenario's clean phase-2).
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


def _make_fair_loader(args, base, poison_set):
    """Build an aligned (benign, malicious) loader over ``base``.

    ``poison_set`` empty -> identical clean views for both sides.
    """
    labels_all = _dataset_labels(base)
    ds = FairCompareDataset(base, labels_all, poison_set, TARGET_LABEL,
                            FAIR_DISTRIBUTION, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    return ds, loader


def train_single_clean(args, device, epochs, base, init_state=None):
    """Train one model on clean ``base`` data; return its state dict."""
    model = make_resnet(args, device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print("  [phase1-clean] training on %d images for %d epochs" % (len(base), epochs))
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x01, labels, _ in loader:
            x01 = x01.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(_normalize(x01))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        print("    [phase1-clean] epoch %2d/%d  loss = %.4f"
              % (epoch + 1, epochs, running / max(len(loader), 1)))
    return copy.deepcopy(model.state_dict())


def train_pair_source(args, device, epochs, base, poison_set):
    """Pre-train benign (clean) + malicious (poisoned) from an IDENTICAL init on
    the aligned ``base``; return both state dicts (reverse scenario phase-1)."""
    fix_all_seed(args.seed)
    benign = make_resnet(args, device)
    malicious = make_resnet(args, device)
    init_state = copy.deepcopy(benign.state_dict())
    malicious.load_state_dict(init_state)

    opt_b = make_optimizer(benign, args)
    opt_m = make_optimizer(malicious, args)
    sch_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=max(1, int(epochs * 0.8)))
    sch_m = torch.optim.lr_scheduler.StepLR(opt_m, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    ds, loader = _make_fair_loader(args, base, poison_set)
    if ds.fair:
        identical = (ds.benign_labels() == ds.malicious_labels())
        print("  [phase1-pair] poison %d | flipped %d | per-position labels identical: %s"
              % (len(poison_set), len(ds.flipped_set), identical))
    print("  [phase1-pair] training benign + malicious for %d epochs" % epochs)
    for epoch in range(epochs):
        benign.train(); malicious.train()
        lb = lm = 0.0
        for img_b, lab_b, img_m, lab_m, _ in loader:
            img_b = img_b.to(device); lab_b = lab_b.to(device)
            img_m = img_m.to(device); lab_m = lab_m.to(device)

            opt_b.zero_grad()
            out_b, _ = benign(_normalize(img_b))
            loss_b = criterion(out_b, lab_b); loss_b.backward(); opt_b.step()
            lb += loss_b.item()

            opt_m.zero_grad()
            out_m, _ = malicious(_normalize(img_m))
            loss_m = criterion(out_m, lab_m); loss_m.backward(); opt_m.step()
            lm += loss_m.item()
        sch_b.step(); sch_m.step()
        nb = max(len(loader), 1)
        print("    [phase1-pair] epoch %2d/%d  benign_loss = %.4f  malicious_loss = %.4f"
              % (epoch + 1, epochs, lb / nb, lm / nb))
    return copy.deepcopy(benign.state_dict()), copy.deepcopy(malicious.state_dict())


def train_compare(args, device, init_b, init_m, base, poison_set, epochs):
    """Train benign + malicious (init from ``init_b`` / ``init_m``) on the aligned
    ``base`` for ``epochs``, recording per-epoch update magnitudes for each model
    relative to its OWN starting weights. Returns (benign, malicious, history).

    ``poison_set`` non-empty -> malicious sees poisoned, benign sees matched-clean
    (forward scenario phase-2). ``poison_set`` empty -> both clean (reverse
    scenario phase-2).
    """
    benign = make_resnet(args, device)
    malicious = make_resnet(args, device)
    benign.load_state_dict(init_b)
    malicious.load_state_dict(init_m)

    opt_b = make_optimizer(benign, args)
    opt_m = make_optimizer(malicious, args)
    sch_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=max(1, int(epochs * 0.8)))
    sch_m = torch.optim.lr_scheduler.StepLR(opt_m, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    ds, loader = _make_fair_loader(args, base, poison_set)
    if len(poison_set) > 0 and ds.fair:
        identical = (ds.benign_labels() == ds.malicious_labels())
        print("  [phase2] poison %d | flipped %d | per-position labels identical: %s"
              % (len(poison_set), len(ds.flipped_set), identical))

    theta_S_b = flat_params(benign).clone()
    theta_S_m = flat_params(malicious).clone()
    history = {
        'epoch': [],
        'benign_step': [], 'malicious_step': [],
        'benign_disp': [], 'malicious_disp': [],
        'benign_drift': [], 'malicious_drift': [],
        'benign_loss': [], 'malicious_loss': [],
    }

    for epoch in range(epochs):
        benign.train(); malicious.train()
        start_b = flat_params(benign).clone()
        start_m = flat_params(malicious).clone()
        prev_b = start_b.clone(); prev_m = start_m.clone()
        step_b = step_m = 0.0
        lb = lm = 0.0

        for img_b, lab_b, img_m, lab_m, _ in loader:
            img_b = img_b.to(device); lab_b = lab_b.to(device)
            img_m = img_m.to(device); lab_m = lab_m.to(device)

            opt_b.zero_grad()
            out_b, _ = benign(_normalize(img_b))
            loss_b = criterion(out_b, lab_b); loss_b.backward(); opt_b.step()
            lb += loss_b.item()

            opt_m.zero_grad()
            out_m, _ = malicious(_normalize(img_m))
            loss_m = criterion(out_m, lab_m); loss_m.backward(); opt_m.step()
            lm += loss_m.item()

            cur_b = flat_params(benign); cur_m = flat_params(malicious)
            step_b += torch.norm(cur_b - prev_b).item()
            step_m += torch.norm(cur_m - prev_m).item()
            prev_b = cur_b.clone(); prev_m = cur_m.clone()

        sch_b.step(); sch_m.step()
        end_b = flat_params(benign); end_m = flat_params(malicious)
        nb = max(len(loader), 1)

        history['epoch'].append(epoch + 1)
        history['benign_step'].append(step_b)
        history['malicious_step'].append(step_m)
        history['benign_disp'].append(torch.norm(end_b - start_b).item())
        history['malicious_disp'].append(torch.norm(end_m - start_m).item())
        history['benign_drift'].append(torch.norm(end_b - theta_S_b).item())
        history['malicious_drift'].append(torch.norm(end_m - theta_S_m).item())
        history['benign_loss'].append(lb / nb)
        history['malicious_loss'].append(lm / nb)

        print("    epoch %2d/%d | step-len  benign=%.3f malicious=%.3f "
              "(x%.2f) | disp b=%.3f m=%.3f | drift b=%.3f m=%.3f"
              % (epoch + 1, epochs,
                 history['benign_step'][-1], history['malicious_step'][-1],
                 history['malicious_step'][-1] / max(history['benign_step'][-1], 1e-8),
                 history['benign_disp'][-1], history['malicious_disp'][-1],
                 history['benign_drift'][-1], history['malicious_drift'][-1]))

    return benign, malicious, history


# ---------------------------------------------------------------------------
# Evaluation (clean accuracy + attack success rate on the CIFAR test set).
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_models(benign, malicious, args, device, test_base):
    benign.eval(); malicious.eval()
    base = _maybe_limit(test_base, EVAL_LIMIT, args.seed + 7)
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    correct_b = correct_m = total = 0
    asr_hit = asr_total = 0
    for x01, labels, _ in loader:
        x01 = x01.to(device); labels = labels.to(device)
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

def plot_history(history, out_path, title, benign_label, malicious_label):
    e = history['epoch']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    series = [
        ('benign_step', 'malicious_step',
         'Per-epoch step-wise update length\n(sum of ||theta_{t+1}-theta_t||)',
         'L2 update length'),
        ('benign_disp', 'malicious_disp',
         'Per-epoch displacement\n||theta_end - theta_start||', 'L2 displacement'),
        ('benign_drift', 'malicious_drift',
         'Drift from pre-trained weights\n||theta_end - theta_S||', 'L2 drift'),
    ]
    for ax, (bk, mk, t, ylabel) in zip(axes, series):
        ax.plot(e, history[bk], 'o-', label=benign_label)
        ax.plot(e, history[mk], 's-', label=malicious_label)
        ax.set_title(t); ax.set_xlabel('epoch'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


def plot_scenario_summary(results, out_path):
    """Bar chart: total step-wise update length (benign vs malicious) per scenario."""
    names = list(results.keys())
    tot_b = [sum(results[s]['history']['benign_step']) for s in names]
    tot_m = [sum(results[s]['history']['malicious_step']) for s in names]
    x = range(len(names))
    width = 0.38
    plt.figure(figsize=(max(7, 2.4 * len(names)), 5))
    plt.bar([i - width / 2 for i in x], tot_b, width, label='benign')
    plt.bar([i + width / 2 for i in x], tot_m, width, label='malicious')
    for i in x:
        ratio = tot_m[i] / max(tot_b[i], 1e-8)
        plt.text(i, max(tot_b[i], tot_m[i]), 'x%.2f' % ratio,
                 ha='center', va='bottom', fontsize=9)
    plt.xticks(list(x), names, fontsize=9)
    plt.ylabel('Total step-wise update length (summed over epochs)')
    plt.title('%s: phase-2 parameter movement by scenario' % DATASET)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved summary figure: %s" % out_path)


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------

def run_clean_to_backdoor(args, device, train_base, test_base, poison_set, epochs1, epochs2):
    """Forward: clean pre-trained model -> benign(clean) vs malicious(backdoor)."""
    print("\n==== [clean_to_backdoor] phase-1: pre-train ONE clean model ====")
    theta_S = train_single_clean(args, device, epochs1, train_base)

    print("\n==== [clean_to_backdoor] phase-2: benign clean vs malicious poisoned ====")
    benign, malicious, history = train_compare(
        args, device, theta_S, theta_S, train_base, poison_set, epochs2)
    metrics = evaluate_models(benign, malicious, args, device, test_base)
    return {'history': history, 'metrics': metrics,
            'benign_label': 'benign (clean)',
            'malicious_label': 'malicious (backdoor)'}


def run_poison_to_clean(args, device, train_base, test_base, poison_set, epochs1, epochs2):
    """Reverse: benign(clean) & malicious(poisoned) pre-trained -> both on clean."""
    print("\n==== [poison_to_clean] phase-1: pre-train benign clean + malicious poisoned ====")
    benign_state, malicious_state = train_pair_source(
        args, device, epochs1, train_base, poison_set)

    print("\n==== [poison_to_clean] phase-2: fine-tune BOTH on clean data ====")
    benign, malicious, history = train_compare(
        args, device, benign_state, malicious_state, train_base, set(), epochs2)
    metrics = evaluate_models(benign, malicious, args, device, test_base)
    return {'history': history, 'metrics': metrics,
            'benign_label': 'benign (clean-pretrained)',
            'malicious_label': 'malicious (poison-pretrained)'}


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    args.n_classes = _n_classes()
    args.source = ['cifar']  # placeholders; CIFAR has no domain concept
    args.target = ['cifar']

    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    e1 = PHASE1_EPOCHS if PHASE1_EPOCHS is not None else args.epochs
    e2 = PHASE2_EPOCHS if PHASE2_EPOCHS is not None else args.epochs

    print("Dataset: %s (%d classes) | scenarios: %s"
          % (DATASET, args.n_classes, ", ".join(SCENARIOS)))
    print("Phase-1 epochs: %d | Phase-2 epochs: %d | poison rate: %.2f | target label: %d"
          % (e1, e2, POISON_RATE, TARGET_LABEL))

    train_base = RawCIFAR(train=True, image_size=args.image_size)
    test_base = RawCIFAR(train=False, image_size=args.image_size)
    train_base = _maybe_limit(train_base, LIMIT_TRAIN_IMAGES, args.seed)

    n = len(train_base)
    poison_set, n_poison = build_poison_set(n, POISON_RATE, args.seed)
    print("Train images: %d | poison set: %d (%.1f%%) -> label %d"
          % (n, n_poison, 100.0 * n_poison / max(n, 1), TARGET_LABEL))

    runners = {
        'clean_to_backdoor': run_clean_to_backdoor,
        'poison_to_clean': run_poison_to_clean,
    }

    results = {}
    for name in SCENARIOS:
        if name not in runners:
            print("  [warn] unknown scenario '%s', skipping." % name)
            continue
        results[name] = runners[name](
            args, device, train_base, test_base, poison_set, e1, e2)

        r = results[name]
        plot_history(
            r['history'],
            os.path.join(OUTPUT_DIR, "same_domain_%s_%s.png" % (DATASET, name)),
            "%s same-domain: %s" % (DATASET, name),
            r['benign_label'], r['malicious_label'])
        m = r['metrics']
        print("  -- eval (%s) -- benign acc %.2f%% | malicious acc %.2f%% | ASR %.2f%%"
              % (name, 100 * m['benign_clean_acc'], 100 * m['malicious_clean_acc'],
                 100 * m['attack_success_rate']))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(results) >= 1:
        plot_scenario_summary(results, os.path.join(
            OUTPUT_DIR, "same_domain_%s_summary.png" % DATASET))

    # --- Summary table. ---
    print("\n==== %s same-domain summary ====" % DATASET)
    print("%-20s %14s %14s %10s %12s %12s %10s"
          % ("scenario", "tot_benign", "tot_malic", "ratio",
             "benign_acc", "malic_acc", "ASR"))
    for name in results:
        h = results[name]['history']
        m = results[name]['metrics']
        tot_b = sum(h['benign_step']); tot_m = sum(h['malicious_step'])
        print("%-20s %14.4f %14.4f %10.3f %11.2f%% %11.2f%% %9.2f%%"
              % (name, tot_b, tot_m, tot_m / max(tot_b, 1e-8),
                 100 * m['benign_clean_acc'], 100 * m['malicious_clean_acc'],
                 100 * m['attack_success_rate']))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
