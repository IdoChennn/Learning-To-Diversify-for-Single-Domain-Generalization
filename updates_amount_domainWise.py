"""Domain-wise parameter-drift comparison (no poisoning).

Question
--------
Starting from a single source-domain-trained classifier, how far do the model
parameters move when we fine-tune (transfer) it to each *different* target
domain? Some target domains are "closer" to the source than others, and we
expect the parameter drift to reflect that distance.

Protocol
--------
1. Train ONE classifier on the clean source-domain training samples -> theta_S.
2. For every other PACS domain, fine-tune a fresh copy initialised from the
   SAME theta_S on that domain's clean training samples.
3. Each epoch, record the parameter drift from the source weights:
       drift(epoch) = || theta_epoch_end - theta_S ||_2
4. Produce a SINGLE figure with one drift-vs-epoch curve per target domain.

No poisoning is involved here. This file is self-contained and reuses the
repo's data pipeline (``data.data_helper``), model (``models.resnet.resnet18``)
and ``train.get_args``. It does not modify any existing code.
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

# Source domain used to pre-train the shared weights theta_S.
SOURCE_DOMAIN = 'photo'
# Target domains to transfer to. None = every task domain except the source;
# otherwise a list of domain names to restrict to.
TARGET_DOMAINS = None

# Training length (epochs). Fall back to args.epochs if these are None.
SOURCE_EPOCHS = 20
TARGET_EPOCHS = 20

# Cap the number of training images per phase (None = all). Useful for quick runs.
LIMIT_TRAIN_IMAGES = None

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
    """Deterministic dataset of (image[0,1], 0, label) for ``domain``'s train split.

    Mirrors the repo's training-split selection for both the HuggingFace PACS
    backend and the local-file backend.
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


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    running = 0.0
    for x01, labels, _ in loader:
        x01 = x01.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(_normalize(x01))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


def train_source(args, device, epochs):
    """Train a single classifier on clean source-domain data; return its weights."""
    fix_all_seed(args.seed)
    model = make_resnet(args, device)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, SOURCE_DOMAIN)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print("  [source] training on %d images for %d epochs" % (len(base), epochs))
    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, criterion, loader, device)
        scheduler.step()
        print("    [source] epoch %2d/%d  loss = %.4f" % (epoch + 1, epochs, loss))
    return copy.deepcopy(model.state_dict())


def train_target_drift(args, device, source_state, epochs, target_domain):
    """Fine-tune a copy initialised from ``source_state`` on ``target_domain``;
    return the per-epoch parameter drift from the source weights theta_S."""
    model = make_resnet(args, device)
    model.load_state_dict(source_state)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.8)))
    criterion = nn.CrossEntropyLoss()

    base = get_raw_train_dataset(args, target_domain)
    base = _maybe_limit(base, LIMIT_TRAIN_IMAGES, args.seed)
    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    theta_S = flat_params(model).clone()  # == source weights
    epochs_axis, drift = [], []
    print("  [target %s] fine-tuning on %d images for %d epochs"
          % (target_domain, len(base), epochs))
    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, criterion, loader, device)
        scheduler.step()
        d = torch.norm(flat_params(model) - theta_S).item()
        epochs_axis.append(epoch + 1)
        drift.append(d)
        print("    [%s] epoch %2d/%d  loss = %.4f  drift = %.4f"
              % (target_domain, epoch + 1, epochs, loss, d))
    return epochs_axis, drift


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------

def plot_domain_drift(drifts, src, out_path):
    """One figure: parameter drift from theta_S vs epoch, one curve per target."""
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(8, 5.5))
    for i, (tgt, (epochs_axis, drift)) in enumerate(drifts.items()):
        plt.plot(epochs_axis, drift, 'o-', color=cmap(i % 10), label=tgt)
    plt.xlabel('target-training epoch')
    plt.ylabel(r'parameter drift  $\|\theta_{epoch} - \theta_S\|_2$')
    plt.title('Parameter drift from source (%s) across target domains'
              % src.capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend(title='target domain')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


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
    # data_helper uses args.source for backend split selection.
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

    print("Task: %s | Source: %s | Targets: %s" % (args.task, src, ", ".join(targets)))
    print("Source epochs: %d | Target epochs: %d (no poisoning)"
          % (src_epochs, tgt_epochs))

    # --- Step 1: shared source-domain weights theta_S (trained ONCE). ---
    print("\n==== Step 1: train source model on clean '%s' ====" % src)
    source_state = train_source(args, device, src_epochs)

    # --- Step 2-3: transfer to each target domain from the SAME theta_S. ---
    drifts = {}
    for tgt in targets:
        args.target = [tgt]
        print("\n==== Transfer to '%s' ====" % tgt)
        epochs_axis, drift = train_target_drift(
            args, device, source_state, tgt_epochs, tgt)
        drifts[tgt] = (epochs_axis, drift)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 4: single drift figure. ---
    plot_domain_drift(drifts, src, os.path.join(
        OUTPUT_DIR, "param_drift_%s_domainwise.png" % src))

    # --- Summary table. ---
    print("\n==== Final parameter drift from source %s ====" % src)
    print("%-14s %12s" % ("target", "final_drift"))
    for tgt in targets:
        print("%-14s %12.4f" % (tgt, drifts[tgt][1][-1]))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
