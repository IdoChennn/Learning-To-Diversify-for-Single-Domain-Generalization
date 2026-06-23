"""Measure how much the augmented model vs. the baseline model drift (in
parameter space) when adapted from a single source domain to each target domain.

Pipeline
--------
1. Train BOTH models on the source domain the usual way by reusing the existing
   ``Trainer`` from ``train.py`` (augmented model = ``trainer.extractor`` trained
   with the full Learning-to-Diversify pipeline; baseline = ``trainer.baseline``
   trained with plain cross-entropy). The resulting weights are the reference
   point ``theta_source``.
2. For every domain, start fresh copies from ``theta_source`` and fine-tune
   them on that domain the way each model was trained:
     * the augmented model continues with the FULL Learning-to-Diversify loss
       (the same two-stage extractor/convertor min-max update as
       ``train.py``'s ``_do_epoch``), and
     * the baseline continues with plain cross-entropy.
   Each model therefore keeps adapting with its own training objective.
3. After each fine-tuning epoch t, record (a) the L2 norm of the parameter
   update ``|| theta_target(t) - theta_source ||_2`` (over all learnable
   parameters) and (b) the test accuracy on the target domain's held-out test
   split, for both models.
4. Save, per domain, a drift figure and a test-accuracy figure, each with the
   augmented and baseline curves.

This file is self-contained and does not modify any existing code; it only
imports the existing ``Trainer`` / ``get_args`` and the data/model helpers.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; we only save figures to disk
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from data import data_helper
from utils.util import fix_all_seed, loglikeli, club, conditional_mmd_rbf
from utils.contrastive_loss import SupConLoss
from train import Trainer, get_args


# Domains available per task (used to build the per-target fine-tuning runs).
TASK_DOMAINS = {
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'HOME': ['art', 'clip', 'product', 'real'],
}

# The single source domain that both models are first trained on.
SOURCE_DOMAIN = 'photo'

# Number of fine-tuning epochs per target domain. ``None`` -> use ``args.epochs``.
FINETUNE_EPOCHS = None

# Where the figures are written.
OUTPUT_DIR = 'param_drift_figures'


def snapshot_named_params(model):
    """Detached clone of every learnable parameter, keyed by name.

    BatchNorm running stats / counters are intentionally excluded (they live in
    buffers, not ``named_parameters``) so the metric reflects the actual
    optimized weights only.
    """
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def param_l2_diff(model, reference):
    """L2 norm of (current params - reference params) over all learnable params."""
    sq_sum = 0.0
    for name, p in model.named_parameters():
        sq_sum += torch.sum((p.detach() - reference[name]) ** 2).item()
    return sq_sum ** 0.5


def evaluate(model, loader, device):
    """Top-1 accuracy of ``model`` on a labeled test loader (eval mode)."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ((data, _, class_l), _, _) in loader:
            data, class_l = data.to(device), class_l.to(device)
            logits = model(data, train=False)[0]
            _, pred = logits.max(dim=1)
            correct += torch.sum(pred == class_l.data).item()
            total += class_l.size(0)
    if was_training:
        model.train()
    return float(correct) / max(total, 1)


def standard_augment(data, args):
    """Generic per-sample geometric augmentation (random zoom/crop, translation,
    horizontal flip) on normalized tensors. Mirrors ``Trainer._standard_augment``
    in ``train.py`` so the baseline gets a count-matched second view.
    """
    n, c, h, w = data.shape
    device = data.device
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
    grid = F.affine_grid(theta, data.size(), align_corners=False)
    return F.grid_sample(data, grid, align_corners=False, padding_mode='reflection')


def finetune_one_epoch(model, optimizer, loader, device, criterion, args):
    """One epoch of plain cross-entropy fine-tuning on a labeled loader.

    Used for the baseline model only. To match the source-training setup in
    ``train.py``, the baseline sees a count-matched ``[generic_aug, original]``
    batch (2N samples per step) instead of a single view, controlling for the
    2x gradient signal of the L2D-augmented model.
    """
    model.train()
    for ((data, _, class_l), _, _) in loader:
        data, class_l = data.to(device), class_l.to(device)
        optimizer.zero_grad()
        base_view = standard_augment(data, args)
        base_input = torch.cat([base_view, data])
        base_labels = torch.cat([class_l, class_l])
        logits, _ = model(base_input)
        loss = criterion(logits, base_labels)
        loss.backward()
        optimizer.step()


def l2d_finetune_one_epoch(extractor, convertor, optimizer, convertor_opt,
                           loader, args, device, con, criterion, tran):
    """One epoch of full Learning-to-Diversify fine-tuning on a labeled loader.

    This mirrors the two-stage min-max update in ``train.py``'s ``_do_epoch``:
    Stage 1 updates the classifier (``extractor``) with CE + likelihood + a
    contrastive (MI) term on a [generated, original] batch, and Stage 2 updates
    the augmentation generator (``convertor``) with the CLUB MI upper bound and a
    conditional-MMD semantic-consistency term.
    """
    extractor.train()
    for ((data, _, class_l), _, _) in loader:
        data, class_l = data.to(device), class_l.to(device)

        # Stage 1: classifier update on [augmented, original] views.
        optimizer.zero_grad()
        inputs_max = tran(torch.sigmoid(convertor(data)))
        inputs_max = inputs_max * 0.6 + data * 0.4
        data_aug = torch.cat([inputs_max, data])
        labels = torch.cat([class_l, class_l])

        logits, feats = extractor(data_aug)

        emb_src = F.normalize(feats['Embedding'][:class_l.size(0)]).unsqueeze(1)
        emb_aug = F.normalize(feats['Embedding'][class_l.size(0):]).unsqueeze(1)
        con_loss = con(torch.cat([emb_src, emb_aug], dim=1), class_l)

        mu = feats['mu'][class_l.size(0):]
        logvar = feats['logvar'][class_l.size(0):]
        y_samples = feats['Embedding'][:class_l.size(0)]
        likeli = -loglikeli(mu, logvar, y_samples)

        class_loss = criterion(logits, labels)
        loss = class_loss + args.alpha2 * likeli + args.alpha1 * con_loss
        loss.backward()
        optimizer.step()

        # Stage 2: augmentation-generator update.
        inputs_max = tran(torch.sigmoid(convertor(data, estimation=True)))
        inputs_max = inputs_max * 0.6 + data * 0.4
        data_aug = torch.cat([inputs_max, data])

        _, feats2 = extractor(x=data_aug)
        mu = feats2['mu'][class_l.size(0):]
        logvar = feats2['logvar'][class_l.size(0):]
        y_samples = feats2['Embedding'][:class_l.size(0)]
        div = club(mu, logvar, y_samples)

        e = feats2['Embedding']
        e1 = e[:class_l.size(0)]
        e2 = e[class_l.size(0):]
        dist = conditional_mmd_rbf(e1, e2, class_l, num_class=args.n_classes)

        convertor_opt.zero_grad()
        (dist + args.beta * div).backward()
        convertor_opt.step()


def make_optimizer(model, args):
    """Mirror the optimizer used for source training in ``train.py``."""
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        nesterov=True,
        momentum=0.9,
        weight_decay=0.0005,
    )


def get_domain_train_loader(args, domain):
    """Labeled training loader for a single domain (reuses the data pipeline)."""
    saved_source = args.source
    try:
        args.source = [domain]
        loader, _ = data_helper.get_train_dataloader(args, patches=False)
    finally:
        args.source = saved_source
    return loader


def get_domain_test_loader(args, domain):
    """Held-out test loader for a single domain (disjoint from its train split).

    Reuses ``get_source_test_dataloader`` (PACS ``*_test_kfold.txt`` for local
    files, or the held-out HF split), so there is no overlap with the
    fine-tuning train loader.
    """
    saved_source = args.source
    try:
        args.source = [domain]
        loader = data_helper.get_source_test_dataloader(args, patches=False)
    finally:
        args.source = saved_source
    return loader


def track_drift(model_template, source_ref, loader, test_loader, args, device,
                finetune_epochs):
    """Fine-tune a fresh copy of ``model_template`` with plain CE (baseline path).

    Returns ``(diffs, accs)``, each of length ``finetune_epochs + 1``. Index 0 is
    the pre-fine-tuning state (drift 0.0; accuracy = source model on target test
    set), and index t is the value after fine-tuning epoch t.
    """
    model = copy.deepcopy(model_template).to(device)
    optimizer = make_optimizer(model, args)
    criterion = nn.CrossEntropyLoss()

    diffs = [0.0]  # epoch 0: no update yet
    accs = [evaluate(model, test_loader, device)]
    print("    epoch  0/%d  drift = %.4f  test_acc = %.2f%%"
          % (finetune_epochs, diffs[0], 100 * accs[0]))
    for t in range(1, finetune_epochs + 1):
        finetune_one_epoch(model, optimizer, loader, device, criterion, args)
        diff = param_l2_diff(model, source_ref)
        acc = evaluate(model, test_loader, device)
        diffs.append(diff)
        accs.append(acc)
        print("    epoch %2d/%d  drift = %.4f  test_acc = %.2f%%"
              % (t, finetune_epochs, diff, 100 * acc))

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diffs, accs


def track_drift_l2d(extractor_template, convertor_template, source_ref, loader,
                    test_loader, args, device, finetune_epochs):
    """Fine-tune a fresh copy of the augmented model with the full L2D loss.

    Both the classifier (``extractor``) and the augmentation generator
    (``convertor``) are carried over from the source-trained state and keep being
    optimized via the two-stage L2D objective. Drift is still measured only on
    the classifier's parameters, matching the baseline metric.

    Returns ``(diffs, accs)`` (see ``track_drift``).
    """
    extractor = copy.deepcopy(extractor_template).to(device)
    convertor = copy.deepcopy(convertor_template).to(device)
    optimizer = make_optimizer(extractor, args)
    convertor_opt = torch.optim.SGD(convertor.parameters(), lr=args.lr_sc)
    con = SupConLoss()
    criterion = nn.CrossEntropyLoss()
    tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    diffs = [0.0]  # epoch 0: no update yet
    accs = [evaluate(extractor, test_loader, device)]
    print("    epoch  0/%d  drift = %.4f  test_acc = %.2f%%"
          % (finetune_epochs, diffs[0], 100 * accs[0]))
    for t in range(1, finetune_epochs + 1):
        l2d_finetune_one_epoch(extractor, convertor, optimizer, convertor_opt,
                               loader, args, device, con, criterion, tran)
        diff = param_l2_diff(extractor, source_ref)
        acc = evaluate(extractor, test_loader, device)
        diffs.append(diff)
        accs.append(acc)
        print("    epoch %2d/%d  drift = %.4f  test_acc = %.2f%%"
              % (t, finetune_epochs, diff, 100 * acc))

    del extractor, convertor, optimizer, convertor_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diffs, accs


def setup_args():
    """Build args mirroring ``train.main`` but pinned to the single-source setup."""
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
    # Targets for the source-training Trainer = every other domain.
    args.target = [d for d in domains if d != source]
    return args, domains, source


def plot_target(source, target, diffs_aug, diffs_base, out_path):
    epochs = list(range(len(diffs_aug)))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, diffs_aug, marker='o', label='Augmented model')
    plt.plot(epochs, diffs_base, marker='s', label='Baseline model')
    plt.xlabel('Fine-tuning epoch')
    plt.ylabel(r'Parameter difference  $\|\theta_{target}(t) - \theta_{source}\|_2$')
    plt.title('Source: %s \u2192 Target: %s' % (source.capitalize(), target.capitalize()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


def plot_target_acc(source, target, accs_aug, accs_base, out_path):
    epochs = list(range(len(accs_aug)))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [100 * a for a in accs_aug], marker='o', label='Augmented model')
    plt.plot(epochs, [100 * a for a in accs_base], marker='s', label='Baseline model')
    plt.xlabel('Fine-tuning epoch')
    plt.ylabel('Target test accuracy (%)')
    plt.title('Source: %s \u2192 Target: %s' % (source.capitalize(), target.capitalize()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved figure: %s" % out_path)


def main():
    args, domains, source = setup_args()
    finetune_epochs = FINETUNE_EPOCHS if FINETUNE_EPOCHS is not None else args.epochs

    print("Source domain: %s" % source)
    print("Fine-tuning targets: %s" % domains)
    print("Fine-tuning epochs per target: %d" % finetune_epochs)

    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage 1: train both models on the source domain (the usual way). ---
    print("\n==== Training augmented + baseline models on source domain '%s' ====" % source)
    trainer = Trainer(args, device)
    trainer.do_training()

    # Reference weights theta_source (snapshot AFTER source training).
    source_aug_ref = snapshot_named_params(trainer.extractor)
    source_base_ref = snapshot_named_params(trainer.baseline)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {}

    # --- Stage 2: fine-tune fresh copies on each domain and track drift + acc. ---
    for target in domains:
        print("\n==== Fine-tuning on target domain '%s' ====" % target)
        loader = get_domain_train_loader(args, target)
        test_loader = get_domain_test_loader(args, target)

        print("  [Augmented model - L2D fine-tuning]")
        diffs_aug, accs_aug = track_drift_l2d(trainer.extractor, trainer.convertor,
                                              source_aug_ref, loader, test_loader,
                                              args, device, finetune_epochs)
        print("  [Baseline model - CE fine-tuning]")
        diffs_base, accs_base = track_drift(trainer.baseline, source_base_ref,
                                            loader, test_loader, args, device,
                                            finetune_epochs)

        drift_path = os.path.join(
            OUTPUT_DIR, "drift_%s_to_%s.png" % (source, target))
        plot_target(source, target, diffs_aug, diffs_base, drift_path)
        acc_path = os.path.join(
            OUTPUT_DIR, "acc_%s_to_%s.png" % (source, target))
        plot_target_acc(source, target, accs_aug, accs_base, acc_path)
        summary[target] = (diffs_aug[-1], diffs_base[-1], accs_aug[-1], accs_base[-1])

    # --- Final comparison summary. ---
    print("\n==== Final parameter drift (after %d fine-tuning epochs) ====" % finetune_epochs)
    print("%-14s %12s %12s   %s" % ("target", "augmented", "baseline", "which drifts less"))
    for target, (d_aug, d_base, _, _) in summary.items():
        less = "augmented" if d_aug < d_base else "baseline"
        print("%-14s %12.4f %12.4f   %s" % (target, d_aug, d_base, less))

    print("\n==== Final target test accuracy (after %d fine-tuning epochs) ====" % finetune_epochs)
    print("%-14s %12s %12s   %s" % ("target", "augmented", "baseline", "which is better"))
    for target, (_, _, a_aug, a_base) in summary.items():
        better = "augmented" if a_aug > a_base else "baseline"
        print("%-14s %11.2f%% %11.2f%%   %s" % (target, 100 * a_aug, 100 * a_base, better))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
