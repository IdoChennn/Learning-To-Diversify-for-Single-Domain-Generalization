from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# ----------------------------------------------------------------------------
# HuggingFace-backed PACS loading (dataset: "flwrlabs/pacs").
# Lets PACS download & cache automatically (like torchvision's CIFAR10) instead
# of requiring a local PACS image folder. Enabled by default for the PACS task;
# pass --data_backend=files to use the original local txt-list pipeline.
# ----------------------------------------------------------------------------
_HF_PACS_NAME = "flwrlabs/pacs"
_hf_pacs_cache = {}


def use_hf_backend(args):
    backend = getattr(args, 'data_backend', 'auto')
    if backend == 'hf':
        return True
    if backend == 'files':
        return False
    # auto: HF for PACS (no local files needed), local files for other tasks
    return args.task == 'PACS'


def _load_hf_pacs():
    if 'split' not in _hf_pacs_cache:
        from datasets import load_dataset
        print("Loading PACS via HuggingFace datasets ('%s'). "
              "First run downloads ~191MB and caches it under ~/.cache/huggingface." % _HF_PACS_NAME)
        split = load_dataset(_HF_PACS_NAME, split="train")
        idx_by_domain = {}
        for i, d in enumerate(split['domain']):
            idx_by_domain.setdefault(d, []).append(i)
        _hf_pacs_cache['split'] = split
        _hf_pacs_cache['idx_by_domain'] = idx_by_domain
        print("PACS loaded: %d images across domains %s"
              % (len(split), {k: len(v) for k, v in sorted(idx_by_domain.items())}))
    return _hf_pacs_cache['split'], _hf_pacs_cache['idx_by_domain']


def _hf_domain_indices(idx_by_domain, domain):
    if domain not in idx_by_domain:
        raise ValueError("Domain '%s' not found in %s. Available domains: %s"
                         % (domain, _HF_PACS_NAME, sorted(idx_by_domain.keys())))
    return idx_by_domain[domain]


def _hf_split_train_val(indices, val_size, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(len(indices), generator=g).tolist()
    shuffled = [indices[p] for p in perm]
    n_val = int(round(len(shuffled) * val_size))
    return shuffled[n_val:], shuffled[:n_val]


class HFImageDataset(torch.utils.data.Dataset):
    """Wraps the flwrlabs/pacs HF dataset to mirror JigsawNewDataset output:
    returns (transformed_image, 0, label) with 0-indexed labels."""

    def __init__(self, hf_split, indices, img_transformer):
        self.hf_split = hf_split
        self.indices = list(indices)
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        example = self.hf_split[self.indices[index]]
        img = example['image'].convert('RGB')
        return self._image_transformer(img), 0, int(example['label'])


def _get_train_dataloader_hf(args):
    img_transformer, _ = get_train_transformers(args)
    val_transformer = get_val_transformer(args)
    split, idx_by_domain = _load_hf_pacs()
    datasets = []
    val_datasets = []
    for dname in args.source:
        idx = _hf_domain_indices(idx_by_domain, dname)
        train_idx, val_idx = _hf_split_train_val(idx, args.val_size, args.seed)
        datasets.append(HFImageDataset(split, train_idx, img_transformer))
        val_datasets.append(HFImageDataset(split, val_idx, val_transformer))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def _get_val_dataloader_hf(args):
    img_tr = get_val_transformer(args)
    split, idx_by_domain = _load_hf_pacs()
    idx = _hf_domain_indices(idx_by_domain, args.target[0])
    dataset = ConcatDataset([HFImageDataset(split, idx, img_tr)])
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


def _get_multiple_val_dataloader_hf(args):
    img_tr = get_val_transformer(args)
    split, idx_by_domain = _load_hf_pacs()
    loaders = []
    for dname in args.target:
        idx = _hf_domain_indices(idx_by_domain, dname)
        if args.limit_target and len(idx) > args.limit_target:
            idx = idx[:args.limit_target]
            print("Using %d subset of val dataset" % args.limit_target)
        dataset = ConcatDataset([HFImageDataset(split, idx, img_tr)])
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False))
    return loaders


def get_train_dataloader(args, patches):
    if use_hf_backend(args):
        return _get_train_dataloader_hf(args)
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)

    if args.task == 'PACS':
        for dname in dataset_list:
            # name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
            name_train, labels_train = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train_kfold.txt' % dname))
            name_val, labels_val = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_crossval_kfold.txt' % dname))

            train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                          tile_transformer=tile_transformer, jig_classes=30, bias_whole_image=args.bias_whole_image)
            datasets.append(train_dataset)
            val_datasets.append(
                JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
                                  patches=patches, jig_classes=30))
    elif args.task == 'VLCS':
        for dname in dataset_list:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train.txt' % dname), args.val_size)
            train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                             tile_transformer=tile_transformer, jig_classes=30,
                                             bias_whole_image=args.bias_whole_image)
            datasets.append(train_dataset)
            val_datasets.append(
                JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
                                     patches=patches, jig_classes=30))
    elif args.task == 'HOME':
        for dname in dataset_list:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % dname), 0)#args.val_size
            train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches,
                                             img_transformer=img_transformer,
                                             tile_transformer=tile_transformer, jig_classes=30,
                                             bias_whole_image=args.bias_whole_image)
            datasets.append(train_dataset)
            val_datasets.append(
                JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
                                     patches=patches, jig_classes=30))
    else:
        raise NotImplementedError('DATA LOADER NOT IMPLEMENTED.')

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args, patches=False):
    if use_hf_backend(args):
        return _get_val_dataloader_hf(args)
    if args.task == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test_kfold.txt' % args.target[0]))
    elif args.task == 'VLCS':
        names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test.txt' % args.target[0]))
    elif args.task == 'HOME':
        names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % args.target[0]))
    else:
        raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestNewDataset(args,names, labels, patches=patches, img_transformer=img_tr, jig_classes=30)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader

def get_multiple_val_dataloader(args, patches=False):
    if use_hf_backend(args):
        return _get_multiple_val_dataloader_hf(args)
    loaders = []

    for dname in args.target:
        if args.task == 'PACS':
            names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test_kfold.txt' % dname))
        elif args.task == 'VLCS':
            names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test.txt' % dname))
        elif args.task == 'HOME':
            names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % dname))
        else:
            raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')
        img_tr = get_val_transformer(args)
        val_dataset = JigsawTestNewDataset(args,names, labels, patches=patches, img_transformer=img_tr, jig_classes=30)
        if args.limit_target and len(val_dataset) > args.limit_target:
            val_dataset = Subset(val_dataset, args.limit_target)
            print("Using %d subset of val dataset" % args.limit_target)
        dataset = ConcatDataset([val_dataset])
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        loaders.append(loader)
    return loaders


# JIGSAW
def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tile_tr = []
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)

# RSC
# def get_train_transformers(args):
#     img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
#     #img_tr = [transforms.Resize((args.image_size, args.image_size))]
#     #img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
#     if args.random_horiz_flip > 0.0:
#         img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
#     if args.jitter > 0.0:
#         img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
#     img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
#     img_tr.append(transforms.ToTensor())
#     img_tr.append(transforms.Normalize([0.5] * 3, [0.5] * 3))#[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


# def get_target_jigsaw_loader(args):
#     img_transformer, tile_transformer = get_train_transformers(args)
#     name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
#     dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
#                             tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     return loader
