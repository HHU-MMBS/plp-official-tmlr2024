import argparse
import fnmatch
from ast import literal_eval
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset
from torchvision import datasets as tds, transforms

import loaders 

# data sets with number of classes
import model_builders
from loaders.synthetic import SyntheticData, DatasetWithLabels
from loaders.imagenet import IN_C

_DEFAULT_PATH = '/home/shared/DataSets/vision_benchmarks'
_PRECOMPUTED_PATH = '/home/shared/embeddings'  #'./data'
_IMAGENET_PATH = '/home/shared/DataSets/ILSVRC/Data/CLS-LOC/'

_TORCHVISION_DS = ["CIFAR10","CIFAR100", "STL10", "SVHN", "LSUN", "Places365"]
_OOD_DATA_ONLY = ["NINCO", "SUN", "inat", "Places", "texture", 
                 "IN_O", "IN_A", "SVHN", "Places365",
                  "food", "tiny_imagenet", "STL10", "CIFAR10", 
                  "CIFAR100", "IN30", "flowers", "dogs", "CUB200",  "pets",  "LSUN"]
_DATASET_PATHS={
    "IN1K": [f"{_IMAGENET_PATH}/train", f"{_IMAGENET_PATH}/val_structured"],
    # Clustering datasets
    "IN50": [f"{_IMAGENET_PATH}/train", f"{_IMAGENET_PATH}/val_structured"],
    "IN100": [f"{_IMAGENET_PATH}/train", f"{_IMAGENET_PATH}/val_structured"],
    "IN200": [f"{_IMAGENET_PATH}/train", f"{_IMAGENET_PATH}/val_structured"],
    "tiny_imagenet":[f"{_DEFAULT_PATH}/tiny-imagenet-200/train", f"{_DEFAULT_PATH}/tiny-imagenet-200/validation"],
    # Small scale datasets
    'IN30': [ f"{_DEFAULT_PATH}/ImageNet30/train", f"{_DEFAULT_PATH}/ImageNet30/test"],
    "dogs": [f"{_DEFAULT_PATH}/StanfordDogs/images", f"{_DEFAULT_PATH}/StanfordDogs/images"],
    "CUB200": [f"{_DEFAULT_PATH}/CUB_200_2011/images", f"{_DEFAULT_PATH}/CUB_200_2011/images"],
    "LSUN": [f"{_DEFAULT_PATH}/LSUN"]*2,
    "flowers": [f"{_DEFAULT_PATH}/flowers102"]*2,
    "Places365": [f"{_DEFAULT_PATH}/Places365"]*2,
    "food": [f"{_DEFAULT_PATH}/food-101/images"]*2,
    "CIFAR10": [_DEFAULT_PATH]*2,
    "CIFAR20": [_DEFAULT_PATH]*2,
    "CIFAR100": [_DEFAULT_PATH]*2,
    "STL10": [_DEFAULT_PATH]*2,
    "SVHN": [_DEFAULT_PATH]*2,  
    "pets": [f"{_DEFAULT_PATH}/pets"]*2,
    # Imagenet domain generalization benchmarks
    "IN_O": [f"{_DEFAULT_PATH}/imagenet-o"]*2,
    "IN_A": [f"{_DEFAULT_PATH}/imagenet-a"]*2,
    # New OOD banchmarks
    "texture": [f"{_DEFAULT_PATH}/dtd_test"]*2,
    "SUN": [f"{_DEFAULT_PATH}/SUN"]*2,
    "inat": [f"{_DEFAULT_PATH}/iNaturalist"]*2,
    "Places": [f"{_DEFAULT_PATH}/Places"]*2,
    "NINCO" : [f"{_DEFAULT_PATH}/NINCO/NINCO_OOD_classes"]*2,
}

_DATASETS = {
    "CIFAR100": 100,
    "CIFAR10": 10,
    "CIFAR20": 20,
    "STL10": 10,
    "SVHN": 10,
    "IN1K": 1000,
    "IN50": 50,
    "IN100": 100,
    "IN200": 200,
    "IN30":30,
    "tiny_imagenet":200,
    "flowers": 102,
    "food": 101,
    "dogs": 120,
    "CUB200": 200,
    "Places365": 365,
    "pets": 37,
    "LSUN": 10,
    # NEW OOD benchmarks
    "texture": 47,
    "Places": 50,
    "inat": 110,
    "NINCO": 64,
    "SUN": 50,
    "sketch":1000,
    "IN_A": 200,
    "IN_O": 200,
    "IN_R": 200,
    "IN_V2": 1000,
}

def get_default_path(dataset_name):
    if "IN" in dataset_name:
        return _IMAGENET_PATH
    return _DEFAULT_PATH

def get_datapath(dataset, train=True):
    if dataset.startswith('synthetic'):
        dataset = _synthetic_base_dataset(dataset)
    if dataset.startswith('pseudo') or dataset.startswith('Pseudo'):
        dataset = dataset.replace('pseudo', '').replace('Pseudo', '')
    if dataset.startswith('Gen') or dataset.startswith('gen'):
        dataset = dataset.replace('gen', '').replace('Gen', '')
    idx = 0 if train else 1
    return _DATASET_PATHS[dataset][idx]


def get_embeds_path():
    return _PRECOMPUTED_PATH


def _synthetic_base_dataset(dataset):
    return dataset.split('/')[-1]


def _synthetic_dataset_path(dataset):
    return Path(get_datapath(dataset)) / dataset


def available_datasets(pattern=None, ood_only=False, all_datasets=False):
    dsets = list(_DATASETS) if not ood_only else list(_OOD_DATA_ONLY)
    dsets = list(_OOD_DATA_ONLY) + list(_DATASET_PATHS) if all_datasets else dsets
    if pattern is None:
        return dsets
    return tuple(fnmatch.filter(dsets, pattern))


def get_num_classes(dataset):
    dataset = _synthetic_base_dataset(dataset)
    dataset = remove_dataset_prefix(dataset)
    return _DATASETS[dataset]


def get_class_names(dataset):
    if 'CIFAR' in dataset:
        dset = get_dataset(dataset)
        return dset.classes
    elif 'IN1K' in dataset:
        class_name_path = Path(__file__).parent / 'imagenet1000_clsidx_to_labels.txt'
        with open(class_name_path, 'r') as f:
            class_name_dict = literal_eval(f.read())
        return [class_name_dict[i].split(',')[0] for i in range(1000)]
    else:
        raise NotImplementedError


def get_dataset(indist, datapath=None, train=True, transform=None, download=True, precompute_arch=None,
                pseudo_path=None, gen_split='all'):
    if indist.startswith('synthetic'):
        path = _synthetic_dataset_path(indist)
        dataset = SyntheticData(path, transform=transform)
        base_dataset = get_dataset(_synthetic_base_dataset(indist), datapath=datapath, train=False, download=True)
        return DatasetWithLabels(dataset, base_dataset)
    elif indist.startswith('pseudo') or indist.startswith('Pseudo'):
        dataset_name = indist.replace('pseudo', '').replace('Pseudo', '')
        # Fix for precomputed pseudo labels
        if not train and pseudo_path is not None:
            pseudo_path = str(pseudo_path).replace('train', 'val')
        if not precompute_arch:
            return PseudoDataset(dataset_name, pseudo_path, transform=transform, train=train )        
        else:
            return PrecomputedPseudoDataset(pseudo_path, dataset_name, 
                                            precompute_arch, train, 
                                            datapath=get_embeds_path())

    new_resolution = re.findall('^resolution([0-9]+)/', indist)
    if new_resolution:
        new_resolution = int(new_resolution[0])
        indist = re.sub('^resolution[0-9]+/', '', indist)
        resize = transforms.Resize(new_resolution)
        transform = resize if transform is None else transforms.Compose([resize, transform])

    datapath = get_datapath(indist, train) if datapath is None else datapath
    if precompute_arch:
        return PrecomputedEmbeddingDataset(
            indist=indist,
            arch=precompute_arch,
            datapath=get_embeds_path(), # assumes embeddings are saved 
            train=train)

    load_obj = tds if indist in _TORCHVISION_DS else loaders
    if indist == "STL10":
        split = 'train' if train else 'test'
        return getattr(load_obj, indist)(root=datapath,
                        split=split,
                        download=download, transform=transform)
    elif "Places365" in indist:
        split = 'val' if not train else 'train-standard'
        small = True if "Places365"==indist else False
        return getattr(load_obj, "Places365")(root=datapath,
                        split=split, small=small,
                        download=False, transform=transform)
    elif "LSUN"==indist:
        classes = 'test'
        return getattr(load_obj, indist)(root=datapath,
                classes=classes, transform=transform)
    elif "CIFAR20"==indist:
        return getattr(load_obj, indist)(root=datapath,
                train=train,
                download=download, transform=transform)
    # Corrupted datasets from Hendrycks et al. 2019 https://github.com/hendrycks/robustness/
    elif "CIFAR100C" == indist or "CIFAR10C" ==indist:
        # Gaussian noise is supported for now
        path = get_datapath(indist, train=False)
        return getattr(loaders.numpy_loader, indist)(path=path, transform=transform)
    elif "IN_C" == indist:
        return IN_C(datapath, transform=transform)        
    else:
        if indist in _TORCHVISION_DS:
            if indist == "SVHN":
                split = "train" if train else "test"
                return getattr(load_obj, indist)(root=datapath,
                    split=split, download=download, transform=transform)
        
            return getattr(load_obj, indist)(root=datapath,
                train=train,
                download=download, transform=transform)
        else:
            # Assumes ImageFolder class-based folder structure
            return get_vision_ds(root=datapath,
               transform=transform)


def remove_dataset_prefix(dataset_name, prefixes=["Pseudo", "pseudo", "Gen", "gen"]):
    for prefix in prefixes:
        if dataset_name.startswith(prefix):
            dataset_name = dataset_name.replace(prefix, "")
    return dataset_name
    
def check_dataset(dataset):
    dataset_name = remove_dataset_prefix(dataset)
    if dataset_name.startswith('synthetic'):
        path = _synthetic_dataset_path(dataset_name)
        if not path.is_file():
            raise argparse.ArgumentError(
                None, f"File {path} not found.")
    elif _synthetic_base_dataset(dataset_name) not in available_datasets(all_datasets=True):
        raise argparse.ArgumentError(
            None, f"Dataset {dataset} not available. "
            f"Available datasets are: {available_datasets()}")    
    return dataset


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, indist, arch, train, datapath):
        super().__init__()
        self.emb, self.targets = model_builders.load_embeds(
            arch=arch,
            dataset=indist,
            datapath=datapath,
            with_label=True,
            test=not train,
            label_key="label_split") 
        self.original = get_dataset(indist, 
                                    train=train, 
                                    transform=transforms.Compose([
                                            transforms.Resize(size=(224,224)),
                                            transforms.ToTensor()]))

    def __getitem__(self, index):
        return self.emb[index], self.targets[index]

    def __len__(self):
        return len(self.emb)
    
class PrecomputedPseudoDataset(PrecomputedEmbeddingDataset):
    def __init__(self, pseudo_path, indist, arch, train, datapath):
        super().__init__(indist, arch, train, datapath)
        if indist.startswith('pseudo') or indist.startswith('Pseudo'):
            indist = indist.replace('pseudo', '').replace('Pseudo', '')
        self.emb = model_builders.load_embeds(
            arch=arch,
            dataset=indist,
            datapath=datapath,
            with_label=False,
            test=not train)
        try:
            if pseudo_path is not None:
                if "train" in pseudo_path and not train:
                    pseudo_path = pseudo_path.replace('train', 'val')
                self.targets = torch.load(pseudo_path)
        except:
            raise ValueError(f"Could not load pseudo labels from {pseudo_path}")


def get_vision_ds(**kwargs):
    return tds.ImageFolder(**kwargs)    

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, pseudo_label_path, transform=None, train=True):
        super().__init__()
        self.dataset = get_dataset(dataset_name, transform=transform, train=train)
        try:
            self.pseudo_labels = torch.load(pseudo_label_path)
        except:
            raise ValueError(f"Could not load pseudo labels from {pseudo_label_path}")
        self.labels = self.pseudo_labels # for compatibility with evaluation code
        assert len(self.dataset) == len(self.pseudo_labels), f"Pseudo labels and dataset have different lengths {len(self.pseudo_labels)} vs {len(self.dataset)}"

    def __getitem__(self, idx):
        x, _ = self.dataset[idx] # ignore real label
        y = self.pseudo_labels[idx]
        return x, y

    def __len__(self):
        return len(self.dataset)

def get_ood(dataset, return_all=True):
    """Returns a list of OOD datasets for a given in-distribution dataset
    Args:
        dataset (str): Name of in-distribution dataset
    """
    dataset = remove_dataset_prefix(dataset)
    if dataset == "CIFAR100":
        out_dist = ["CIFAR10"]
    elif dataset == "CIFAR10":
        out_dist = ["CIFAR100"]
    elif dataset == "IN1K":
        out_dist = ["inat", "SUN", "Places", "IN_O" ,"texture", "NINCO"] if return_all else ["NINCO"]
    return out_dist
