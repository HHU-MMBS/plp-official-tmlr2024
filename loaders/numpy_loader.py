import argparse
import fnmatch
from ast import literal_eval
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets as tds, transforms


class NumpyDataset(Dataset):
    """
    similar to torch.utils.data.TensorDataset(dataset, dummy_targets)
    """
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.dataset = torch.from_numpy(np.float32(np.load(path).transpose((0,3,1,2))))/255.
        self.targets = torch.zeros(len(self.dataset)).long()

    def __getitem__(self, index):
        x = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[index]

    def __len__(self):
        return len(self.dataset)

class CIFAR100C(NumpyDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)

class CIFAR10C(NumpyDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)