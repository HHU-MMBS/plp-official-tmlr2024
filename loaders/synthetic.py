from pathlib import Path

import torch
import numpy as np
from torchvision import transforms


class SyntheticData(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        path = Path(path)
        if ".npy" in path.name:
            self.data = np.load(path)
            # switch to channels first ordering
            if int(self.data.shape[-1])==3:
                self.data = self.data.transpose(0,3,1,2)
            self.np = True
        else:
            self.data = torch.load(path)
            self.np = False
        self.transform = None
        if transform:
            # transform includes ToTensor() already
            self.transform = transforms.Compose([transforms.ToPILImage(), transform])

    def __getitem__(self, item):
        x = torch.from_numpy(self.data[item, ...])/255.0 if self.np else self.data[item, ...]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.data.shape[0]


class DatasetWithLabels(torch.utils.data.Dataset):
    """ Combines data from one dataset with labels from another dataset. """

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        _, y = self.dataset2[index]
        x = self.dataset1[index]
        return x, y

