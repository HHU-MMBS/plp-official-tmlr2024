import random
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path

from loaders import get_dataset,get_embeds_path

class EmbedNN(Dataset):
    def __init__(self,
                 knn_path,
                 transform,
                 k=-1, # -1 means use all neighbors
                 dataset_name="CIFAR100",
                 datapath=get_embeds_path(),
                 p_self=0.0,
                 precompute_arch=None):
        super().__init__()
        self.transform = transform
        self.neighbors = self.define_knn(knn_path, k)
        self.p_self = p_self
        self.datapath = datapath
        self.precompute_arch = precompute_arch

        self.dataset = get_dataset(
            dataset_name,
            datapath=datapath,
            transform=None,
            train=True,
            download=True,
            precompute_arch=precompute_arch)
        
    def define_knn(self, knn_path, k):
        complete_neighbors = torch.load(knn_path)
        if k < 0 and torch.is_tensor(complete_neighbors):
            k = complete_neighbors.size(1) 
        
        if torch.is_tensor(complete_neighbors):
            return complete_neighbors[:, :k].cpu()   
        else:
            # ignoring k. NN is a list of lists due to varying number of neighbors
            return complete_neighbors

    def get_transformed_imgs(self, idx, *idcs):
        img_indist, label = self.dataset[idx]
        rest_imgs = (self.dataset[i][0] for i in idcs)
        return self.transform(img_indist, *rest_imgs), label

    def draw_pair(self, idx):
        # Knn pair
        if random.random() > self.p_self:
            pair_idx = np.random.choice(self.neighbors[idx], 1)[0]
        # self-pair TODO should be removed
        else:
            pair_idx = idx
        return pair_idx
    
    def __getitem__(self, idx):
        if len(self.neighbors[idx])!=0:
            pair_idx = self.draw_pair(idx)
        else:
            raise ValueError("No KNN pairs found for this image")
        return self.get_transformed_imgs(idx, pair_idx)

    def __len__(self):
        return len(self.dataset) #if not self.mutual_knn else len(self.dataset) - len(self.empty_indices)


class HardPosNN(EmbedNN):
    def __init__(self, knn_path, *args, **kwargs):
        super().__init__(knn_path, *args, **kwargs)
        p = Path(knn_path).parent
        nn_p = p / 'hard_pos_nn.pt'
        if nn_p.is_file():
            self.complete_neighbors = torch.load(nn_p)
        else:
            emb = torch.load(p / 'embeddings.pt')
            emb /= emb.norm(dim=-1, keepdim=True)
            d = emb @ emb.T
            labels = torch.tensor(self.dataset.targets)
            same_label = labels.view(1, -1) == labels.view(-1, 1)
            # Find minimum number of images per class
            k_max = same_label.sum(dim=1).min()
            d.fill_diagonal_(-2)
            d[torch.logical_not(same_label)] = -torch.inf
            self.complete_neighbors = d.topk(k_max, dim=-1)[1]
            torch.save(self.complete_neighbors, nn_p)
        self.neighbors = self.complete_neighbors[:, :self.k]


class HardPosFarN(HardPosNN):
    """Farthest neighbors with same label"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors = self.complete_neighbors[:, self.k:]


class WeightedEmbedNN(EmbedNN):
    def __init__(self, knn_path, *args, **kwargs):
        super().__init__(knn_path, *args, **kwargs)
        self.weight = torch.load(Path(knn_path).parent / 'knn_dists.pt')[:, :self.k].numpy()
        self.weight -= self.weight.min()
        self.weight /= self.weight.sum(axis=-1).reshape(-1, 1)

    def __getitem__(self, idx):
        # KNN pair
        if random.random() > self.p_self:
            pair_idx = np.random.choice(self.neighbors[idx], 1, p=self.weight[idx])[0]
        # self-pair
        else:
            pair_idx = idx

        if not self.sample_id_neg:
            return self.get_transformed_imgs(idx, pair_idx)

        # samples a random indist img
        return self.get_transformed_imgs(*self.sample_neg(idx, pair_idx))



class SSEmbedNN(EmbedNN):
    def __init__(self,
                 knn_path,
                 knn_path_true_pos,
                 transform,
                 k=-1, # -1 means use all neighbors
                 dataset_name="CIFAR100",
                 datapath=get_embeds_path(),
                 p_self=0.0,
                 precompute_arch=None):
        super().__init__(knn_path, transform, k=k, 
                            dataset_name=dataset_name,datapath=datapath, 
                            p_self=p_self, precompute_arch=precompute_arch)
        self.true_neighbors =  self.define_knn(knn_path_true_pos, k=-1)
        self.neighbors = self.define_knn(knn_path, k)
        # see if the NN come from all the dataset or just the UNlabelled part
        self.all_knns = True if len(self.neighbors)==len(self.dataset) else False
        
        if self.precompute_arch is not None:
            self.p_self = 0.0
            self.num_labeled_data = self.dataset.original.len_labelled
        else:
            # image dataset has labelled and unlabelled data (MergeDataset class)
            self.num_labeled_data = self.dataset.len_labelled
    def create_sampler(self, distributed=False):
        return self.dataset.original.create_sampler(distributed)
            
    def draw_pair(self, idx, true_pos):
        nn_set_all = self.true_neighbors if true_pos else self.neighbors
        # Knn pair comes from self.dataset.labelled_dataset[idx] or self.dataset.unlabelled_dataset[idx]
        if true_pos and idx < self.num_labeled_data:
            nn_set = nn_set_all[idx] 
            pair_idx = np.random.choice(nn_set, 1)[0]
        else:
            # when NN are computed on the whole dataset
            if self.all_knns:
                nn_set = nn_set_all[idx] 
            else:
                # when NN are computed ONLY on the unlabelled part
                idx_new = idx - self.num_labeled_data
                nn_set = nn_set_all[idx_new]
            
            # Always draw a random pair from the NN set
            pair_idx = np.random.choice(nn_set, 1)[0]
        return pair_idx
    
    def __getitem__(self, idx):
        datapoint = self.dataset[idx]
        # Image is always first and true_pos bool val is always last
        ref_img, true_pos = datapoint[0], datapoint[-1]
        pair_idx = self.draw_pair(idx, true_pos)
        pair_img = self.dataset[pair_idx][0]
        if self.transform is not None and self.precompute_arch is None:
            try:
                ref_img, pair_img = self.transform(ref_img, pair_img)
            except:
                ref_img, pair_img = self.transform(ref_img), self.transform(pair_img)
        return (ref_img, pair_img), true_pos
        