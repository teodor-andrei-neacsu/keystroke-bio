import torch
import torch.nn as nn
import torch.utils.data as data


class K136M_Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)