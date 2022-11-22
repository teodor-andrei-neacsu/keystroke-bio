
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class KaggleDS(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def preprocess_data(self, data):
        """
        Feature engineering/selection
        
        """



        return data
    

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
