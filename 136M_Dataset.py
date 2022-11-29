import torch
import torch.nn as nn
import torch.utils.data as data


# https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
# https://gombru.github.io/2019/04/03/ranking_loss

class K136M_Base(data.Dataset):
    
    def __init__(self, ):
        pass
        
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class K136M_Contrastive(data.Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        """
        Returns a tuple containing (seq1, seq2, label)
        where label is 1 if images are from the same class and 0 otherwise.
        """
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


    def preprocess(self,):
        pass


class K136M_Triplet(data.Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        """
        Returns a tuple containing (seq1, seq2, seq3) - Anchor, Positive, Negative
        """
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


    def preprocess(self,):
        pass