import os
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader


# https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
# https://gombru.github.io/2019/04/03/ranking_loss



# TODO:
# - how to store the sequences of the dataset?
#       - for each user, store the list of sequences

class K136M_Base(Dataset):
    """
    Classic dataset - used for softmax loss
    """
    
    def __init__(self, sessions_folder):
        self.sessions_folder = sessions_folder
        self.sessions = []
        self.users = []
        
        for session in os.listdir(sessions_folder):
            df = pd.read_csv(os.path.join(sessions_folder, session))
            self.sessions.append(df["SEQUENCE"].to_list())
            self.labels.append(df["PARTICIPANT_ID"].iloc[0])
            


    def __getitem__(self, index):




        pass

    def __len__(self):
        pass


class K136M_Contrastive(Dataset):
    
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


class K136M_Triplet(Dataset):
    
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


if __name__ == "__main__":

    df = pd.read_csv('./K136M_subset/5_keystrokes.txt')
    print(df.head())