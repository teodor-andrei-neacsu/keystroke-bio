import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

from K136M_ds import K136M_Base


class TypeNet(nn.Module):
    """
    Implementation of the TypeNet model from the paper:
    https://arxiv.org/pdf/2101.05570.pdf
    
    """
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_feat)
        self.lstm1 = nn.LSTM(in_feat, 128, 1, batch_first=True, dropout=0.2, bias=True)
        self.drop = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 128, 1, batch_first=True, dropout=0.2, bias=True)
        

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn1(x)
        # move features to the end
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)        
        # get last output
        x = x[:, -1, :]

        return x

class ClfNet(pl.LightningModule):
    
    def __init__(self, emb_generator, num_classes):
        super().__init__()
        self.emb_generator = emb_generator
        self.fc = nn.Linear(128, num_classes)
        self.cross_loss = nn.CrossEntropyLoss()

    def step(self, batch):
        x, y = batch
        embs = self.emb_generator(x)
        preds = self.fc(embs)

        loss = self.cross_loss(preds, y)
        return loss, embs

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, embs = self.step(batch)
        
        

        self.log('train_loss', loss)


        return loss

    def test_step(self, batch, batch_idx):
        pass

    # Optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer    

class SiameseNet(pl.LightningModule):

    def __init__(self, emb_generator):
        super().__init__()
        self.emb_generator = emb_generator
        self.contrastive_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def step(self, batch):
        x1, x2, label = batch
        emb1 = self.emb_generator(x1)
        emb2 = self.emb_generator(x2)
        loss = self.contrastive_loss(emb1, emb2, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
  
class TripletNet(pl.LightningModule):

    def __init__(self, emb_generator):
        super().__init__()
        self.emb_generator = emb_generator
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def step(self, batch):
        anch, pos, neg = batch
        anch_emb = self.emb_generator(anch)
        pos_emb = self.emb_generator(pos)
        neg_emb = self.emb_generator(neg)
        loss = self.triplet_loss(anch_emb, pos_emb, neg_emb)
        return loss

    def training_step(self, batch, batch_idx): 
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    max_seq_len = 75
    enroll_cnt = 5
    
    


    train_dataset = K136M_Base("./K136M_train/", max_len=75)
    test_dataset = K136M_Base("./K136M_test/", max_len=75)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


    emb_generator = TypeNet(5, 10)
    model = ClfNet(emb_generator, 10)
    trainer = pl.Trainer(gpus=0, max_epochs=20, log_every_n_steps=10)
    trainer.fit(model, train_dataloader)
