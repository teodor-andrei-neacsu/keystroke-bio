import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl


class TypeNet(pl.LightningModule):
    """
    Implementation of the TypeNet model from the paper:
    https://arxiv.org/pdf/2101.05570.pdf
    
    """
    
    def __init__(self, in_feat):
        super().__init__()
        self.save_hyperparameters()

        
        self.bn1 = nn.BatchNorm1d(in_feat)
        self.lstm1 = nn.LSTM(in_feat, 128, 1, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 128, 1, batch_first=True)


    def forward(self, x):
        x = self.bn1(x)
        # move features to the end
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        return x[-1]

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    model = TypeNet(5)
    print(model)

    x = torch.randn(1, 5, 10)
    print(x.shape)
    print(model(x).shape)

    






