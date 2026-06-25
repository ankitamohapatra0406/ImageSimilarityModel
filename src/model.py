import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self,x):
        x=self.network(x)
        return x.view(x.size(0),-1)


class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder=Encoder()

    def forward(self,img1,img2):

        feat1=self.encoder(img1)

        feat2=self.encoder(img2)

        return feat1,feat2