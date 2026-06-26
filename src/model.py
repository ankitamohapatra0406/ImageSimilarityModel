import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.projection = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.projection(x)
        return x
    
class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self,img1,img2):

        feature1 = self.encoder(img1)
        feature2 = self.encoder(img2)
        return feature1, feature2