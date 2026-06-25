from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms


class SimilarityDataset(Dataset):

    def __init__(self,csv_file):

        self.data=pd.read_csv(csv_file)
        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        img1=Image.open(self.data.iloc[index,0])
        img2=Image.open(self.data.iloc[index,1])
        label=self.data.iloc[index,2]

        return (
            self.transform(img1),
            self.transform(img2),
            label
        )