from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms


class SimilarityDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        if "train" in csv_file:

            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor()
         ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img1 = Image.open(
            self.data.iloc[index,0]
        ).convert("RGB")

        img2 = Image.open(
            self.data.iloc[index,1]
        ).convert("RGB")


        label = float(self.data.iloc[index, 2])

        return (
            self.transform(img1),
            self.transform(img2),
            label
        )