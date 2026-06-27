from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms


class SimilarityDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        anchor = Image.open(
            self.data.iloc[index, 0]
        ).convert("RGB")

        positive = Image.open(
            self.data.iloc[index, 1]
        ).convert("RGB")

        negative = Image.open(
            self.data.iloc[index, 2]
        ).convert("RGB")

        return (
            self.transform(anchor),
            self.transform(positive),
            self.transform(negative)
        )