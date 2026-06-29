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

        self.cache = {}

    def __len__(self):
        return len(self.data)

    def _get_image(self, path):
        path = path.replace('\\', '/')
        if path not in self.cache:
            img = Image.open(path).convert("RGB")
            self.cache[path] = self.transform(img)
        return self.cache[path]

    def __getitem__(self, index):

        anchor = self._get_image(self.data.iloc[index, 0])
        positive = self._get_image(self.data.iloc[index, 1])
        negative = self._get_image(self.data.iloc[index, 2])

        return anchor, positive, negative