import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import SimilarityDataset
from model import SiameseNetwork

# Load Dataset
dataset = SimilarityDataset("dataset/pairs.csv")

loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Initialize Model
model=SiameseNetwork()
criterion=nn.CosineEmbeddingLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs=10
print("Training Started...\n")

for epoch in range(epochs):
    total_loss=0
    for img1,img2,label in loader:

        label=label.float()
        optimizer.zero_grad()
        output1,output2=model(img1,img2)

        loss=criterion(output1,output2,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss : {total_loss:.4f}")

torch.save(
    model.state_dict(),
    "saved_models/siamese_model.pth"
)

print("\nModel Saved Successfully!")