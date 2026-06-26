import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import SimilarityDataset
from model import SiameseNetwork

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SimilarityDataset("dataset/train_pairs.csv")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = SiameseNetwork().to(device)

criterion = nn.CosineEmbeddingLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 30
print("Training Started...\n")


# Training
for epoch in range(epochs):

    model.train()
    total_loss = 0
    for img1, img2, label in loader:
        # Move data to device
        img1 = img1.to(device)
        img2 = img2.to(device)

        # CosineEmbeddingLoss expects labels: +1 or -1
        label = label.float()
        label[label == 0] = -1
        label = label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch + 1}/{epochs}]  Average Loss: {avg_loss:.4f}")

# Save Model
torch.save(
    model.state_dict(),
    "saved_models/siamese_model.pth"
)

print("\nTraining Completed Successfully!")
print("Model saved to saved_models/siamese_model.pth")