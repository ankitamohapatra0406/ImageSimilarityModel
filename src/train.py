import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import SimilarityDataset
from model import SiameseNetwork


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = SimilarityDataset("dataset/train_triplets.csv")

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = SiameseNetwork().to(device)

    criterion = nn.TripletMarginLoss(margin=0.5, p=2)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    epochs = 3

    print("Training Started...\n")

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for anchor, positive, negative in loop:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_embedding = model.encoder(anchor)
            positive_embedding = model.encoder(positive)
            negative_embedding = model.encoder(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch + 1}/{epochs}]  Average Loss: {avg_loss:.4f}")

        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/siamese_model.pth")
        print(f"Model checkpoint saved to saved_models/siamese_model.pth")

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    main()
