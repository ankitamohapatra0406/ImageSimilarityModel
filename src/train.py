import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import SimilarityDataset
from model import SiameseNetwork


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SimilarityDataset(
        "dataset/train_triplets.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    model = SiameseNetwork().to(device)

    criterion = nn.TripletMarginLoss(
        margin=0.5,
        p=2
    )

    optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4
)

    epochs = 30

    print("Training Started...\n")

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for anchor, positive, negative in loader:

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_embedding = model.encoder(anchor)
            positive_embedding = model.encoder(positive)
            negative_embedding = model.encoder(negative)

            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch+1}/{epochs}]  Average Loss: {avg_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "saved_models/siamese_model.pth"
    )

    print("\nTraining Completed Successfully!")
    print("Model saved to saved_models/siamese_model.pth")


if __name__ == "__main__":
    main()