import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from model import SiameseNetwork


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <image1_path> <image2_path> [model_path]")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else "saved_models/siamese_model.pth"

    # Validate paths
    for path in (img1_path, img2_path):
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            sys.exit(1)

    # Setup device and transform
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess and score
    try:
        img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
        img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = model.encoder(img1).cpu().numpy()
            emb2 = model.encoder(img2).cpu().numpy()

        similarity = cosine_similarity(emb1, emb2)[0][0]
        score = ((similarity + 1) / 2) * 100
        print(f"Similarity Score: {score:.2f}%")
    except Exception as e:
        print(f"Error processing images: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

