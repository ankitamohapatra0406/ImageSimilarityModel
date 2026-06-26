import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from model import SiameseNetwork

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Model
model = SiameseNetwork().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/siamese_model.pth",
        map_location=device
    )
)

model.eval()

# Load Test CSV
df = pd.read_csv("dataset/test_pairs.csv")

correct = 0

positive_scores = []
negative_scores = []

print("-" * 80)

for index, row in df.iterrows():

    img1 = Image.open(row["img1"]).convert("RGB")
    img2 = Image.open(row["img2"]).convert("RGB")

    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model.encoder(t1)
        emb2 = model.encoder(t2)

    similarity = cosine_similarity(
        emb1.cpu().numpy(),
        emb2.cpu().numpy()
    )[0][0]

    score = ((similarity + 1) / 2) * 100

    prediction = 1 if score >= 70 else 0

    label = row["label"]

    if prediction == label:
        correct += 1

    if label == 1:
        positive_scores.append(score)
    else:
        negative_scores.append(score)

    print(
        f"{index+1:02d}. "
        f"{score:6.2f}%   "
        f"Pred={prediction}   "
        f"True={label}"
    )

print("-" * 80)

accuracy = correct / len(df) * 100
print(f"\nAccuracy : {accuracy:.2f}%")
print(f"Positive Avg : {sum(positive_scores)/len(positive_scores):.2f}%")
print(f"Negative Avg : {sum(negative_scores)/len(negative_scores):.2f}%")