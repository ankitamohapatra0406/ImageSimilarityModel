import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

from model import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = SiameseNetwork().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/siamese_model.pth",
        map_location=device
    )
)

model.eval()

df = pd.read_csv("dataset/test_pairs.csv")

scores = []
labels = []

print("-"*90)

for i, row in df.iterrows():

    img1 = Image.open(row["img1"]).convert("RGB")
    img2 = Image.open(row["img2"]).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():

        emb1 = model.encoder(img1)
        emb2 = model.encoder(img2)

    similarity = cosine_similarity(
        emb1.cpu().numpy(),
        emb2.cpu().numpy()
    )[0][0]

    score = ((similarity + 1)/2)*100

    scores.append(score)
    labels.append(row["label"])

scores = np.array(scores)
labels = np.array(labels)

best_acc = 0
best_threshold = 0

for threshold in np.arange(40,95,0.5):

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels,preds)

    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

preds = (scores >= best_threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary"
)

roc = roc_auc_score(labels, scores)

print(f"\nBest Threshold : {best_threshold:.2f}%")
print(f"Accuracy       : {best_acc*100:.2f}%")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC-AUC        : {roc:.4f}")

print("\nPositive Avg :", scores[labels==1].mean())
print("Negative Avg :", scores[labels==0].mean())