import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from model import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Load Model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("saved_models/siamese_model.pth", map_location=device))
model.eval()


image1_path = "dataset/original/PREIMG_20.png"
image2_path = "dataset/generated/AI_20 - Rishav.png"

image1 = Image.open(image1_path).convert("RGB")
image2 = Image.open(image2_path).convert("RGB")

tensor1 = transform(image1).unsqueeze(0).to(device)
tensor2 = transform(image2).unsqueeze(0).to(device)

with torch.no_grad():
    embedding1 = model.encoder(tensor1)
    embedding2 = model.encoder(tensor2)

similarity = cosine_similarity(
    embedding1.cpu().numpy(),
    embedding2.cpu().numpy()
)

score = ((similarity[0][0] + 1) / 2) * 100

print(f"Image 1 : {image1_path}")
print(f"Image 2 : {image2_path}")
print(f"Similarity : {score:.2f}%")
