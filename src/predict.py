import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from model import SiameseNetwork

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#trained model loading
model = SiameseNetwork()
model.load_state_dict(torch.load("saved_models/siamese_model.pth"))
model.eval()

# Img loading
reference_image = Image.open("dataset/original/fight.jpg")
generated_image = Image.open("dataset/generated/war_edited.jpg")

# Img Preprocessing
reference_tensor = transform(reference_image).unsqueeze(0)
generated_tensor = transform(generated_image).unsqueeze(0)

# Feature Embeddings generated
with torch.no_grad():
    reference_embedding = model.encoder(reference_tensor)
    generated_embedding = model.encoder(generated_tensor)

# Similarity calc
similarity = cosine_similarity(
    reference_embedding.numpy(),
    generated_embedding.numpy()
)
score = ((similarity[0][0] + 1) / 2) * 100
print(f"Similarity Score : {score:.2f}%")
print(similarity)

print(reference_embedding)
print(generated_embedding)