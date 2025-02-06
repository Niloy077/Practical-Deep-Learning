from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

app = Flask(__name__, static_folder="static")  # Ensures static files are served

# Load Pretrained ResNet18 Model
model = models.resnet18(pretrained=True)
model.eval()

# Define Image Transformations (Wihttps://chatgpt.com/c/67a306b1-030c-8000-89f9-feb7ff3dde8cth Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet Class Labels Correctly
with open("imagenet_classes.txt") as f:
    LABELS = [line.strip() for line in f.readlines()]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = outputs.max(1)

    predicted_idx = predicted_class.item()
    class_label = LABELS[predicted_idx] if 0 <= predicted_idx < len(LABELS) else "Unknown Class"

    return jsonify({"prediction": class_label})

if __name__ == "__main__":
    app.run(debug=True)
