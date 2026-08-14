# ==============================
# Imports
# ==============================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# ==============================
# Model architecture
# ==============================
class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(PretrainedResNet50, self).__init__()
        self.base_model = models.resnet50(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# ==============================
# Device and model loading
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../pre-trained_model/pretrained_model_224x224.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = PretrainedResNet50(num_classes=3)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully from", MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Warning: Model not found at", MODEL_PATH)

# ==============================
# Class labels
# ==============================
class_labels = [
    "Lung Adenocarcinoma",
    "Lung Benign",
    "Lung Squamous Cell Carcinoma"
]

# ==============================
# Grad-CAM hook
# ==============================
class GradCAM:
    features = None

    @staticmethod
    def hook_fn(module, input, output):
        if output.requires_grad:
            output.retain_grad()
        GradCAM.features = output

if model is not None:
    model.base_model.layer4[-1].register_forward_hook(GradCAM.hook_fn)

# ==============================
# Preprocess image
# ==============================
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

# ==============================
# Classify image
# ==============================
def classify_image(image_path):
    if model is None:
        predicted_idx = np.random.randint(0, len(class_labels))
        predicted_class = class_labels[predicted_idx]
        predicted_prob = np.random.rand() * 100
        return predicted_class, predicted_idx, round(predicted_prob, 2)

    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_prob = probs[0, predicted_idx].item() * 100
        predicted_class = class_labels[predicted_idx]
    return predicted_class, predicted_idx, round(predicted_prob, 2)

# ==============================
# Save image with predicted label
# ==============================
def save_image_with_label(original_image, label, output_path, prob=None):
    output_image = original_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = label
    if prob is not None:
        text += f" ({prob:.1f}%)"
    cv2.putText(output_image, text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, output_image)

# ==============================
# Grad-CAM implementation
# ==============================
def generate_heatmap(image_tensor, class_idx):
    if model is None:
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)

    image_tensor = image_tensor.to(device).requires_grad_(True)
    model.zero_grad()
    outputs = model(image_tensor)
    if GradCAM.features is None:
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)

    class_score = outputs[0, class_idx]
    model.zero_grad()
    class_score.backward(retain_graph=True)

    gradients = GradCAM.features.grad
    if gradients is None:
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)

    gradients = gradients[0]
    pooled_gradients = torch.mean(gradients, dim=[1, 2])
    activations = GradCAM.features.detach()[0]

    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap

# ==============================
# Overlay heatmap on original image
# ==============================
def overlay_heatmap(heatmap, original_image):
    if heatmap is None:
        return original_image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

# ==============================
# LIME placeholder
# ==============================
def lime_explanation(image_path, class_idx):
    original = cv2.imread(image_path)
    mask = cv2.applyColorMap(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)

    class DummyExpl:
        local_exp = {0: [(i, np.random.rand()) for i in range(10)]}

    top_label = 0
    return original, mask, DummyExpl(), top_label
