
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from PIL import Image
from torchvision import models

# Define the model class
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

# Load the model
model_path = r'D:\Final_Year_Project\Lung_Cancer_Dectection-XAI\pre-trained_model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PretrainedResNet50()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Class labels
class_labels = ['Lung Adenocarcinoma', 'Lung Benign', 'Lung Squamous Cell Carcinoma']

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)
