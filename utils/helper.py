import numpy as np
import io
import torch
import torchvision
import collections
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EPSILON = 0.25
NUM_STEPS = 5
ALPHA = 0.025


def load_model():
    """Load and return the pre-trained ResNet50 model."""
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path, mean=MEAN, std=STD):
    """Preprocess the input image."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(image_path)
    input_image = preprocess(img)
    input_image = input_image.unsqueeze(0)
    input_image.requires_grad = True
    return input_image

