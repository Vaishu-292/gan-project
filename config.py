import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GALLERY_DIR = os.path.join(BASE_DIR, "static", "gallery")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 1024

