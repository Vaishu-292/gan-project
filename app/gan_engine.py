import torch
import numpy as np
from PIL import Image
import os
import config
from torchvision.utils import save_image
from app.model import Generator 

class GANEngine:
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.mode = "REAL_GAN"

        self.model = Generator(nz=100, ngf=64, nc=3).to(self.device)
      
        self.load_weights()

    def load_weights(self):
        weights_path = os.path.join(config.BASE_DIR, "models", "dcgan_weights.pth")
        if os.path.exists(weights_path):
            print(f" Loading Trained Weights from {weights_path}")
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f" Error loading weights: {e}")
        else:
            print("No weights file found. Running with RANDOM WEIGHTS (Untrained Mode).")
            self.model.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    def generate_image(self, seed: int, truncation: float):
        filename = f"seed_{seed}.png"
        save_path = os.path.join(config.GALLERY_DIR, filename)

        if os.path.exists(save_path):
            return f"/static/gallery/{filename}"

        try:

            torch.manual_seed(seed)
            noise = torch.randn(1, 100, 1, 1, device=self.device)

            with torch.no_grad():
                fake_image_tensor = self.model(noise)

            fake_image_tensor = (fake_image_tensor + 1) / 2.0
            fake_image_tensor = fake_image_tensor.clamp(0, 1)

            ndarr = fake_image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr[0])
            
            im = im.resize((512, 512), Image.NEAREST)

            im.save(save_path)
            
            return f"/static/gallery/{filename}"

        except Exception as e:
            print(f"Generation Error: {e}")
            return None