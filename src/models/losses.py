import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGGPerceptual(nn.Module):
    """L1 distance between VGG19 conv5_4 features (before activation)."""
    def __init__(self):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for p in self.features.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        f = lambda x: self.features((x - self.mean) / self.std)
        with torch.no_grad(): target = f(hr)
        return F.l1_loss(f(sr), target)