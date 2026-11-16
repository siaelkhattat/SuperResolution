import torch.nn as nn
import torch.nn.functional as F

class Bicubic(nn.Module):
    """Baseline without learned parameters. Same interface as the future models:
    input LR tensor (B,3,h,w) in [0,1] -> output SR tensor (B,3,h*s,w*s)."""
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, lr):
        return F.interpolate(lr, scale_factor=self.scale, mode="bicubic",
                             align_corners=False).clamp(0, 1)