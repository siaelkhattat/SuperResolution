import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

class PatchDiscriminator(nn.Module):
    """Fully convolutional: outputs a map of real/fake logits (one per patch)."""
    def __init__(self, c=64):
        super().__init__()
        def blk(i, o, s): return [sn(nn.Conv2d(i, o, 3 if s == 1 else 4, s, 1)), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(
            *blk(3, c, 1), *blk(c, c, 2),
            *blk(c, 2 * c, 1), *blk(2 * c, 2 * c, 2),
            *blk(2 * c, 4 * c, 1), *blk(4 * c, 4 * c, 2),
            sn(nn.Conv2d(4 * c, 1, 3, 1, 1)))

    def forward(self, x):
        return self.net(x)