import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(c, c, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSRLite(nn.Module):
    """Input LR (B,3,h,w) in [0,1] -> output (B,3,h*s,w*s). Predicts a residual over bicubic."""
    def __init__(self, scale=4, n_blocks=8, channels=64, bicubic_skip=True):
        super().__init__()
        assert scale in (2, 4)
        self.scale, self.bicubic_skip = scale, bicubic_skip
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)],
                                  nn.Conv2d(channels, channels, 3, padding=1))
        up = []
        for _ in range(scale // 2):
            up += [nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*up)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, lr):
        f = self.head(lr)
        f = f + self.body(f)
        out = self.tail(self.upsample(f))
        if self.bicubic_skip:
            out = out + F.interpolate(lr, scale_factor=self.scale, mode="bicubic",
                                      align_corners=False)
        return out.clamp(0, 1)