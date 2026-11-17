import math, torch, torch.nn as nn, torch.nn.functional as F
from src.models.unet_diffusion import UNetLite

def cosine_alphas_cumprod(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()          # length T

def bicubic_up(lr, scale):
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)

class DiffusionSR(nn.Module):
    """Residual diffusion SR. forward(lr) -> SR in [0,1], same interface as the other methods."""
    def __init__(self, scale=4, base=48, mults=(1, 2, 4), T=1000, residual_scale=4.0, steps=10, seed=0):
        super().__init__()
        self.scale, self.T, self.rs, self.steps, self.seed = scale, T, residual_scale, steps, seed
        self.arch = dict(scale=scale, base=base, mults=list(mults), T=T, residual_scale=residual_scale)
        self.net = UNetLite(6, 3, base, tuple(mults))
        self.register_buffer("ac", cosine_alphas_cumprod(T))

    def predict_x0(self, x_t, cond, t):
        return self.net(torch.cat([x_t, (cond - 0.5) * 2], dim=1), t)

    @torch.no_grad()
    def forward(self, lr):
        cond = bicubic_up(lr, self.scale)
        g = torch.Generator(device=lr.device).manual_seed(self.seed)
        x = torch.randn(cond.shape, generator=g, device=lr.device)
        ts = torch.linspace(self.T - 1, 0, self.steps).round().long().tolist()
        for i, t in enumerate(ts):
            ab_t = self.ac[t]
            ab_prev = self.ac[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(ab_t)
            tt = torch.full((cond.shape[0],), t, device=lr.device)
            x0 = self.predict_x0(x, cond, tt).clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps      # DDIM, eta = 0
        return (cond + x0 / self.rs).clamp(0, 1)

def load_diffusion_sr(path, steps=10, device="cuda"):
    ck = torch.load(path, map_location=device)
    m = DiffusionSR(steps=steps, **ck["arch"])
    m.net.load_state_dict(ck["ema"]); m.to(device).eval()
    return m, ck