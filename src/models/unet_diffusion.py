import math
import torch, torch.nn as nn, torch.nn.functional as F

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c_in); self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)
        self.n2 = nn.GroupNorm(8, c_out); self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class UNetLite(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, base=48, mults=(1, 2, 4)):
        super().__init__()
        self.base = base
        t_dim = base * 4
        self.temb = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        chs = [base * m for m in mults]; self.L = len(chs)
        self.conv_in = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down, self.down_s = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, ci in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, ci, t_dim), ResBlock(ci, ci, t_dim)]))
            c = ci
            self.down_s.append(nn.Conv2d(ci, ci, 3, stride=2, padding=1) if i < self.L - 1 else nn.Identity())
        self.mid = nn.ModuleList([ResBlock(c, c, t_dim), ResBlock(c, c, t_dim)])
        self.up, self.up_s = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(self.L)):
            ci = chs[i]
            self.up.append(nn.ModuleList([ResBlock(2 * ci, ci, t_dim), ResBlock(ci, ci, t_dim)]))
            self.up_s.append(Up(ci, chs[i - 1]) if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(8, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], out_ch, 3, padding=1))

    def forward(self, x, t):
        H, W = x.shape[-2:]
        m = 2 ** (self.L - 1)
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.conv_in(x); skips = []
        for blocks, s in zip(self.down, self.down_s):
            for b in blocks: h = b(h, temb)
            skips.append(h); h = s(h)
        for b in self.mid: h = b(h, temb)
        for blocks, s in zip(self.up, self.up_s):
            h = torch.cat([h, skips.pop()], dim=1)
            for b in blocks: h = b(h, temb)
            h = s(h)
        return self.out(h)[..., :H, :W]