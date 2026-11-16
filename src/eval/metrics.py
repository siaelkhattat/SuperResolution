import numpy as np, torch
from skimage.metrics import structural_similarity

def rgb_to_y(img):                      # img : ndarray float [0,1], HxWx3
    return 16 / 255 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255

def _prep(a, b, shave):
    a, b = rgb_to_y(a), rgb_to_y(b)
    if shave: a, b = a[shave:-shave, shave:-shave], b[shave:-shave, shave:-shave]
    return a, b

def psnr(sr, hr, shave=4):
    a, b = _prep(sr, hr, shave)
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)

def ssim(sr, hr, shave=4):
    a, b = _prep(sr, hr, shave)
    return structural_similarity(a, b, data_range=1.0)

_lpips = None
def lpips_score(sr, hr, device="cuda"):
    global _lpips
    import lpips
    if _lpips is None: _lpips = lpips.LPIPS(net="alex").to(device).eval()
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) * 2 - 1
    with torch.no_grad(): return _lpips(t(sr), t(hr)).item()