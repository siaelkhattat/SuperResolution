import random
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset

def bicubic_down(img, scale):
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)

class SRDataset(Dataset):
    def __init__(self, img_dir, names=None, scale=4, patch=128, train=True, repeat=10):
        d = Path(img_dir)
        self.files = [d / n for n in names] if names else sorted(p for p in d.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".bmp"})
        self.scale, self.patch, self.train = scale, patch, train
        self.repeat = repeat if train else 1

    def __len__(self): return len(self.files) * self.repeat

    def __getitem__(self, i):
        hr = Image.open(self.files[i % len(self.files)]).convert("RGB")
        if self.train:
            w, h = hr.size; p = self.patch
            x, y = random.randint(0, w - p), random.randint(0, h - p)
            hr = hr.crop((x, y, x + p, y + p))
            if random.random() < .5: hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < .5: hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        else:                                   # rogner pour que H, W soient divisibles par scale
            w, h = hr.size; hr = hr.crop((0, 0, w - w % self.scale, h - h % self.scale))
        lr = bicubic_down(hr, self.scale)
        to_t = lambda im: torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        return to_t(lr), to_t(hr)