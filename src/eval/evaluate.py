import numpy as np, pandas as pd, torch, yaml
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import SRDataset
from src.eval.metrics import psnr, ssim, lpips_score

def to_np(t):
    return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)

def get_datasets(cfg_path="configs/data.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    raw, s = Path(cfg["raw_dir"]), cfg["scale"]
    test_names = Path("data/splits/test.txt").read_text().split()
    return {
        "Set5": SRDataset(raw / "Set5", scale=s, train=False),
        "Set14": SRDataset(raw / "Set14", scale=s, train=False),
        "DIV2K_test_c512": SRDataset(raw / "DIV2K_train_HR", names=test_names, scale=s, train=False, center_crop=512),    }

@torch.no_grad()
def evaluate_method(name, model, datasets, device="cuda", shave=4):
    model = model.to(device).eval()
    rows = []
    for dname, ds in datasets.items():
        for i in tqdm(range(len(ds)), desc=f"{name} on {dname}"):
            lr, hr = ds[i]
            sr = model(lr[None].to(device))[0]
            sr_np, hr_np = to_np(sr), to_np(hr)
            rows.append({
                "method": name, "dataset": dname, "image": ds.files[i % len(ds.files)].name,
                "psnr": psnr(sr_np, hr_np, shave), "ssim": ssim(sr_np, hr_np, shave),
                "lpips": lpips_score(sr_np, hr_np, device),
            })
    return pd.DataFrame(rows)

def summarize(df):
    return (df.groupby(["method", "dataset"])[["psnr", "ssim", "lpips"]]
              .mean().round(4).reset_index())