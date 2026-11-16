import argparse, random, time, yaml
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import SRDataset, SRTrainCached
from src.models.edsr_lite import EDSRLite
from src.models.discriminator import PatchDiscriminator
from src.models.losses import VGGPerceptual
from src.eval.metrics import psnr, lpips_score
from src.eval.evaluate import to_np

@torch.no_grad()
def validate(G, ds, device):
    G.eval(); p, l = [], []
    for i in range(len(ds)):
        lr, hr = ds[i]
        sr, h = to_np(G(lr[None].to(device))[0]), to_np(hr)
        p.append(psnr(sr, h)); l.append(lpips_score(sr, h, device))
    G.train()
    return float(np.mean(p)), float(np.mean(l))

def train(cfg_path="configs/gan.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda"
    raw = Path(cfg["raw_dir"]) / "DIV2K_train_HR"
    tr = Path("data/splits/train.txt").read_text().split()
    va = Path("data/splits/val.txt").read_text().split()
    train_ds = SRTrainCached(raw, tr, cfg["scale"], cfg["patch"], cfg["repeat"])
    val_ds = SRDataset(raw, names=va, scale=cfg["scale"], train=False)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)

    ck = torch.load(cfg["init_ckpt"], map_location=device)
    arch = ck["arch"]
    G = EDSRLite(**arch).to(device); G.load_state_dict(ck["model"]); G.train()
    D = PatchDiscriminator().to(device)
    vgg = VGGPerceptual().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.9, 0.99))
    bce = F.binary_cross_entropy_with_logits
    print("G params (M):", sum(p.numel() for p in G.parameters()) / 1e6,
          "| D params (M):", sum(p.numel() for p in D.parameters()) / 1e6)

    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], 1e9, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        acc = dict(d=0.0, pix=0.0, per=0.0, adv=0.0)
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            sr = G(lr_b)
            # discriminator step
            lr_real, lr_fake = D(hr_b), D(sr.detach())
            loss_d = 0.5 * (bce(lr_real, torch.ones_like(lr_real)) + bce(lr_fake, torch.zeros_like(lr_fake)))
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
            # generator step
            fake = D(sr)
            l_pix, l_per = F.l1_loss(sr, hr_b), vgg(sr, hr_b)
            l_adv = bce(fake, torch.ones_like(fake))
            loss_g = cfg["w_pix"] * l_pix + cfg["w_per"] * l_per + cfg["w_adv"] * l_adv
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            for k, v in zip(acc, (loss_d, l_pix, l_per, l_adv)): acc[k] += v.item()
        n = len(loader)
        row = {"epoch": ep, **{k: round(v / n, 4) for k, v in acc.items()},
               "val_psnr": np.nan, "val_lpips": np.nan, "minutes": round((time.time() - t0) / 60, 1)}
        if ep % cfg["val_every"] == 0 or ep == cfg["epochs"]:
            row["val_psnr"], row["val_lpips"] = validate(G, val_ds, device)
            torch.save({"model": G.state_dict(), "arch": arch, "epoch": ep}, ckpt_dir / f"{cfg['name']}_ep{ep}.pth")
            if row["val_lpips"] < best:
                best = row["val_lpips"]
                torch.save({"model": G.state_dict(), "arch": arch, "epoch": ep,
                            "val_lpips": best, "val_psnr": row["val_psnr"]}, ckpt_dir / f"{cfg['name']}_best.pth")
        log.append(row); print(row)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(f"results/tables/train_log_{cfg['name']}.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/gan.yaml")
    train(ap.parse_args().config)