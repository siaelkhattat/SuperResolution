import argparse, random, time, yaml
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.data.dataset import SRDataset, SRTrainCached
from src.models.edsr_lite import EDSRLite
from src.eval.metrics import psnr
from src.eval.evaluate import to_np

@torch.no_grad()
def validate(model, ds, device):
    model.eval(); vals = []
    for i in range(len(ds)):
        lr, hr = ds[i]
        vals.append(psnr(to_np(model(lr[None].to(device))[0]), to_np(hr)))
    model.train()
    return float(np.mean(vals))

def train(cfg_path="configs/cnn.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = Path(cfg["raw_dir"]) / "DIV2K_train_HR"
    tr_names = Path("data/splits/train.txt").read_text().split()
    va_names = Path("data/splits/val.txt").read_text().split()

    train_ds = SRTrainCached(raw, tr_names, cfg["scale"], cfg["patch"], cfg["repeat"])
    val_ds = SRDataset(raw, names=va_names, scale=cfg["scale"], train=False)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=0, drop_last=True)

    arch = dict(scale=cfg["scale"], n_blocks=cfg["n_blocks"], channels=cfg["channels"])
    model = EDSRLite(**arch).to(device)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=cfg["lr_min"])
    loss_fn = torch.nn.L1Loss()

    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        run = 0.0
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            loss = loss_fn(model(lr_b), hr_b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        row = {"epoch": ep, "l1": run / len(loader), "val_psnr": np.nan,
               "minutes": round((time.time() - t0) / 60, 1)}
        if ep % cfg["val_every"] == 0 or ep == cfg["epochs"]:
            row["val_psnr"] = validate(model, val_ds, device)
            if row["val_psnr"] > best:
                best = row["val_psnr"]
                torch.save({"model": model.state_dict(), "arch": arch, "epoch": ep,
                            "val_psnr": best}, ckpt_dir / f"{cfg['name']}_best.pth")
        log.append(row); print(row)
    torch.save({"model": model.state_dict(), "arch": arch, "epoch": cfg["epochs"]},
               ckpt_dir / f"{cfg['name']}_last.pth")
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(f"results/tables/train_log_{cfg['name']}.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/cnn.yaml")
    train(ap.parse_args().config)