import argparse, copy, math, random, time, yaml
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import SRDataset, SRTrainCached
from src.models.diffusion import DiffusionSR, bicubic_up
from src.eval.metrics import psnr, lpips_score
from src.eval.evaluate import to_np

@torch.no_grad()
def validate(ema_model, ds, device):
    ema_model.eval(); p, l = [], []
    for i in range(len(ds)):
        lr, hr = ds[i]
        sr, h = to_np(ema_model(lr[None].to(device))[0]), to_np(hr)
        p.append(psnr(sr, h)); l.append(lpips_score(sr, h, device))
    return float(np.mean(p)), float(np.mean(l))

def infinite(loader):
    while True:
        for b in loader: yield b

def train(cfg_path="configs/diffusion.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda"
    raw = Path(cfg["raw_dir"]) / "DIV2K_train_HR"
    tr = Path("data/splits/train.txt").read_text().split()
    va = Path("data/splits/val.txt").read_text().split()[:cfg["val_images"]]
    train_ds = SRTrainCached(raw, tr, cfg["scale"], cfg["patch"], cfg["repeat"])
    val_ds = SRDataset(raw, names=va, scale=cfg["scale"], train=False, center_crop=cfg["val_crop"])
    data = infinite(DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True))

    model = DiffusionSR(scale=cfg["scale"], base=cfg["base"], mults=cfg["mults"], T=cfg["timesteps"],
                        residual_scale=cfg["residual_scale"], steps=cfg["val_steps"]).to(device)
    net = model.net
    ema = DiffusionSR(scale=cfg["scale"], base=cfg["base"], mults=cfg["mults"], T=cfg["timesteps"],
                      residual_scale=cfg["residual_scale"], steps=cfg["val_steps"]).to(device)
    ema.net.load_state_dict(net.state_dict()); ema.eval()
    print("U-Net params (M):", sum(p.numel() for p in net.parameters()) / 1e6)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_p, best_p = ckpt_dir / f"{cfg['name']}_last.pth", ckpt_dir / f"{cfg['name']}_best.pth"

    it, best, log = 0, 1e9, []
    if last_p.exists():                                    # automatic resume
        st = torch.load(last_p, map_location=device)
        net.load_state_dict(st["net"]); ema.net.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        it, best, log = st["it"], st["best"], st["log"]
        print(f"Resumed from iteration {it}")

    ac, T, rs = model.ac, cfg["timesteps"], cfg["residual_scale"]
    t0, run = time.time(), 0.0
    while it < cfg["iters"]:
        lr_b, hr_b = next(data)
        lr_b, hr_b = lr_b.to(device), hr_b.to(device)
        cond = bicubic_up(lr_b, cfg["scale"])
        x0 = ((hr_b - cond) * rs).clamp(-1, 1)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        a = ac[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        for g in opt.param_groups:
            g["lr"] = cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * it / cfg["iters"]))
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.predict_x0(x_t, cond, t)
        loss = F.mse_loss(pred.float(), x0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        d = min(cfg["ema"], (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, pn in zip(ema.net.parameters(), net.parameters()): pe.mul_(d).add_(pn.detach(), alpha=1 - d)
        it += 1; run += loss.item()
        if it % 200 == 0:
            print(f"it {it} | loss {run / 200:.5f} | {(time.time() - t0) / 60:.1f} min"); run = 0.0
        if it % cfg["val_every"] == 0 or it == cfg["iters"]:
            vp, vl = validate(ema, val_ds, device)
            log.append({"it": it, "val_psnr": vp, "val_lpips": vl})
            print(f"== VAL it {it}: PSNR {vp:.3f} | LPIPS {vl:.4f}")
            arch = ema.arch
            if vl < best:
                best = vl
                torch.save({"ema": ema.net.state_dict(), "arch": arch, "it": it, "val_lpips": vl, "val_psnr": vp}, best_p)
            torch.save({"net": net.state_dict(), "ema": ema.net.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "it": it, "best": best, "log": log, "arch": arch}, last_p)
            Path("results/tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(log).to_csv(f"results/tables/train_log_{cfg['name']}.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/diffusion.yaml")
    train(ap.parse_args().config)