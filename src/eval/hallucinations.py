import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.ndimage import gaussian_filter
from src.eval.metrics import rgb_to_y
from src.eval.evaluate import to_np
from src.models.bicubic import Bicubic

TILE = 64

def _hf(y, sigma=1.5):
    return y - gaussian_filter(y, sigma)

def tile_stats(sr, hr, y0, x0, size=TILE):
    s = rgb_to_y(sr[y0:y0 + size, x0:x0 + size]); h = rgb_to_y(hr[y0:y0 + size, x0:x0 + size])
    hs, hh = _hf(s), _hf(h)
    e_sr, e_hr = float(hs.std()), float(hh.std())
    corr = float(np.corrcoef(hs.ravel(), hh.ravel())[0, 1]) if e_sr > 1e-6 and e_hr > 1e-6 else 0.0
    mse = float(np.mean((s - h) ** 2))
    return dict(e_sr=e_sr, e_hr=e_hr, corr=corr,
                psnr=100.0 if mse == 0 else 10 * np.log10(1.0 / mse),
                score=e_sr * (1 - max(corr, 0.0)))

def auto_type(st):
    if st["corr"] < 0.5 and st["e_sr"] > 1.3 * st["e_hr"]: return "texture_added"
    if st["corr"] < 0.5 and st["e_sr"] >= 0.7 * st["e_hr"]: return "structure_wrong"
    return "other"

@torch.no_grad()
def run_models(lr, models, device):
    return {m: to_np(f(lr[None].to(device))[0]) for m, f in models.items()}

@torch.no_grad()
def seed_std_map(diff, lr, device, seeds=(0, 1, 2, 3)):
    runs = []
    for s in seeds:
        diff.seed = s; runs.append(rgb_to_y(to_np(diff(lr[None].to(device))[0])))
    diff.seed = 0
    return np.std(np.stack(runs), axis=0)

def analyze_dataset(dname, ds, models, diff, device, tile=TILE):
    rows = []
    for idx in range(len(ds)):
        lr, hr = ds[idx]; hr_np = to_np(hr); H, W = hr_np.shape[:2]
        diff.seed = 0
        outs = run_models(lr, models, device)
        sstd = seed_std_map(diff, lr, device)
        for y0 in range(0, H - tile + 1, tile):
            for x0 in range(0, W - tile + 1, tile):
                for m, sr in outs.items():
                    st = tile_stats(sr, hr_np, y0, x0, tile)
                    rows.append(dict(dataset=dname, image=ds.files[idx % len(ds.files)].name, idx=idx,
                                     method=m, y=y0, x=x0, size=tile, **st, auto_type=auto_type(st),
                                     seed_std=float(sstd[y0:y0 + tile, x0:x0 + tile].mean()) if m == "diffusion" else np.nan))
    return pd.DataFrame(rows)

def top_candidates(df, method, k=12, per_image=2):
    d = df[df.method == method].sort_values("score", ascending=False)
    out, count = [], {}
    for _, r in d.iterrows():
        key = (r.dataset, r.image)
        if count.get(key, 0) >= per_image: continue
        out.append(r); count[key] = count.get(key, 0) + 1
        if len(out) == k: break
    return pd.DataFrame(out)

def mark_joint(df, q=0.90):
    g = df[df.method == "gan"].set_index(["dataset", "image", "y", "x"]).score
    d = df[df.method == "diffusion"].set_index(["dataset", "image", "y", "x"]).score
    both = (g >= g.quantile(q)) & (d.reindex(g.index) >= d.quantile(q))
    keys = set(both[both].index)
    df["joint"] = [(r.dataset, r.image, r.y, r.x) in keys for r in df.itertuples()]
    return df

@torch.no_grad()
def make_card(title, lr, hr_np, box, models, diff, device, path, tile_psnr=True):
    y0, x0, s = box
    diff.seed = 0
    outs = run_models(lr, models, device)
    sstd = seed_std_map(diff, lr, device)
    up = lambda a: np.repeat(np.repeat(a, 4, axis=0), 4, axis=1)
    lr_np = to_np(lr)
    panels = {"LR (nearest)": up(lr_np)[y0:y0 + s, x0:x0 + s],
              "Bicubic": to_np(Bicubic(4)(lr[None])[0])[y0:y0 + s, x0:x0 + s]}
    panels.update({k.upper() if k == "gan" else k.capitalize(): v[y0:y0 + s, x0:x0 + s] for k, v in outs.items()})
    panels["HR (truth)"] = hr_np[y0:y0 + s, x0:x0 + s]
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    for j, (t, im) in enumerate(panels.items()):
        ax[0, j].imshow(im, interpolation="nearest"); ax[0, j].axis("off")
        if tile_psnr and t not in ("HR (truth)", "LR (nearest)"):
            mse = np.mean((rgb_to_y(im) - rgb_to_y(hr_np[y0:y0 + s, x0:x0 + s])) ** 2)
            t += f"  {10 * np.log10(1 / max(mse, 1e-10)):.1f} dB"
        ax[0, j].set_title(t, fontsize=10)
    ax[1, 0].imshow(hr_np); ax[1, 0].add_patch(Rectangle((x0, y0), s, s, fill=False, ec="red", lw=2))
    ax[1, 0].set_title("location", fontsize=10)
    ax[1, 1].imshow(sstd[y0:y0 + s, x0:x0 + s], cmap="magma"); ax[1, 1].set_title("diffusion: std over 4 seeds", fontsize=10)
    hy = rgb_to_y(hr_np[y0:y0 + s, x0:x0 + s])
    for j, k in enumerate(["edsr", "gan", "diffusion"]):
        err = np.abs(rgb_to_y(outs[k][y0:y0 + s, x0:x0 + s]) - hy)
        ax[1, 2 + j].imshow(err, cmap="inferno", vmin=0, vmax=0.25); ax[1, 2 + j].set_title(f"|{k} - HR|", fontsize=10)
    ax[1, 5].axis("off")
    for a in ax[1]: a.axis("off")
    fig.suptitle(title, fontsize=11); plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130); plt.close(fig)