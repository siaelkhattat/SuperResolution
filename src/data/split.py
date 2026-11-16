import random, yaml
from pathlib import Path
import random, yaml
from pathlib import Path

def make_splits(cfg_path="configs/data.yaml", force=False):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path("data/splits")
    if not force and (out / "train.txt").exists() and (out / "train.txt").read_text().strip():
        print("Splits already exist: not overwriting (use force=True to regenerate).")
        return None
    files = sorted(p.name for p in (Path(cfg["raw_dir"]) / "DIV2K_train_HR").glob("*.png"))
    assert len(files) >= cfg["n_train"] + cfg["n_val"] + cfg["n_test"], \
        f"Only {len(files)} DIV2K images found: refusing to create splits."
    rng = random.Random(cfg["seed"]); rng.shuffle(files)
    n_tr, n_va, n_te = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    splits = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va],
              "test": files[n_tr + n_va:n_tr + n_va + n_te]}
    out.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        (out / f"{k}.txt").write_text("\n".join(sorted(v)))
    return splits

if __name__ == "__main__":
    s = make_splits()
    if s: print({k: len(v) for k, v in s.items()})