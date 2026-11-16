"""Télécharge DIV2K (train HR) et Set5/Set14. Rien de lourd n'est versionné."""
import zipfile, shutil, argparse, urllib.request
from pathlib import Path
import yaml

def download_div2k(cfg):
    raw = Path(cfg["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "DIV2K_train_HR.zip"
    if not zip_path.exists():
        print("Téléchargement DIV2K (~3.5 Go)...")
        urllib.request.urlretrieve(cfg["div2k_url"], zip_path)
    out = raw / "DIV2K_train_HR"
    if not out.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw)
    print("DIV2K prêt :", len(list(out.glob("*.png"))), "images")

def download_benchmarks(cfg):
    from huggingface_hub import snapshot_download
    raw = Path(cfg["raw_dir"])
    for name in ["Set5", "Set14"]:
        dst = raw / name
        if dst.exists() and any(dst.glob("*.png")):
            continue
        tmp = Path(snapshot_download(repo_id=f"eugenesiow/{name}", repo_type="dataset"))
        dst.mkdir(parents=True, exist_ok=True)
        for z in tmp.rglob("*.zip"):          # décompresser les archives éventuelles
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp / "_unzipped")
        for f in list(tmp.rglob("*.png")) + list(tmp.rglob("*.jpg")) + list(tmp.rglob("*.bmp")):
            if "HR" in f.as_posix() and "LR" not in f.name:
                shutil.copy(f, dst / f.name)
        print(name, ":", len(list(dst.iterdir())), "images HR")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/data.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    download_div2k(cfg); download_benchmarks(cfg)