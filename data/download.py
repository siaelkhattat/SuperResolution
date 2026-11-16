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

import tarfile

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def download_benchmarks(cfg):
    from huggingface_hub import hf_hub_download
    raw = Path(cfg["raw_dir"])
    for name in ["Set5", "Set14"]:
        dst = raw / name
        if dst.exists() and any(p.suffix.lower() in IMG_EXT for p in dst.iterdir()):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        archive = hf_hub_download(repo_id=f"eugenesiow/{name}", repo_type="dataset",
                                  filename=f"data/{name}_HR.tar.gz")
        tmp = raw / f"_{name}_extract"
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp)
        for f in tmp.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                shutil.copy(f, dst / f.name)
        shutil.rmtree(tmp)
        print(name, ":", len(list(dst.iterdir())), "HR images")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/data.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    download_div2k(cfg); download_benchmarks(cfg)