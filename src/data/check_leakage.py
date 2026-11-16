import hashlib
from pathlib import Path

def img_hash(p):
    from PIL import Image
    im = Image.open(p).convert("RGB").resize((64, 64))   # robuste au format/taille
    return hashlib.md5(im.tobytes()).hexdigest()

def check(raw_dir="data/raw"):
    raw = Path(raw_dir)
    bench = {img_hash(p) for n in ["Set5", "Set14"] for p in (raw / n).iterdir()}
    names = []
    for s in ["train", "val", "test"]:
        names += [raw / "DIV2K_train_HR" / n for n in Path(f"data/splits/{s}.txt").read_text().split()]
    leaks = [p.name for p in names if img_hash(p) in bench]
    train = set(Path("data/splits/train.txt").read_text().split())
    val = set(Path("data/splits/val.txt").read_text().split())
    test = set(Path("data/splits/test.txt").read_text().split())
    assert not (train & val or train & test or val & test), "Chevauchement entre splits"
    assert not leaks, f"Fuite avec Set5/Set14 : {leaks}"
    print("OK : aucun chevauchement, aucune fuite vers Set5/Set14")

if __name__ == "__main__":
    check()