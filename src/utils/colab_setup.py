import shutil, subprocess, tarfile
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/sr_low_cost")
DATA_TAR = DRIVE_ROOT / "data" / "sr_data.tar"
CKPT_DIR = DRIVE_ROOT / "checkpoints"
RESULTS_DIR = DRIVE_ROOT / "results"
RAW = Path("data/raw")
EXPECTED = {"DIV2K_train_HR": 400, "Set5": 5, "Set14": 14}


def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")


def _complete(raw):
    # ">=": a session that still holds the full 800 DIV2K images counts as complete
    return all((raw / d).is_dir() and len(list((raw / d).glob("*"))) >= n
               for d, n in EXPECTED.items())


def setup_data(raw=RAW):
    """Restore the 400 DIV2K images + Set5 + Set14 from the Drive archive (~2 min)."""
    raw = Path(raw)
    if _complete(raw):
        print("Data already present, nothing to do.")
        return
    for d in EXPECTED:                                   # remove any partial leftovers
        shutil.rmtree(raw / d, ignore_errors=True)
    if not DATA_TAR.exists():
        raise FileNotFoundError(f"{DATA_TAR} not found: build it once with build_data_archive().")
    raw.mkdir(parents=True, exist_ok=True)
    local = Path("/content/sr_data.tar")
    shutil.copy(DATA_TAR, local)                         # local disk is much faster than Drive
    chk = subprocess.run(["tar", "-tf", str(local)], capture_output=True, text=True)
    if chk.returncode != 0:
        local.unlink(missing_ok=True)
        raise RuntimeError("Archive on Drive is corrupt: rebuild it.\n" + chk.stderr[-300:])
    subprocess.run(["tar", "-xf", str(local), "-C", str(raw)], check=True)
    local.unlink()
    assert _complete(raw), "Extraction done but file counts are wrong."
    print("Data restored: 400 DIV2K, 5 Set5, 14 Set14.")


def build_data_archive(raw=RAW):
    """ONE-TIME: pack the 400 split images + Set5/Set14 into one tar on Drive.
    Needs the full DIV2K download on disk. Written to a .part file and renamed only
    after verification, so a half-written archive can never look valid."""
    raw = Path(raw)
    names = []
    for s in ["train", "val", "test"]:
        names += Path(f"data/splits/{s}.txt").read_text().split()
    assert len(names) == 400 and len(set(names)) == 400, f"Bad splits: {len(names)} names"
    for d, n in (("Set5", 5), ("Set14", 14)):
        assert len(list((raw / d).glob("*"))) == n, f"{d} incomplete"
    DATA_TAR.parent.mkdir(parents=True, exist_ok=True)
    part = DATA_TAR.with_name("sr_data.tar.part")
    with tarfile.open(part, "w") as tf:
        for n in names:
            tf.add(raw / "DIV2K_train_HR" / n, arcname=f"DIV2K_train_HR/{n}")
        for d in ["Set5", "Set14"]:
            tf.add(raw / d, arcname=d)
    with tarfile.open(part) as tf:
        files = [m.name for m in tf.getmembers() if m.isfile()]
    got = (sum(f.startswith("DIV2K_train_HR/") for f in files),
           sum(f.startswith("Set5/") for f in files), sum(f.startswith("Set14/") for f in files))
    assert got == (400, 5, 14), f"Archive check failed: {got}"
    part.replace(DATA_TAR)
    print("Archive OK:", got, "|", round(DATA_TAR.stat().st_size / 2**30, 2), "GB")


def load_generator(name="edsr_lite_best", device="cuda"):
    """Load a trained EDSR-lite or GAN generator from Drive. NO training involved."""
    import torch
    from src.models.edsr_lite import EDSRLite
    path = CKPT_DIR / f"{name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: {sorted(p.name for p in CKPT_DIR.glob('*.pth'))}")
    ck = torch.load(path, map_location=device)
    model = EDSRLite(**ck["arch"]); model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def sync_results_to_drive():
    """Run after producing tables/figures: a disconnect can't lose them anymore."""
    for sub in ["tables", "figures", "hallucinations"]:
        src = Path("results") / sub
        if src.exists():
            shutil.copytree(src, RESULTS_DIR / sub, dirs_exist_ok=True)
    print("Results synced to Drive.")


def restore_results_from_drive():
    for sub in ["tables", "figures", "hallucinations"]:
        src = RESULTS_DIR / sub
        if src.exists():
            shutil.copytree(src, Path("results") / sub, dirs_exist_ok=True)


def verify_repo():
    """Fail early if a file was never pushed to GitHub."""
    required = ["src/models/bicubic.py", "src/models/edsr_lite.py", "src/eval/metrics.py",
                "src/eval/evaluate.py", "src/eval/benchmark_speed.py", "src/data/dataset.py",
                "src/data/split.py", "src/data/check_leakage.py", "src/train/train_cnn.py",
                "configs/data.yaml", "configs/cnn.yaml", "requirements.txt"]
    missing = [f for f in required if not Path(f).exists()]
    assert not missing, f"Not in the cloned repo (commit and push them): {missing}"
    for s in ["train", "val", "test"]:
        assert Path(f"data/splits/{s}.txt").read_text().strip(), f"data/splits/{s}.txt is empty"
    print("Repo OK.")