"""
Train one Diffusion1D per discovered label and save to
checkpoints/ddpm_<Label>.pt

Run:
    python -m raman_ddpm.ddpm_batch_trainer --epochs 100
"""
import argparse, pathlib, torch
from .datasets import RamanDataset, discover_labels
from .ddpm_trainer import DDPMTrainer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH  = 64                  # GPU-friendly

# ── CLI ----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=100,
               help="max epochs for normal-size classes (tiny ones auto-halve)")
p.add_argument("--min-spectra", type=int, default=10,
               help="skip labels with fewer spectra than this")
args = p.parse_args()

# ── discover labels ----------------------------------------------------
proc   = pathlib.Path("data/processed")
labels = discover_labels(proc)
print("Found", len(labels), "labels")

ckpt_dir = pathlib.Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)

for lbl in labels:
    spectra = list(proc.glob(f"{lbl}_*.npy"))
    if len(spectra) < args.min_spectra:
        print(lbl, f"✗ only {len(spectra)} spectra – skipped")
        continue

    ck_path = ckpt_dir / f"ddpm_{lbl}.pt"
    if ck_path.exists():
        print(lbl, "✓ checkpoint exists – skipping")
        continue

    epochs = args.epochs if len(spectra) > 64 else args.epochs // 2
    print(f"\n┏━ Training {lbl}  ({len(spectra)} spectra, {epochs} epochs) ━┓")

    ds = RamanDataset(proc, lbl)             # single-label subset
    trainer = DDPMTrainer(ds, device=DEVICE,
                          batch_size=BATCH, label=lbl)
    trainer.ckpt_path = ck_path
    trainer.fit(epochs)
