"""
Generate --per 1000 synthetic spectra for every class.
Run: python -m raman_ddpm.gen_all_synth --per 1000
"""
import argparse, pathlib, torch, numpy as np
from .datasets import discover_labels
from .sampler import sample_spectra

p = argparse.ArgumentParser(); p.add_argument("--per", type=int, default=1000)
args = p.parse_args()

ckpt_dir = pathlib.Path("checkpoints")
out_root = pathlib.Path("data/synthetic"); out_root.mkdir(parents=True, exist_ok=True)
labels = discover_labels(pathlib.Path("data/processed"))

for lbl in labels:
    ckpt = ckpt_dir / f"ddpm_{lbl}.pt"
    dest = out_root / lbl; dest.mkdir(exist_ok=True)
    synth = sample_spectra(ckpt, n_samples=args.per, device="mps" if torch.backends.mps.is_available() else "cpu")
    for i, arr in enumerate(synth):
        np.save(dest / f"{lbl}_synthetic_{i:04d}.npy", arr.cpu().numpy())
    print(f"{lbl} ✓  {args.per} synthetic saved → {dest}")
