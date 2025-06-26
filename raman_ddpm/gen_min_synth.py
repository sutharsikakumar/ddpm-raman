"""
Generate just enough synthetic spectra for each label to reach TARGET total
(real + synthetic).  Usage:

    python -m raman_ddpm.gen_min_synth --target 200
"""
import argparse, pathlib, numpy as np, torch
from collections import Counter
from .datasets import discover_labels
from .sampler  import sample_spectra

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── CLI ────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=200,
               help="desired total spectra per label (real + synthetic)")
args = p.parse_args()

PROC   = pathlib.Path("data/processed")
SYNTH  = pathlib.Path("data/synthetic"); SYNTH.mkdir(parents=True, exist_ok=True)
CKPTS  = pathlib.Path("checkpoints")

# ---------------------------------------------------------------------
#  Count how many spectra each label already has  (real + synthetic)
# ---------------------------------------------------------------------
real_cnt  = Counter(p.name.split('_', 1)[0]            # label from filename
                    for p in PROC.glob("*.npy"))

synth_cnt = Counter(p.parent.name                      # label = folder name
                    for p in SYNTH.rglob("*.npy"))

total_cnt = {lbl: real_cnt.get(lbl, 0) + synth_cnt.get(lbl, 0)
             for lbl in discover_labels(PROC)}
# ---------------------------------------------------------------------

for lbl in discover_labels(PROC):
    need = args.target - total_cnt.get(lbl, 0)
    if need <= 0:
        print(f"{lbl:18} ✓ already ≥ {args.target}")
        continue

    ckpt = CKPTS / f"ddpm_{lbl}.pt"
    if not ckpt.exists():
        print(f"{lbl:18} ✗ checkpoint missing – skipped")
        continue

    print(f"{lbl:18} → generating {need:4d} synthetic")
    out_dir = SYNTH / lbl; out_dir.mkdir(exist_ok=True)
    synth = sample_spectra(ckpt, n_samples=need, device=DEVICE)
    for i, arr in enumerate(synth):
        np.save(out_dir / f"{lbl}_synth_{i:04d}.npy", arr.cpu().numpy())
