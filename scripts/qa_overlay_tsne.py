#!/usr/bin/env python
"""
Visual sanity checks
────────────────────
1) Overlay real vs synthetic mean ±1σ for every class that has BOTH data types
2) 2-D t-SNE of up to SUBSAMPLE real + synthetic spectra / class
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

# ╭────────────────────── adjustable constants ──────────────────────╮
REAL_ROOT = Path("data/processed")      # flat directory of real .npy
SYN_ROOT  = Path("data/synthetic")      # dirs or flat; either works
WN        = np.linspace(50, 1750, 571)  # Raman-shift grid
SUBSAMPLE = 120                         # per class for t-SNE
RESULTS   = Path("results")
RESULTS.mkdir(exist_ok=True)
# ╰───────────────────────────────────────────────────────────────────╯


# -------------------------------------------------------------------
# helper: collect .npy files, keyed by class name
# -------------------------------------------------------------------
def collect_real(root: Path):
    d = defaultdict(list)
    for f in root.glob("*.npy"):
        cls = f.name.split("__", 1)[0]
        d[cls].append(f)
    return d

def collect_syn(root: Path):
    d = defaultdict(list)
    for sub in root.iterdir():
        if sub.is_dir():
            d[sub.name].extend(sub.glob("*.npy"))
        elif sub.suffix == ".npy":
            cls = sub.name.split("__", 1)[0]
            d[cls].append(sub)
    return d

def load_stack(files):
    """Load → squeeze any (1,571) → stack into (N,571)."""
    return np.stack(
        [np.load(f, mmap_mode="r").squeeze() for f in files]
    )

real_dict = collect_real(REAL_ROOT)
syn_dict  = collect_syn(SYN_ROOT)

common = sorted(set(real_dict) & set(syn_dict))
if not common:
    print("✖  No element has both real and synthetic spectra — nothing to plot.")
    sys.exit(0)

# -------------------------------------------------------------------
# 1) Overlay plots
# -------------------------------------------------------------------
for cls in common:
    real_stack = load_stack(real_dict[cls])
    syn_stack  = load_stack(syn_dict [cls])

    plt.figure(figsize=(5, 3))
    for arr, col, lab in [(real_stack, "k", "real"),
                          (syn_stack,  "C1", "synthetic")]:
        mu, sd = arr.mean(0), arr.std(0)
        plt.plot(WN, mu, lw=2, c=col, label=lab)
        plt.fill_between(WN, mu - sd, mu + sd, alpha=.15, color=col)

    plt.title(cls)
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Normalised intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / f"overlay_{cls}.png")
    plt.close()

# -------------------------------------------------------------------
# 2) t-SNE
# -------------------------------------------------------------------
X, labels = [], []
for cls in common:
    real = load_stack(real_dict[cls][:SUBSAMPLE])
    syn  = load_stack(syn_dict [cls][:SUBSAMPLE])
    X.append(real); labels += [f"{cls}_real"]*len(real)
    X.append(syn);  labels += [f"{cls}_syn"] *len(syn)

X = np.vstack(X)
emb = TSNE(n_components=2,
           perplexity=30,
           init="pca",
           learning_rate="auto",
           random_state=0).fit_transform(X)

plt.figure(figsize=(6, 5))
for lab in sorted(set(labels)):
    idx = [i for i,l in enumerate(labels) if l == lab]
    plt.scatter(emb[idx,0], emb[idx,1], s=8, alpha=.6, label=lab)
plt.axis("off")
plt.legend(fontsize=6, ncol=4)
plt.tight_layout()
plt.savefig(RESULTS / "tsne_real_vs_syn.png")

print(f"✓  Plotted {len(common)} classes  →  results/overlay_*.png"
      "\n✓  Saved 2-D t-SNE to           results/tsne_real_vs_syn.png")
