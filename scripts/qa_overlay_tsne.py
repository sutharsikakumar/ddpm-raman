
"""
Visual sanity checks
────────────────────
1) Overlay real vs synthetic mean ±1σ for every class that has BOTH data types
2) 2-D t-SNE of up to SUBSAMPLE real + synthetic spectra per class
"""

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


REAL_ROOT = Path("data/processed")   
SYN_ROOT  = Path("data/synthetic")    
WN        = np.linspace(50, 1750, 571) 
SUBSAMPLE = 120                        
RESULTS   = Path("results"); RESULTS.mkdir(exist_ok=True)


def collect_real(root: Path):
    d = defaultdict(list)
    for f in root.glob("*.npy"):
        d[f.name.split("__", 1)[0]].append(f)
    return d

def collect_syn(root: Path):
    d = defaultdict(list)
    for sub in root.iterdir():
        if sub.is_dir():
            d[sub.name].extend(sub.glob("*.npy"))
        elif sub.suffix == ".npy":
            d[sub.name.split("__", 1)[0]].append(sub)
    return d

def load_stack(files):
    """Load each .npy → flatten → stack, ensuring (N,571)."""
    return np.stack([np.load(f, mmap_mode="r").reshape(-1) for f in files])


real_dict = collect_real(REAL_ROOT)
syn_dict  = collect_syn(SYN_ROOT)
common    = sorted(set(real_dict) & set(syn_dict))

if not common:
    print("✖  No mineral has both real AND synthetic spectra. Exiting.")
    sys.exit(0)

for cls in common:
    try:
        real_stack = load_stack(real_dict[cls])
        syn_stack  = load_stack(syn_dict[cls])

        if real_stack.shape[1] != 571 or syn_stack.shape[1] != 571:
            raise ValueError(f"unexpected vector length "
                             f"{real_stack.shape[1]} / {syn_stack.shape[1]}")

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

    except Exception as e:
        print(f"[skip] {cls:25s}  ({type(e).__name__}: {e})")
        continue   # move on to next mineral


good_classes = [cls for cls in common
                if (RESULTS / f"overlay_{cls}.png").exists()]

if not good_classes:
    print("✖  All classes were skipped. No t-SNE produced.")
    sys.exit(0)

X, labels = [], []
for cls in good_classes:
    real = load_stack(real_dict[cls][:SUBSAMPLE])
    syn  = load_stack(syn_dict [cls][:SUBSAMPLE])
    X.append(real); labels += [f"{cls}_real"] * len(real)
    X.append(syn);  labels += [f"{cls}_syn"]  * len(syn)

X = np.vstack(X)
emb = TSNE(n_components=2, perplexity=30,
           init="pca", learning_rate="auto",
           random_state=0).fit_transform(X)

plt.figure(figsize=(6, 5))
for lab in sorted(set(labels)):
    idx = [i for i, l in enumerate(labels) if l == lab]
    plt.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=.6, label=lab)
plt.axis("off")
plt.legend(fontsize=6, ncol=4)
plt.tight_layout()
plt.savefig(RESULTS / "tsne_real_vs_syn.png")

print(f"✓  Overlay PNGs for {len(good_classes)} minerals → results/overlay_*.png"
      "\n✓  2-D t-SNE written to                results/tsne_real_vs_syn.png")
