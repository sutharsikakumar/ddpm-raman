from pathlib import Path
import numpy as np, torch, sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


REAL_ROOT = Path("data/processed")  
SYN_ROOT  = Path("data/synthetic")  
N_REAL    = 10                   
N_SYN     = 10 

def _gather_files():
    """
    Build a dict {class: (real_files, syn_files)}.
    Keep only minerals with ≥ N_REAL real AND ≥ N_SYN synthetic spectra.
    Down-sample each side to N_REAL / N_SYN.
    """
    rng   = np.random.default_rng(0)
    table = {}

    for cls in sorted({p.name for p in SYN_ROOT.iterdir() if p.is_dir()}):
        real = list(REAL_ROOT.glob(f"{cls}__*.npy"))    
        syn  = list((SYN_ROOT / cls).glob("*.npy"))

        if len(real) < N_REAL or len(syn) < N_SYN:
            print(f"[skip] {cls:<25s} real={len(real):3d}  syn={len(syn):3d}")
            continue

        table[cls] = (
            rng.choice(real, N_REAL, replace=False),
            rng.choice(syn,  N_SYN,  replace=False),
        )
    return table

class RamanDataset(Dataset):
    """
    Balanced Raman dataset.
      split      : 'train' or 'test'
      test_size  : 0.0 – 1.0 (fraction for test). 0.0 → no split.
    """
    def __init__(self, split="train", test_size=.2, random_state=0):
        file_table = _gather_files()

        X_all, y_all, classes = [], [], []
        for idx, (cls, (real_f, syn_f)) in enumerate(file_table.items()):
            classes.append(cls)
            for f in (*real_f, *syn_f):
                X_all.append(np.load(f, mmap_mode="r").reshape(-1))
                y_all.append(idx)

        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        X_all = np.asarray(X_all, np.float32)
        y_all = np.asarray(y_all, np.int64)

        if test_size and test_size > 0.0:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=test_size, stratify=y_all,
                random_state=random_state)
            self.X, self.y = (X_tr, y_tr) if split == "train" else (X_te, y_te)
        else:
            self.X, self.y = X_all, y_all 


    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), int(self.y[idx])

def get_loaders(batch=32, **kw):
    train_ds = RamanDataset(split="train", **kw)
    test_ds  = RamanDataset(split="test",  **kw)
    return (DataLoader(train_ds, batch,   shuffle=True,  num_workers=2),
            DataLoader(test_ds,  batch*2, shuffle=False, num_workers=2))

def discover_labels():
    """Return list of mineral names in the balanced dataset (train split)."""
    return list(RamanDataset(split="train", test_size=0.0).cls_to_idx.keys())

sys.modules['raman_ddpm.datasets'] = sys.modules[__name__] 
