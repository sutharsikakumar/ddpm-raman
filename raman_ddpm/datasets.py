from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

REAL_ROOT = Path("data/processed")
SYN_ROOT  = Path("data/synthetic")

class RamanDataset(Dataset):
    def __init__(self, split="train", test_size=.2, random_state=0):
        X, y, cls_ok = [], [], []

        # —— gather classes that have BOTH real and synthetic ——
        for cls in sorted({p.name for p in SYN_ROOT.iterdir()}):
            real_files = list((REAL_ROOT/cls).glob("*.npy"))
            syn_files  = list((SYN_ROOT /cls).glob("*.npy"))
            if not real_files or not syn_files:
                print(f"[skip] {cls:20s} (real={len(real_files)}  syn={len(syn_files)})")
                continue
            cls_ok.append(cls)
            for f in real_files:
                X.append(np.load(f, mmap_mode="r")); y.append(len(cls_ok)-1)
            for f in syn_files:
                X.append(np.load(f, mmap_mode="r")); y.append(len(cls_ok)-1)

        self.cls_to_idx = {c:i for i,c in enumerate(cls_ok)}
        X, y = np.asarray(X, np.float32), np.asarray(y, np.int64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)

        self.X, self.y = (X_tr, y_tr) if split=="train" else (X_te, y_te)

    def __len__(self):      return len(self.X)
    def __getitem__(self,i): return torch.from_numpy(self.X[i]), int(self.y[i])

def get_loaders(batch=32, **kw):
    tr = RamanDataset(split="train", **kw)
    te = RamanDataset(split="test",  **kw)
    return (DataLoader(tr, batch, shuffle=True,  num_workers=2),
            DataLoader(te, batch*2, shuffle=False, num_workers=2))
