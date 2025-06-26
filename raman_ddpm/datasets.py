from pathlib import Path
from typing import Optional
import numpy as np, torch
from torch.utils.data import Dataset

def discover_labels(folder: Path):
    return sorted({p.name.split('_', 1)[0] for p in folder.glob('*.npy')})

class RamanDataset(Dataset):
    """
    If label is None → multi-class mode (returns x, y)
    If label='BP'    → single-class subset (returns x, 0)
    """
    def __init__(self, folder: str, label: Optional[str] = None):
        folder = Path(folder)
        if label is None:
            self.files = sorted(folder.glob('*.npy'))
            self.label_names = discover_labels(folder)
            self.label_to_idx = {l:i for i,l in enumerate(self.label_names)}
            self.multi = True
        else:
            self.files = sorted(folder.glob(f"{label}_*.npy"))
            if not self.files:
                raise FileNotFoundError(f"no spectra starting with {label}_")
            self.multi = False
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        x = torch.from_numpy(np.load(self.files[idx])).unsqueeze(0) * 2 - 1
        if self.multi:
            label_str = self.files[idx].name.split('_',1)[0]
            y = self.label_to_idx[label_str]
            return x.float(), y
        return x.float(), 0
