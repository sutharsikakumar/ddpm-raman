#!/usr/bin/env python
import argparse, torch, random
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, random_split
from raman_ddpm import RamanDataset, CNNTrainer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=40)
args = p.parse_args()

real_ds  = RamanDataset("data/processed")               
synth_ds = RamanDataset("data/synthetic")             

full = ConcatDataset([real_ds, synth_ds])
n_val = int(0.2 * len(full))
train_ds, val_ds = random_split(full, [len(full)-n_val, n_val],
                                generator=torch.Generator().manual_seed(1))

trainer = CNNTrainer(train_ds, n_cls=len(real_ds.label_names), device=DEVICE)
trainer.fit(args.epochs)
