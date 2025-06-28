#!/usr/bin/env python
"""
Train a 1-D CNN on Raman spectra (balanced N_REAL + N_SYN / class).

Run from repo root, venv active:
    python -m raman_ddpm.cnn_trainer
"""

from pathlib import Path
import time, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .cnn_model   import RamanCNN
from .dataset     import RamanDataset

# -------------------------------------------------------------------- #
#  reproducibility & logging paths
# -------------------------------------------------------------------- #
SEED   = 42
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)

RUN_DIR = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
writer  = SummaryWriter(RUN_DIR.as_posix())

Path("models").mkdir(exist_ok=True)

# -------------------------------------------------------------------- #
#  trainer
# -------------------------------------------------------------------- #
class CNNTrainer:
    def __init__(self,
                 full_dataset,
                 n_cls: int,
                 lr: float = 2e-4,
                 batch: int = 64,
                 device: str = "cpu"):

        # 80 / 20 split into train / val
        n_val   = int(0.2 * len(full_dataset))
        n_train = len(full_dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED))

        self.train_dl = DataLoader(self.train_ds, batch,   shuffle=True,  num_workers=2)
        self.val_dl   = DataLoader(self.val_ds,   batch*2, shuffle=False, num_workers=2)

        self.model  = RamanCNN(n_cls).to(device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device

        # AMP only on CUDA
        self.amp_enabled = (device == "cuda")
        self.scaler      = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.best_val_acc = 0
        self.wait         = 0        # early-stopping counter
        self.patience     = 15

    # ---------------------------------------------------------------
    def _loop(self, dl, train=True):
        self.model.train(mode=train)
        total_loss, correct, n = 0.0, 0, 0

        for x, y in dl:
            x = x.unsqueeze(1).to(self.device)   # (B,1,571)
            y = y.to(self.device)

            with torch.amp.autocast(
                device_type="cuda", enabled=self.amp_enabled):
                logits = self.model(x)
                loss   = F.cross_entropy(logits, y)

            if train:
                self.opt.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward(); self.opt.step()

            total_loss += loss.item() * len(x)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(x)

        return total_loss / n, correct / n      # mean loss, accuracy

    # ---------------------------------------------------------------
    def fit(self, epochs: int = 100):
        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = self._loop(self.train_dl, train=True)
            val_loss, val_acc = self._loop(self.val_dl,   train=False)

            writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, ep)
            writer.add_scalars("acc",  {"train": tr_acc,  "val": val_acc},  ep)

            print(f"ep {ep:03d}  "
                  f"tr_loss {tr_loss:.4f}  tr_acc {tr_acc:.3%}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.3%}")

            # checkpoint & early-stop
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.wait = 0
                torch.save(self.model.state_dict(), "models/cnn_best.pt")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"⏹  early stop at epoch {ep} (no improvement in {self.patience})")
                    break

        # always save final weights and FLUSH TensorBoard
        torch.save(self.model.state_dict(), "models/cnn_last.pt")
        writer.flush(); writer.close()
        print("✓ logs flushed, cnn_last.pt written")

# -------------------------------------------------------------------- #
#  script entry-point
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    full_ds = RamanDataset(split="train", test_size=0.0)   # we split inside trainer
    n_cls   = len(full_ds.cls_to_idx)

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")

    trainer = CNNTrainer(full_ds, n_cls, lr=2e-4, batch=64, device=device)
    trainer.fit(epochs=200)
