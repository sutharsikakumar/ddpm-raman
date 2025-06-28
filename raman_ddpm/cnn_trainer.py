"""
Train a 1-D CNN and log metrics to TensorBoard *and* a live CSV,
so scalar export works even with protobuf-6.
"""

from pathlib import Path
import csv, time, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .cnn_model   import RamanCNN
from .dataset     import RamanDataset


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RUN_DIR = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(RUN_DIR.as_posix())

Path("models").mkdir(exist_ok=True)
CSV_PATH = Path("results/metrics_live.csv")
CSV_PATH.parent.mkdir(exist_ok=True)

class CNNTrainer:
    def __init__(self, full_ds, n_cls, lr=2e-4, batch=64, device="cpu"):

        n_val = int(0.2 * len(full_ds))
        n_trn = len(full_ds) - n_val
        self.tr_ds, self.val_ds = random_split(
            full_ds, [n_trn, n_val],
            generator=torch.Generator().manual_seed(SEED))

        self.tr_dl = DataLoader(self.tr_ds, batch,   shuffle=True,  num_workers=2)
        self.va_dl = DataLoader(self.val_ds, batch*2, shuffle=False, num_workers=2)

        self.model  = RamanCNN(n_cls).to(device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device

        self.amp    = device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.best_val = 0
        self.wait = 0
        self.patience = 15

        # CSV header
        with CSV_PATH.open("w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["epoch", "loss_train", "loss_val",
                                 "acc_train", "acc_val"])


    def _run_epoch(self, dl, train=True):
        self.model.train(mode=train)
        tot_loss = correct = n = 0

        for x, y in dl:
            x = x.unsqueeze(1).to(self.device)
            y = y.to(self.device)
            with torch.amp.autocast(device_type="cuda", enabled=self.amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

            if train:
                self.opt.zero_grad(set_to_none=True)
                (self.scaler if self.amp else loss).backward()
                if self.amp:
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    self.opt.step()

            tot_loss += loss.item() * len(x)
            correct  += (logits.argmax(1) == y).sum().item()
            n        += len(x)

        return tot_loss / n, correct / n


    def fit(self, epochs=100):
        for ep in range(1, epochs+1):
            l_tr, a_tr = self._run_epoch(self.tr_dl, train=True)
            l_va, a_va = self._run_epoch(self.va_dl, train=False)

            writer.add_scalars("loss", {"train": l_tr, "val": l_va}, ep)
            writer.add_scalars("acc",  {"train": a_tr, "val": a_va}, ep)

            # CSV append
            with CSV_PATH.open("a", newline="") as f:
                csv.writer(f).writerow([ep, l_tr, l_va, a_tr, a_va])

            print(f"ep {ep:03d}  "
                  f"tr_loss {l_tr:.4f}  tr_acc {a_tr:.3%}  "
                  f"val_loss {l_va:.4f}  val_acc {a_va:.3%}")

            if a_va > self.best_val:
                self.best_val = a_va; self.wait = 0
                torch.save(self.model.state_dict(), "models/cnn_best.pt")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"⏹ early-stop at epoch {ep}")
                    break

        torch.save(self.model.state_dict(), "models/cnn_last.pt")
        writer.flush(); writer.close()
        print("✓ CSV + TensorBoard logs flushed")


if __name__ == "__main__":
    ds   = RamanDataset(split="train", test_size=0.0)
    ncls = len(ds.cls_to_idx)
    dev  = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
    CNNTrainer(ds, ncls, lr=2e-4, batch=64, device=dev).fit(epochs=200)
