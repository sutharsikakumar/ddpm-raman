import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

# NEW ───── import the dataset helper we wrote
from raman_ddpm.datasets import RamanDataset, get_loaders  
from .cnn_model import RamanCNN

class CNNTrainer:
    def __init__(self, dataset, n_cls, lr=1e-3, device="cpu"):
        self.dl     = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model  = RamanCNN(n_cls).to(device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device

    def fit(self, epochs):
        for ep in range(epochs):
            for x, y in self.dl:
                # NEW ───── add channel dimension
                x = x.unsqueeze(1).to(self.device)      # (B,1,571)
                y = y.to(self.device)

                logits = self.model(x)
                loss   = F.cross_entropy(logits, y)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            print(f"epoch {ep+1:03d}   loss {loss.item():.4f}")

# ------------------------------------------------------------------
# EXAMPLE “main” so you can run:  python -m raman_ddpm.cnn_trainer
# ------------------------------------------------------------------
if __name__ == "__main__":
    train_ds = RamanDataset(split="train")
    n_cls    = len(train_ds.cls_to_idx)

    trainer  = CNNTrainer(train_ds, n_cls,
                          lr=2e-4, device="cuda" if torch.cuda.is_available() else "cpu")
    trainer.fit(epochs=100)
