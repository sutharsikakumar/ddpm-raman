import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from .cnn_model import RamanCNN

class CNNTrainer:
    def __init__(self, dataset, n_cls, lr=1e-3, device="cpu"):
        self.dl = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = RamanCNN(n_cls).to(device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device
    def fit(self, epochs):
        for ep in range(epochs):
            for x,y in self.dl:
                x,y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            print(f"epoch {ep+1} loss {loss.item():.3f}")
