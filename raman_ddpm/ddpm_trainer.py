import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from .ddpm_model import Diffusion1D, linear_beta_schedule

class DDPMTrainer:
    """
    Train a 1-D DDPM on a single label.
      • batch_size = 64   (fits easily in Apple-M GPU RAM)
      • drop_last = False so tiny classes still get a batch
      • self.ckpt_path  – set by caller; checkpoint written at the end
      • self.label      – used only for nicer log lines
    """
    def __init__(self, dataset, n_steps=50, lr=2e-4,
                 device="cpu", batch_size=64, label=""):
        self.dl       = DataLoader(dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=False)
        self.n_steps  = n_steps
        self.device   = device
        self.label    = label
        self.model    = Diffusion1D().to(device)
        self.opt      = torch.optim.Adam(self.model.parameters(), lr)

        betas  = linear_beta_schedule(n_steps).to(device)
        self.alphas     = 1.0 - betas
        self.alphas_cum = torch.cumprod(self.alphas, 0)

        self.ckpt_path = None 

    def step(self, x0):
        t   = torch.randint(0, self.n_steps, (x0.size(0),), device=self.device)
        a_t = self.alphas_cum[t][:, None, None]
        noise = torch.randn_like(x0)
        x_t   = (a_t.sqrt() * x0 + (1 - a_t).sqrt() * noise)
        pred  = self.model(x_t, t.float() / self.n_steps)

        loss = F.mse_loss(pred, noise)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, epochs):
        for ep in range(epochs):
            epoch_loss = n = 0
            for x, _ in self.dl:
                epoch_loss += self.step(x.to(self.device)); n += 1
            avg = epoch_loss / max(n, 1)
            print(f"[{self.label}] epoch {ep+1}/{epochs}  "
                  f"loss {avg:.4f}  batches {n}")

   
        if self.ckpt_path:
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)
            print(f"checkpoint saved → {self.ckpt_path}")
