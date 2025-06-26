import torch
from .ddpm_model import Diffusion1D, linear_beta_schedule

@torch.no_grad()
def sample_spectra(ckpt, n_samples=8, n_steps=50, device="cpu"):
    betas = linear_beta_schedule(n_steps).to(device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, 0)
    model = Diffusion1D().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    x = torch.randn(n_samples, 1, 571, device=device)
    for i in reversed(range(n_steps)):
        a_t = alphas_cum[i]
        eps = model(x, torch.full((n_samples,), i, device=device) / n_steps)
        x = (1 / a_t.sqrt()) * (x - (1 - a_t).sqrt() * eps)
        if i:
            x += betas[i].sqrt() * torch.randn_like(x)
    return (x + 1) / 2   
