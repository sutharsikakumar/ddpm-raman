import math, torch, torch.nn as nn

def linear_beta_schedule(n, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, n)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.act    = nn.GELU()
    def forward(self, x):
        h = self.act(self.conv1(x))
        return x + self.conv2(h)

class Diffusion1D(nn.Module):
    """8-block ResNet, 128 ch, gated time embedding (Qi et al., 2023)."""
    def __init__(self, ch=128, depth=8):
        super().__init__()
        self.proj = nn.Conv1d(1, ch, 1)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(depth)])
        self.time_fc = nn.Sequential(nn.Linear(1, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.out = nn.Conv1d(ch, 1, 1)
    def forward(self, x, t):
        t_emb = self.time_fc(t[:, None]).unsqueeze(-1)
        h = self.blocks(self.proj(x) + t_emb)
        return self.out(h)
