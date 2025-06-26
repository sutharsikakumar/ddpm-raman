import torch.nn as nn

class RamanCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv1d(64,128, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv1d(128,256, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, n_cls)
        )
    def forward(self, x): return self.net(x)
