from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt
from itertools import islice
from raman_ddpm import RamanCNN, get_loaders

WN = np.linspace(50, 1750, 571)
_, test_dl = get_loaders(batch=128, test_size=0.2)
n_cls = len(test_dl.dataset.cls_to_idx)
idx_to_cls = {v:k for k,v in test_dl.dataset.cls_to_idx.items()}

model = RamanCNN(n_cls).eval()
model.load_state_dict(torch.load("models/cnn_best.pt"))

mis = []
with torch.no_grad():
    for x, y in test_dl:
        logits = model(x.unsqueeze(1))
        preds  = logits.argmax(1)
        wrong  = preds != y
        for spec, t, p in zip(x[wrong], y[wrong], preds[wrong]):
            mis.append((spec.numpy(), idx_to_cls[int(t)], idx_to_cls[int(p)]))
        if len(mis) > 5: break               

for i,(spec,true_lbl,pred_lbl) in enumerate(mis[:5],1):
    plt.figure(figsize=(4,2))
    plt.plot(WN, spec, c="k")
    plt.title(f"true: {true_lbl}  pred: {pred_lbl}", fontsize=8)
    plt.xlabel("Raman shift (cm⁻¹)"); plt.tight_layout()
    plt.savefig(f"results/mis_{i}.png", dpi=300); plt.close()

print("✓ mis-classified overlays saved to results/mis_*.png")
