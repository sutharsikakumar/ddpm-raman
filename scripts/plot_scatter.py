import torch, matplotlib.pyplot as plt, numpy as np
from raman_ddpm import RamanCNN, get_loaders

_, te = get_loaders(batch=512); te.num_workers = 0
n_cls = len(te.dataset.cls_to_idx)
model = RamanCNN(n_cls).eval()
model.load_state_dict(torch.load("models/cnn_best.pt"))

conf, err = [], []
with torch.no_grad():
    for x,y in te:
        prob = torch.softmax(model(x.unsqueeze(1)), 1)
        c    = prob.max(1).values.cpu().numpy()
        e    = (prob.argmax(1).cpu() != y).numpy().astype(int)
        conf.extend(c); err.extend(e)

plt.scatter(conf, err, s=6, alpha=.4)
plt.yticks([0,1], ["correct","wrong"]); plt.xlabel("Confidence")
plt.tight_layout(); plt.savefig("results/conf_vs_err.png", dpi=300)
