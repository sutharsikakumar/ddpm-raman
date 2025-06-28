"""
UMAP of CNN penultimate-layer embeddings
• colours:   blue = real   orange = synthetic
• output:    results/umap_real_vs_syn.png
"""

from pathlib import Path
import warnings, random, torch, numpy as np, matplotlib.pyplot as plt, umap


warnings.filterwarnings("ignore",
                        message=".*force_all_finite.*",   category=FutureWarning)
warnings.filterwarnings("ignore",
                        message="Graph is not fully connected.*", category=UserWarning)


from raman_ddpm import RamanCNN, get_loaders

random.seed(42); torch.manual_seed(42)


dl, _ = get_loaders(batch=512, test_size=0.0) 
dl.num_workers = 0

n_cls = len(dl.dataset.cls_to_idx)
model = RamanCNN(n_cls).eval()
model.load_state_dict(torch.load("models/cnn_best.pt"))

embeddings, origin = [], []  

with torch.no_grad():
    for x, _ in dl:
        h = model.net[:-1](x.unsqueeze(1)).cpu() 
        embeddings.append(h)
        
        origin.extend([0 if i < len(x)//2 else 1 for i in range(len(x))])

emb = torch.cat(embeddings).numpy()
origin = np.array(origin)


if len(emb) > 4000:
    idx = np.random.choice(len(emb), 4000, replace=False)
    emb, origin = emb[idx], origin[idx]

umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine",
                     random_state=42).fit_transform(emb)


plt.figure(figsize=(6,5))
plt.scatter(umap_emb[:,0], umap_emb[:,1],
            c=origin, cmap="coolwarm", s=6, alpha=.8)
plt.xticks([]); plt.yticks([])
plt.title("UMAP of CNN embeddings\nblue=real  orange=synthetic")
Path("results").mkdir(exist_ok=True)
plt.tight_layout(); plt.savefig("results/umap_real_vs_syn.png", dpi=300)
print("✓ results/umap_real_vs_syn.png written")
