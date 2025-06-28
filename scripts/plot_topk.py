from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt

def main():

    from raman_ddpm import RamanCNN, get_loaders


    _, test_dl = get_loaders(batch=128, test_size=0.2)
    test_dl.num_workers = 0

    n_cls = len(test_dl.dataset.cls_to_idx)
    model = RamanCNN(n_cls).eval()
    model.load_state_dict(torch.load("models/cnn_best.pt"))

    topk_hits = np.zeros(5)
    n = 0
    with torch.no_grad():
        for x, y in test_dl:
            probs = model(x.unsqueeze(1))
            topk  = probs.topk(5, dim=1).indices.cpu().numpy()
            y     = y.numpy()
            for k in range(5):
                topk_hits[k] += (topk[:, :k+1] == y[:, None]).any(1).sum()
            n += len(y)

    acc_k = topk_hits / n
    plt.plot(range(1,6), acc_k, marker="o")
    plt.xticks(range(1,6)); plt.xlabel("k"); plt.ylabel("Top-k accuracy")
    plt.grid(True); plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/topk_curve.png", dpi=300)
    print("✓ results/topk_curve.png written")

if __name__ == "__main__":
    main()
