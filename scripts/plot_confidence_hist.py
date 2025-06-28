"""
Confidence-histogram visualisation

• Top panel – full histogram on a **log y-scale** (so the big spike at
  confidence ≈ 0 is visible but doesn’t dwarf everything else).

• Bottom panel – linear-scale zoom on the interesting range
  0.1 … 1.0 so you can compare correct vs wrong when the model is
  relatively “sure”.

Works on macOS/Win/Linux because the DataLoader uses num_workers = 0 and
everything runs inside a  __main__  guard.
"""

from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt


def main():
   
    from raman_ddpm import RamanCNN, get_loaders


    _, test_dl = get_loaders(batch=256, test_size=0.20)
    test_dl.num_workers = 0

    n_cls = len(test_dl.dataset.cls_to_idx)
    model = RamanCNN(n_cls).eval()
    model.load_state_dict(torch.load("models/cnn_best.pt"))

    conf_good, conf_bad = [], []
    with torch.no_grad():
        for x, y in test_dl:
            probs = torch.softmax(model(x.unsqueeze(1)), 1)
            conf  = probs.max(1).values.cpu().numpy()
            preds = probs.argmax(1).cpu().numpy()
            good  = preds == y.numpy()
            conf_good.extend(conf[good])
            conf_bad.extend(conf[~good])


    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1]}
    )


    ax_full.hist(conf_good, bins=20, alpha=.6, label="correct")
    ax_full.hist(conf_bad,  bins=20, alpha=.6, label="wrong")
    ax_full.set_yscale("log")
    ax_full.set_ylabel("#Spectra (log)")
    ax_full.legend()

    bins_zoom = np.linspace(0.1, 1.0, 19)
    ax_zoom.hist([c for c in conf_good if c >= .1], bins=bins_zoom,
                 alpha=.6, label="correct (>0.1)")
    ax_zoom.hist([c for c in conf_bad  if c >= .1], bins=bins_zoom,
                 alpha=.6, label="wrong  (>0.1)")
    ax_zoom.set_xlabel("Soft-max confidence")
    ax_zoom.set_ylabel("#Spectra")
    ax_zoom.legend(fontsize=8)

    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    out = Path("results/confidence_hist.png")
    fig.savefig(out, dpi=300)
    print("✓", out, "written")


if __name__ == "__main__":
    main()
