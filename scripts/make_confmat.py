"""
Confusion matrix for a *subset* of classes.

Choose ONE of two selection methods below:

    • SUBSET_NAMES : explicit list of mineral names to keep
    • TOP_N        : keep the N classes with the largest test-set support

Outputs
-------
results/conf_matrix_subset.csv   – raw integer matrix for the subset
results/conf_matrix_subset.png   – row-normalised heat-map
"""


SUBSET_NAMES = [  
    #"Quartz", "Calcite", "Forsterite", "Tremolite", "Anhydrite",
    #"Dolomite", "Gypsum", "Halite", "Hematite", "Goethite",
    #"Magnetite", "Pyrite", "Siderite", "Aragonite", "Albite",
    #"Kyanite", "Muscovite", "Hematite", "Goethite", "Magnetite", "Pyrite",
]
TOP_N = 266           

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from sklearn.metrics import confusion_matrix


def main():
    
    from raman_ddpm import RamanCNN, get_loaders

    _, test_dl = get_loaders(batch=256, test_size=0.20)
    test_dl.num_workers = 0

    idx_to_cls = {v: k for k, v in test_dl.dataset.cls_to_idx.items()}
    n_total = len(idx_to_cls)

   
    model = RamanCNN(n_total).eval()
    model.load_state_dict(torch.load("models/cnn_best.pt"))

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dl:
            y_true.extend(y.numpy())
            y_pred.extend(model(x.unsqueeze(1)).argmax(1).cpu().numpy())

    full_cm = confusion_matrix(y_true, y_pred,
                               labels=list(range(n_total))) 


    if SUBSET_NAMES:
        keep_idx = [i for i, name in idx_to_cls.items() if name in SUBSET_NAMES]
    else: 
        class_support = full_cm.sum(1)       
        top = np.argsort(class_support)[-TOP_N:]  
        keep_idx = sorted(top.tolist())

    if not keep_idx:
        raise SystemExit("No classes selected – check SUBSET_NAMES or TOP_N")

    sub_cm = full_cm[np.ix_(keep_idx, keep_idx)]
    idx_to_cls_sub = {i: idx_to_cls[j] for i, j in enumerate(keep_idx)}

    
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(sub_cm,
                 index=[idx_to_cls_sub[i] for i in range(len(keep_idx))],
                 columns=[idx_to_cls_sub[i] for i in range(len(keep_idx))]\
                ).to_csv("results/conf_matrix_subset.csv")
    print("✓ results/conf_matrix_subset.csv written")


    cm = sub_cm.astype(np.float32)
    row_sum = cm.sum(1, keepdims=True)
    cm = cm / np.where(row_sum == 0, 1, row_sum)

    n_cls = len(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, pad=.02, label="Recall")

    ax.set_title(f"Confusion matrix ({n_cls} classes)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels([idx_to_cls_sub[i] for i in range(n_cls)],
                       rotation=90, fontsize=6)
    ax.set_yticklabels([idx_to_cls_sub[i] for i in range(n_cls)],
                       fontsize=6)

    plt.tight_layout()
    fig.savefig("results/conf_matrix_subset.png", dpi=300)
    print("✓ results/conf_matrix_subset.png written")


if __name__ == "__main__":
    main()
