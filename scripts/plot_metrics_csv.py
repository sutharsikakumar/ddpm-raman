# scripts/plot_metrics_csv.py
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("results/metrics_live.csv")
if not csv_path.exists():
    raise SystemExit(f"{csv_path} not found – train first!")

df = pd.read_csv(csv_path)

plt.figure(figsize=(6,4))
plt.subplot(2,1,1)
plt.plot(df.epoch, df.loss_train, label="train")
plt.plot(df.epoch, df.loss_val,   label="val")
plt.ylabel("Cross-entropy loss"); plt.legend()

plt.subplot(2,1,2)
plt.plot(df.epoch, df.acc_train, label="train")
plt.plot(df.epoch, df.acc_val,   label="val")
plt.ylabel("Accuracy"); plt.xlabel("Epoch")

plt.tight_layout()
out = Path("results/curves_loss_acc.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300)
print("✓ figure saved to", out)
