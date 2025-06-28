#!/usr/bin/env python
"""
Make a static PNG of loss / accuracy curves from the latest TensorBoard run.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

RUNS = Path("runs")

# --------------------------------------------------------------------
# find latest run directory that contains an events file
# --------------------------------------------------------------------
run_dirs = sorted([p for p in RUNS.iterdir() if p.is_dir()])
if not run_dirs:
    raise SystemExit("No runs/ sub-folders found.")

latest = run_dirs[-1]                       # newest timestamp
events = list(latest.glob("events.out.tfevents.*"))
if not events:
    raise SystemExit(f"No .tfevents file in {latest}")
print(f"Using log dir: {latest}")

# --------------------------------------------------------------------
# load scalars
# --------------------------------------------------------------------
ea = event_accumulator.EventAccumulator(latest.as_posix())
ea.Reload()

print("Available scalar tags:", ea.Tags()["scalars"])
# tags created by CNNTrainer
train_loss = [x.value for x in ea.Scalars("loss/train")]
val_loss   = [x.value for x in ea.Scalars("loss/val")]
train_acc  = [x.value for x in ea.Scalars("acc/train")]
val_acc    = [x.value for x in ea.Scalars("acc/val")]

epochs = range(1, len(train_loss)+1)

# --------------------------------------------------------------------
# plot
# --------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.subplot(2,1,1)
plt.plot(epochs, train_loss, label="train"); plt.plot(epochs, val_loss, label="val")
plt.ylabel("Cross-entropy loss"); plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs, train_acc,  label="train"); plt.plot(epochs, val_acc,  label="val")
plt.ylabel("Accuracy"); plt.xlabel("Epoch")

plt.tight_layout()
out = Path("results/curves_loss_acc.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300)
print(f"✓ curves saved to {out}")
