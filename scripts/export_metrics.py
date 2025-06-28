"""
Export loss/accuracy curves from the newest TensorBoard run
that actually contains scalar data.
"""

from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

RUNS = Path("runs")
run_dirs = sorted([p for p in RUNS.iterdir() if p.is_dir()])

if not run_dirs:
    raise SystemExit("No runs/* folders found.")

ea = None
for run in reversed(run_dirs):
    if not any(run.glob("events.out.tfevents.*")):
        continue
    ea_tmp = event_accumulator.EventAccumulator(run.as_posix(),
                                                size_guidance={'scalars': 0})
    ea_tmp.Reload()
    if ea_tmp.Tags()["scalars"]:
        ea = ea_tmp
        print("Using run:", run.name)
        break

if ea is None:
    raise SystemExit("No scalars found in ANY run folder. "
                     "Did the trainer write SummaryWriter.add_scalars(...) ?")

def scalars(tag):
    return [(ev.step, ev.value) for ev in ea.Scalars(tag)]

# confirm available tags
print("Scalar tags:", ea.Tags()["scalars"])

train_loss = scalars("loss/train")
val_loss   = scalars("loss/val")
train_acc  = scalars("acc/train")
val_acc    = scalars("acc/val")

data = {
    "epoch":       [s for s,_ in train_loss],
    "loss_train":  [v for _,v in train_loss],
    "loss_val":    [v for _,v in val_loss],
    "acc_train":   [v for _,v in train_acc],
    "acc_val":     [v for _,v in val_acc],
}

out = Path("results/metrics_by_epoch.csv")
out.parent.mkdir(exist_ok=True)
pd.DataFrame(data).to_csv(out, index=False)
print("CSV written to", out)
