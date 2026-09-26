"""Train and evaluate the two-stage IDS on UNSW-NB15, then save it for the API.

Run from the repository root:
    python scripts/train_two_stage.py --data <folder with the two UNSW-NB15 CSVs>

The alert threshold is chosen on a held-out slice of the TRAINING partition
(never on the test set): the highest attack recall with false alarms <= --max-fpr.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from backend.two_stage import TwoStageIDS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--max-fpr", type=float, default=0.05)
args = ap.parse_args()

MODEL_DIR = os.path.join(ROOT, "model")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
meta = json.load(open(os.path.join(MODEL_DIR, "preprocessing_meta.json")))
class_names = json.load(open(os.path.join(MODEL_DIR, "class_names.json")))
redundant = set(meta["redundant_columns_dropped"])
enc_cols = meta["encoded_column_order"]
# features actually used by the new model: the encoded layout minus the redundant columns
use_cols = [c for c in enc_cols if c not in redundant]

train = pd.read_csv(os.path.join(args.data, "UNSW_NB15_training-set.csv"))  # 175,341 flows
test = pd.read_csv(os.path.join(args.data, "UNSW_NB15_testing-set.csv"))    # 82,332 flows


def encode(df):
    d = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")
    d = pd.get_dummies(d, columns=meta["categorical_columns"], drop_first=True)
    for c in use_cols:
        if c not in d.columns:
            d[c] = 0
    return d[use_cols].astype(float)


Xtr, Xte = encode(train), encode(test)
ytr = train.attack_cat.map({c: i for i, c in enumerate(class_names)}).values
yte = test.attack_cat.map({c: i for i, c in enumerate(class_names)}).values
NORMAL = class_names.index("Normal")
attack_classes = [c for c in class_names if c != "Normal"]
attack_ids = [class_names.index(c) for c in attack_classes]

PARAMS = dict(num_leaves=15, min_child_samples=50, reg_lambda=5, n_estimators=300,
              learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)


def fit_stages(X, y):
    det = LGBMClassifier(**PARAMS).fit(X, (y != NORMAL).astype(int))
    m = y != NORMAL
    remap = {cid: i for i, cid in enumerate(attack_ids)}
    fam = LGBMClassifier(class_weight="balanced", **PARAMS).fit(X[m], np.array([remap[v] for v in y[m]]))
    return det, fam


# ---- 1) choose the threshold on a held-out slice of the training partition
a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
det_a, _ = fit_stages(a, ya)
pv = det_a.predict_proba(b)[:, 1]
isatk = yb != NORMAL
best = None
for thr in np.round(np.arange(0.05, 0.996, 0.005), 3):
    pred = pv >= thr
    fpr = (pred & ~isatk).sum() / (~isatk).sum()
    rec = (pred & isatk).sum() / isatk.sum()
    if fpr <= args.max_fpr and (best is None or rec > best[1]):
        best = (float(thr), rec, fpr)
thr = best[0]
print(f"threshold chosen on training-partition validation: {thr:.3f} "
      f"(val recall={best[1]:.3f}, val false-alarm={best[2]:.3f})")

# ---- 2) refit on the full training partition and evaluate on the unseen test partition
det, fam = fit_stages(Xtr, ytr)
model = TwoStageIDS(use_cols, det, fam, thr, class_names, attack_classes)
pred = model.predict(Xte)
prob = model.predict_proba(Xte)

acc = accuracy_score(yte, pred)
mf1 = f1_score(yte, pred, average="macro")
wf1 = f1_score(yte, pred, average="weighted")
tb = yte != NORMAL
pb = pred != NORMAL
recall = (pb & tb).sum() / tb.sum()
fpr = (pb & ~tb).sum() / (~tb).sum()
precision = (pb & tb).sum() / max(pb.sum(), 1)
bin_acc = (pb == tb).mean()
auc = roc_auc_score(tb, det.predict_proba(Xte)[:, 1])
fam_acc = accuracy_score(yte[tb], pred[tb])
print(f"\n10-class:  accuracy={acc:.3f}  macroF1={mf1:.3f}  weightedF1={wf1:.3f}")
print(f"Binary  :  accuracy={bin_acc:.3f}  attack recall={recall:.3f}  precision={precision:.3f}  "
      f"false-alarm={fpr:.3f}  ROC-AUC={auc:.4f}")
print(f"Family accuracy on true attacks (incl. missed): {fam_acc:.3f}\n")
print(classification_report(yte, pred, target_names=class_names, digits=2))

# threshold sweep for the report
sweep = []
ps = det.predict_proba(Xte)[:, 1]
for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    pr = ps >= t
    sweep.append({"threshold": t, "accuracy": float((pr == tb).mean()),
                  "attack_recall": float((pr & tb).sum() / tb.sum()),
                  "false_alarm": float((pr & ~tb).sum() / (~tb).sum())})

# ---- 3) figures used by the dashboard (ids: twostage)
os.makedirs(FIG_DIR, exist_ok=True)


def cm_fig(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax,
                                            cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


cm_fig(yte, pred, "Confusion Matrix - Two-Stage IDS (Unseen Test Set)",
       os.path.join(FIG_DIR, "confusion_matrix_test_twostage.png"))
det_v = det_a  # validation matrix comes from the model that did not see the slice
fam_a = fit_stages(a, ya)[1]
val_model = TwoStageIDS(use_cols, det_a, fam_a, thr, class_names, attack_classes)
cm_fig(yb, val_model.predict(b), "Confusion Matrix - Two-Stage IDS (Validation Slice)",
       os.path.join(FIG_DIR, "confusion_matrix_val_twostage.png"))

fpr_c, tpr_c, _ = roc_curve(tb, ps)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr_c, tpr_c, lw=2, label=f"Normal vs attack (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey")
ax.scatter([fpr], [recall], color="red", zorder=5, label=f"Operating point (thr {thr:.2f})")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC - Two-Stage IDS detector (Test Set)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "roc_auc_test_twostage.png"), dpi=150); plt.close(fig)

# ---- 4) save
joblib.dump(model, os.path.join(MODEL_DIR, "pipeline_twostage.pkl"))
report = classification_report(yte, pred, target_names=class_names, output_dict=True, digits=4)
json.dump({
    "id": "twostage", "name": "Two-Stage IDS",
    "test_accuracy": round(acc, 4), "test_weighted_f1": round(wf1, 4), "test_macro_f1": round(mf1, 4),
    "binary": {"accuracy": round(float(bin_acc), 4), "attack_recall": round(float(recall), 4),
               "attack_precision": round(float(precision), 4), "false_alarm_rate": round(float(fpr), 4),
               "roc_auc": round(float(auc), 4)},
    "threshold": thr, "threshold_chosen_on": "held-out 20% of training partition, max recall with false alarms <= "
                                              f"{args.max_fpr:.0%}",
    "threshold_sweep_on_test": sweep,
    "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                  for c in class_names},
}, open(os.path.join(MODEL_DIR, "twostage_metrics.json"), "w"), indent=2)
print("saved model/pipeline_twostage.pkl, model/twostage_metrics.json and figures")
