"""Two-stage intrusion detector.

Stage 1: a binary LightGBM (Normal vs attack) with a tunable alert threshold.
Stage 2: a LightGBM that names the attack family, run on flows stage 1 flags.

The object exposes predict_proba() over the 10 classes so it can sit next to the
other pipelines. The returned probabilities are built so that argmax equals the
two-stage decision (attack iff stage-1 probability >= threshold).
"""
import numpy as np


class TwoStageIDS:
    def __init__(self, columns, detector, family, threshold, class_names, family_classes):
        self.columns = list(columns)
        self.detector = detector
        self.family = family
        self.threshold = float(threshold)
        self.class_names = list(class_names)
        self.normal_idx = self.class_names.index("Normal")
        # stage-2 output column -> index in the 10-class ordering
        self.family_idx = [self.class_names.index(c) for c in family_classes]

    def _attack_score(self, X):
        return self.detector.predict_proba(X[self.columns])[:, 1]

    def predict_proba(self, X):
        Xc = X[self.columns]
        p = np.clip(self.detector.predict_proba(Xc)[:, 1], 1e-6, 1 - 1e-6)
        thr = min(max(self.threshold, 1e-6), 1 - 1e-6)
        # re-centre so that 0.5 corresponds to the alert threshold
        logit = np.log(p / (1 - p)) - np.log(thr / (1 - thr))
        p_adj = 1 / (1 + np.exp(-logit))

        q = self.family.predict_proba(Xc)
        out = np.zeros((len(Xc), len(self.class_names)))
        out[:, self.family_idx] = q * p_adj[:, None]
        normal = 1 - p_adj
        # flagged rows must resolve to an attack family, not Normal
        flagged = p_adj >= 0.5
        cap = out.max(axis=1) * 0.5
        normal = np.where(flagged, np.minimum(normal, cap), normal)
        out[:, self.normal_idx] = normal
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
