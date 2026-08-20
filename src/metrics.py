"""PAD metrics: AUC, EER and HTER, at frame and video level.

Conventions follow the paper (Sec. 4, "Metrics"):
  * score = P(bona fide); label 1 = bona fide, 0 = attack.
  * APCER = fraction of attacks accepted, BPCER = fraction of bona fide rejected.
  * HTER = (APCER + BPCER) / 2, evaluated at the target set's EER threshold.
  * Frame scores belonging to the same source video are averaged before
    evaluation; a still image is a one-frame presentation.
"""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import auc as sk_auc, roc_curve


def eer_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    return thr[np.nanargmin(np.abs(fpr - (1 - tpr)))]


def metrics_at(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    bf, atk = scores[labels == 1], scores[labels == 0]
    apcer = float(np.mean(atk >= threshold)) if len(atk) else float("nan")
    bpcer = float(np.mean(bf < threshold)) if len(bf) else float("nan")
    return {"APCER": apcer * 100, "BPCER": bpcer * 100,
            "HTER": (apcer + bpcer) / 2 * 100, "TH": float(threshold)}


def performances(scores, labels):
    """AUC, EER and HTER at this set's own EER threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    at_eer = metrics_at(scores, labels, eer_threshold(scores, labels))
    return {"AUC": sk_auc(fpr, tpr) * 100,
            "EER": (at_eer["APCER"] + at_eer["BPCER"]) / 2,
            "HTER": at_eer["HTER"], "APCER": at_eer["APCER"],
            "BPCER": at_eer["BPCER"], "TH": at_eer["TH"]}


def video_id_of(path):
    """Frames of one presentation share `<dir>/<stem minus the trailing _<k>>`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(path), stem.rsplit("_", 1)[0])


def video_level(paths, scores, labels):
    agg = defaultdict(lambda: [[], None])
    for p, s, l in zip(paths, scores, labels):
        vid = video_id_of(p)
        agg[vid][0].append(s)
        agg[vid][1] = l
    vids = sorted(agg)
    return (vids,
            np.array([np.mean(agg[v][0]) for v in vids]),
            np.array([agg[v][1] for v in vids]))
