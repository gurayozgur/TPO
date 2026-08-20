"""Scoring helpers shared by `train.py` and `infer.py`."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PADCsvDataset
from metrics import performances, video_level


@torch.no_grad()
def score_dataset(model, csv_file, batch_size=64, device="cuda", num_workers=6):
    """Return (scores, labels, paths); score = P(bona fide) per frame."""
    dl = DataLoader(PADCsvDataset(csv_file, train=False, return_path=True),
                    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    scores, labels, paths = [], [], []
    for img, lab, pth in dl:
        logits = model(img.to(device, non_blocking=True))
        scores.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(lab.numpy())
        paths.extend(pth)
    return np.concatenate(scores), np.concatenate(labels), paths


def evaluate(model, csv_file, batch_size=64, device="cuda", scores_out=None,
             num_workers=6):
    """Frame- and video-level AUC/EER/HTER for one protocol CSV."""
    scores, labels, paths = score_dataset(model, csv_file, batch_size, device,
                                          num_workers)
    _, v_scores, v_labels = video_level(paths, scores, labels)
    if scores_out:
        os.makedirs(os.path.dirname(os.path.abspath(scores_out)), exist_ok=True)
        pd.DataFrame({"image_path": paths, "score": scores, "label": labels}
                     ).to_csv(scores_out, index=False)
    return {"frame": performances(scores, labels),
            "video": performances(v_scores, v_labels),
            "n_frames": int(len(scores)), "n_videos": int(len(v_scores))}
