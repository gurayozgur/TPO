"""Run the TPO-trained FoundPAD detector on a protocol CSV or a folder of images.

The score is P(bona fide) in [0, 1]: high means bona fide, low means attack.

Labelled protocol CSV -> frame- and video-level AUC, EER and HTER:
    python src/infer.py --ckpt pretrained/foundpad_tpo.pth \
        --csv data/protocols/tpo_all.csv

Folder of images -> per-image scores only (no metrics, nothing to compare to):
    python src/infer.py --ckpt pretrained/foundpad_tpo.pth \
        --images my_frames/ --scores_out my_scores.csv
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import build_transforms  # noqa: E402
from evaluate import evaluate  # noqa: E402
from models import load_detector  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@torch.no_grad()
def score_images(model, paths, batch_size, device):
    tf = build_transforms(train=False)
    scores = []
    model.eval()
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i + batch_size]:
            img = cv2.imread(p)
            if img is None:
                raise SystemExit(f"unreadable image: {p}")
            batch.append(tf(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"])
        logits = model(torch.stack(batch).to(device))
        scores.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser(description="FoundPAD (TPO) inference")
    ap.add_argument("--ckpt", default="pretrained/foundpad_tpo.pth")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="labelled protocol CSV -> reports metrics")
    src.add_argument("--images", help="image file or folder -> reports scores")
    ap.add_argument("--scores_out", default=None, help="write per-frame scores here")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        device = "cpu"

    model, ckpt = load_detector(args.ckpt, device)
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    if args.csv:
        res = evaluate(model, args.csv, args.batch_size, device, args.scores_out,
                       args.num_workers)
        for level in ("frame", "video"):
            m = res[level]
            print(f"{level:5s} (n={res['n_' + level + 's']:6d})  "
                  f"AUC {m['AUC']:6.2f}  EER {m['EER']:6.2f}  HTER {m['HTER']:6.2f}")
        print(json.dumps(res, indent=2))
        return

    if os.path.isdir(args.images):
        paths = sorted(p for p in glob.glob(os.path.join(args.images, "**", "*"),
                                            recursive=True)
                       if p.lower().endswith(IMAGE_EXT))
    else:
        paths = [args.images]
    if not paths:
        raise SystemExit(f"no images found under {args.images}")

    scores = score_images(model, paths, args.batch_size, device)
    out = pd.DataFrame({"image_path": paths, "score_bonafide": scores})
    if args.scores_out:
        out.to_csv(args.scores_out, index=False)
        print(f"wrote {args.scores_out} ({len(out)} images)")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
