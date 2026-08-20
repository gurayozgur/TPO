"""Verify that an unpacked TPO release matches the shipped protocols.

    python src/check_data.py

Checks that every frame referenced by every protocol CSV exists, that the
per-protocol frame and presentation counts are the expected ones, and that a
random sample of frames decodes at the expected 256x256 resolution.
"""

import csv
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import REPO_ROOT, resolve  # noqa: E402
from metrics import video_id_of  # noqa: E402

PROTOCOLS = {
    "tpo_all": (57408, 12480),
    "tpo_tomatoes": (19136, 4160),
    "tpo_potatoes": (19136, 4160),
    "tpo_onions": (19136, 4160),
    "tpo_print": (32448, 7488),
    "tpo_replay": (32448, 7488),
}
SAMPLE = 200


def main():
    proto_dir = os.path.join(REPO_ROOT, "data", "protocols")
    ok = True
    all_paths = []

    for name, (n_frames, n_videos) in sorted(PROTOCOLS.items()):
        path = os.path.join(proto_dir, name + ".csv")
        if not os.path.exists(path):
            print(f"[FAIL] {name}: protocol CSV missing at {path}")
            ok = False
            continue
        rows, missing, vids = 0, 0, set()
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows += 1
                vids.add(video_id_of(row[0]))
                if not os.path.exists(resolve(row[0])):
                    missing += 1
                    if missing == 1:
                        first_missing = row[0]
                all_paths.append(row[0])
        status = "ok  " if (missing == 0 and rows == n_frames
                            and len(vids) == n_videos) else "FAIL"
        ok &= status == "ok  "
        print(f"[{status}] {name:14s} {rows:6d}/{n_frames} frames, "
              f"{len(vids):5d}/{n_videos} presentations, {missing} missing")
        if missing:
            print(f"         first missing: {resolve(first_missing)}")

    if all_paths:
        random.seed(0)
        bad = []
        for rel in random.sample(all_paths, min(SAMPLE, len(all_paths))):
            img = cv2.imread(resolve(rel))
            if img is None or img.shape[:2] != (256, 256):
                bad.append((rel, None if img is None else img.shape))
        if bad:
            ok = False
            print(f"[FAIL] {len(bad)}/{SAMPLE} sampled frames unreadable or not "
                  f"256x256, e.g. {bad[0]}")
        else:
            print(f"[ok  ] {min(SAMPLE, len(all_paths))} sampled frames decode at 256x256")

    if ok:
        print("\nTPO looks complete.")
    else:
        print("\nSomething is missing. Unpack the TPO archive into data/ so that "
              "the frames land in data/TPO/ -- see README.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
