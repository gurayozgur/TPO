"""CSV-driven PAD dataset and the paper's pre-processing pipeline.

Protocol CSVs have a header row; column 0 is `image_path`, column 1 is `label`
("bonafide" or "attack"). Any further columns are metadata and are ignored here.
Labels map to 1 = bona fide, 0 = attack, so the network score
softmax[:, 1] = P(bona fide) is directly comparable across the code base.

Image paths are stored relative to the repository root, so a checkout plus an
unpacked `data/TPO` works from any working directory.

Pre-processing follows Sec. 4 of the paper: every frame is read as RGB and
resized to 224x224; no face detector or landmark step is involved. Training adds
the DADM/MMDA photometric protocol, each transform applied independently with
probability 0.5. Evaluation uses resize and normalisation only.
"""

import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import albumentations
from albumentations.pytorch import ToTensorV2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SIZE = 224

# CLIP's own channel statistics. FoundPAD's public pipeline applied ImageNet
# statistics to a CLIP encoder; the paper corrects this (Sec. 4).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(train):
    if train:
        return albumentations.Compose([
            albumentations.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.GaussNoise(std_range=(0.0, 0.2), p=0.5),
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.12, contrast_limit=0.0, p=0.5),
            albumentations.RGBShift(r_shift_limit=40, g_shift_limit=40,
                                    b_shift_limit=40, p=0.5),
            albumentations.RandomGamma(gamma_limit=(50, 150), p=0.5),
            albumentations.Normalize(CLIP_MEAN, CLIP_STD),
            ToTensorV2(),
        ])
    return albumentations.Compose([
        albumentations.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        albumentations.Normalize(CLIP_MEAN, CLIP_STD),
        ToTensorV2(),
    ])


def resolve(path):
    """Repo-relative paths resolve against the checkout; absolute paths pass through."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


class PADCsvDataset(Dataset):
    def __init__(self, csv_file, train, fraction=1.0, seed=777, return_path=False):
        self.df = pd.read_csv(csv_file)
        if fraction < 1.0:
            self.df = self.df.sample(frac=fraction, random_state=seed
                                     ).reset_index(drop=True)
        self.tf = build_transforms(train)
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def labels(self):
        return self.df.iloc[:, 1]

    def __getitem__(self, idx):
        rel = self.df.iloc[idx, 0]
        img = cv2.imread(resolve(rel))
        if img is None:
            raise RuntimeError(
                f"unreadable image: {resolve(rel)}\n"
                "Did you unpack the TPO archive into data/ ? See data/README.md")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = 1 if self.df.iloc[idx, 1] == "bonafide" else 0
        img = self.tf(image=img)["image"]
        if self.return_path:
            return img, torch.tensor(label, dtype=torch.long), rel
        return img, torch.tensor(label, dtype=torch.long)


def balanced_sampler(dataset):
    """Inverse-class-frequency sampling (FoundPAD convention).

    One epoch draws as many samples as the source contains, with replacement and
    equal expected weight for bona fide and attack frames.
    """
    labels = dataset.labels()
    counts = labels.value_counts()
    weights = [1.0 / counts[l] for l in labels.values]
    return WeightedRandomSampler(weights=weights, num_samples=len(labels),
                                 replacement=True)
