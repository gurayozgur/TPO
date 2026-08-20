"""Train FoundPAD on TPO.

Reproduces the released checkpoint and the TPO row of Table 1 in the paper.
Defaults are the settings that model was trained with (Sec. 4): AdamW without a
schedule, beta = (0.9, 0.999), weight decay 5e-5, gradient clipping at norm 5,
batch 48, LoRA lr 5e-6, head lr 1e-4, inverse-class-frequency sampling, 9
epochs, seed 777.

TPO is used in its entirety as a training source: the paper carves no train,
development or test partition out of it, so there is no checkpoint selection and
the final epoch is the released model.

Example
-------
python src/train.py --train_csv data/protocols/tpo_all.csv --out runs/tpo
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PADCsvDataset, balanced_sampler  # noqa: E402
from evaluate import evaluate  # noqa: E402
from models import build_model, trainable_state_dict  # noqa: E402


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser(description="Train FoundPAD on TPO")
    ap.add_argument("--train_csv", default="data/protocols/tpo_all.csv",
                    help="TPO protocol CSV to train on")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--test_csv", nargs="+", default=[],
                    help="optional held-out protocol CSVs to score after training")
    ap.add_argument("--test_name", nargs="+", default=[])
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr_lora", type=float, default=5e-6)
    ap.add_argument("--lr_header", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--max_norm", type=float, default=5.0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.4)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="random fraction of training frames (data-scale ablation)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    names = args.test_name or [os.path.splitext(os.path.basename(p))[0]
                               for p in args.test_csv]
    if len(names) != len(args.test_csv):
        raise SystemExit("--test_name must match --test_csv in length")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    model = build_model(args.device, args.lora_rank, args.lora_alpha,
                        args.lora_dropout)

    train_ds = PADCsvDataset(args.train_csv, train=True, fraction=args.fraction,
                             seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=balanced_sampler(train_ds),
                          num_workers=args.num_workers, drop_last=True)

    header = [p for n, p in model.named_parameters()
              if n.startswith("header.") and p.requires_grad]
    lora = [p for n, p in model.named_parameters()
            if not n.startswith("header.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": args.lr_lora},
         {"params": header, "lr": args.lr_header}],
        betas=(0.9, 0.999), weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"train {len(train_ds)} frames | {len(train_dl)} updates/epoch "
          f"| {args.epochs} epochs", flush=True)

    history = os.path.join(args.out, "history.jsonl")
    for epoch in range(args.epochs):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for images, target in train_dl:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            loss = criterion(model(images), target)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            n += 1
        rec = {"epoch": epoch, "loss": loss_sum / max(n, 1),
               "time_s": round(time.time() - t0, 1)}
        with open(history, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ep {epoch}] loss {rec['loss']:.4f} ({rec['time_s']}s)", flush=True)

    ckpt_path = os.path.join(args.out, "foundpad_tpo.pth")
    torch.save({"model": trainable_state_dict(model),
                "epoch": args.epochs - 1,
                "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                         "dropout": args.lora_dropout},
                "train_csv": args.train_csv}, ckpt_path)
    print(f"saved {ckpt_path}", flush=True)

    if args.test_csv:
        summary = {}
        for name, csvf in zip(names, args.test_csv):
            summary[name] = evaluate(model, csvf, args.batch_size, args.device,
                                     os.path.join(args.out, f"scores_{name}.csv"),
                                     num_workers=args.num_workers)
            v = summary[name]["video"]
            print(f"{name}: video AUC {v['AUC']:.2f} HTER {v['HTER']:.2f}", flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
