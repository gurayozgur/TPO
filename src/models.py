"""FoundPAD detector: frozen CLIP ViT-B/16 + rsLoRA + linear two-class head.

Architecture (Sec. 4 of the paper):
  * CLIP ViT-B/16 image encoder, frozen.
  * Rank-stabilised LoRA on the query and value projections of all 12 attention
    blocks (r=8, alpha=8, dropout=0.4).
  * Linear two-class head on the L2-normalised image embedding.

Only the LoRA matrices and the head are trained: ~0.30M parameters.
The score reported everywhere is softmax[:, 1] = P(bona fide).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

from lora import PlainMultiheadAttentionLoRA

BACKBONE = "ViT-B/16"
EMBED_DIM = 512


class FoundPAD(nn.Module):
    def __init__(self, visual, embed_dim=EMBED_DIM):
        super().__init__()
        self.visual = visual
        self.header = nn.Linear(embed_dim, 2)

    def forward(self, x):
        feats = self.visual(x.to(torch.float32)).float()
        feats = F.normalize(feats, dim=-1)
        return self.header(feats)


def _apply_lora(visual, rank, alpha, dropout, device):
    n = 0
    for block in visual.transformer.resblocks:
        for name, sub in block.named_children():
            if isinstance(sub, nn.MultiheadAttention):
                setattr(block, name, PlainMultiheadAttentionLoRA(
                    sub, enable_lora=["q", "v"], r=rank,
                    lora_alpha=alpha, dropout_rate=dropout).to(device))
                n += 1
    return n


def build_model(device="cuda", lora_rank=8, lora_alpha=8, lora_dropout=0.4,
                clip_download_root=None, verbose=True):
    """Build the FoundPAD detector.

    CLIP ViT-B/16 (~350 MB) is downloaded on first use into `$CLIP_CACHE_DIR`,
    or `~/.cache/clip` if that variable is unset.
    """
    if clip_download_root is None:
        clip_download_root = os.environ.get(
            "CLIP_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "clip"))

    clip_model, _ = clip.load(BACKBONE, device="cpu", jit=False,
                              download_root=clip_download_root)
    for p in clip_model.parameters():          # CLIP ships fp16 weights
        if p.dtype == torch.float16:
            p.data = p.data.float()

    model = FoundPAD(clip_model.to(device).visual).to(device)
    n_blocks = _apply_lora(model.visual, lora_rank, lora_alpha, lora_dropout, device)

    # freeze everything except LoRA and the head
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name) or name.startswith("header.")

    if verbose:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[FoundPAD] CLIP {BACKBONE} + rsLoRA on {n_blocks} blocks "
              f"(r={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
        print(f"[FoundPAD] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M")
    return model


def trainable_state_dict(model):
    """The ~0.30M LoRA + head parameters -- everything else is frozen CLIP."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_" in k or k.startswith("header.")}


def load_detector(path, device="cuda", verbose=True):
    """Build the detector and load a trained checkpoint into it.

    The LoRA geometry is taken from the checkpoint, so models trained with a
    non-default rank load correctly. Returns (model, checkpoint metadata).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("lora", {})
    model = build_model(device,
                        lora_rank=cfg.get("rank", 8),
                        lora_alpha=cfg.get("alpha", 8),
                        lora_dropout=cfg.get("dropout", 0.4),
                        verbose=verbose)

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    stray = [k for k in missing if "lora_" in k or k.startswith("header.")]
    if stray:
        raise RuntimeError(f"checkpoint is missing trainable weights: {stray[:5]}")
    return model, ckpt
