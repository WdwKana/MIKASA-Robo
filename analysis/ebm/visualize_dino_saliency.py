"""
Day 1 sanity check: visualize DINOv2 CLS-to-patch attention on MIKASA-Robo demo frames.

Tests assumption H1: pretrained DINOv2 CLS attention highlights task-relevant
objects (purple cube, ball, target ring, candidate blocks) without any
training, and stays low on empty wood-grain frames.

Outputs:
  analysis/ebm/saliency_maps/{task}/step{NNN}_overlay.png   side-by-side raw + heatmap overlay
  analysis/ebm/saliency_maps/_summary.json                  per-frame max/mean saliency for D1 decision

Run:
  HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub \
    /home/s4176650/.conda/envs/mikasa/bin/python \
    analysis/ebm/visualize_dino_saliency.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
OUT_DIR = REPO_ROOT / "analysis/ebm/saliency_maps"
MODEL_NAME = "facebook/dinov2-small"

FRAMES_TO_PROCESS = [
    ("RememberColor9-v0", ["000", "005", "010", "020", "030", "040", "050"]),
    ("InterceptFast-v0",  ["000", "005", "010", "020", "030", "040", "050"]),
    ("RememberShapeAndColor3x2-v0", ["000", "005", "010", "020", "030", "040", "050"]),
]

# Frames are 640x512. Layout:
#   x=0..480, y=0..512  -> base camera (large, contains arm + table + objects)
#   x=480..640, y=0..256 -> small gripper inset (we discard for single-camera setup)
#   x=480..640, y=256..512 -> black padding
# We crop the base camera region only, mirroring the user's single-camera setup.
CROP_BASE = (0, 0, 480, 512)  # (left, upper, right, lower) in PIL convention


def load_model(device: torch.device):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def cls_to_patch_attention(model, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Returns:
      attn  (Hp*Wp,)  head-mean of last-layer CLS->patch attention
      grid  (Hp, Wp)  patch grid shape
    """
    out = model(pixel_values=pixel_values, output_attentions=True)
    # last layer attentions: (B, num_heads, S, S), S = 1 + Hp*Wp (no register tokens for dinov2-small)
    attn = out.attentions[-1]                  # (1, H, S, S)
    cls_attn = attn[0, :, 0, 1:]               # (H, Hp*Wp)  drop CLS->CLS
    cls_attn = cls_attn.mean(dim=0)            # (Hp*Wp,) head-mean
    # Determine grid shape from input
    _, _, h, w = pixel_values.shape
    p = model.config.patch_size
    grid = (h // p, w // p)
    assert grid[0] * grid[1] == cls_attn.shape[0], (grid, cls_attn.shape)
    return cls_attn, grid


def render_overlay(raw_pil: Image.Image, sal_2d: np.ndarray, save_path: Path,
                   title: str) -> None:
    """Side-by-side: raw | heatmap | overlay."""
    sal_norm = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)

    raw_np = np.array(raw_pil.convert("RGB"))
    H, W = raw_np.shape[:2]
    sal_up = np.array(Image.fromarray((sal_norm * 255).astype(np.uint8))
                      .resize((W, H), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(raw_np); ax[0].set_title("raw"); ax[0].axis("off")
    ax[1].imshow(sal_up, cmap="hot", vmin=0, vmax=1); ax[1].set_title("CLS->patch attn"); ax[1].axis("off")
    ax[2].imshow(raw_np); ax[2].imshow(sal_up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax[2].set_title("overlay"); ax[2].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model(device)
    print(f"loaded {MODEL_NAME} on {device}, patch_size={model.config.patch_size}")

    summary = {"model": MODEL_NAME, "frames": []}

    for task, steps in FRAMES_TO_PROCESS:
        frames_dir = REPO_ROOT / f"{task}_frames"
        if not frames_dir.is_dir():
            print(f"[skip] {frames_dir} missing")
            continue
        for step in steps:
            src = frames_dir / f"render_env0_step{step}.png"
            if not src.exists():
                print(f"[skip] {src} missing")
                continue

            raw = Image.open(src).convert("RGB")
            base = raw.crop(CROP_BASE)  # single base-camera region

            inputs = processor(images=base, return_tensors="pt").to(device)
            attn, grid = cls_to_patch_attention(model, inputs.pixel_values)
            sal_2d = attn.reshape(grid).float().cpu().numpy()

            out_path = OUT_DIR / task / f"step{step}_overlay.png"
            render_overlay(base, sal_2d, out_path, f"{task} step{step}")

            summary["frames"].append({
                "task": task,
                "step": int(step),
                "out": str(out_path.relative_to(REPO_ROOT)),
                "saliency_max": float(sal_2d.max()),
                "saliency_mean": float(sal_2d.mean()),
                "saliency_p95":  float(np.percentile(sal_2d, 95)),
                "max_over_mean": float(sal_2d.max() / (sal_2d.mean() + 1e-8)),
                "grid": list(grid),
            })
            print(f"  {task} step{step}: max={sal_2d.max():.4f} "
                  f"mean={sal_2d.mean():.4f} max/mean={sal_2d.max()/sal_2d.mean():.2f}")

    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {summary_path}")
    print(f"overlays -> {OUT_DIR}/<task>/step*_overlay.png")


if __name__ == "__main__":
    main()
