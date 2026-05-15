"""
CORRECTED Day 1 saliency check: run DINOv2 attention on the ACTUAL agent
training input — i.e. the 128x128 base_camera view (top inset of the demo
frame), NOT the 512x512 render_camera (left big panel) which the agent never
sees.

Layout of demo frames (640x512, render_mode="all"):
  x=0..512, y=0..512    -> render_camera (scenic view, NOT seen by agent)
  x=512..640, y=0..128  -> base_camera 128x128  (this IS the agent input)
  x=512..640, y=128..256-> hand_camera 128x128  (Panda default; possibly used too)

Outputs to: analysis/ebm/saliency_maps_basecam/<task>/step{NNN}_overlay.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
OUT_DIR = REPO_ROOT / "analysis/ebm/saliency_maps_basecam"
MODEL_NAME = "facebook/dinov2-small"

# The two 128x128 inset views we care about
CROPS = {
    "base_cam":  (512, 0, 640, 128),
    "hand_cam":  (512, 128, 640, 256),
}

TOP_K = 8

FRAMES_TO_PROCESS = [
    ("RememberColor9-v0",          ["000", "005", "010", "020", "030", "040", "050"]),
    ("InterceptFast-v0",           ["000", "005", "010", "020", "030", "040", "050"]),
    ("RememberShapeAndColor3x2-v0", ["000", "005", "010", "020", "030", "040", "050"]),
]


def load_model(device):
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    m = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return proc, m


@torch.no_grad()
def cls_attention(model, pixel_values):
    out = model(pixel_values=pixel_values, output_attentions=True)
    attn = out.attentions[-1][0, :, 0, 1:].mean(dim=0)
    _, _, h, w = pixel_values.shape
    p = model.config.patch_size
    grid = (h // p, w // p)
    return attn.reshape(grid).float().cpu().numpy(), grid


def render_overlay(views, save_path, title):
    """views: list of (name, raw_pil, sal_2d, top_idx, grid)."""
    n = len(views)
    fig, ax = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        ax = ax.reshape(2, 1)
    for c, (name, raw, sal, top_idx, grid) in enumerate(views):
        H, W = np.array(raw).shape[:2]
        Hp, Wp = grid
        ph, pw = H // Hp, W // Wp
        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        sal_up = np.array(Image.fromarray((sal_norm*255).astype(np.uint8))
                          .resize((W, H), Image.BILINEAR)) / 255.0
        ax[0, c].imshow(raw); ax[0, c].set_title(f"{name} raw"); ax[0, c].axis("off")
        ax[1, c].imshow(raw); ax[1, c].imshow(sal_up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        for (i, j) in top_idx:
            rect = mpatches.Rectangle((j*pw, i*ph), pw, ph, linewidth=1.4,
                                      edgecolor="lime", facecolor="none")
            ax[1, c].add_patch(rect)
        ax[1, c].set_title(f"{name} attn + top-{TOP_K}"); ax[1, c].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, model = load_model(device)
    print(f"loaded {MODEL_NAME} on {device}, patch={model.config.patch_size}")

    summary = {"model": MODEL_NAME, "frames": []}

    for task, steps in FRAMES_TO_PROCESS:
        fdir = REPO_ROOT / f"{task}_frames"
        if not fdir.is_dir():
            continue
        for step in steps:
            src = fdir / f"render_env0_step{step}.png"
            if not src.exists():
                continue
            full = Image.open(src).convert("RGB")
            views = []
            entry = {"task": task, "step": int(step), "views": {}}
            for name, box in CROPS.items():
                raw = full.crop(box)                # 128x128
                inp = proc(images=raw, return_tensors="pt").to(device)
                sal, grid = cls_attention(model, inp.pixel_values)
                flat = sal.ravel()
                top_idx_flat = np.argsort(flat)[-TOP_K:][::-1]
                top_idx = [(int(i // grid[1]), int(i % grid[1])) for i in top_idx_flat]
                views.append((name, raw, sal, top_idx, grid))
                entry["views"][name] = {
                    "max": float(sal.max()),
                    "mean": float(sal.mean()),
                    "max_over_mean": float(sal.max() / sal.mean()),
                    "top_k_idx": top_idx,
                }
            out = OUT_DIR / task / f"step{step}_overlay.png"
            render_overlay(views, out, f"{task} step{step}")
            summary["frames"].append(entry)
            base_mm = entry["views"]["base_cam"]["max_over_mean"]
            hand_mm = entry["views"]["hand_cam"]["max_over_mean"]
            print(f"  {task} step{step}: base max/mean={base_mm:.2f}  hand max/mean={hand_mm:.2f}")

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
