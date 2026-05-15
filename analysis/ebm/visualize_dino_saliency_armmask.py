"""
Path B diagnostic: re-evaluate DINOv2 CLS->patch attention AFTER masking out
the (white) robot arm region.

Hypothesis under test (B): if the arm weren't dominating attention, would
DINOv2 highlight task-relevant objects (purple cube, ball, target ring,
candidate blocks)?

Method:
  1. HSV-based pixel mask for arm: S < S_thr AND V > V_thr  (whitish/light-grey)
  2. Aggregate to patch level: patch = arm if >40% pixels arm-colored
  3. Zero saliency at arm patches; renormalize for visualization
  4. Find top-K=8 patches in the masked map; render with bright bounding boxes

Outputs:
  analysis/ebm/saliency_maps_armmasked/{task}/step{NNN}_overlay.png
  analysis/ebm/saliency_maps_armmasked/_summary.json
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
OUT_DIR = REPO_ROOT / "analysis/ebm/saliency_maps_armmasked"
MODEL_NAME = "facebook/dinov2-small"
CROP_BASE = (0, 0, 480, 512)

S_THR, V_THR = 0.18, 0.55           # HSV thresholds for "whitish arm-like"
PATCH_ARM_FRAC = 0.40                # patch labeled arm if >40% pixels are arm-like
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


def rgb_to_hsv_arr(rgb):
    """rgb in [0,1] HxWx3 -> hsv HxWx3 in [0,1]."""
    rgb = rgb.astype(np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    delta = mx - mn
    h = np.zeros_like(mx)
    s = np.where(mx > 0, delta / (mx + 1e-8), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def arm_pixel_mask(rgb_pil_resized: np.ndarray) -> np.ndarray:
    """
    Returns boolean HxW mask, True where pixel looks like the white robot arm.
    rgb in [0,1].
    """
    hsv = rgb_to_hsv_arr(rgb_pil_resized)
    s, v = hsv[..., 1], hsv[..., 2]
    return (s < S_THR) & (v > V_THR)


def patch_arm_mask_from_pixel(arm_pix: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Hp x Wp boolean mask: True if patch is mostly arm-colored."""
    H, W = arm_pix.shape
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp
    out = np.zeros(grid, dtype=bool)
    for i in range(Hp):
        for j in range(Wp):
            block = arm_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            out[i, j] = block.mean() > PATCH_ARM_FRAC
    return out


def render_overlay(raw_pil, sal_raw, sal_masked, arm_patch, top_patch_idx,
                   grid, save_path, title):
    H, W = np.array(raw_pil).shape[:2]
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp

    def upsample(m):
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return np.array(Image.fromarray((m*255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0

    sal_raw_u = upsample(sal_raw)
    sal_masked_u = upsample(sal_masked)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(raw_pil); ax[0].set_title("raw"); ax[0].axis("off")
    ax[1].imshow(raw_pil); ax[1].imshow(sal_raw_u, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax[1].set_title("DINOv2 attn (no mask)"); ax[1].axis("off")
    ax[2].imshow(raw_pil); ax[2].imshow(arm_patch.repeat(ph, 0).repeat(pw, 1).astype(float),
                                        cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    ax[2].set_title("arm-patch mask"); ax[2].axis("off")
    ax[3].imshow(raw_pil); ax[3].imshow(sal_masked_u, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    # draw top-K boxes
    for (i, j) in top_patch_idx:
        rect = mpatches.Rectangle((j*pw, i*ph), pw, ph, linewidth=1.4,
                                  edgecolor="lime", facecolor="none")
        ax[3].add_patch(rect)
    ax[3].set_title(f"masked attn + top-{TOP_K} (lime)"); ax[3].axis("off")
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

            raw = Image.open(src).convert("RGB").crop(CROP_BASE)
            inp = proc(images=raw, return_tensors="pt").to(device)
            sal_raw, grid = cls_attention(model, inp.pixel_values)

            # build arm mask in the *processed* image space (224x224 default)
            proc_img = inp.pixel_values[0].permute(1, 2, 0).cpu().numpy()
            # de-normalize back to [0,1] roughly
            mean = np.array(proc.image_mean); std = np.array(proc.image_std)
            proc_img = np.clip(proc_img * std + mean, 0, 1)

            arm_pix = arm_pixel_mask(proc_img)
            arm_patch = patch_arm_mask_from_pixel(arm_pix, grid)
            arm_frac = arm_patch.mean()

            sal_masked = sal_raw.copy()
            sal_masked[arm_patch] = 0.0

            # top-K (excluding arm)
            flat = sal_masked.ravel()
            top_idx_flat = np.argsort(flat)[-TOP_K:][::-1]
            top_idx = [(int(i // grid[1]), int(i % grid[1])) for i in top_idx_flat]

            out = OUT_DIR / task / f"step{step}_overlay.png"
            render_overlay(raw, sal_raw, sal_masked, arm_patch, top_idx,
                           grid, out, f"{task} step{step}")

            summary["frames"].append({
                "task": task, "step": int(step),
                "arm_patch_fraction": float(arm_frac),
                "raw_max": float(sal_raw.max()),
                "raw_mean": float(sal_raw.mean()),
                "masked_max": float(sal_masked.max()),
                "masked_mean": float(sal_masked[~arm_patch].mean()) if (~arm_patch).any() else 0.0,
                "masked_max_over_mean": float(
                    sal_masked.max() / (sal_masked[~arm_patch].mean() + 1e-8)
                    if (~arm_patch).any() else 0.0),
                "top_k_idx": top_idx,
            })
            print(f"  {task} step{step}: arm_frac={arm_frac:.2f} "
                  f"masked_max/mean={summary['frames'][-1]['masked_max_over_mean']:.2f}")

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
