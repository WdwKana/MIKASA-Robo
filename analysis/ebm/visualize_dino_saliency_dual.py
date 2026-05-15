"""
Dual-camera DINOv2 saliency verification (Day 1 redo with correct cameras).

For each frame, run DINOv2 on BOTH base_cam and hand_cam (as separate forward
passes), then compute a JOINT top-K=8 across the union of patches from both
views (this is what EBM-Robo's buffer write decision will operate on).

Outputs:
  analysis/ebm/saliency_maps_dual/{task}/step{NNN}_overlay.png
  analysis/ebm/saliency_maps_dual/_summary.json

For each frame, also report:
  - share of top-K coming from hand_cam vs base_cam (expect hand_cam-heavy)
  - per-camera saliency max/mean (sanity)
  - whether the empty-table delay frames have lower max than info frames
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
OUT_DIR   = REPO_ROOT / "analysis/ebm/saliency_maps_dual"
MODEL_NAME = "facebook/dinov2-small"

CROPS = {
    "base_cam":  (512, 0,   640, 128),
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


def render_dual_overlay(base_pil, base_sal, base_grid,
                        hand_pil, hand_sal, hand_grid,
                        joint_top, save_path, title):
    """joint_top: list of (cam, i, j) for top-K across both views."""

    def upsample(raw, sal):
        H, W = np.array(raw).shape[:2]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return np.array(Image.fromarray((sal_n * 255).astype(np.uint8))
                        .resize((W, H), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    bH, bW = np.array(base_pil).shape[:2]
    bph, bpw = bH // base_grid[0], bW // base_grid[1]
    hH, hW = np.array(hand_pil).shape[:2]
    hph, hpw = hH // hand_grid[0], hW // hand_grid[1]

    ax[0, 0].imshow(base_pil); ax[0, 0].set_title("base_cam raw"); ax[0, 0].axis("off")
    ax[0, 1].imshow(hand_pil); ax[0, 1].set_title("hand_cam raw"); ax[0, 1].axis("off")

    ax[1, 0].imshow(base_pil)
    ax[1, 0].imshow(upsample(base_pil, base_sal), cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax[1, 0].set_title(f"base_cam attn  (joint top-{TOP_K} marked lime)"); ax[1, 0].axis("off")

    ax[1, 1].imshow(hand_pil)
    ax[1, 1].imshow(upsample(hand_pil, hand_sal), cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax[1, 1].set_title(f"hand_cam attn  (joint top-{TOP_K} marked lime)"); ax[1, 1].axis("off")

    for cam, i, j in joint_top:
        if cam == "base_cam":
            rect = mpatches.Rectangle((j*bpw, i*bph), bpw, bph,
                                      linewidth=1.6, edgecolor="lime", facecolor="none")
            ax[1, 0].add_patch(rect)
        else:
            rect = mpatches.Rectangle((j*hpw, i*hph), hpw, hph,
                                      linewidth=1.6, edgecolor="lime", facecolor="none")
            ax[1, 1].add_patch(rect)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


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
            base = full.crop(CROPS["base_cam"])
            hand = full.crop(CROPS["hand_cam"])

            base_inp = proc(images=base, return_tensors="pt").to(device)
            hand_inp = proc(images=hand, return_tensors="pt").to(device)
            base_sal, base_grid = cls_attention(model, base_inp.pixel_values)
            hand_sal, hand_grid = cls_attention(model, hand_inp.pixel_values)

            # JOINT top-K over union of patches from both views.
            base_flat = [("base_cam", int(i // base_grid[1]), int(i % base_grid[1]),
                          float(base_sal.ravel()[i])) for i in range(base_sal.size)]
            hand_flat = [("hand_cam", int(i // hand_grid[1]), int(i % hand_grid[1]),
                          float(hand_sal.ravel()[i])) for i in range(hand_sal.size)]
            all_flat = base_flat + hand_flat
            all_flat.sort(key=lambda x: -x[3])
            joint_top = [(c, i, j) for c, i, j, _ in all_flat[:TOP_K]]

            n_hand = sum(1 for c, _, _ in joint_top if c == "hand_cam")
            n_base = TOP_K - n_hand

            out = OUT_DIR / task / f"step{step}_overlay.png"
            render_dual_overlay(base, base_sal, base_grid,
                                hand, hand_sal, hand_grid,
                                joint_top, out,
                                f"{task} step{step}  (joint top-{TOP_K}: hand={n_hand} base={n_base})")

            summary["frames"].append({
                "task": task, "step": int(step),
                "base_max": float(base_sal.max()), "base_mean": float(base_sal.mean()),
                "base_max_over_mean": float(base_sal.max() / base_sal.mean()),
                "hand_max": float(hand_sal.max()), "hand_mean": float(hand_sal.mean()),
                "hand_max_over_mean": float(hand_sal.max() / hand_sal.mean()),
                "joint_top_n_hand": n_hand,
                "joint_top_n_base": n_base,
                "joint_top": joint_top,
            })
            print(f"  {task} step{step}:  hand_max/mean={hand_sal.max()/hand_sal.mean():5.2f}"
                  f"  base_max/mean={base_sal.max()/base_sal.mean():5.2f}"
                  f"   joint_top: hand={n_hand} base={n_base}")

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
