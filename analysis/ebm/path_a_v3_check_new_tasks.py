"""
Quick saliency-quality check: run path_a_head_v3.pt on demo frames of the
new tasks (InterceptMedium, RememberShape*, RememberShapeAndColor*) to
verify it generalizes beyond RC9 training data.
"""
from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")
from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from baselines.ppo.modules.saliency_head import load_saliency_head

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
HEAD = ROOT / "analysis/ebm/path_a_head_v3.pt"
OUT  = ROOT / "analysis/ebm/path_a_v3_new_task_vis"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")
TOP_K = 8
CROPS = {"base_cam": (512, 0, 640, 128), "hand_cam": (512, 128, 640, 256)}

TASKS = [
    ("InterceptMedium-v0",          ["000", "010", "020", "030", "050", "080"]),
    # No frames yet for: RememberShape5/9, ShellGameTouch
    # but visualizing what we have already establishes the pattern
    ("RememberShapeAndColor3x2-v0", ["000", "005", "010", "020", "030", "040"]),
    ("RememberColor9-v0",           ["000", "005", "030", "050"]),  # control
]


@torch.no_grad()
def main():
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    head, xy_concat = load_saliency_head(HEAD, device=DEVICE)
    Hp, Wp = vit.grid; Nv = Hp * Wp

    summary = []
    for task, steps in TASKS:
        fdir = ROOT / f"{task}_frames"
        if not fdir.is_dir():
            print(f"[skip] {fdir}")
            continue
        for step in steps:
            src = fdir / f"render_env0_step{step}.png"
            if not src.exists():
                continue
            full = Image.open(src).convert("RGB")
            base_pil = full.crop(CROPS["base_cam"])
            hand_pil = full.crop(CROPS["hand_cam"])
            base = np.array(base_pil); hand = np.array(hand_pil)
            rgb6 = torch.from_numpy(np.concatenate([base[None], hand[None]], axis=-1)).to(DEVICE)
            tok_b, tok_h, _, _ = vit(rgb6)
            cat = torch.cat([tok_b, tok_h], dim=1)
            sal = head(cat, xy_concat)[0].cpu().numpy()
            base_sal = sal[:Nv].reshape(Hp, Wp)
            hand_sal = sal[Nv:].reshape(Hp, Wp)
            order = np.argsort(-sal)
            joint_top = []
            for ix in order[:TOP_K]:
                if ix < Nv:
                    joint_top.append(("base_cam", int(ix // Wp), int(ix % Wp)))
                else:
                    j = ix - Nv
                    joint_top.append(("hand_cam", int(j // Wp), int(j % Wp)))

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].imshow(base_pil); ax[0, 0].set_title("base raw"); ax[0, 0].axis("off")
            ax[0, 1].imshow(hand_pil); ax[0, 1].set_title("hand raw"); ax[0, 1].axis("off")
            for cidx, (raw, smap) in enumerate([(base_pil, base_sal), (hand_pil, hand_sal)]):
                Wpx, Hpx = raw.size
                ph, pw = Hpx // Hp, Wpx // Wp
                snorm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)
                up = np.array(Image.fromarray((snorm*255).astype(np.uint8)).resize(raw.size, Image.BILINEAR)) / 255.0
                ax[1, cidx].imshow(raw); ax[1, cidx].imshow(up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
                cam = "base_cam" if cidx == 0 else "hand_cam"
                for c, i, j in joint_top:
                    if c == cam:
                        rect = mpatches.Rectangle((j*pw, i*ph), pw, ph,
                                                  linewidth=1.6, edgecolor="lime", facecolor="none")
                        ax[1, cidx].add_patch(rect)
                ax[1, cidx].set_title(f"{cam} HEAD-v3 + top-{TOP_K}"); ax[1, cidx].axis("off")
            fig.suptitle(f"{task} step{step}  max sal: {sal.max():.2f}  mean: {sal.mean():.2f}",
                         fontsize=10)
            fig.tight_layout()
            (OUT / task).mkdir(exist_ok=True)
            fig.savefig(OUT / task / f"step{step}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            summary.append((task, step, float(sal.max()), float(sal.mean()), joint_top))
            print(f"  {task} step{step}: max={sal.max():.2f} mean={sal.mean():.2f}")
    print(f"\nvis -> {OUT}")


if __name__ == "__main__":
    main()
