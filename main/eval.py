import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import torch
import numpy as np
from env import GateRacingEnv
from models import ActorCritic

CKPT = r"C:\clones\tinywhoop_bc_racing\checkpoints\impala_circle_small_1775453960\ckpt_320.pt"
TRACK = "easy_circle"
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NUM_EPISODES = 50

ckpt_path = Path(CKPT)
if "LATEST" in str(ckpt_path):
    ckpt_dirs = sorted(Path("checkpoints").glob("impala_*"), key=lambda p: p.stat().st_mtime)
    if not ckpt_dirs:
        ckpt_dirs = sorted(Path("checkpoints").glob("ppo_*"), key=lambda p: p.stat().st_mtime)
    if ckpt_dirs:
        best = ckpt_dirs[-1] / "best.pt"
        if best.exists():
            ckpt_path = best
        else:
            ckpts = sorted(ckpt_dirs[-1].glob("ckpt_*.pt"))
            if ckpts:
                ckpt_path = ckpts[-1]
    print(f"Auto-detected: {ckpt_path}")

ac = ActorCritic()
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
ac.load_state_dict(ckpt["model_state_dict"])
ac.eval()
print(f"Loaded: {ckpt_path}")
print(f"  Step: {ckpt.get('global_step', '?')} | Updates: {ckpt.get('num_updates', '?')} | C: {ckpt.get('curriculum_level', '?')}")

env = GateRacingEnv(track=TRACK, render_mode="human", academy_level=LEVEL)
print(f"  Level: {LEVEL}")

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        obs_t = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            action, _, _ = ac.get_action(obs_t, deterministic=True)
            action = action.squeeze(0).numpy()
        obs, reward, term, trunc, info = env.step(action)
        total_r += reward
        steps += 1
        done = term or trunc
        last_mask = obs["masks"][-1]
        green = last_mask[0]
        red = last_mask[1]
        mask_rgb = np.stack([np.zeros_like(green), green, red], axis=-1)
        mask_img = (mask_rgb * 255).astype(np.uint8)
        cv2.imshow("Gate Mask", cv2.resize(mask_img, (320, 240), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)
        time.sleep(1.0 / 50.0)
    print(f"Ep {ep+1}/{NUM_EPISODES}: R={total_r:+.1f} G={info['gates_passed']} Steps={steps}")

env.close()
