import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "main"))

import numpy as np
from env import GateRacingEnv

env = GateRacingEnv(track="circle_small", academy_level=0)
print(f"gate_size={env.gate_size}")

passes = 0
for ep in range(20):
    obs, info = env.reset()
    g0 = env.gate_positions[0]
    n0 = env.gate_normals[0]
    pos0, _ = env._get_drone_pos_orn()
    rel0 = np.array(pos0) - g0
    print(f"ep{ep}: spawn={np.array(pos0).round(2)} g0={g0.round(2)} n0={n0.round(2)} dot0={np.dot(rel0,n0):+.3f}")
    last_info = info
    for _ in range(120):
        obs, r, term, trunc, last_info = env.step(np.zeros(4, dtype=np.float32))
        if last_info["gates_passed"] > 0 or term or trunc:
            break
    pos, _ = env._get_drone_pos_orn()
    print(f"   end pos={np.array(pos).round(2)} gates={last_info['gates_passed']} term={term} trunc={trunc}")
    if last_info["gates_passed"] > 0:
        passes += 1

print(f"\n{passes}/20 episodes passed gate by free-fall")
