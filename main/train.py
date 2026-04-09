import math
import sys
import time
from collections import deque
from pathlib import Path

_MAIN_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _MAIN_DIR)

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from env import GateRacingEnv
from models import ActorCritic
from metrics import MetricsLogger

ACADEMY = {
    "levels": [
        {"name": "easy_circle",        "unlock_at": 0, "threshold": 3.0,  "metric": "avg_gates"},
        {"name": "fast_ring",          "unlock_at": 1, "threshold": 4.0,  "metric": "avg_gates"},
        {"name": "technical",          "unlock_at": 2, "threshold": 5.0,  "metric": "avg_gates"},
        {"name": "championship",       "unlock_at": 3, "threshold": 6.0,  "metric": "avg_gates"},
        {"name": "random",             "unlock_at": 4, "threshold": 7.0,  "metric": "avg_gates"},
        {"name": "random_dr",          "unlock_at": 5, "threshold": 8.0,  "metric": "avg_gates"},
    ],
    "focus_newest": 0.5,
}

CONFIG = {
    "track": "circle_small",
    "lr": 0.001,
    "gamma": 0.99,
    "rho_bar": 1.0,
    "c_bar": 1.0,
    "value_coef": 0.5,
    "max_grad_norm": 10.0,
    "num_workers": 16,
    "trajectory_length": 256, 
    "learner_batch": 128,
    "total_timesteps": 500_000_000,
    "sil_coef": 0.25,
    "golden_capacity": 256,
    "golden_max_uses": 4,
    "sil_samples_per_update": 8,
    "aux_coef": 1.0,
    "target_entropy": 6.3,
    "alpha_lr": 3e-3,
    "alpha_up_init": 0.3,
    "alpha_down_init": 0.05,
    "alpha_up_clamp": (0.01, 0.5),
    "alpha_down_clamp": (0.01, 0.5),
    "alpha_blend_width": 0.5,
    "reset_popart_on_load": True,
    "value_warmup_policy_coef": 0.1,
    "value_warmup_v_loss_threshold": 1.0,
    "resume": r"C:\clones\tinywhoop_bc_racing\lake\impala_circle_small_1775679280\checkpoints\ckpt_2480.pt",
}

@ray.remote
class EnvWorker:
    def __init__(self, main_dir, track, academy_level, trajectory_length):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import sys as _sys
        if main_dir not in _sys.path:
            _sys.path.insert(0, main_dir)
        import torch as _torch
        from env import GateRacingEnv
        from models import ActorCritic
        self._torch = _torch
        self.env = GateRacingEnv(track=track, academy_level=academy_level)
        self.policy = ActorCritic()
        self.policy.eval()
        self.trajectory_length = trajectory_length
        self.obs, _info = self.env.reset()
        self._current_track_gates = _info.get("track_gates", [])
        self.ep_reward = 0.0

    def set_weights(self, state_dict):
        self.policy.load_state_dict(state_dict)

    def set_academy_level(self, level):
        self.env.academy_level = level

    def set_dr_scale(self, dr_scale):
        self.env.dr_scale = dr_scale

    def set_alive_disabled(self, val):
        self.env.set_alive_disabled(val)


    def collect_trajectory(self):
        th = self._torch
        T = self.trajectory_length
        masks = np.zeros((T, 4, 2, 60, 80), dtype=np.float32)
        imu = np.zeros((T, 4, 10), dtype=np.float32)
        cv2_feats = np.zeros((T, 10), dtype=np.float32)
        nav_feats = np.zeros((T, 6), dtype=np.float32)
        act_hist = np.zeros((T, 4, 4), dtype=np.float32)
        state = np.zeros((T, 38), dtype=np.float32)
        actions = np.zeros((T, 4), dtype=np.float32)
        bin_indices = np.zeros((T, 4), dtype=np.int64)
        blp = np.zeros(T, dtype=np.float32)
        rewards = np.zeros(T, dtype=np.float32)
        dones = np.zeros(T, dtype=np.float32)
        gate_idx = np.zeros(T, dtype=np.int32)
        ep_rewards = []
        ep_gates = []
        ep_dists = []
        ep_diag = []
        for t in range(T):
            masks[t] = self.obs["masks"]
            imu[t] = self.obs["imu"]
            cv2_feats[t] = self.obs["cv2"]
            nav_feats[t] = self.obs["nav"]
            act_hist[t] = self.obs["actions"]
            state[t] = self.obs["state"]
            with th.no_grad():
                obs_t = {
                    "masks": th.as_tensor(self.obs["masks"], dtype=th.float32).unsqueeze(0),
                    "imu": th.as_tensor(self.obs["imu"], dtype=th.float32).unsqueeze(0),
                    "cv2": th.as_tensor(self.obs["cv2"], dtype=th.float32).unsqueeze(0),
                    "nav": th.as_tensor(self.obs["nav"], dtype=th.float32).unsqueeze(0),
                    "actions": th.as_tensor(self.obs["actions"], dtype=th.float32).unsqueeze(0),
                    "state": th.as_tensor(self.obs["state"], dtype=th.float32).unsqueeze(0),
                }
                action, lp, idx = self.policy.get_action(obs_t)
            actions[t] = action.squeeze(0).numpy()
            bin_indices[t] = idx.squeeze(0).numpy()
            blp[t] = lp.item()
            next_obs, reward, term, trunc, info = self.env.step(actions[t])
            rewards[t] = reward
            gate_idx[t] = info.get("gate_idx", 0)
            self.ep_reward += reward
            done = term or trunc
            dones[t] = float(done)
            if done:
                ep_rewards.append(self.ep_reward)
                ep_gates.append(info.get("gates_passed", 0))
                ep_dists.append(info.get("distance_to_target", 0.0))
                ep_diag.append({
                    "reason": info.get("term_reason", ""),
                    "final_pos": info.get("final_pos", [0, 0, 0]),
                    "final_euler": info.get("final_euler", [0, 0, 0]),
                    "steps": info.get("steps", 0),
                    "dist": float(info.get("distance_to_target", 0.0)),
                    "gates": int(info.get("gates_passed", 0)),
                    "reward": float(self.ep_reward),
                    "track_gates": self._current_track_gates,
                })
                self.ep_reward = 0.0
                next_obs, _info = self.env.reset()
                self._current_track_gates = _info.get("track_gates", [])
            self.obs = next_obs
        act_diff = np.diff(actions, axis=0)
        act_jerk = float(np.sqrt((act_diff ** 2).mean())) if len(act_diff) else 0.0
        gyro = imu[:, -1, 0:3]
        gyro_diff = np.diff(gyro, axis=0)
        gyro_jerk = float(np.sqrt((gyro_diff ** 2).mean())) if len(gyro_diff) else 0.0
        return {
            "masks": masks, "imu": imu, "cv2": cv2_feats, "nav": nav_feats,
            "actions_hist": act_hist, "state": state,
            "actions": actions, "bin_indices": bin_indices,
            "behavior_log_probs": blp,
            "act_jerk": act_jerk, "gyro_jerk": gyro_jerk,
            "rewards": rewards, "dones": dones,
            "gate_idx": gate_idx,
            "ep_rewards": ep_rewards, "ep_gates": ep_gates,
            "ep_dists": ep_dists,
            "ep_diag": ep_diag,
            "academy_level": self.env.academy_level,
        }

    def close(self):
        self.env.close()

class AcademyManager:
    def __init__(self):
        self.highest_unlocked = 0
        self.level_rewards = {i: deque(maxlen=100) for i in range(6)}
        self.level_success = {i: deque(maxlen=100) for i in range(6)}

    def record(self, level, ep_reward, gates_passed):
        self.level_rewards[level].append(ep_reward)
        self.level_success[level].append(gates_passed)

    def check_advance(self):
        old = self.highest_unlocked
        for next_lvl in range(old + 1, len(ACADEMY["levels"])):
            prereq_lvl = ACADEMY["levels"][next_lvl]["unlock_at"] - 1
            stats = self.level_success[prereq_lvl]
            if len(stats) < 50:
                break
            val = np.mean(stats)
            thr = ACADEMY["levels"][prereq_lvl]["threshold"]
            if val < thr:
                break
            self.highest_unlocked = next_lvl
        return self.highest_unlocked > old

    def sample_level(self):
        h = self.highest_unlocked
        unlocked = list(range(h + 1))
        if len(unlocked) == 1:
            return unlocked[0]
        focus = ACADEMY["focus_newest"]
        rest = (1.0 - focus) / max(len(unlocked) - 1, 1)
        probs = np.array([focus if l == h else rest for l in unlocked])
        return int(np.random.choice(unlocked, p=probs))

    def status_str(self):
        parts = []
        for i in range(self.highest_unlocked + 1):
            stats = self.level_success[i]
            val = np.mean(stats) if stats else 0.0
            mx = int(max(stats)) if stats else 0
            n = len(stats)
            thr = ACADEMY["levels"][i]["threshold"]
            met = val >= thr
            ok = "*" if n >= 50 and met else ""
            parts.append(f"L{i}:{val:.1f}{ok}({mx})/{thr:.0f}")
        return " ".join(parts)

class GoldenMemory:
    def __init__(self, capacity, max_uses):
        self.capacity = capacity
        self.max_uses = max_uses
        self.buffer = [None] * capacity
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.gates = np.zeros(capacity, dtype=np.float32)
        self.levels = np.zeros(capacity, dtype=np.int32)
        self.uses = np.zeros(capacity, dtype=np.int32)
        self.dr_scales = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def _score(self, n=None):
        n = n or self.size
        g = self.gates[:n]
        lvl = (1 + self.levels[:n]).astype(np.float32)
        dr = (1 + self.dr_scales[:n]).astype(np.float32)
        gate_score = (g ** 2) * lvl * dr
        r_shift = np.maximum(self.returns[:n] + 110.0, 1.0)
        return_score = r_shift * 0.01 * lvl * dr
        return np.where(g > 0, gate_score, return_score)

    @staticmethod
    def _compute_score(gates, ret, level, dr_scale):
        lvl = 1 + level
        dr = 1 + dr_scale
        if gates > 0:
            return (gates ** 2) * lvl * dr
        r_shift = max(ret + 110.0, 1.0)
        return r_shift * 0.01 * lvl * dr

    def _evict_idx(self):
        scores = self._score() + self.returns[:self.size] * 1e-6
        return int(np.argmin(scores))

    def add(self, traj, ret, gates, level, dr_scale=0.0):
        if ret <= 0 and gates == 0:
            return False
        new_score = self._compute_score(gates, ret, level, dr_scale) + ret * 1e-6
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = self._evict_idx()
            evict_score = self._score()[idx] + self.returns[idx] * 1e-6
            if new_score <= evict_score:
                return False
        self.buffer[idx] = traj
        self.returns[idx] = ret
        self.gates[idx] = gates
        self.levels[idx] = level
        self.uses[idx] = 0
        self.dr_scales[idx] = dr_scale
        return True

    def sample(self, n):
        if self.size == 0:
            return []
        valid_mask = self.uses[:self.size] < self.max_uses
        high_gate_mask = self.gates[:self.size] >= 8
        valid_mask = valid_mask | (high_gate_mask & (self.uses[:self.size] < self.max_uses * 4))
        valid_idxs = np.where(valid_mask)[0]
        if len(valid_idxs) == 0:
            return []
        n = min(n, len(valid_idxs))
        all_scores = self._score()
        weights = np.maximum(all_scores[valid_idxs], 0.1)
        probs = weights / weights.sum()
        chosen = np.random.choice(valid_idxs, size=n, replace=False, p=probs)
        self.uses[chosen] += 1
        return [self.buffer[i] for i in chosen]

    def purge_weak(self):
        if self.size == 0:
            return
        scores = self._score()
        threshold = np.percentile(scores, 25)
        keep_mask = scores > threshold
        keep_idxs = np.where(keep_mask)[0]
        new_size = len(keep_idxs)
        new_buffer = [self.buffer[i] for i in keep_idxs]
        self.returns[:new_size] = self.returns[keep_idxs]
        self.gates[:new_size] = self.gates[keep_idxs]
        self.levels[:new_size] = self.levels[keep_idxs]
        self.uses[:new_size] = self.uses[keep_idxs]
        self.dr_scales[:new_size] = self.dr_scales[keep_idxs]
        self.buffer = new_buffer + [None] * (self.capacity - new_size)
        self.size = new_size

    def stats(self):
        if self.size == 0:
            return 0, 0
        return self.size, int(self.gates[:self.size].max())

    def gate_fraction(self):
        if self.size == 0:
            return 0.0
        return float(np.count_nonzero(self.gates[:self.size] > 0)) / float(self.size)

def extract_episodes(traj):
    episodes = []
    ep_start = 0
    dones = traj["dones"]
    for t in range(len(dones)):
        if dones[t]:
            ep = {k: traj[k][ep_start:t + 1].copy() for k in ["masks", "imu", "cv2", "nav", "actions_hist", "state", "actions", "bin_indices", "rewards", "dones"]}
            episodes.append(ep)
            ep_start = t + 1
    return episodes

def compute_vtrace(behavior_log_probs, current_log_probs, rewards, values,
                   dones, bootstrap_values, gamma, rho_bar=1.0, c_bar=1.0):
    B, T = rewards.shape
    log_rho = current_log_probs - behavior_log_probs
    is_ratio = torch.exp(log_rho)
    rho = is_ratio.clamp(max=rho_bar)
    c = is_ratio.clamp(max=c_bar)
    values_ext = torch.cat([values, bootstrap_values.unsqueeze(1)], dim=1)
    vs = torch.zeros(B, T + 1, device=rewards.device)
    vs[:, T] = bootstrap_values
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        delta = rho[:, t] * (rewards[:, t] + gamma * not_done * values_ext[:, t + 1] - values_ext[:, t])
        vs[:, t] = values_ext[:, t] + delta + gamma * not_done * c[:, t] * (vs[:, t + 1] - values_ext[:, t + 1])
    vtrace_targets = vs[:, :T]
    next_vs = torch.cat([vtrace_targets[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    advantages = rho * (rewards + gamma * (1.0 - dones) * next_vs - values_ext[:, :T])
    return vtrace_targets.reshape(-1), advantages.reshape(-1), rho.mean().item()

def compute_sil_loss(agent, trajectories, device, gamma):
    if not trajectories:
        return torch.tensor(0.0, device=device), {}
    total_pi_loss = 0.0
    total_v_loss = 0.0
    total_adv = 0.0
    total_frac = 0.0
    for traj in trajectories:
        obs = {
            "masks": torch.as_tensor(traj["masks"], dtype=torch.float32, device=device),
            "imu": torch.as_tensor(traj["imu"], dtype=torch.float32, device=device),
            "cv2": torch.as_tensor(traj["cv2"], dtype=torch.float32, device=device),
            "nav": torch.as_tensor(traj["nav"], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(traj["actions_hist"], dtype=torch.float32, device=device),
            "state": torch.as_tensor(traj["state"], dtype=torch.float32, device=device),
        }
        indices = torch.as_tensor(traj["bin_indices"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(traj["rewards"], dtype=torch.float32, device=device)
        dones = torch.as_tensor(traj["dones"], dtype=torch.float32, device=device)
        log_probs, _, _, values, _, _ = agent.evaluate(obs, indices)
        values = values.squeeze(-1)
        T = len(rewards)
        mc_returns = torch.zeros(T, device=device)
        R = 0.0
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R * (1.0 - dones[t])
            mc_returns[t] = R
        advantages = mc_returns - values.detach()
        pos_adv = torch.clamp(advantages, min=0.0, max=10.0)
        if pos_adv.sum() > 0:
            pos_adv = pos_adv / (pos_adv.mean() + 1e-8)
        total_pi_loss += -(log_probs * pos_adv).mean()
        total_v_loss += 0.5 * pos_adv.pow(2).mean()
        total_adv += pos_adv.mean().item()
        total_frac += (pos_adv > 0).float().mean().item()
    n = len(trajectories)
    loss = (total_pi_loss + total_v_loss) / n
    stats = {
        "sil_loss": loss.item(),
        "sil_pi": (total_pi_loss / n).item(),
        "sil_v": (total_v_loss / n).item(),
        "sil_adv": total_adv / n,
        "sil_frac": total_frac / n,
    }
    return loss, stats

def compute_bootstrap_batched(agent, batch, B, T, device):
    dones_bt = np.stack([tr["dones"] for tr in batch])
    flipped = dones_bt[:, ::-1]
    first_nondone = np.argmax(flipped == 0.0, axis=1)
    has_nondone = np.any(dones_bt == 0.0, axis=1)
    boot_idx = T - 1 - first_nondone
    boot_idx[~has_nondone] = -1
    valid_bi = np.where(has_nondone)[0]
    bootstrap_vals = torch.zeros(B, device=device)
    if len(valid_bi) > 0:
        boot_obs = {}
        for key, tkey in [("masks", "masks"), ("imu", "imu"), ("cv2", "cv2"),
                          ("nav", "nav"), ("actions", "actions_hist"), ("state", "state")]:
            boot_obs[key] = torch.as_tensor(
                np.stack([batch[i][tkey][boot_idx[i]] for i in valid_bi]),
                dtype=torch.float32, device=device,
            )
        with torch.no_grad():
            bv = agent.get_value(boot_obs).squeeze(-1).float()
        bootstrap_vals[torch.as_tensor(valid_bi, dtype=torch.long, device=device)] = bv
    return bootstrap_vals

@torch.no_grad()
def evaluate_policy(agent, track, num_episodes=20, device="cpu", academy_level=0):
    eval_env = GateRacingEnv(track=track, academy_level=academy_level)
    total_reward = 0.0
    total_gates = 0
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            action, _, _ = agent.get_action(obs_t, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
        total_gates += info.get("gates_passed", 0)
    eval_env.close()
    return total_reward / num_episodes, total_gates / num_episodes

def fmt_steps(n):
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)

def fmt_time(seconds):
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    if seconds >= 60:
        return f"{seconds/60:.0f}m"
    return f"{seconds:.0f}s"

def train(cfg):
    from track_gen import generate_all_tracks
    generate_all_tracks(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray.init(ignore_reinit_error=True)
    N = cfg["num_workers"]
    T = cfg["trajectory_length"]
    B = cfg["learner_batch"]
    print(f"Device: {device} | Workers: {N} | Batch: {B} | Track: {cfg['track']} | Academy IMPALA+V-Trace+SIL")
    run_id = f"impala_{cfg['track']}_{int(time.time())}"
    logger = MetricsLogger(run_id)
    logger.build_tracks()
    ckpt_dir = logger.ckpt_dir
    agent = ActorCritic().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"])
    alpha_up = torch.tensor(cfg["alpha_up_init"], device=device, requires_grad=True)
    alpha_down = torch.tensor(cfg["alpha_down_init"], device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([alpha_up, alpha_down], lr=cfg["alpha_lr"])
    scaler = GradScaler()
    golden = GoldenMemory(capacity=cfg["golden_capacity"], max_uses=cfg["golden_max_uses"])
    academy = AcademyManager()
    dr_scale = 0.0
    dr_gates = deque(maxlen=200)
    global_step = 0
    num_updates = 0
    alive_disabled = False
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        saved = ckpt["model_state_dict"]
        current = agent.state_dict()
        compatible = {k: v for k, v in saved.items() if k in current and current[k].shape == v.shape}
        agent.load_state_dict(compatible, strict=False)
        print(f"  Loaded {len(compatible)}/{len(current)} params, skipped {len(saved)-len(compatible)} incompatible")
        academy.highest_unlocked = ckpt.get("academy_highest", 0)
        global_step = ckpt.get("global_step", 0)
        num_updates = ckpt.get("num_updates", 0)
        dr_scale = ckpt.get("dr_scale", 0.0)
        alive_disabled = ckpt.get("alive_disabled", False)
        if "alpha_up" in ckpt:
            alpha_up.data.copy_(torch.tensor(ckpt["alpha_up"], device=device))
        if "alpha_down" in ckpt:
            alpha_down.data.copy_(torch.tensor(ckpt["alpha_down"], device=device))
        if cfg.get("reset_popart_on_load", False):
            with torch.no_grad():
                agent.critic.popart.mu.zero_()
                agent.critic.popart.sigma.fill_(1.0)
                nn.init.kaiming_uniform_(agent.critic.popart.weight, a=math.sqrt(5))
                bound = 1.0 / math.sqrt(agent.critic.popart.weight.shape[1])
                nn.init.uniform_(agent.critic.popart.bias, -bound, bound)
            print("PopArt reset: mu=0, sigma=1, head re-initialized")
    warmup_active = bool(cfg.get("reset_popart_on_load", False)) and bool(cfg.get("resume"))
    if warmup_active:
        print(f"Value-Warmup active: policy_coef={cfg['value_warmup_policy_coef']} until v_loss<{cfg['value_warmup_v_loss_threshold']}")
        print(f"Resumed from {cfg['resume']} | Step: {global_step} | Updates: {num_updates} | L:{academy.highest_unlocked} | DR:{dr_scale:.2f}")
        academy.highest_unlocked = 0
        academy.level_success = {i: deque(maxlen=200) for i in range(len(ACADEMY["levels"]))}
        print(f"Academy reset to L:0 (re-earn all levels)")
    init_level = academy.sample_level()
    workers = [
        EnvWorker.remote(_MAIN_DIR, cfg["track"], init_level, T)
        for _ in range(N)
    ]
    weights = {k: v.cpu() for k, v in agent.state_dict().items()}
    ray.get([w.set_weights.remote(weights) for w in workers])
    pending = {}
    worker_levels = {}
    for w in workers:
        lvl = academy.sample_level()
        w.set_academy_level.remote(lvl)
        w.set_dr_scale.remote(dr_scale if academy.highest_unlocked >= 5 else 0.0)
        fut = w.collect_trajectory.remote()
        pending[fut] = w
        worker_levels[fut] = lvl
    recent_rewards = deque(maxlen=200)
    recent_gates = deque(maxlen=200)
    best_avg_gates = -1.0
    train_start = time.time()
    sil_stats = {}
    batch = []
    batch_levels = []
    if alive_disabled:
        ray.get([w.set_alive_disabled.remote(True) for w in workers])
    while global_step < cfg["total_timesteps"]:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        future = ready[0]
        worker = pending.pop(future)
        traj_level = worker_levels.pop(future)
        traj = ray.get(future)
        for k in ["masks", "imu", "cv2", "nav", "actions_hist", "state", "actions", "bin_indices", "behavior_log_probs", "rewards", "dones"]:
            if isinstance(traj[k], np.ndarray):
                traj[k] = traj[k].copy()
        weights = {k: v.cpu() for k, v in agent.state_dict().items()}
        worker.set_weights.remote(weights)
        new_lvl = academy.sample_level()
        worker.set_academy_level.remote(new_lvl)
        worker.set_dr_scale.remote(dr_scale if academy.highest_unlocked >= 5 else 0.0)
        new_fut = worker.collect_trajectory.remote()
        pending[new_fut] = worker
        worker_levels[new_fut] = new_lvl
        global_step += T
        for ep_r, ep_g in zip(traj["ep_rewards"], traj["ep_gates"]):
            recent_rewards.append(ep_r)
            recent_gates.append(ep_g)
            academy.record(traj_level, ep_r, ep_g)
            if traj_level == 5:
                dr_gates.append(ep_g)
        for d in traj.get("ep_diag", []):
            fp = d["final_pos"]
            fe = d["final_euler"]
            ep_id = logger.log_episode(
                update=num_updates + 1,
                level=int(traj_level),
                reward=float(d["reward"]),
                gates=int(d["gates"]),
                steps=int(d["steps"]),
                dist=float(d["dist"]),
                reason=d["reason"],
                fx=float(fp[0]), fy=float(fp[1]), fz=float(fp[2]),
                roll=float(fe[0]), pitch=float(fe[1]), yaw=float(fe[2]),
                dr_scale=dr_scale,
            )
            tg = d.get("track_gates", [])
            if tg:
                logger.log_track(ep_id, num_updates + 1, int(traj_level), tg)
        episodes = extract_episodes(traj)
        ep_meta = list(zip(traj["ep_rewards"], traj["ep_gates"]))
        for ep, (ep_ret, ep_g) in zip(episodes, ep_meta):
            golden.add(ep, ep_ret, ep_g, traj.get("academy_level", 0), dr_scale)
        if not alive_disabled and golden.gate_fraction() >= 0.25:
            ray.get([w.set_alive_disabled.remote(True) for w in workers])
            alive_disabled = True
            print(f"\n[R_ALIVE] Disabled (GM gate_fraction={golden.gate_fraction():.2f})")
        n_steps = len(traj["rewards"])
        logger.log_steps(
            update=num_updates + 1,
            level_arr=np.full(n_steps, int(traj_level), dtype=np.int32),
            state=traj["state"],
            actions=traj["actions"],
            rewards=traj["rewards"],
            dones=traj["dones"],
            gate_idx=traj["gate_idx"],
        )
        batch.append(traj)
        batch_levels.append(traj_level)
        if len(batch) < B:
            continue
        update_start = time.time()
        all_masks = torch.as_tensor(np.concatenate([tr["masks"] for tr in batch]), dtype=torch.float32, device=device)
        all_imu = torch.as_tensor(np.concatenate([tr["imu"] for tr in batch]), dtype=torch.float32, device=device)
        all_cv2 = torch.as_tensor(np.concatenate([tr["cv2"] for tr in batch]), dtype=torch.float32, device=device)
        all_nav = torch.as_tensor(np.concatenate([tr["nav"] for tr in batch]), dtype=torch.float32, device=device)
        all_act_hist = torch.as_tensor(np.concatenate([tr["actions_hist"] for tr in batch]), dtype=torch.float32, device=device)
        all_state = torch.as_tensor(np.concatenate([tr["state"] for tr in batch]), dtype=torch.float32, device=device)
        all_indices = torch.as_tensor(np.concatenate([tr["bin_indices"] for tr in batch]), dtype=torch.int64, device=device)
        all_blp = torch.as_tensor(np.concatenate([tr["behavior_log_probs"] for tr in batch]), dtype=torch.float32, device=device)
        all_rewards = torch.as_tensor(np.concatenate([tr["rewards"] for tr in batch]), dtype=torch.float32, device=device)
        all_dones = torch.as_tensor(np.concatenate([tr["dones"] for tr in batch]), dtype=torch.float32, device=device)
        all_obs = {"masks": all_masks, "imu": all_imu, "cv2": all_cv2, "nav": all_nav, "actions": all_act_hist, "state": all_state}
        with autocast("cuda"):
            all_lp, all_entropy, _, all_vd, all_vn, all_aux_pred = agent.evaluate(all_obs, all_indices)
            all_vd = all_vd.squeeze(-1)
            all_vn = all_vn.squeeze(-1)
        bootstrap_vals = compute_bootstrap_batched(agent, batch, B, T, device)
        with torch.no_grad():
            vtargets_cat, advantages_cat, avg_rho = compute_vtrace(
                all_blp.view(B, T), all_lp.detach().float().view(B, T),
                all_rewards.view(B, T), all_vd.detach().float().view(B, T),
                all_dones.view(B, T), bootstrap_vals,
                cfg["gamma"], cfg["rho_bar"], cfg["c_bar"],
            )
        agent.update_popart(vtargets_cat)
        normalized_targets = agent.normalize_targets(vtargets_cat)
        aux_targets = torch.stack([
            all_state[:, 0] / 10.0,
            all_state[:, 1] / 10.0,
            all_state[:, 2] / 4.0,
            all_state[:, 12] / 10.0,
            all_state[:, 19],
            all_state[:, 3] / 5.0,
            all_state[:, 4] / 5.0,
            all_state[:, 5] / 3.0,
        ], dim=-1).detach()
        with autocast("cuda"):
            policy_loss = -(advantages_cat * all_lp).mean()
            value_loss = 0.5 * (all_vn - normalized_targets.detach()).pow(2).mean()
            entropy_mean = all_entropy.mean()
            delta = entropy_mean.detach() - cfg["target_entropy"]
            blend = torch.sigmoid(delta / cfg["alpha_blend_width"])
            alpha_eff = (1.0 - blend) * (-alpha_up.detach()) + blend * alpha_down.detach()
            entropy_loss = alpha_eff * entropy_mean
            aux_loss = F.mse_loss(all_aux_pred, aux_targets)
            policy_coef = cfg["value_warmup_policy_coef"] if warmup_active else 1.0
            loss = (policy_coef * policy_loss
                    + cfg["value_coef"] * value_loss
                    + entropy_loss
                    + cfg["aux_coef"] * aux_loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]).item()
        if math.isfinite(raw_norm):
            scaler.step(optimizer)
            scaler.update()
            clipped_norm = min(raw_norm, cfg["max_grad_norm"])
            grad_str = f"g:{clipped_norm:.1f}({raw_norm:.1f})"
        else:
            scaler.update()
            grad_str = "g:SKIP"
        delta_f = (entropy_mean.detach().float() - cfg["target_entropy"])
        alpha_loss = -alpha_up * F.relu(-delta_f) - alpha_down * F.relu(delta_f)
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            alpha_up.clamp_(*cfg["alpha_up_clamp"])
            alpha_down.clamp_(*cfg["alpha_down_clamp"])
        avg_pl = policy_loss.item()
        avg_vl = value_loss.item()
        if warmup_active and avg_vl < cfg["value_warmup_v_loss_threshold"]:
            warmup_active = False
            print(f"[Update {num_updates}] Value-Warmup deactivated: v_loss={avg_vl:.3f} < {cfg['value_warmup_v_loss_threshold']}")
        avg_aux = aux_loss.item()
        avg_ent = all_entropy.mean().item()
        avg_alpha_up = alpha_up.item()
        avg_alpha_down = alpha_down.item()
        avg_alpha = avg_alpha_down if delta_f.item() > 0 else avg_alpha_up
        num_updates += 1
        sil_stats = {}
        if golden.size >= 4:
            sil_trajs = golden.sample(min(cfg["sil_samples_per_update"], golden.size))
            if sil_trajs:
                with autocast("cuda"):
                    sil_loss, sil_stats = compute_sil_loss(agent, sil_trajs, device, cfg["gamma"])
                optimizer.zero_grad()
                scaler.scale(cfg["sil_coef"] * sil_loss).backward()
                scaler.unscale_(optimizer)
                sil_raw = nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]).item()
                if math.isfinite(sil_raw):
                    scaler.step(optimizer)
                    scaler.update()
                    sil_clipped = min(sil_raw, cfg["max_grad_norm"])
                    sil_stats["sg"] = f"sg:{sil_clipped:.1f}({sil_raw:.1f})"
                else:
                    scaler.update()
                    sil_stats["sg"] = "sg:SKIP"
        advanced = academy.check_advance()
        if advanced:
            old_size = golden.size
            golden.purge_weak()
            print(f"\n[Academy] Level {academy.highest_unlocked} unlocked ({ACADEMY['levels'][academy.highest_unlocked]['name']}) | GM {old_size}->{golden.size} | {academy.status_str()}")
        old_dr = dr_scale
        if academy.highest_unlocked >= 5 and len(dr_gates) >= 50:
            dr_avg = np.mean(dr_gates)
            if dr_avg > 5.0:
                dr_scale = min(dr_scale + 0.005, 1.0)
            elif dr_avg < 3.0:
                dr_scale = max(dr_scale - 0.01, 0.0)
            if abs(dr_scale - old_dr) >= 0.05:
                print(f"[DR] {old_dr:.2f} \u2192 {dr_scale:.2f} (avg_gates {dr_avg:.1f})")
        elapsed = time.time() - train_start
        fps = (T * B) / max(time.time() - update_start, 1e-6)
        avg_r = np.mean(recent_rewards) if recent_rewards else 0.0
        max_r = max(recent_rewards) if recent_rewards else 0.0
        max_g = int(max(recent_gates)) if recent_gates else 0
        pa_mu = agent.critic.popart.mu.item()
        pa_sigma = agent.critic.popart.sigma.item()
        gm_size, gm_top = golden.stats()
        avg_act_jerk = float(np.mean([tr["act_jerk"] for tr in batch]))
        avg_gyro_jerk = float(np.mean([tr["gyro_jerk"] for tr in batch]))
        if sil_stats:
            sil_str = (
                f"SIL:{sil_stats['sil_loss']:.2f} "
                f"p:{sil_stats['sil_pi']:.2f} v:{sil_stats['sil_v']:.2f}"
                f"({sil_stats['sil_frac']:.0%}) {sil_stats.get('sg', '')}"
            )
        else:
            sil_str = "SIL:--"
        print(
            f"[{num_updates:5d}] {fmt_steps(global_step)} {fps:.0f}/s {fmt_time(elapsed)} | "
            f"R:{avg_r:+.1f}({max_r:+.1f}) G:{max_g} | "
            f"VT p:{avg_pl:.3f} v:{avg_vl:.3f} aux:{avg_aux:.3f} H:{avg_ent:.2f} au:{avg_alpha_up:.3f} ad:{avg_alpha_down:.3f} rho:{avg_rho:.2f} {grad_str} | "
            f"{sil_str} | mu:{pa_mu:.2f} sig:{pa_sigma:.2f} | "
            f"GM:{gm_size}({golden.capacity}) top:{gm_top} | {academy.status_str()} | L:{academy.highest_unlocked} DR:{dr_scale:.2f} | "
            f"smooth a:{avg_act_jerk:.3f} g:{avg_gyro_jerk:.2f}"
        )
        level_stats = []
        for i in range(academy.highest_unlocked + 1):
            stats = academy.level_success[i]
            if stats:
                level_stats.append({"level": i, "avg": float(np.mean(stats)), "max": int(max(stats)), "n": len(stats)})
            else:
                level_stats.append({"level": i, "avg": 0.0, "max": 0, "n": 0})
        gn_raw = raw_norm if math.isfinite(raw_norm) else None
        gn_clip = (min(raw_norm, cfg["max_grad_norm"]) if math.isfinite(raw_norm) else None)
        if sil_stats:
            sil_loss_v = sil_stats.get("sil_loss")
            sil_pi_v = sil_stats.get("sil_pi")
            sil_v_v = sil_stats.get("sil_v")
            sil_frac_v = sil_stats.get("sil_frac")
            sil_gn = sil_stats.get("sil_grad_norm")
        else:
            sil_loss_v = sil_pi_v = sil_v_v = sil_frac_v = sil_gn = None
        logger.log_update(
            update=num_updates, global_step=global_step, elapsed_sec=elapsed, fps=fps,
            avg_reward=avg_r, max_reward=max_r, max_gates=max_g,
            policy_loss=avg_pl, value_loss=avg_vl, aux_loss=avg_aux, entropy=avg_ent,
            alpha=avg_alpha, target_entropy=cfg["target_entropy"], ema_slope=0.0,
            rho_mean=avg_rho, grad_norm_raw=gn_raw, grad_norm_clipped=gn_clip,
            sil_loss=sil_loss_v, sil_pi=sil_pi_v, sil_v=sil_v_v, sil_frac=sil_frac_v, sil_grad_norm=sil_gn,
            popart_mu=pa_mu, popart_sigma=pa_sigma,
            gm_size=gm_size, gm_top_gates=gm_top,
            academy_highest=academy.highest_unlocked, dr_scale=dr_scale,
            level_stats=level_stats,
        )
        logger.flush()
        if num_updates % 20 == 0:
            ckpt_path = ckpt_dir / f"ckpt_{num_updates}.pt"
            torch.save({
                "num_updates": num_updates,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "academy_highest": academy.highest_unlocked,
                "dr_scale": dr_scale,
                "alive_disabled": alive_disabled,
                "alpha_up": alpha_up.detach().cpu().item(),
                "alpha_down": alpha_down.detach().cpu().item(),
            }, str(ckpt_path))
            avg_reward, avg_gates = evaluate_policy(
                agent, cfg["track"], num_episodes=20, device=device,
                academy_level=academy.highest_unlocked,
            )
            print(f"  [Eval] R:{avg_reward:.1f} G:{avg_gates:.1f} | ckpt:{ckpt_path.name}")
            if avg_gates > best_avg_gates:
                best_avg_gates = avg_gates
                torch.save({
                    "num_updates": num_updates,
                    "global_step": global_step,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "academy_highest": academy.highest_unlocked,
                    "avg_gates": avg_gates,
                    "dr_scale": dr_scale,
                    "alive_disabled": alive_disabled,
                    "alpha_up": alpha_up.detach().cpu().item(),
                    "alpha_down": alpha_down.detach().cpu().item(),
                }, str(ckpt_dir / "best.pt"))
        batch = []
        batch_levels = []
    for w in workers:
        ray.kill(w)
    ray.shutdown()
    logger.close()
    print(f"Done. {fmt_steps(global_step)} steps in {fmt_time(time.time() - train_start)}")

if __name__ == "__main__":
    train(CONFIG)
