import time
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import duckdb


METRICS_SCHEMA = pa.schema([
    ("update_idx", pa.int64()),
    ("global_step", pa.int64()),
    ("elapsed_sec", pa.float64()),
    ("fps", pa.float64()),
    ("avg_reward", pa.float64()),
    ("max_reward", pa.float64()),
    ("max_gates", pa.float64()),
    ("policy_loss", pa.float64()),
    ("value_loss", pa.float64()),
    ("aux_loss", pa.float64()),
    ("entropy", pa.float64()),
    ("alpha", pa.float64()),
    ("target_entropy", pa.float64()),
    ("ema_slope", pa.float64()),
    ("rho_mean", pa.float64()),
    ("grad_norm_raw", pa.float64()),
    ("grad_norm_clipped", pa.float64()),
    ("sil_loss", pa.float64()),
    ("sil_pi", pa.float64()),
    ("sil_v", pa.float64()),
    ("sil_frac", pa.float64()),
    ("sil_grad_norm", pa.float64()),
    ("popart_mu", pa.float64()),
    ("popart_sigma", pa.float64()),
    ("gm_size", pa.int64()),
    ("gm_top_gates", pa.float64()),
    ("academy_highest", pa.int32()),
    ("dr_scale", pa.float64()),
    ("l0_avg", pa.float64()), ("l0_max", pa.float64()), ("l0_n", pa.int64()),
    ("l1_avg", pa.float64()), ("l1_max", pa.float64()), ("l1_n", pa.int64()),
    ("l2_avg", pa.float64()), ("l2_max", pa.float64()), ("l2_n", pa.int64()),
    ("l3_avg", pa.float64()), ("l3_max", pa.float64()), ("l3_n", pa.int64()),
    ("l4_avg", pa.float64()), ("l4_max", pa.float64()), ("l4_n", pa.int64()),
    ("l5_avg", pa.float64()), ("l5_max", pa.float64()), ("l5_n", pa.int64()),
    ("timestamp", pa.float64()),
])

EPISODES_SCHEMA = pa.schema([
    ("episode_id", pa.int64()),
    ("update_idx", pa.int64()),
    ("level", pa.int32()),
    ("reward", pa.float64()),
    ("gates", pa.int32()),
    ("steps", pa.int32()),
    ("dist", pa.float64()),
    ("reason", pa.string()),
    ("fx", pa.float64()),
    ("fy", pa.float64()),
    ("fz", pa.float64()),
    ("roll", pa.float64()),
    ("pitch", pa.float64()),
    ("yaw", pa.float64()),
    ("dr_scale", pa.float64()),
    ("timestamp", pa.float64()),
])

TRACKS_SCHEMA = pa.schema([
    ("track_id", pa.int64()),
    ("episode_id", pa.int64()),
    ("update_idx", pa.int64()),
    ("level", pa.int32()),
    ("gate_idx", pa.int32()),
    ("gate_x", pa.float64()),
    ("gate_y", pa.float64()),
    ("gate_z", pa.float64()),
    ("gate_yaw_deg", pa.float64()),
    ("gate_normal_x", pa.float64()),
    ("gate_normal_y", pa.float64()),
    ("gate_normal_z", pa.float64()),
    ("gate_size", pa.float64()),
])

STEPS_SCHEMA = pa.schema([
    ("update_idx", pa.int64()),
    ("level", pa.int32()),
    ("pos_rel_x", pa.float32()), ("pos_rel_y", pa.float32()), ("pos_rel_z", pa.float32()),
    ("vel_x", pa.float32()), ("vel_y", pa.float32()), ("vel_z", pa.float32()),
    ("roll", pa.float32()), ("pitch", pa.float32()), ("yaw", pa.float32()),
    ("angvel_x", pa.float32()), ("angvel_y", pa.float32()), ("angvel_z", pa.float32()),
    ("dist", pa.float32()),
    ("dir_next_x", pa.float32()), ("dir_next_y", pa.float32()), ("dir_next_z", pa.float32()),
    ("gate_nx", pa.float32()), ("gate_ny", pa.float32()), ("gate_nz", pa.float32()),
    ("dot", pa.float32()),
    ("rpm_0", pa.float32()), ("rpm_1", pa.float32()), ("rpm_2", pa.float32()), ("rpm_3", pa.float32()),
    ("prev_act_0", pa.float32()), ("prev_act_1", pa.float32()), ("prev_act_2", pa.float32()), ("prev_act_3", pa.float32()),
    ("gf_0", pa.float32()), ("gf_1", pa.float32()), ("gf_2", pa.float32()), ("gf_3", pa.float32()), ("gf_4", pa.float32()),
    ("gf_5", pa.float32()), ("gf_6", pa.float32()), ("gf_7", pa.float32()), ("gf_8", pa.float32()), ("gf_9", pa.float32()),
    ("action_0", pa.float32()), ("action_1", pa.float32()), ("action_2", pa.float32()), ("action_3", pa.float32()),
    ("reward", pa.float32()),
    ("done", pa.float32()),
    ("gate_idx", pa.int32()),
])


class _Stream:
    def __init__(self, base_dir: Path, name: str, schema: pa.Schema):
        self.base_dir = base_dir
        self.name = name
        self.schema = schema
        self.roll_idx = 0
        self.arrow_path = base_dir / f"{name}.live.arrow"
        self._sink = pa.OSFile(str(self.arrow_path), "wb")
        self._writer = ipc.new_stream(self._sink, schema)

    def write(self, tbl: pa.Table):
        for batch in tbl.to_batches():
            self._writer.write_batch(batch)

    def roll_to_parquet(self):
        self._writer.close()
        self._sink.close()
        try:
            with pa.memory_map(str(self.arrow_path), "r") as src:
                reader = ipc.open_stream(src)
                tbl = reader.read_all()
            if tbl.num_rows > 0:
                out = self.base_dir / f"{self.name}_{self.roll_idx:04d}.parquet"
                pq.write_table(tbl, str(out), compression="zstd")
                self.roll_idx += 1
        except Exception:
            pass
        try:
            self.arrow_path.unlink()
        except Exception:
            pass
        self._sink = pa.OSFile(str(self.arrow_path), "wb")
        self._writer = ipc.new_stream(self._sink, self.schema)

    def close(self):
        try:
            self._writer.close()
            self._sink.close()
        except Exception:
            pass
        try:
            with pa.memory_map(str(self.arrow_path), "r") as src:
                reader = ipc.open_stream(src)
                tbl = reader.read_all()
            if tbl.num_rows > 0:
                out = self.base_dir / f"{self.name}_{self.roll_idx:04d}.parquet"
                pq.write_table(tbl, str(out), compression="zstd")
        except Exception:
            pass
        try:
            self.arrow_path.unlink()
        except Exception:
            pass


class MetricsLogger:
    def __init__(self, run_id, log_dir="lake", roll_interval_sec=600):
        self.run_id = run_id
        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.roll_interval = roll_interval_sec
        self._last_roll = time.time()
        self._metrics = _Stream(self.run_dir, "metrics", METRICS_SCHEMA)
        self._episodes = _Stream(self.run_dir, "episodes", EPISODES_SCHEMA)
        self._tracks = _Stream(self.run_dir, "tracks", TRACKS_SCHEMA)
        self._steps = _Stream(self.run_dir, "steps", STEPS_SCHEMA)
        self._episode_counter = 0
        self._track_counter = 0
        self._pending_episodes = []
        self._pending_tracks = []
        self._pending_steps = []
        self._pending_metrics = []

    def log_update(self, update, global_step, elapsed_sec, fps, avg_reward, max_reward, max_gates,
                   policy_loss, value_loss, aux_loss, entropy,
                   alpha, target_entropy, ema_slope,
                   rho_mean, grad_norm_raw, grad_norm_clipped,
                   sil_loss, sil_pi, sil_v, sil_frac, sil_grad_norm,
                   popart_mu, popart_sigma,
                   gm_size, gm_top_gates,
                   academy_highest, dr_scale,
                   level_stats):
        level_cols = {i: (None, None, None) for i in range(6)}
        for s in (level_stats or []):
            lv = s.get("level")
            if lv is not None and 0 <= lv < 6:
                level_cols[lv] = (
                    float(s["avg"]) if s.get("avg") is not None else None,
                    float(s["max"]) if s.get("max") is not None else None,
                    int(s["n"]) if s.get("n") is not None else None,
                )
        row = {
            "update_idx": int(update),
            "global_step": int(global_step),
            "elapsed_sec": float(elapsed_sec),
            "fps": float(fps),
            "avg_reward": float(avg_reward),
            "max_reward": float(max_reward),
            "max_gates": float(max_gates),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "aux_loss": float(aux_loss),
            "entropy": float(entropy),
            "alpha": float(alpha),
            "target_entropy": float(target_entropy),
            "ema_slope": float(ema_slope),
            "rho_mean": float(rho_mean),
            "grad_norm_raw": float(grad_norm_raw) if grad_norm_raw is not None else None,
            "grad_norm_clipped": float(grad_norm_clipped) if grad_norm_clipped is not None else None,
            "sil_loss": float(sil_loss) if sil_loss is not None else None,
            "sil_pi": float(sil_pi) if sil_pi is not None else None,
            "sil_v": float(sil_v) if sil_v is not None else None,
            "sil_frac": float(sil_frac) if sil_frac is not None else None,
            "sil_grad_norm": float(sil_grad_norm) if sil_grad_norm is not None else None,
            "popart_mu": float(popart_mu),
            "popart_sigma": float(popart_sigma),
            "gm_size": int(gm_size),
            "gm_top_gates": float(gm_top_gates),
            "academy_highest": int(academy_highest),
            "dr_scale": float(dr_scale),
            "timestamp": time.time(),
        }
        for i in range(6):
            avg, mx, n = level_cols[i]
            row[f"l{i}_avg"] = avg
            row[f"l{i}_max"] = mx
            row[f"l{i}_n"] = n
        self._pending_metrics.append(row)

    def log_episode(self, update, level, reward, gates, steps, dist, reason,
                    fx, fy, fz, roll, pitch, yaw, dr_scale):
        self._episode_counter += 1
        ep_id = self._episode_counter
        self._pending_episodes.append({
            "episode_id": ep_id,
            "update_idx": int(update),
            "level": int(level),
            "reward": float(reward),
            "gates": int(gates),
            "steps": int(steps),
            "dist": float(dist),
            "reason": str(reason),
            "fx": float(fx), "fy": float(fy), "fz": float(fz),
            "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
            "dr_scale": float(dr_scale),
            "timestamp": time.time(),
        })
        return ep_id

    def log_track(self, episode_id, update, level, gates):
        for g in gates:
            self._track_counter += 1
            self._pending_tracks.append({
                "track_id": self._track_counter,
                "episode_id": int(episode_id),
                "update_idx": int(update),
                "level": int(level),
                "gate_idx": int(g["idx"]),
                "gate_x": float(g["x"]),
                "gate_y": float(g["y"]),
                "gate_z": float(g["z"]),
                "gate_yaw_deg": float(g["yaw_deg"]),
                "gate_normal_x": float(g["nx"]),
                "gate_normal_y": float(g["ny"]),
                "gate_normal_z": float(g["nz"]),
                "gate_size": float(g["size"]),
            })

    def log_steps(self, update, level_arr, state, actions, rewards, dones, gate_idx):
        n = state.shape[0]
        batch = {
            "update_idx": np.full(n, int(update), dtype=np.int64),
            "level": level_arr.astype(np.int32),
            "pos_rel_x": state[:, 0], "pos_rel_y": state[:, 1], "pos_rel_z": state[:, 2],
            "vel_x": state[:, 3], "vel_y": state[:, 4], "vel_z": state[:, 5],
            "roll": state[:, 6], "pitch": state[:, 7], "yaw": state[:, 8],
            "angvel_x": state[:, 9], "angvel_y": state[:, 10], "angvel_z": state[:, 11],
            "dist": state[:, 12],
            "dir_next_x": state[:, 13], "dir_next_y": state[:, 14], "dir_next_z": state[:, 15],
            "gate_nx": state[:, 16], "gate_ny": state[:, 17], "gate_nz": state[:, 18],
            "dot": state[:, 19],
            "rpm_0": state[:, 20], "rpm_1": state[:, 21], "rpm_2": state[:, 22], "rpm_3": state[:, 23],
            "prev_act_0": state[:, 24], "prev_act_1": state[:, 25], "prev_act_2": state[:, 26], "prev_act_3": state[:, 27],
            "gf_0": state[:, 28], "gf_1": state[:, 29], "gf_2": state[:, 30], "gf_3": state[:, 31], "gf_4": state[:, 32],
            "gf_5": state[:, 33], "gf_6": state[:, 34], "gf_7": state[:, 35], "gf_8": state[:, 36], "gf_9": state[:, 37],
            "action_0": actions[:, 0], "action_1": actions[:, 1], "action_2": actions[:, 2], "action_3": actions[:, 3],
            "reward": rewards, "done": dones, "gate_idx": gate_idx.astype(np.int32),
        }
        self._pending_steps.append(batch)

    def flush(self):
        if self._pending_metrics:
            tbl = pa.Table.from_pylist(self._pending_metrics, schema=METRICS_SCHEMA)
            self._metrics.write(tbl)
            self._pending_metrics = []
        if self._pending_episodes:
            tbl = pa.Table.from_pylist(self._pending_episodes, schema=EPISODES_SCHEMA)
            self._episodes.write(tbl)
            self._pending_episodes = []
        if self._pending_tracks:
            tbl = pa.Table.from_pylist(self._pending_tracks, schema=TRACKS_SCHEMA)
            self._tracks.write(tbl)
            self._pending_tracks = []
        if self._pending_steps:
            merged = {k: np.concatenate([b[k] for b in self._pending_steps]) for k in self._pending_steps[0]}
            tbl = pa.Table.from_pydict(merged, schema=STEPS_SCHEMA)
            self._steps.write(tbl)
            self._pending_steps = []
        now = time.time()
        if now - self._last_roll >= self.roll_interval:
            self._metrics.roll_to_parquet()
            self._episodes.roll_to_parquet()
            self._tracks.roll_to_parquet()
            self._steps.roll_to_parquet()
            self._last_roll = now

    def build_tracks(self, seed=42):
        from track_gen import build_tracks
        dst = self.run_dir / "tracks_src"
        build_tracks(dst, seed=seed)

    def query(self, sql):
        con = duckdb.connect()
        return con.execute(sql).fetchall()

    def close(self):
        self.flush()
        self._metrics.close()
        self._episodes.close()
        self._tracks.close()
        self._steps.close()
