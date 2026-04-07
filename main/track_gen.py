import json
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

PRESETS = {
    "easy_circle": dict(shape="circle", radius=3.0, num_gates=7, height_var=0.0, max_turn=55, gate_size=1.0, description="Sanfter Kreis"),
    "fast_ring": dict(shape="circle", radius=6.0, num_gates=8, height_var=0.0, max_turn=50, gate_size=1.0, description="Großer Ring, Speed-Track"),
    "technical": dict(shape="random", num_control=8, radius=5.0, num_gates=8, height_var=0.3, max_turn=55, perturbation=1.5, gate_size=1.2, description="Unregelmäßig mit Höhenvariation"),
    "championship": dict(shape="random", num_control=10, radius=5.5, num_gates=10, height_var=0.4, max_turn=55, perturbation=1.2, gate_size=1.0, description="Großer unregelmäßiger Kurs, viele Gates"),
}

DEFAULTS = {
    "base_height": 1.5,
    "min_spacing": 2.0,
    "max_spacing": 8.0,
    "max_turn_angle": 60.0,
    "min_height": 1.0,
    "max_height": 2.0,
    "clearance": 2.0,
    "fov_half": 55.0,
    "max_retries": 100,
    "spline_samples": 1000,
}


def generate_control_points(preset):
    shape = preset["shape"]
    if shape == "circle":
        r = preset["radius"]
        n = max(preset.get("num_control", preset["num_gates"]), 4)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    elif shape == "ellipse":
        rx = preset["radius_x"]
        ry = preset["radius_y"]
        n = max(preset.get("num_control", preset["num_gates"]), 4)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
    elif shape == "random":
        n = preset.get("num_control", 8)
        r = preset["radius"]
        pert = preset.get("perturbation", 1.0)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = r + np.random.uniform(-pert, pert, n)
        angle_noise = np.random.uniform(-0.3, 0.3, n)
        angles = angles + angle_noise
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    raise ValueError(f"Unknown shape: {shape}")


def make_closed_spline(pts, num_samples):
    pts_closed = np.vstack([pts, pts[0]])
    t = np.zeros(len(pts_closed))
    for i in range(1, len(pts_closed)):
        t[i] = t[i - 1] + np.linalg.norm(pts_closed[i] - pts_closed[i - 1])
    total_len = t[-1]
    t_norm = t / total_len
    cs_x = CubicSpline(t_norm, pts_closed[:, 0], bc_type="periodic")
    cs_y = CubicSpline(t_norm, pts_closed[:, 1], bc_type="periodic")
    s = np.linspace(0, 1, num_samples, endpoint=False)
    path = np.column_stack([cs_x(s), cs_y(s)])
    dx = cs_x(s, 1)
    dy = cs_y(s, 1)
    tangents = np.column_stack([dx, dy])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.clip(norms, 1e-8, None)
    arc_lengths = np.zeros(num_samples)
    for i in range(1, num_samples):
        arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(path[i] - path[i - 1])
    total_arc = arc_lengths[-1] + np.linalg.norm(path[0] - path[-1])
    return path, tangents, arc_lengths, total_arc


def sample_gates_along_path(path, tangents, arc_lengths, total_arc, num_gates, preset):
    gate_arc_positions = np.linspace(0, total_arc, num_gates, endpoint=False)
    gates = []
    base_h = DEFAULTS["base_height"]
    h_var = preset.get("height_var", 0.0)
    for arc_pos in gate_arc_positions:
        idx = np.searchsorted(arc_lengths, arc_pos, side="right") - 1
        idx = max(0, min(idx, len(path) - 1))
        x, y = path[idx]
        tx, ty = tangents[idx]
        yaw_rad = math.atan2(ty, tx)
        yaw_deg = math.degrees(yaw_rad)
        z = base_h + np.random.uniform(-h_var, h_var)
        z = np.clip(z, DEFAULTS["min_height"], DEFAULTS["max_height"])
        gates.append([float(x), float(y), float(z), float(yaw_deg)])
    return gates


def validate_track(gates, preset):
    n = len(gates)
    positions = np.array([g[:3] for g in gates])
    yaws_rad = np.radians([g[3] for g in gates])
    normals = np.column_stack([np.cos(yaws_rad), np.sin(yaws_rad), np.zeros(n)])
    max_turn_limit = preset.get("max_turn", DEFAULTS["max_turn_angle"])
    for i in range(n):
        j = (i + 1) % n
        d = np.linalg.norm(positions[j] - positions[i])
        if d < DEFAULTS["min_spacing"]:
            return False, f"Gate {i}->{j} spacing {d:.2f} < {DEFAULTS['min_spacing']}"
        if d > DEFAULTS["max_spacing"]:
            return False, f"Gate {i}->{j} spacing {d:.2f} > {DEFAULTS['max_spacing']}"
    for i in range(n):
        j = (i + 1) % n
        angle_diff = abs(math.degrees(math.atan2(
            math.sin(yaws_rad[j] - yaws_rad[i]),
            math.cos(yaws_rad[j] - yaws_rad[i]),
        )))
        if angle_diff > max_turn_limit:
            return False, f"Gate {i}->{j} turn {angle_diff:.1f}° > {max_turn_limit}°"
    fov_half = DEFAULTS["fov_half"]
    for i in range(n):
        j = (i + 1) % n
        vec_to_next = positions[j] - positions[i]
        dist = np.linalg.norm(vec_to_next)
        if dist < 1e-6:
            return False, f"Gate {i} and {j} overlap"
        direction = vec_to_next / dist
        normal_2d = normals[i][:2]
        dir_2d = direction[:2]
        dir_2d_norm = np.linalg.norm(dir_2d)
        if dir_2d_norm < 1e-6:
            continue
        dir_2d = dir_2d / dir_2d_norm
        cos_angle = np.clip(np.dot(normal_2d, dir_2d), -1, 1)
        angle = math.degrees(math.acos(cos_angle))
        if angle > fov_half:
            return False, f"Gate {j} not visible from {i}: angle {angle:.1f}° > {fov_half}°"
    for i in range(n):
        for k in range(i + 2, n):
            if i == 0 and k == n - 1:
                continue
            d = np.linalg.norm(positions[k] - positions[i])
            if d < DEFAULTS["clearance"]:
                return False, f"Gate {i} and {k} too close: {d:.2f} < {DEFAULTS['clearance']}"
    return True, "OK"


def compute_max_turn(gates):
    n = len(gates)
    yaws_rad = np.radians([g[3] for g in gates])
    max_t = 0.0
    for i in range(n):
        j = (i + 1) % n
        diff = abs(math.degrees(math.atan2(
            math.sin(yaws_rad[j] - yaws_rad[i]),
            math.cos(yaws_rad[j] - yaws_rad[i]),
        )))
        max_t = max(max_t, diff)
    return max_t


def generate_track(name, preset, seed=42):
    np.random.seed(seed)
    num_samples = DEFAULTS["spline_samples"]
    for attempt in range(DEFAULTS["max_retries"]):
        ctrl = generate_control_points(preset)
        path, tangents, arc_lengths, total_arc = make_closed_spline(ctrl, num_samples)
        gates = sample_gates_along_path(path, tangents, arc_lengths, total_arc, preset["num_gates"], preset)
        valid, reason = validate_track(gates, preset)
        if valid:
            max_turn = compute_max_turn(gates)
            track = {
                "name": name,
                "gates": [[round(v, 3) for v in g] for g in gates],
                "gate_size": preset.get("gate_size", 1.0),
                "metadata": {
                    "preset": name,
                    "total_length": round(total_arc, 1),
                    "max_turn": round(max_turn, 1),
                    "all_visible": True,
                    "attempts": attempt + 1,
                },
            }
            print(f"  {name}: {len(gates)} gates, length={total_arc:.1f}m, max_turn={max_turn:.1f}°, visible=YES (attempt {attempt+1})")
            return track, path
        if attempt == DEFAULTS["max_retries"] - 1:
            print(f"  {name}: FAILED after {DEFAULTS['max_retries']} attempts. Last: {reason}")
    return None, None


def generate_random_track(gate_size=1.0, rng=None):
    if rng is None:
        rng = np.random
    gate_count = rng.choice([7, 8, 10])
    arena_half = 4.0
    min_spacing, max_spacing = 1.5, 4.0
    min_h, max_h = 1.0, 2.5
    max_turn_rad = math.radians(90.0)
    fov_half_rad = math.radians(60.0)
    gates = []
    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-1.0, 1.0)
    z = rng.uniform(min_h, max_h)
    yaw = rng.uniform(0, 2 * math.pi)
    gates.append([x, y, z, yaw])
    for _ in range(1, gate_count):
        best, best_score = None, -1e9
        prev = gates[-1]
        px, py, pz, pyaw = prev
        for _ in range(20):
            turn = rng.uniform(-max_turn_rad, max_turn_rad)
            nyaw = pyaw + turn
            dist = rng.uniform(min_spacing, max_spacing)
            nx = px + math.cos(nyaw) * dist
            ny = py + math.sin(nyaw) * dist
            nz = rng.uniform(min_h, max_h)
            if abs(nx) > arena_half or abs(ny) > arena_half:
                continue
            vec = np.array([nx - px, ny - py])
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-6:
                continue
            fwd = np.array([math.cos(pyaw), math.sin(pyaw)])
            cos_a = np.dot(fwd, vec / vec_norm)
            if cos_a < math.cos(fov_half_rad):
                continue
            too_close = False
            for g in gates[:-1]:
                if math.hypot(nx - g[0], ny - g[1]) < 1.5:
                    too_close = True
                    break
            if too_close:
                continue
            score = cos_a + dist / max_spacing
            if score > best_score:
                best = [nx, ny, nz, nyaw]
                best_score = score
        if best is None:
            turn = rng.uniform(-max_turn_rad * 0.5, max_turn_rad * 0.5)
            nyaw = pyaw + turn
            dist = rng.uniform(min_spacing, max_spacing * 0.8)
            best = [px + math.cos(nyaw) * dist, py + math.sin(nyaw) * dist,
                    rng.uniform(min_h, max_h), nyaw]
        gates.append(best)
    last = gates[-1]
    first = gates[0]
    close_vec = np.array([first[0] - last[0], first[1] - last[1]])
    close_yaw = math.atan2(close_vec[1], close_vec[0])
    gates[-1][3] = close_yaw
    gates_deg = [[round(g[0], 3), round(g[1], 3), round(g[2], 3),
                  round(math.degrees(g[3]) % 360, 1)] for g in gates]
    return {"name": "_random", "gates": gates_deg, "gate_size": gate_size}


def generate_all_tracks(out_dir=None, seed=42):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "tracks"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    missing = [name for name in PRESETS if not (out_dir / f"{name}.json").exists()]
    if not missing:
        return
    print(f"Generating {len(missing)} missing tracks:")
    for name in missing:
        track, _ = generate_track(name, PRESETS[name], seed=seed)
        if track:
            out_path = out_dir / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(track, f, indent=2)
            print(f"    -> {out_path}")


def plot_tracks(tracks_with_paths):
    import matplotlib.pyplot as plt
    n = len(tracks_with_paths)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
    for idx, item in enumerate(tracks_with_paths):
        track, path = item[0], item[1]
        ax = axes[idx // cols][idx % cols]
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], "k--", alpha=0.3, linewidth=1)
        gates = np.array(track["gates"])
        positions = gates[:, :2]
        yaws_rad = np.radians(gates[:, 3])
        normals = np.column_stack([np.cos(yaws_rad), np.sin(yaws_rad)])
        perps = np.column_stack([-np.sin(yaws_rad), np.cos(yaws_rad)])
        gs = track["gate_size"]
        fov_half_rad = math.radians(DEFAULTS["fov_half"])
        fov_len = 2.0
        for i in range(len(gates)):
            p = positions[i]
            perp = perps[i]
            p1 = p + perp * gs * 0.5
            p2 = p - perp * gs * 0.5
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="green", linewidth=3, solid_capstyle="round")
            ax.annotate(str(i), xy=p, fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8))
            n_dir = normals[i]
            ax.arrow(p[0], p[1], n_dir[0] * 0.5, n_dir[1] * 0.5,
                     head_width=0.15, head_length=0.1, fc="orange", ec="orange", alpha=0.8)
            left = np.array([
                n_dir[0] * math.cos(fov_half_rad) - n_dir[1] * math.sin(fov_half_rad),
                n_dir[0] * math.sin(fov_half_rad) + n_dir[1] * math.cos(fov_half_rad),
            ])
            right = np.array([
                n_dir[0] * math.cos(-fov_half_rad) - n_dir[1] * math.sin(-fov_half_rad),
                n_dir[0] * math.sin(-fov_half_rad) + n_dir[1] * math.cos(-fov_half_rad),
            ])
            tri = plt.Polygon([p, p + left * fov_len, p + right * fov_len],
                              alpha=0.08, color="blue", linewidth=0)
            ax.add_patch(tri)
        ax.set_aspect("equal")
        ax.set_title(f"{track['name']} ({len(gates)}g, {track['metadata']['total_length']}m, max_turn={track['metadata']['max_turn']}°)")
        ax.grid(True, alpha=0.3)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    out_path = tracks_with_paths[0][2] if len(tracks_with_paths[0]) > 2 else "tracks/tracks_overview.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Plot saved: {out_path}")
    plt.close(fig)


def build_tracks(out_dir, seed=42, show=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, preset in PRESETS.items():
        track, path = generate_track(name, preset, seed=seed)
        if track:
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(track, f, indent=2)
            results.append((track, path, out_dir / "tracks_overview.png"))
    if results:
        plot_tracks(results)
    return [r[0]["name"] for r in results]


if __name__ == "__main__":
    out_dir = Path("tracks")
    out_dir.mkdir(exist_ok=True)
    results = []
    print("Generating tracks:")
    for name, preset in PRESETS.items():
        track, path = generate_track(name, preset, seed=42)
        if track:
            out_path = out_dir / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(track, f, indent=2)
            print(f"    -> {out_path}")
            results.append((track, path))
    if results:
        print(f"\nenv.py TRACKS format:")
        for track, _ in results:
            gates_str = ",\n            ".join([str(g) for g in track["gates"]])
            print(f'    "{track["name"]}": {{\n        "gates": [\n            {gates_str},\n        ],\n        "gate_size": {track["gate_size"]},\n    }},')
        print()
        plot_tracks(results)
