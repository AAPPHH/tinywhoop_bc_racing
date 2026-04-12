import json
import math
import collections
from pathlib import Path
import numpy as np
import cv2
import gymnasium
from gymnasium import spaces
import pybullet as p
import pybullet_data

TRACKS = {
    "circle_small": {
        "gates": [
            [3.0, 0.0, 1.5, 0.0],
            [0.0, 3.0, 1.5, 90.0],
            [-3.0, 0.0, 1.5, 180.0],
            [0.0, -3.0, 1.5, 270.0],
        ],
        "gate_size": 1.0,
    },
    "oval": {
        "gates": [
            [4.0, 0.0, 1.5, 0.0],
            [2.0, 2.5, 1.5, 60.0],
            [-2.0, 2.5, 1.5, 120.0],
            [-4.0, 0.0, 1.5, 180.0],
            [-2.0, -2.5, 1.5, 240.0],
            [2.0, -2.5, 1.5, 300.0],
        ],
        "gate_size": 1.0,
    },
"visible_loop": {
        "gates": [
            [3.0, 0.0, 1.5, 0.0],
            [6.0, 1.5, 1.5, 20.0],
            [8.0, 4.0, 1.5, 45.0],
            [8.0, 7.0, 1.5, 75.0],
            [6.0, 9.5, 1.5, 110.0],
            [3.0, 10.0, 1.5, 145.0],
            [0.0, 9.0, 1.5, 180.0],
            [-2.0, 6.5, 1.5, 210.0],
            [-2.0, 3.5, 1.5, 250.0],
            [0.0, 1.0, 1.5, 290.0],
        ],
        "gate_size": 1.2,
    },
}

_tracks_dir = Path(__file__).resolve().parent.parent / "tracks"
if _tracks_dir.exists():
    for _f in sorted(_tracks_dir.glob("*.json")):
        with open(_f) as _fh:
            _t = json.load(_fh)
            TRACKS[_t["name"]] = {"gates": _t["gates"], "gate_size": _t["gate_size"]}

CF2X_MASS = 0.027
CF2X_ARM_LENGTH = 0.0397
CF2X_MAX_RPM = 21700.0
CF2X_HOVER_RPM = 14500.0
HOVER_DELTA = 5000.0
CF2X_IXX = 1.4e-5
CF2X_IYY = 1.4e-5
CF2X_IZZ = 2.17e-5
CF2X_KF = 3.16e-10
CF2X_KM = 7.94e-12
GRAVITY = 9.81

PHYSICS_HZ = 240
CONTROL_HZ = 50
SIM_STEPS_PER_CTRL = PHYSICS_HZ // CONTROL_HZ

CAM_FOV = 120.0
CAM_NEAR = 0.1
CAM_FAR = 20.0
CAM_RENDER_W = 160
CAM_RENDER_H = 120
CAM_MASK_W = 80
CAM_MASK_H = 60

GYRO_NOISE_STD = 0.01
ACCEL_NOISE_STD = 0.05

DR_BASELINE = {
    'mass': CF2X_MASS,
    'ixx': CF2X_IXX,
    'iyy': CF2X_IYY,
    'izz': CF2X_IZZ,
    'kf': CF2X_KF,
    'motor_tau': 0.030,
    'drag_coef': 0.01,
}
DR_RANGES = {
    'mass':         (-0.15, +0.15),
    'thrust':       (-0.30, +1.50),
    'inertia':      (-0.75, +0.30),
    'drag_coef':    (-0.50, +1.00),
    'motor_tau':    (-0.70, +0.70),
    'motor_noise':  0.05,
    'action_delay': 3,
    'wind_sigma':   0.3,
    'wind_theta':   0.1,
    'com_offset':   0.003,
}
PHYSICS_DT = 1.0 / PHYSICS_HZ

WINDOW_LEN = 16
TEMPORAL_DIM = 30
MASK_WINDOW_LEN = 1
CEILING = 4.0
RED_THRESHOLD = 0.6

ACADEMY_LEVELS = [
    {"name": "easy_circle",        "max_steps": 500},
    {"name": "fast_ring",          "max_steps": 600},
    {"name": "technical",          "max_steps": 700},
    {"name": "championship",      "max_steps": 800},
    {"name": "random",            "max_steps": 800},
    {"name": "random_dr",         "max_steps": 800},
]


class GateRacingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, track="circle_small", render_mode=None, academy_level=0, dr_scale=0.0):
        super().__init__()
        self.render_mode = render_mode
        self.track_name = track
        self.dr_scale = dr_scale
        self.track_cfg = TRACKS[track]
        self.gate_size = self.track_cfg["gate_size"]
        self.gates_raw = np.array(self.track_cfg["gates"], dtype=np.float64)
        self.num_gates = len(self.gates_raw)
        self.gate_positions = self.gates_raw[:, :3].copy()
        self.gate_yaws_rad = np.radians(self.gates_raw[:, 3])
        self.gate_normals = np.stack([
            np.cos(self.gate_yaws_rad),
            np.sin(self.gate_yaws_rad),
            np.zeros(self.num_gates),
        ], axis=-1)
        self.observation_space = spaces.Dict({
            "masks":    spaces.Box(0.0, 1.0, shape=(MASK_WINDOW_LEN, 2, CAM_MASK_H, CAM_MASK_W), dtype=np.float32),
            "temporal": spaces.Box(-np.inf, np.inf, shape=(WINDOW_LEN, TEMPORAL_DIM), dtype=np.float32),
            "state":    spaces.Box(-np.inf, np.inf, shape=(38,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.academy_level = academy_level
        self._active_track = track
        self._physics_client = None
        self._drone_id = None
        self._gate_visual_ids = []
        self._mask_window = None
        self._temporal_window = None
        self._step_count = 0
        self._current_gate_idx = 0
        self._gates_passed = 0
        self._prev_dist = 0.0
        self._prev_dot = 0.0
        self._last_rpms = np.zeros(4, dtype=np.float64)
        self._start_xy = np.zeros(2)
        self._prev_action = np.zeros(4, dtype=np.float64)
        self.r_prog = 5.0
        self._alive_disabled = False
        self._yaw_at_gate_pass = 0.0
        self._step_at_gate_pass = 0
        self._arrived = False
        self._max_steps = 500
        self._num_active_gates = self.num_gates
        self._dr_mass = CF2X_MASS
        self._dr_kf = CF2X_KF
        self._dr_ixx = CF2X_IXX
        self._dr_iyy = CF2X_IYY
        self._dr_izz = CF2X_IZZ
        self._dr_drag = DR_BASELINE['drag_coef']
        self._dr_motor_tau = DR_BASELINE['motor_tau']
        self._dr_motor_noise = 0.0
        self._dr_action_delay = 0
        self._dr_wind_sigma = 0.0
        self._dr_com_offset = np.zeros(3, dtype=np.float64)
        self._motor_state = np.zeros(4, dtype=np.float64)
        self._action_delay_buf = collections.deque(maxlen=4)
        self._wind = np.zeros(3, dtype=np.float64)

    def _init_pybullet(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._physics_client)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=self._physics_client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client)
        self._build_drone()
        self._gate_visual_ids = []
        for i in range(self.num_gates):
            ids = self._build_gate(i)
            self._gate_visual_ids.append(ids)

    def _build_drone(self):
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CF2X_ARM_LENGTH, CF2X_ARM_LENGTH, 0.01],
            physicsClientId=self._physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CF2X_ARM_LENGTH, CF2X_ARM_LENGTH, 0.01],
            rgbaColor=[0.2, 0.2, 0.8, 1.0],
            physicsClientId=self._physics_client,
        )
        self._drone_id = p.createMultiBody(
            baseMass=CF2X_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, 1.5],
            physicsClientId=self._physics_client,
        )
        p.changeDynamics(
            self._drone_id, -1,
            localInertiaDiagonal=[CF2X_IXX, CF2X_IYY, CF2X_IZZ],
            linearDamping=0.0,
            angularDamping=0.0,
            physicsClientId=self._physics_client,
        )

    def _build_gate(self, gate_idx):
        x, y, z = self.gate_positions[gate_idx]
        yaw = self.gate_yaws_rad[gate_idx]
        gs = self.gate_size
        t = 0.05
        gate_orn = list(p.getQuaternionFromEuler([0, 0, yaw]))
        sy, cy = math.sin(yaw), math.cos(yaw)
        red = [1.0, 0.0, 0.0, 1.0]
        ids = []
        post_top = z + gs / 2.0
        post_hz = post_top / 2.0
        post_offset = gs / 2.0 + t / 2.0
        lp_pos = [x - sy * post_offset, y + cy * post_offset, post_hz]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, t/2, post_hz], physicsClientId=self._physics_client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, t/2, post_hz], rgbaColor=red, physicsClientId=self._physics_client)
        ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                     basePosition=lp_pos, baseOrientation=gate_orn, physicsClientId=self._physics_client))
        rp_pos = [x + sy * post_offset, y - cy * post_offset, post_hz]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, t/2, post_hz], physicsClientId=self._physics_client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, t/2, post_hz], rgbaColor=red, physicsClientId=self._physics_client)
        ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                     basePosition=rp_pos, baseOrientation=gate_orn, physicsClientId=self._physics_client))
        bar_hy = gs / 2.0 + t
        tp_pos = [x, y, z + gs / 2.0 + t / 2.0]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, bar_hy, t/2], physicsClientId=self._physics_client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, bar_hy, t/2], rgbaColor=red, physicsClientId=self._physics_client)
        ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                     basePosition=tp_pos, baseOrientation=gate_orn, physicsClientId=self._physics_client))
        return ids

    def _get_drone_pos_orn(self):
        pos, orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._physics_client)
        return np.array(pos, dtype=np.float64), np.array(orn, dtype=np.float64)

    def _get_drone_vel(self):
        lin, ang = p.getBaseVelocity(self._drone_id, physicsClientId=self._physics_client)
        return np.array(lin, dtype=np.float64), np.array(ang, dtype=np.float64)

    @staticmethod
    def _quat_to_euler(q):
        x, y, z, w = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _quat_to_rot(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
            [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
        ], dtype=np.float64)

    def _apply_motors(self, rpms):
        rpms = np.clip(rpms, 0.0, CF2X_MAX_RPM)
        self._last_rpms = rpms.copy()
        if self._dr_motor_noise > 0.0:
            noise = 1.0 + np.random.normal(0, self._dr_motor_noise, size=4)
            rpms = rpms * noise
            rpms = np.clip(rpms, 0.0, CF2X_MAX_RPM)
        thrusts = self._dr_kf * rpms ** 2
        torques_z = CF2X_KM * rpms ** 2
        total_thrust = np.sum(thrusts)
        d = CF2X_ARM_LENGTH * math.sqrt(2) / 2.0
        tau_x = d * (thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3])
        tau_y = d * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])
        tau_z = -torques_z[0] + torques_z[1] - torques_z[2] + torques_z[3]
        force_body = [0.0, 0.0, float(total_thrust)]
        torque_body = [float(tau_x), float(tau_y), float(tau_z)]
        if self._dr_com_offset.any():
            grav_force = np.array([0.0, 0.0, -self._dr_mass * GRAVITY])
            com_torque = np.cross(self._dr_com_offset, grav_force)
            torque_body[0] += com_torque[0]
            torque_body[1] += com_torque[1]
            torque_body[2] += com_torque[2]
        p.applyExternalForce(self._drone_id, -1, force_body, [0, 0, 0],
                             p.LINK_FRAME, physicsClientId=self._physics_client)
        p.applyExternalTorque(self._drone_id, -1, torque_body,
                              p.LINK_FRAME, physicsClientId=self._physics_client)
        if self._dr_wind_sigma > 0.0 or self._dr_drag > 0.0:
            pos, orn = self._get_drone_pos_orn()
        if self._dr_wind_sigma > 0.0:
            self._wind += DR_RANGES['wind_theta'] * (0.0 - self._wind) * PHYSICS_DT + \
                          self._dr_wind_sigma * math.sqrt(PHYSICS_DT) * np.random.standard_normal(3)
            p.applyExternalForce(self._drone_id, -1, self._wind.tolist(), pos.tolist(),
                                 p.WORLD_FRAME, physicsClientId=self._physics_client)
        if self._dr_drag > 0.0:
            lin_vel, _ = self._get_drone_vel()
            drag_force = -self._dr_drag * lin_vel
            p.applyExternalForce(self._drone_id, -1, drag_force.tolist(), pos.tolist(),
                                 p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _get_track_gates(self):
        out = []
        for i in range(self.num_gates):
            out.append({
                "idx": i,
                "x": float(self.gate_positions[i][0]),
                "y": float(self.gate_positions[i][1]),
                "z": float(self.gate_positions[i][2]),
                "yaw_deg": float(math.degrees(self.gate_yaws_rad[i])),
                "nx": float(self.gate_normals[i][0]),
                "ny": float(self.gate_normals[i][1]),
                "nz": float(self.gate_normals[i][2]),
                "size": float(self.gate_size),
            })
        return out

    def _load_track(self, track_name):
        if track_name == self._active_track:
            return
        self._active_track = track_name
        cfg = TRACKS[track_name]
        self.gate_size = cfg["gate_size"]
        self.gates_raw = np.array(cfg["gates"], dtype=np.float64)
        self.num_gates = len(self.gates_raw)
        self.gate_positions = self.gates_raw[:, :3].copy()
        self.gate_yaws_rad = np.radians(self.gates_raw[:, 3])
        self.gate_normals = np.stack([
            np.cos(self.gate_yaws_rad),
            np.sin(self.gate_yaws_rad),
            np.zeros(self.num_gates),
        ], axis=-1)
        for ids in self._gate_visual_ids:
            for body_id in ids:
                p.removeBody(body_id, physicsClientId=self._physics_client)
        self._gate_visual_ids = []
        for i in range(self.num_gates):
            ids = self._build_gate(i)
            self._gate_visual_ids.append(ids)

    def _load_random_track(self):
        from track_gen import generate_random_track
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        cfg = generate_random_track(gate_size=1.0, rng=rng)
        self._active_track = "_random"
        self.gate_size = cfg["gate_size"]
        self.gates_raw = np.array(cfg["gates"], dtype=np.float64)
        self.num_gates = len(self.gates_raw)
        self.gate_positions = self.gates_raw[:, :3].copy()
        self.gate_yaws_rad = np.radians(self.gates_raw[:, 3])
        self.gate_normals = np.stack([
            np.cos(self.gate_yaws_rad),
            np.sin(self.gate_yaws_rad),
            np.zeros(self.num_gates),
        ], axis=-1)
        for ids in self._gate_visual_ids:
            for body_id in ids:
                p.removeBody(body_id, physicsClientId=self._physics_client)
        self._gate_visual_ids = []
        for i in range(self.num_gates):
            ids = self._build_gate(i)
            self._gate_visual_ids.append(ids)

    def _update_gate_colors(self):
        for i in range(self.num_gates):
            color = [0.0, 1.0, 0.0, 1.0] if i == self._current_gate_idx else [1.0, 0.0, 0.0, 1.0]
            for body_id in self._gate_visual_ids[i]:
                p.changeVisualShape(body_id, -1, rgbaColor=color, physicsClientId=self._physics_client)

    def _render_mask(self):
        pos, orn = self._get_drone_pos_orn()
        rot = self._quat_to_rot(orn)
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])
        target = pos + forward
        view_mat = p.computeViewMatrix(
            cameraEyePosition=pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._physics_client,
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=CAM_RENDER_W / CAM_RENDER_H,
            nearVal=CAM_NEAR,
            farVal=CAM_FAR,
            physicsClientId=self._physics_client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=CAM_RENDER_W,
            height=CAM_RENDER_H,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client,
        )
        img = np.array(rgba, dtype=np.float32).reshape(CAM_RENDER_H, CAM_RENDER_W, 4)
        r = img[:, :, 0] / 255.0
        g = img[:, :, 1] / 255.0
        b = img[:, :, 2] / 255.0
        green_full = ((g > 0.6) & (r < 0.3) & (b < 0.3)).astype(np.float32)
        red_full = ((r > RED_THRESHOLD) & (g < 0.3) & (b < 0.3)).astype(np.float32)
        green_down = green_full.reshape(CAM_MASK_H, 2, CAM_MASK_W, 2).mean(axis=(1, 3))
        green_down = (green_down > 0.25).astype(np.float32)
        red_down = red_full.reshape(CAM_MASK_H, 2, CAM_MASK_W, 2).mean(axis=(1, 3))
        red_down = (red_down > 0.25).astype(np.float32)
        return np.stack([green_down, red_down])

    @staticmethod
    def _extract_channel_features(mask_2d):
        mask_uint8 = (mask_2d * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(5, dtype=np.float32)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 1.0:
            return np.zeros(5, dtype=np.float32)
        M = cv2.moments(largest)
        cx = M["m10"] / (M["m00"] + 1e-6)
        cy = M["m01"] / (M["m00"] + 1e-6)
        _, _, w, h = cv2.boundingRect(largest)
        return np.array([
            (cx / CAM_MASK_W) * 2.0 - 1.0,
            (cy / CAM_MASK_H) * 2.0 - 1.0,
            area / (CAM_MASK_W * CAM_MASK_H),
            w / (h + 1e-6),
            1.0,
        ], dtype=np.float32)

    def _extract_gate_features(self, mask):
        target_f = self._extract_channel_features(mask[0])
        other_f = self._extract_channel_features(mask[1])
        return np.concatenate([target_f, other_f])

    def _read_imu(self):
        _, ang_vel = self._get_drone_vel()
        pos, orn = self._get_drone_pos_orn()
        lin_vel_prev = getattr(self, '_prev_lin_vel', np.zeros(3))
        lin_vel, _ = self._get_drone_vel()
        rot = self._quat_to_rot(orn)
        accel_world = (lin_vel - lin_vel_prev) * CONTROL_HZ
        gravity_world = np.array([0, 0, -GRAVITY])
        accel_body = rot.T @ (accel_world - gravity_world)
        gyro_body = rot.T @ ang_vel
        gyro_body += np.random.normal(0, GYRO_NOISE_STD, size=3)
        accel_body += np.random.normal(0, ACCEL_NOISE_STD, size=3)
        rpms_norm = self._last_rpms / CF2X_MAX_RPM
        self._prev_lin_vel = lin_vel.copy()
        result = np.concatenate([gyro_body, accel_body, rpms_norm]).astype(np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

    def _get_nav_features(self):
        pos, orn = self._get_drone_pos_orn()
        euler = self._quat_to_euler(orn)
        euler_norm = np.clip(np.array(euler) / math.pi, -1.0, 1.0)
        alt = np.clip(pos[2] / 3.0, 0.0, 1.0)
        heading_delta = np.clip((euler[2] - self._yaw_at_gate_pass) / math.pi, -1.0, 1.0)
        time_since = np.clip((self._step_count - self._step_at_gate_pass) / max(self._max_steps, 1), 0.0, 1.0)
        return np.array([euler_norm[0], euler_norm[1], euler_norm[2], alt, heading_delta, time_since], dtype=np.float32)

    def _get_target(self):
        return self.gate_positions[self._current_gate_idx]

    def _get_privileged_state(self):
        pos, orn = self._get_drone_pos_orn()
        lin_vel, ang_vel = self._get_drone_vel()
        euler = self._quat_to_euler(orn)
        target = self._get_target()
        pos_rel = target - pos
        dist = np.linalg.norm(pos_rel)
        gate_normal = self.gate_normals[self._current_gate_idx]
        if dist > 1e-6:
            dot_product = np.dot(pos_rel / dist, gate_normal)
        else:
            dot_product = 0.0
        next_idx = (self._current_gate_idx + 1) % self._num_active_gates
        next_pos = self.gate_positions[next_idx]
        pos_rel_next = next_pos - pos
        rpms_norm = self._last_rpms / CF2X_MAX_RPM
        gate_features = self._extract_gate_features(self._current_mask)
        state = np.concatenate([
            pos_rel, lin_vel, euler, ang_vel,
            [dist], pos_rel_next, gate_normal, [dot_product],
            rpms_norm, self._prev_action,
            gate_features,
        ]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=100.0, neginf=-100.0)

    def _check_gate_pass(self, pos):
        gate_pos = self.gate_positions[self._current_gate_idx]
        normal = self.gate_normals[self._current_gate_idx]
        rel = pos - gate_pos
        dist = np.linalg.norm(rel)
        dot = np.dot(rel, normal)
        passed = False
        offset_mag = 0.0
        if dist < self.gate_size * 1.5 and self._prev_dot < 0 and dot >= 0:
            along_normal = dot * normal
            offset_vec = rel - along_normal
            offset_mag = np.linalg.norm(offset_vec)
            if offset_mag < self.gate_size * 0.5:
                passed = True
        self._prev_dot = dot
        return passed, offset_mag


    def set_r_prog(self, val):
        self.r_prog = float(val)

    def set_alive_disabled(self, val):
        self._alive_disabled = bool(val)

    def _reward_gate_racing(self, pos, vel, euler, ang_vel, passed, offset, action):
        target = self.gate_positions[self._current_gate_idx]
        curr_dist = np.linalg.norm(pos - target)
        r_prog = self._prev_dist - curr_dist
        self._prev_dist = curr_dist
        r_gate = 100.0 if passed else 0.0
        ang_speed_sq = float(np.dot(ang_vel, ang_vel))
        r_angular = -0.0001 * ang_speed_sq
        action_delta = action - self._prev_action
        r_smooth = -0.005 * float(np.dot(action_delta, action_delta))
        return r_prog + r_gate + r_angular + r_smooth

    def _check_termination(self, pos, vel, euler):
        lvl = self.academy_level
        tilt_60 = math.radians(60.0)
        oob = {0: 10.0, 1: 12.0, 2: 12.0, 3: 15.0, 4: 15.0, 5: 15.0}.get(lvl, 15.0)
        if pos[2] < 0.3 and (vel[2] < -1.0 or abs(euler[0]) > tilt_60 or abs(euler[1]) > tilt_60):
            return True, "crash"
        if np.linalg.norm(pos[:2]) > oob:
            return True, "oob"
        if pos[2] > CEILING:
            return True, "ceiling"
        return False, ""

    def _setup_level(self):
        lvl = self.academy_level
        self._max_steps = ACADEMY_LEVELS[min(lvl, len(ACADEMY_LEVELS) - 1)]["max_steps"]
        self._arrived = False
        if lvl >= 4:
            self._load_random_track()
        else:
            track_name = {
                0: "easy_circle",
                1: "fast_ring",
                2: "technical",
                3: "championship",
            }.get(lvl, self.track_name)
            self._load_track(track_name)
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        if lvl == 0:
            self._spawn_gate_idx = int(rng.integers(0, self.num_gates))
        else:
            self._spawn_gate_idx = 0
        g0 = self.gate_positions[self._spawn_gate_idx]
        n0 = self.gate_normals[self._spawn_gate_idx]
        spawn_offset = 1.0 if lvl == 0 else 2.0
        self._spawn_pos = g0 - n0 * spawn_offset
        self._spawn_noise_xy = 0.2
        self._spawn_noise_rot = 5.0
        self._num_active_gates = self.num_gates
        self._reward_fn = self._reward_gate_racing

    def _sample_dr(self):
        s = self.dr_scale
        if s <= 0.0:
            self._dr_mass = DR_BASELINE['mass']
            self._dr_kf = DR_BASELINE['kf']
            self._dr_ixx = DR_BASELINE['ixx']
            self._dr_iyy = DR_BASELINE['iyy']
            self._dr_izz = DR_BASELINE['izz']
            self._dr_drag = DR_BASELINE['drag_coef']
            self._dr_motor_tau = DR_BASELINE['motor_tau']
            self._dr_motor_noise = 0.0
            self._dr_action_delay = 0
            self._dr_wind_sigma = 0.0
            self._dr_com_offset = np.zeros(3, dtype=np.float64)
            return
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        u = lambda lo, hi: rng.uniform(lo, hi)
        self._dr_mass = DR_BASELINE['mass'] * (1.0 + u(*DR_RANGES['mass']) * s)
        self._dr_kf = DR_BASELINE['kf'] * (1.0 + u(*DR_RANGES['thrust']) * s)
        inertia_scale = 1.0 + u(*DR_RANGES['inertia']) * s
        self._dr_ixx = DR_BASELINE['ixx'] * inertia_scale
        self._dr_iyy = DR_BASELINE['iyy'] * inertia_scale
        self._dr_izz = DR_BASELINE['izz'] * inertia_scale
        self._dr_drag = DR_BASELINE['drag_coef'] * (1.0 + u(*DR_RANGES['drag_coef']) * s)
        self._dr_motor_tau = DR_BASELINE['motor_tau'] * (1.0 + u(*DR_RANGES['motor_tau']) * s)
        self._dr_motor_noise = DR_RANGES['motor_noise'] * s
        self._dr_action_delay = int(rng.integers(0, int(DR_RANGES['action_delay'] * s) + 1))
        self._dr_wind_sigma = DR_RANGES['wind_sigma'] * s
        com_mag = u(0.0, DR_RANGES['com_offset'] * s)
        com_dir = rng.standard_normal(3)
        com_dir /= np.linalg.norm(com_dir) + 1e-12
        self._dr_com_offset = com_dir * com_mag

    def _apply_dr_to_body(self):
        p.changeDynamics(
            self._drone_id, -1,
            mass=self._dr_mass,
            localInertiaDiagonal=[self._dr_ixx, self._dr_iyy, self._dr_izz],
            physicsClientId=self._physics_client,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._physics_client is None:
            self._init_pybullet()
        self._setup_level()
        self._sample_dr()
        self._apply_dr_to_body()
        self._motor_state = np.ones(4, dtype=np.float64) * CF2X_HOVER_RPM
        self._action_delay_buf = collections.deque(maxlen=max(self._dr_action_delay + 1, 1))
        for _ in range(self._dr_action_delay + 1):
            self._action_delay_buf.append(np.zeros(4, dtype=np.float64))
        self._wind = np.zeros(3, dtype=np.float64)
        start_pos = self._spawn_pos.copy()
        noise_xy = self.np_random.normal(0, self._spawn_noise_xy, size=2)
        start_pos[0] += noise_xy[0]
        start_pos[1] += noise_xy[1]
        self._start_xy = start_pos[:2].copy()
        noise_rot = self.np_random.normal(0, math.radians(self._spawn_noise_rot))
        yaw = self.gate_yaws_rad[self._spawn_gate_idx] + noise_rot
        start_orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self._drone_id, start_pos.tolist(), list(start_orn),
            physicsClientId=self._physics_client,
        )
        p.resetBaseVelocity(
            self._drone_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self._physics_client,
        )
        self._step_count = 0
        self._current_gate_idx = self._spawn_gate_idx
        self._gates_passed = 0
        self._yaw_at_gate_pass = yaw
        self._step_at_gate_pass = 0
        self._update_gate_colors()
        self._last_rpms = np.zeros(4, dtype=np.float64)
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._prev_lin_vel = np.zeros(3, dtype=np.float64)
        rel = start_pos - self.gate_positions[self._spawn_gate_idx]
        self._prev_dot = np.dot(rel, self.gate_normals[self._spawn_gate_idx])
        target = self._get_target()
        self._prev_dist = np.linalg.norm(start_pos - target)
        self._mask_window = collections.deque(
            [np.zeros((2, CAM_MASK_H, CAM_MASK_W), dtype=np.float32) for _ in range(MASK_WINDOW_LEN)],
            maxlen=MASK_WINDOW_LEN,
        )
        self._temporal_window = collections.deque(
            [np.zeros(TEMPORAL_DIM, dtype=np.float32) for _ in range(WINDOW_LEN)],
            maxlen=WINDOW_LEN,
        )
        mask = self._render_mask()
        self._current_mask = mask
        imu = self._read_imu()
        cv2_feat = self._extract_gate_features(mask)
        nav_feat = self._get_nav_features()
        act_feat = np.zeros(4, dtype=np.float32)
        self._mask_window.append(mask)
        self._temporal_window.append(np.concatenate([imu, cv2_feat, nav_feat, act_feat]))
        obs = {
            "masks":    np.array(self._mask_window, dtype=np.float32),
            "temporal": np.array(self._temporal_window, dtype=np.float32),
            "state":    self._get_privileged_state(),
        }
        info = {"gates_passed": 0, "academy_level": self.academy_level, "arrived": False, "track_gates": self._get_track_gates()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()[:4]
        clipped_action = np.clip(action, -1.0, 1.0)
        self._action_delay_buf.append(clipped_action.copy())
        delayed_action = self._action_delay_buf[0]
        target_rpms = CF2X_HOVER_RPM + delayed_action * HOVER_DELTA
        target_rpms = np.clip(target_rpms, 0.0, CF2X_MAX_RPM)
        if self.dr_scale <= 0.0:
            self._motor_state = target_rpms
            for _ in range(SIM_STEPS_PER_CTRL):
                self._apply_motors(self._motor_state)
                p.stepSimulation(physicsClientId=self._physics_client)
        else:
            tau = max(self._dr_motor_tau, 1e-6)
            for _ in range(SIM_STEPS_PER_CTRL):
                self._motor_state += (target_rpms - self._motor_state) * (PHYSICS_DT / tau)
                self._apply_motors(self._motor_state)
                p.stepSimulation(physicsClientId=self._physics_client)
        self._step_count += 1
        pos, orn = self._get_drone_pos_orn()
        lin_vel, ang_vel = self._get_drone_vel()
        euler = self._quat_to_euler(orn)
        passed, offset_mag = self._check_gate_pass(pos)
        if passed:
            self._gates_passed += 1
            self._yaw_at_gate_pass = euler[2]
            self._step_at_gate_pass = self._step_count
            self._current_gate_idx = (self._current_gate_idx + 1) % self._num_active_gates
            self._update_gate_colors()
            rel_new = pos - self.gate_positions[self._current_gate_idx]
            self._prev_dot = np.dot(rel_new, self.gate_normals[self._current_gate_idx])
            target = self.gate_positions[self._current_gate_idx]
            self._prev_dist = np.linalg.norm(pos - target)
        terminated, term_reason = self._check_termination(pos, lin_vel, euler)
        truncated = self._step_count >= self._max_steps
        if terminated:
            reward = 0.0
            if self._gates_passed > 0:
                reward -= 50.0
        else:
            reward = self._reward_fn(pos, lin_vel, euler, ang_vel, passed, offset_mag, clipped_action)
        self._prev_action = clipped_action.copy()
        mask = self._render_mask()
        self._current_mask = mask
        imu = self._read_imu()
        cv2_feat = self._extract_gate_features(mask)
        nav_feat = self._get_nav_features()
        act_feat = self._prev_action.astype(np.float32)
        self._mask_window.append(mask)
        self._temporal_window.append(np.concatenate([imu, cv2_feat, nav_feat, act_feat]))
        obs = {
            "masks":    np.array(self._mask_window, dtype=np.float32),
            "temporal": np.array(self._temporal_window, dtype=np.float32),
            "state":    self._get_privileged_state(),
        }
        info = {
            "gates_passed": self._gates_passed,
            "academy_level": self.academy_level,
            "arrived": self._arrived,
            "distance_to_target": np.linalg.norm(pos - self._get_target()),
            "speed": np.linalg.norm(lin_vel),
            "term_reason": term_reason if terminated else ("truncated" if truncated else ""),
            "final_pos": pos.tolist(),
            "final_euler": euler.tolist(),
            "steps": self._step_count,
            "gate_idx": self._current_gate_idx,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pos, orn = self._get_drone_pos_orn()
        rot = self._quat_to_rot(orn)
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])
        target = pos + forward
        view_mat = p.computeViewMatrix(
            cameraEyePosition=pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._physics_client,
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=CAM_RENDER_W / CAM_RENDER_H,
            nearVal=CAM_NEAR,
            farVal=CAM_FAR,
            physicsClientId=self._physics_client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=CAM_RENDER_W,
            height=CAM_RENDER_H,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client,
        )
        img = np.array(rgba, dtype=np.uint8).reshape(CAM_RENDER_H, CAM_RENDER_W, 4)
        return img[:, :, :3]

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None

if __name__ == "__main__":
    for lvl in [0, 1, 2, 4]:
        name = ACADEMY_LEVELS[min(lvl, len(ACADEMY_LEVELS)-1)]["name"]
        env = GateRacingEnv(track="circle_small", academy_level=lvl)
        obs, info = env.reset()
        print(f"Level {lvl} ({name}): masks={obs['masks'].shape} temporal={obs['temporal'].shape} state={obs['state'].shape}")
        total_r = 0.0
        for s in range(50):
            action = np.zeros(4, dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            total_r += reward
            if term or trunc:
                break
        print(f"  R={total_r:.2f} G={info['gates_passed']} arrived={info['arrived']} steps={s+1}")
        env.close()
    print("Academy smoke-test passed.")
