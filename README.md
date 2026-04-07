# Deep Whoop Racing

```
    ____                    _      __  __                    
   / __ \___  ___  ____    | | /| / / / /_  ____  ____  ____ 
  / / / / _ \/ _ \/ __ \   | |/ |/ / / __ \/ __ \/ __ \/ __ \
 / /_/ /  __/  __/ /_/ /   |__/|__/ / / / / /_/ / /_/ / /_/ /
/_____/\___/\___/ .___/             /_/ /_/\____/\____/ .___/ 
               /_/                                   /_/      
                  RL-BASED AUTONOMOUS RACING
```

> Train a neural net to race a 65mm FPV drone through gate tracks using reinforcement learning in simulation, then deploy on real hardware.

---

## Architecture

**Asymmetric Actor-Critic with IMPALA + V-Trace + Self-Imitation Learning**

```
                        ENV (PyBullet CF2X @ 240Hz)
                                  |
                    obs: masks + imu + cv2 + nav
                                  |
                    +-------------+-------------+
                    |                           |
               Actor (deployable)          Critic (training only)
               masks(4,2,60,80)            privileged state(38)
               imu(4,10) + cv2(10)               |
               nav(6) + actions(4,4)         MLP -> PopArt
                    |                           |
             C51 Categorical(4x51)         denormalized value
               -> 4 motor RPMs                  |
                    |                     V-Trace targets
                    +-------------+-------------+
                                  |
                          PPO-style update
                          + SAC adaptive alpha
                          + SIL from GoldenMemory
```

**Key decisions:**
- C51 distributional action head (51 bins per motor, [-1,1] around hover)
- PopArt value normalization for curriculum stability
- SAC-style adaptive entropy: learned alpha auto-tunes exploration vs exploitation
- Dual-EMA on PopArt mu tracks learning slope, shifts entropy target dynamically
- GoldenMemory stores best trajectories, vectorized scoring by `gates^2 * (1+level)`

---

## Academy Curriculum

| Level | Track | Gates | Unlock | Max Steps |
|-------|-------|-------|--------|-----------|
| 0 | easy_circle | 4 | -- | 500 |
| 1 | fast_ring | 4 | L0 avg_gates >= 3.0 | 600 |
| 2 | technical | 4 | L1 avg_gates >= 4.0 | 700 |
| 3 | championship | 4 | L2 avg_gates >= 5.0 | 800 |

50% of training on highest unlocked level, rest distributed across lower levels.

---

## Training Log Format

```
[ 2878] 87.1M 20482/s 5m | R:+243.1(+417.6) | VT p:2.407 v:0.249 aux:0.191 H:6.32 a:0.018 t:6.3 s:+0.01 rho:1.00 g:3.5(3.5) | SIL:8.54 p:7.54 v:1.00(74%) sg:0.4(0.4) | mu:21.80 sig:27.17 | GM:256(256) top:12 | L0:3.9*/3(200) L1:4.1*/4(200) L2:4.0/5(200) | L:2
```

| Field | Meaning |
|-------|---------|
| `R:avg(max)` | Episode reward (200-ep window) |
| `H:6.32` | Policy entropy (4 motors x 51 bins, max ~15.7) |
| `a:0.018` | SAC alpha (learned entropy coefficient) |
| `t:6.3` | Adaptive entropy target (clamped 5.0-8.0) |
| `s:+0.01` | EMA slope (PopArt mu trend: + = improving) |
| `GM:256(256) top:12` | GoldenMemory entries / capacity / best gates |
| `L0:3.9*/3(200)` | Per-level: avg_gates (* = threshold met) / threshold (n_episodes) |

---

## File Structure

```
main/
  env.py        PyBullet CF2X physics, camera masks, rewards, curriculum
  models.py     Actor (C51), Critic (PopArt), RunningMeanStd
  train.py      IMPALA learner, Ray workers, V-Trace, SIL, Academy
  track_gen.py  Procedural track generation
tracks/         Generated track JSON files
checkpoints/    Training checkpoints
```

---

## Hardware Target

| Whoop (65mm) | Ground Station |
|--------------|----------------|
| HDZero Whoop Lite VTX | HDZero Goggles |
| HDZero Nano Lite Camera | HDMI Capture Card |
| ELRS Receiver | ELRS TX Module |
| Betaflight FC | PC + GPU |

**Sim-to-real pipeline:** Train in PyBullet -> export Actor weights -> inference on PC -> ELRS TX -> drone

---

## Sim2Real

*Train in simulation. Deploy on real hardware.*

```
+---------------------+                       +---------------------+
|   PyBullet Sim      |                       |    Real Air65       |
|   CF2X dynamics     | ----- transfer ---->  |   HDZero + ELRS     |
|   240Hz physics     |                       |   PC inference      |
|   50Hz control      |                       |   50Hz control      |
+---------------------+                       +---------------------+
        ^                                              |
        |                                              |
        +-------- domain randomization ----------------+
                  (mass, motor, latency, IMU, wind)
```

**Why sim2real:**
- Crashes in sim are free -- millions of episodes, no broken props
- Curriculum from easy_circle -> technical -> championship -> random tracks
- Domain randomization closes the reality gap

**Domain randomization (dr_scale 0.0 -> 1.0, ramped after L5 unlock):**
- Mass / inertia / battery sag
- Motor response curves + thrust noise
- Camera FOV, exposure, latency
- IMU bias and gyro/accel noise
- Wind disturbances

**Transfer pipeline:**
```
RL Policy (sim) -> Actor weights export -> PC inference (PyTorch)
                -> ELRS TX uplink -> Betaflight FC -> motors
```

The Actor (masks + IMU + cv2 + nav) is the deployable network. The Critic with privileged state stays in training only.

---

## Tech Stack

- **Sim:** PyBullet DIRECT (CF2X dynamics, 240Hz physics / 50Hz control)
- **RL:** Custom IMPALA + V-Trace + SIL + SAC adaptive entropy
- **Model:** PyTorch, C51 categorical actor, asymmetric critic with PopArt
- **Parallelization:** Ray actors (16 workers, CPU-only envs, centralized GPU learner)
- **OS:** Windows 11, Python 3.10+

---

## Progress

- [x] PyBullet CF2X sim with accurate dynamics
- [x] 2-channel mask vision (green=target, red=others)
- [x] IMPALA architecture with Ray workers
- [x] V-Trace off-policy correction
- [x] PopArt value normalization
- [x] C51 distributional action head
- [x] Academy curriculum (4 levels)
- [x] GoldenMemory + SIL (vectorized)
- [x] SAC adaptive entropy with dual-EMA target
- [x] Consistent L0/L1 gate passing (~4.0 avg)
- [ ] L2 breakthrough (target: 5.0 avg gates)
- [ ] L3 championship track
- [ ] Sim-to-real transfer
- [ ] First autonomous lap on real hardware

---

## License

MIT
