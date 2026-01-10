# Deep_tinywhoop_racing_imitation

```
    ____                    _      __  __                    
   / __ \___  ___  ____    | | /| / / / /_  ____  ____  ____ 
  / / / / _ \/ _ \/ __ \   | |/ |/ / / __ \/ __ \/ __ \/ __ \
 / /_/ /  __/  __/ /_/ /   |__/|__/ / / / / /_/ / /_/ / /_/ /
/_____/\___/\___/ .___/             /_/ /_/\____/\____/ .___/ 
               /_/                                   /_/      
                    🏁 IMITATION LEARNING 🏁
```

> A neural network learns to race by watching humans fly.

---

## 💡 The Idea

**Simple concept:**
1. Human races a tiny whoop on a fixed track
2. Record everything: video + stick inputs
3. Train a network: frame → controls
4. Let the network fly

---

## 🔧 Hardware

| Whoop (65mm) | Ground Station |
|--------------|----------------|
| HDZero Whoop Lite VTX | HDZero Goggles |
| HDZero Nano Lite Camera | HDMI Capture Card |
| ELRS Receiver | ELRS TX Module |
| Betaflight FC | PC + GPU |

---

## 📡 Pipeline

### Phase 1 → Data Collection

*Human flies. Machine watches.*

```
╔═══════════════╗   5.8GHz    ╔═══════════════╗    HDMI     ╔═══════════════╗
║  TINY WHOOP   ║ ──────────► ║    HDZERO     ║ ─────────► ║    CAPTURE    ║
║   + Camera    ║             ║    GOGGLES    ║            ║     CARD      ║
╚═══════════════╝             ╚═══════════════╝            ╚═══════╤═══════╝
                                                                   │
                                                                   ▼
╔═══════════════╗             ╔═══════════════╗            ╔═══════════════╗
║    ELRS TX    ║ ─────────► ║     PILOT     ║            ║      PC       ║
║    MODULE     ║    log      ║    INPUTS     ║ ──sync───► ║   RECORDING   ║
╚═══════════════╝             ╚═══════════════╝            ╚═══════════════╝
```

**What gets recorded:**
- `video` → 720p60 from HDZero feed
- `controls` → throttle, roll, pitch, yaw
- `sync` → matched by timestamp

---

### Phase 2 → Training

*Behavioral cloning. Learn by imitation.*

```
┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Video Frame   │ ───► │  Neural Network  │ ───► │  Stick Values  │
│    (pixels)    │      │                  │      │  [T, R, P, Y]  │
└────────────────┘      └──────────────────┘      └────────────────┘
```

The network learns: **"when I see this → do that"**

---

### Phase 3 → Autonomous Flight

*Close the loop. Network takes control.*

```
╔═══════════════╗   5.8GHz    ╔═══════════════╗    HDMI     ╔═══════════════╗
║  TINY WHOOP   ║ ──────────► ║    HDZERO     ║ ─────────► ║    CAPTURE    ║
║   + Camera    ║             ║    GOGGLES    ║            ║     CARD      ║
╚═══════╤═══════╝             ╚═══════════════╝            ╚═══════╤═══════╝
        ▲                                                          │
        │                                                          ▼
        │  2.4GHz             ╔═══════════════╗            ╔═══════════════╗
        └──────────────────── ║    ELRS TX    ║ ◄───────── ║   INFERENCE   ║
                    ELRS      ║    MODULE     ║  control   ║    (model)    ║
                              ╚═══════════════╝            ╚═══════════════╝
```

**The Loop:**
```
camera → vtx → goggles → capture → model → tx → receiver → flight controller → repeat
```

---

## ⏱️ Latency Budget

```
HDZero Link     ████████░░░░░░░░░░░░  ~20ms
Capture Card    ████████████░░░░░░░░  ~30ms  
Inference       ████░░░░░░░░░░░░░░░░  ~10ms
ELRS Link       ████░░░░░░░░░░░░░░░░  ~10ms
─────────────────────────────────────────────
TOTAL           ██████████████████░░  ~70ms   ✓ Target: <100ms
```

---

## ⚠️ Safety

```python
# TODO: ...
pass
```

---

## 📋 Roadmap

- [ ] Data collection pipeline
- [ ] Record dataset on track
- [ ] Baseline model
- [ ] Training loop
- [ ] ELRS control interface
- [ ] First autonomous lap
- [ ] Iterate & improve

---

## 📄 License

MIT
