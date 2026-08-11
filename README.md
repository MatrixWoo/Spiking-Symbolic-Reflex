# Spiking-Symbolic-Reflex

**Event → Concept → Rule / SNN → Reflex Action → Feedback**

An event-driven spiking reflex control framework for robot manipulation slip recovery. SNN serves as a fast reflex bypass (6-20ms) alongside symbolic rule reasoning for interpretable decision-making.

## Quick Start

```bash
conda activate snn-demo
python main.py          # v2: all 5 experiments
python main_v1.py       # v1: baseline 3-way comparison
python plot_architecture.py  # generate architecture figure

# MuJoCo experiments
cd mujoco_reflex
MUJOCO_GL=egl python main.py   # headless server; omit MUJOCO_GL on local machine
```

## Project Structure

```
Spiking-SNN-Reflex/
├── main.py / main_v1.py        # Experiment entry points
├── environment.py              # Toy env + MuJoCo config + 5 experiment runners
├── event_detector.py           # 5 analog event signals (state-difference driven)
├── concept_layer.py            # Event → semantic concept mapping + EMA smoothing
├── rule_layer.py               # Fuzzy symbolic rules (4 rules, configurable delay)
├── snn_reflex.py               # 5→10→3 LIF spiking network (adaptive force output)
├── visualization.py            # 7 figures (baseline/delay/sweep/noise/action/force/raster)
├── plot_architecture.py        # Paper architecture figure
├── summary.csv                 # MuJoCo 6-scenario × 4-mode complete results
├── docs/                       # Design notes, iteration records, thesis outline
├── mujoco_reflex/              # MuJoCo integration
│   ├── main.py                 # 4-mode comparison experiments
│   ├── mujoco_env.py           # Physics environment adapter
│   └── assets/                 # MuJoCo scene XMLs
└── output/
    ├── architecture_overview.png
    ├── v1/                     # Original 3-way comparison
    ├── v2/                     # 5-experiment suite
    └── mujoco/                 # MuJoCo experiment videos
```

## Key Results (MuJoCo Physics Simulation)

| Scenario | NoReflex | Rule | SNN | Key Finding |
|----------|:---:|:---:|:---:|------|
| calibration_ramp | ✗ | ✓ | ✓ | SNN 294ms, Rule 378ms |
| transport_fast | ✓ | ✓ | ✓ (6ms) | SNN 16.7× faster |
| lateral_impulse | ✓ | ✓ | ✓ (10ms) | SNN 9.4× faster |
| **heavy_low_friction** | **✗** | **✗** | **✓ (6ms)** | Rule fails, SNN saves |
| offset_grasp | ✗ | ✓ | ✓ | SNN 2.1× faster |
| no_disturbance | ✓ | ✓ | ✓ (zero FP) | No false triggers |
