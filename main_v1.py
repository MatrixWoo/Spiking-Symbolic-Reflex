#!/usr/bin/env python3
"""
SNN Reflex Demo v1 — Original 3-way baseline.
Output → output/v1/
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

from environment import ToyGraspEnvironment, EnvConfig, run_baseline_experiments
from event_detector import EventDetector
from concept_layer import ConceptLayer
from rule_layer import RuleLayer
from snn_reflex import SNNReflexModule
from visualization import plot_baseline_comparison, plot_snn_raster, plot_force_efficiency

cfg = EnvConfig(slip_time=2.5, slip_duration=0.4, slip_magnitude=8.0,
                gripper_close_force=5.0, gripper_tighten_force=18.0)
ed, cl, rl, snn = EventDetector(cfg), ConceptLayer(), RuleLayer(), SNNReflexModule(cfg)
results, ed, cl, rl, snn = run_baseline_experiments(cfg, ed, cl, rl, snn)

out = os.path.join(os.path.dirname(__file__), 'output', 'v1')
os.makedirs(out, exist_ok=True)

plot_baseline_comparison(results, ed, cl, rl, snn, os.path.join(out, 'baseline_comparison.png'))
plot_snn_raster(snn, os.path.join(out, 'snn_spike_raster.png'))
plot_force_efficiency(results, os.path.join(out, 'force_efficiency.png'))

print("v1 done → output/v1/")
for f in sorted(os.listdir(out)):
    print(f"  {f}")
