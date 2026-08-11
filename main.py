#!/usr/bin/env python3
"""
SNN Reflex Demo v2 — Main Entry Point
======================================
Demonstrates Event → Concept → Rule → Reflex Action closed loop.

Experiments:
  A. Baseline 3-way comparison (fixed: no_reflex now shows events)
  B. Rule Delay vs SNN Reflex (80ms rule delay vs 10-20ms SNN)
  C. Slip Magnitude Sweep (robustness across [4,6,8,10,12])
  D. Noise Robustness (false positive rate under observation noise)
  E. Action Differentiation (tighten vs regrasp)

Usage:
  conda activate snn-demo
  python main.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import (
    ToyGraspEnvironment, EnvConfig,
    run_baseline_experiments,
    run_delay_experiment,
    run_slip_magnitude_sweep,
    run_noise_experiment,
    run_action_differentiation_experiment,
)
from event_detector import EventDetector
from concept_layer import ConceptLayer
from rule_layer import RuleLayer
from snn_reflex import SNNReflexModule
from visualization import generate_all_plots, print_summary


def main():
    print("=" * 70)
    print("  SNN Reflex Demo v2")
    print("  Event → Concept → Rule → Reflex Action")
    print("  Multi-Experiment Validation Suite")
    print("=" * 70)

    np.random.seed(42)

    # Base config
    base_cfg = EnvConfig(
        dt=0.01, total_time=5.0,
        slip_time=2.5, slip_duration=0.4,
        slip_magnitude=8.0,
        gripper_close_force=5.0,
        gripper_tighten_force=18.0,
        gripper_adaptive_max=18.0,
        object_mass=0.5,
    )

    # Shared modules
    ed = EventDetector(base_cfg)
    cl = ConceptLayer()
    rl = RuleLayer()
    snn = SNNReflexModule(base_cfg)

    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'v2')

    # ==========================================
    # Experiment A: Baseline (fixed)
    # ==========================================
    print("\n" + "─" * 50)
    print("  Experiment A: Baseline Comparison")
    print("  (No Reflex now detects events — no more blank graphs)")
    print("─" * 50)

    baseline_results, ed, cl, rl, snn = run_baseline_experiments(base_cfg, ed, cl, rl, snn)

    # ==========================================
    # Experiment B: Rule Delay vs SNN
    # ==========================================
    print("\n" + "─" * 50)
    print("  Experiment B: Rule Delay (80ms) vs SNN Reflex")
    print("─" * 50)
    delay_results, _, _, _, delay_snn = run_delay_experiment(base_cfg, delay_ms=80.0)

    # ==========================================
    # Experiment C: Slip Magnitude Sweep
    # ==========================================
    print("\n" + "─" * 50)
    print("  Experiment C: Slip Magnitude Sweep [4, 6, 8, 10, 12]")
    print("─" * 50)
    sweep_results = run_slip_magnitude_sweep(base_cfg)

    # ==========================================
    # Experiment D: Noise Robustness
    # ==========================================
    print("\n" + "─" * 50)
    print("  Experiment D: Noise Robustness [0, 0.01, 0.02, 0.05]")
    print("─" * 50)
    noise_results = run_noise_experiment(base_cfg, noise_levels=[0.0, 0.01, 0.02, 0.05])

    # ==========================================
    # Experiment E: Action Differentiation
    # ==========================================
    print("\n" + "─" * 50)
    print("  Experiment E: Action Differentiation (tighten vs regrasp)")
    print("─" * 50)
    action_results, action_snn = run_action_differentiation_experiment(base_cfg)

    # ==========================================
    # Generate all plots
    # ==========================================
    print(f"\n{'─' * 50}")
    print("  Generating all visualizations...")
    print("─" * 50)

    generate_all_plots(
        output_dir,
        baseline_results=baseline_results,
        event_detector=ed, concept_layer=cl, rule_layer=rl,
        snn_module=snn,
        delay_results=delay_results,
        sweep_results=sweep_results,
        noise_results=noise_results,
        action_results=action_results,
    )

    # ==========================================
    # Print summary
    # ==========================================
    print_summary(
        baseline_results=baseline_results,
        delay_results=delay_results,
        sweep_results=sweep_results,
        noise_results=noise_results,
        action_results=action_results,
        snn_module=snn,
    )

    print(f"\n  All plots saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    • {f} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
