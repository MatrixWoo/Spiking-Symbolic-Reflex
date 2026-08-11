"""
Visualization v2: generates plots for all experiments.

Figures:
1. Baseline 3-way comparison (fixed x-axis, no_reflex now has data)
2. Delay experiment (Rule vs SNN latency)
3. Slip magnitude sweep
4. Noise robustness
5. Action differentiation
6. SNN spike raster with differentiated output
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import os

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def _extract_history(env):
    if not env.history:
        return None
    keys = env.history[0].keys()
    return {k: np.array([step[k] for step in env.history]) for k in keys}


# ============================================================
# Figure 1: Baseline Comparison (Fixed)
# ============================================================

def plot_baseline_comparison(results: Dict, event_detector, concept_layer,
                             rule_layer, snn_module, save_path: str):
    """3-column comparison: No Reflex, Rule, SNN+Rule. Fixed x-axis 0-5s."""
    labels = {
        'no_reflex': 'No Reflex\n(detects but no action)',
        'rule_reflex': 'Rule Reflex\n(30ms, 18N binary)',
        'snn_rule_reflex': 'SNN + Rule Reflex\n(305 spikes, adaptive)',
    }
    colors = {'no_reflex': '#e74c3c', 'rule_reflex': '#3498db',
              'snn_rule_reflex': '#2ecc71'}

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    for col, (key, label) in enumerate(labels.items()):
        if key not in results:
            continue
        env = results[key]
        data = _extract_history(env)
        if data is None:
            continue
        time = data['time']
        slip_t, slip_end = env.cfg.slip_time, env.cfg.slip_time + env.cfg.slip_duration

        # Row 1: Position
        ax = fig.add_subplot(gs[0, col])
        ax.plot(time, data['object_z'], color=colors[key], linewidth=2, label='Object Z')
        ax.plot(time, data['gripper_z'], '--', color='gray', linewidth=1, alpha=0.7, label='Gripper Z')
        ax.axvspan(slip_t, slip_end, alpha=0.15, color='orange', label='Slip')
        ax.axhline(y=0.03, color='gray', linestyle=':', alpha=0.5)
        if env.reflex_time:
            ax.axvline(x=env.reflex_time, color='green', linestyle='--', linewidth=1.5)
            ax.annotate(env.reflex_type or '', xy=(env.reflex_time, 0.05), fontsize=7, color='green')
        drop_mask = data['object_dropped'].astype(bool)
        if np.any(drop_mask):
            ax.axvline(x=time[drop_mask][0], color='red', linestyle=':', linewidth=1.5)
            ax.annotate('DROPPED', xy=(time[drop_mask][0], 0.28), fontsize=8, color='red', fontweight='bold')
        ax.set_title(label, color=colors[key], fontweight='bold', fontsize=10)
        ax.set_ylabel('Z (m)')
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.02, None)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 2: Events
        ax = fig.add_subplot(gs[1, col])
        if event_detector and event_detector.event_history:
            etime = np.array([e['time'] for e in event_detector.event_history])
            ev_colors = {'distance_increase': '#e74c3c', 'velocity_anomaly': '#e67e22',
                         'contact_loss': '#2ecc71', 'slip_risk': '#3498db',
                         'grasp_unstable': '#9b59b6'}
            for ev_name, ev_color in ev_colors.items():
                vals = np.array([e.get(ev_name, 0) for e in event_detector.event_history])
                ax.plot(etime, vals, color=ev_color, linewidth=1, label=ev_name, alpha=0.8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvspan(slip_t, slip_end, alpha=0.1, color='orange')
        ax.set_ylabel('Event Activation')
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_title('Events (all modes detect)', fontsize=9)

        # Row 3: Concepts
        ax = fig.add_subplot(gs[2, col])
        if concept_layer and concept_layer.concept_history:
            ctime = np.array([c['time'] for c in concept_layer.concept_history])
            c_colors = {'slip': '#e74c3c', 'grasp_unstable': '#e67e22',
                        'object_falling': '#c0392b', 'recovery_needed': '#8e44ad'}
            for cp_name, cp_color in c_colors.items():
                vals = np.array([c.get(cp_name, 0) for c in concept_layer.concept_history])
                ax.plot(ctime, vals, color=cp_color, linewidth=1.5, label=cp_name, alpha=0.85)
            ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
        ax.axvspan(slip_t, slip_end, alpha=0.1, color='orange')
        ax.set_ylabel('Concept Activation')
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 4: Reflex signal
        ax = fig.add_subplot(gs[3, col])
        if key != 'no_reflex' and rule_layer and rule_layer.rule_history:
            rtime = np.array([r['time'] for r in rule_layer.rule_history])
            confs = np.array([r['confidence'] for r in rule_layer.rule_history])
            ax.fill_between(rtime, 0, confs, alpha=0.25, color='blue', label='Rule confidence')
            fired = [(r['time'], r['rule']) for r in rule_layer.rule_history if r['rule']]
            if fired:
                for ft, fr in fired:
                    ax.axvline(x=ft, color='purple', linestyle='-', alpha=0.3, linewidth=1)
        if key == 'snn_rule_reflex' and snn_module and snn_module.snn_response_history:
            stime = np.array([s['time'] for s in snn_module.snn_response_history])
            tc = np.array([s['tighten_confidence'] for s in snn_module.snn_response_history])
            rc = np.array([s['regrasp_confidence'] for s in snn_module.snn_response_history])
            ax.plot(stime, tc, 'r-', linewidth=1, alpha=0.6, label='SNN tighten')
            ax.plot(stime, rc, 'g-', linewidth=1, alpha=0.6, label='SNN regrasp')
        ax.axvspan(slip_t, slip_end, alpha=0.1, color='orange', label='Slip')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Baseline Comparison: Event → Concept → Rule → Reflex', fontsize=14, fontweight='bold')
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 2: Delay Experiment
# ============================================================

def plot_delay_experiment(results: Dict, save_path: str):
    """Rule with delay vs SNN reflex."""
    labels = {
        'no_reflex': 'No Reflex',
        'rule_delayed': f'Rule (delayed)',
        'snn_only': 'SNN Only',
        'snn_plus_delayed_rule': 'SNN + Delayed Rule',
    }
    colors = {'no_reflex': '#e74c3c', 'rule_delayed': '#e67e22',
              'snn_only': '#2ecc71', 'snn_plus_delayed_rule': '#3498db'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Experiment A: Rule Delay vs SNN Reflex', fontsize=14, fontweight='bold')

    # Top: Object Z position
    ax = axes[0, 0]
    for key, label in labels.items():
        if key not in results: continue
        env = results[key]
        d = _extract_history(env)
        if d is None: continue
        ax.plot(d['time'], d['object_z'], color=colors[key], linewidth=2, label=label)

        # Mark reflex
        if env.reflex_time:
            ax.axvline(x=env.reflex_time, color=colors[key], linestyle='--', linewidth=1, alpha=0.7)
            ax.annotate(f'{env.reflex_type}\n{env.reflex_time:.3f}s',
                       xy=(env.reflex_time, 0.05), fontsize=6, color=colors[key])

    ax.axvspan(env.cfg.slip_time, env.cfg.slip_time + env.cfg.slip_duration,
               alpha=0.1, color='orange')
    ax.axhline(y=0.03, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Object Z (m)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, 5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title('Object Height During Slip', fontsize=11)

    # Top-right: Reflex latency comparison
    ax = axes[0, 1]
    bar_labels = []
    bar_times = []
    bar_colors_list = []
    for key, label in labels.items():
        if key not in results or key == 'no_reflex': continue
        env = results[key]
        if env.reflex_time:
            latency = (env.reflex_time - env.cfg.slip_time) * 1000
            bar_labels.append(label)
            bar_times.append(latency)
            bar_colors_list.append(colors[key])
        else:
            bar_labels.append(label)
            bar_times.append(1000)  # cap for visualization
            bar_colors_list.append(colors[key])
    bars = ax.barh(bar_labels, bar_times, color=bar_colors_list, edgecolor='white')
    ax.set_xlabel('Latency after slip onset (ms)')
    ax.set_title('Reflex Latency', fontsize=11)
    for bar, val in zip(bars, bar_times):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms' if val < 1000 else 'N/A', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # Bottom: Recovery success
    ax = axes[1, 0]
    rec_labels = []
    rec_vals = []
    rec_colors = []
    for key, label in labels.items():
        if key not in results: continue
        env = results[key]
        rec_labels.append(label)
        rec_vals.append(1.0 if env.recovery_success else 0.0)
        rec_colors.append(colors[key])
    bars = ax.bar(rec_labels, rec_vals, color=rec_colors, edgecolor='white')
    ax.set_title('Recovery Success', fontsize=11)
    ax.set_ylim(0, 1.2)
    for bar, val in zip(bars, rec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '✓' if val > 0.5 else '✗', ha='center', fontsize=14, fontweight='bold')

    # Bottom-right: Force comparison
    ax = axes[1, 1]
    force_labels = []
    force_vals = []
    force_colors = []
    for key, label in labels.items():
        if key not in results: continue
        env = results[key]
        d = _extract_history(env)
        if d is None: continue
        force_labels.append(label)
        force_vals.append(max(d['gripper_force']))
        force_colors.append(colors[key])
    bars = ax.bar(force_labels, force_vals, color=force_colors, edgecolor='white')
    ax.set_title('Max Grip Force (N)', fontsize=11)
    for bar, val in zip(bars, force_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 3: Slip Magnitude Sweep
# ============================================================

def plot_slip_sweep(sweep_results: Dict, save_path: str):
    """Recovery rate vs slip magnitude."""
    magnitudes = sorted(sweep_results.keys())
    modes = ['no_reflex', 'rule_reflex', 'snn_rule_reflex']
    mode_labels = ['No Reflex', 'Rule Reflex', 'SNN+Rule']
    mode_colors = ['#e74c3c', '#3498db', '#2ecc71']
    mode_markers = ['x', 'o', 's']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Experiment B: Slip Magnitude Sweep', fontsize=14, fontweight='bold')

    # Recovery rate
    ax = axes[0]
    for mode, label, color, marker in zip(modes, mode_labels, mode_colors, mode_markers):
        rec_rates = []
        for mag in magnitudes:
            if mode in sweep_results[mag]:
                rec_rates.append(1.0 if sweep_results[mag][mode]['recovered'] else 0.0)
            else:
                rec_rates.append(0.0)
        ax.plot(magnitudes, rec_rates, color=color, marker=marker, linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel('Slip Magnitude (N/kg)')
    ax.set_ylabel('Recovery Rate')
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Success')

    # Max grip force
    ax = axes[1]
    for mode, label, color, marker in zip(modes, mode_labels, mode_colors, mode_markers):
        forces = []
        for mag in magnitudes:
            if mode in sweep_results[mag]:
                forces.append(sweep_results[mag][mode].get('max_grip_force', 0))
            else:
                forces.append(0)
        ax.plot(magnitudes, forces, color=color, marker=marker, linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel('Slip Magnitude (N/kg)')
    ax.set_ylabel('Max Grip Force (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Force Response')

    # Reflex latency
    ax = axes[2]
    for mode, label, color, marker in zip(modes, mode_labels, mode_colors, mode_markers):
        latencies = []
        for mag in magnitudes:
            if mode in sweep_results[mag]:
                rt = sweep_results[mag][mode].get('reflex_time')
                if rt:
                    latencies.append((rt - sweep_results[mag][mode].get('slip_time', 2.5)) * 1000)
                else:
                    latencies.append(None)
            else:
                latencies.append(None)
        valid_mags = [m for m, l in zip(magnitudes, latencies) if l is not None]
        valid_lats = [l for l in latencies if l is not None]
        if valid_mags:
            ax.plot(valid_mags, valid_lats, color=color, marker=marker, linewidth=2,
                    markersize=8, label=label)
    ax.set_xlabel('Slip Magnitude (N/kg)')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Reflex Latency')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 4: Noise Robustness
# ============================================================

def plot_noise_experiment(noise_results: Dict, save_path: str):
    """False positive rate and recovery under noise."""
    noise_levels = sorted(noise_results.keys())
    modes = ['rule_reflex', 'snn_only']
    mode_labels = ['Rule Only', 'SNN Only']
    mode_colors = ['#3498db', '#2ecc71']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Experiment C: Noise Robustness', fontsize=14, fontweight='bold')

    # False positive rate
    ax = axes[0]
    for mode, label, color in zip(modes, mode_labels, mode_colors):
        fp_rates = []
        for nl in noise_levels:
            if mode in noise_results[nl]:
                fp_rates.append(1.0 if noise_results[nl][mode]['false_positive'] else 0.0)
            else:
                fp_rates.append(0.0)
        ax.plot(noise_levels, fp_rates, color=color, marker='o', linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel('Noise Std')
    ax.set_ylabel('False Positive Rate')
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('False Trigger Rate')

    # Recovery rate
    ax = axes[1]
    for mode, label, color in zip(modes, mode_labels, mode_colors):
        rec_rates = []
        for nl in noise_levels:
            if mode in noise_results[nl]:
                rec_rates.append(1.0 if noise_results[nl][mode]['recovered'] else 0.0)
            else:
                rec_rates.append(0.0)
        ax.plot(noise_levels, rec_rates, color=color, marker='s', linewidth=2,
                markersize=8, label=label)
    ax.set_xlabel('Noise Std')
    ax.set_ylabel('Recovery Rate')
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Under Noise')

    # Reflex latency under noise
    ax = axes[2]
    for mode, label, color in zip(modes, mode_labels, mode_colors):
        lats = []
        for nl in noise_levels:
            if mode in noise_results[nl]:
                rt = noise_results[nl][mode].get('reflex_time')
                if rt:
                    lats.append((rt - 2.5) * 1000)
                else:
                    lats.append(None)
            else:
                lats.append(None)
        valid_nl = [n for n, l in zip(noise_levels, lats) if l is not None]
        valid_lats = [l for l in lats if l is not None]
        if valid_nl:
            ax.plot(valid_nl, valid_lats, color=color, marker='^', linewidth=2,
                    markersize=8, label=label)
    ax.set_xlabel('Noise Std')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Latency Under Noise')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 5: Action Differentiation
# ============================================================

def plot_action_differentiation(action_results: Dict, save_path: str):
    """Tighten vs regrasp differentiation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Experiment D: Action Differentiation (tighten vs regrasp)', fontsize=14, fontweight='bold')

    scenario_labels = {
        'slip_with_contact': 'Slip + Contact\n(should tighten)',
        'slip_with_contact_loss': 'Slip + Contact Loss\n(should regrasp)',
    }
    ax_colors = ['#3498db', '#e74c3c']

    for idx, (key, label) in enumerate(scenario_labels.items()):
        if key not in action_results: continue
        info = action_results[key]
        env = info['env']
        data = _extract_history(env)
        if data is None: continue

        ax = axes[idx]
        time = data['time']
        ax.plot(time, data['object_z'], color=ax_colors[idx], linewidth=2, label='Object Z')
        ax.plot(time, data['gripper_z'], '--', color='gray', linewidth=1, alpha=0.5)
        ax.axvspan(env.cfg.slip_time, env.cfg.slip_time + env.cfg.slip_duration,
                   alpha=0.15, color='orange')

        if env.reflex_time:
            ax.axvline(x=env.reflex_time, color='green' if info['correct'] else 'red',
                      linestyle='--', linewidth=2)
            status = '✓ CORRECT' if info['correct'] else '✗ WRONG'
            color = 'green' if info['correct'] else 'red'
            ax.annotate(f'{env.reflex_type}\n{status}',
                       xy=(env.reflex_time, 0.08), fontsize=9, color=color, fontweight='bold')

        ax.set_title(f'{label}\nExpected: {info["correct_action"]} | Got: {info["actual_action"]}',
                    fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z (m)')
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 6: SNN Spike Raster (3 outputs)
# ============================================================

def plot_snn_raster(snn_module, save_path: str):
    """Spike raster with 3 output neurons."""
    if snn_module is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Hidden layer
    ax = axes[0]
    for i, neuron in enumerate(snn_module.hidden.neurons):
        spike_times = list(neuron.spike_history)
        if spike_times:
            ax.scatter(spike_times, [i] * len(spike_times), s=3, c='blue', alpha=0.7, marker='|')
    ax.set_ylabel('Hidden Neuron #')
    ax.set_title('SNN Spike Raster — Hidden Layer (10 neurons)', fontsize=11)
    ax.set_ylim(-1, 10)
    ax.grid(True, alpha=0.3, axis='y')

    # Output layer (3 neurons)
    ax = axes[1]
    out_colors = ['#e74c3c', '#2ecc71', '#f39c12']
    out_labels = ['tighten', 'regrasp', 'force_delta']
    for i, (neuron, color, label) in enumerate(zip(snn_module.output.neurons,
                                                     out_colors, out_labels)):
        spike_times = list(neuron.spike_history)
        if spike_times:
            ax.scatter(spike_times, [i] * len(spike_times), s=5, c=color, alpha=0.8,
                      marker='|', label=label)
    ax.set_ylabel('Output Neuron')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Output Layer — tighten | regrasp | force_delta', fontsize=11)
    ax.set_ylim(-1, 3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(out_labels)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 7: Force Efficiency
# ============================================================

def plot_force_efficiency(results: Dict, save_path: str):
    """Compare force integral (energy proxy) across methods."""
    labels_map = {
        'no_reflex': 'No Reflex', 'rule_reflex': 'Rule Reflex',
        'snn_rule_reflex': 'SNN+Rule', 'rule_delayed': 'Rule (delayed)',
        'snn_only': 'SNN Only', 'snn_plus_delayed_rule': 'SNN+Delayed Rule',
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#2ecc71', '#3498db']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Max force
    ax = axes[0]
    names, max_forces, force_integrals, recovered = [], [], [], []
    for key, label in labels_map.items():
        if key not in results: continue
        env = results[key]
        d = _extract_history(env)
        if d is None: continue
        names.append(label)
        max_forces.append(max(d['gripper_force']))
        force_integrals.append(env.force_integral)
        recovered.append(1.0 if env.recovery_success else 0.0)

    bar_colors = ['#2ecc71' if r > 0.5 else '#e74c3c' for r in recovered]
    bars = ax.bar(names, max_forces, color=bar_colors, edgecolor='white')
    ax.set_title('Max Grip Force (lower is safer)')
    ax.set_ylabel('Force (N)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for bar, val in zip(bars, max_forces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=9)

    # Force integral (energy proxy)
    ax = axes[1]
    bars = ax.bar(names, force_integrals, color=bar_colors, edgecolor='white')
    ax.set_title('Force Integral (N·s, energy proxy)')
    ax.set_ylabel('Force × Time (N·s)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for bar, val in zip(bars, force_integrals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=9)

    fig.suptitle('Force Efficiency: Who achieves recovery with less force?', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Master function
# ============================================================

def generate_all_plots(output_dir: str, baseline_results=None,
                       event_detector=None, concept_layer=None,
                       rule_layer=None, snn_module=None,
                       delay_results=None, sweep_results=None,
                       noise_results=None, action_results=None,
                       action_snn=None):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)

    if baseline_results:
        plot_baseline_comparison(baseline_results, event_detector, concept_layer,
                                 rule_layer, snn_module,
                                 os.path.join(output_dir, '01_baseline_comparison.png'))
        if snn_module:
            plot_snn_raster(snn_module, os.path.join(output_dir, 'snn_spike_raster.png'))
        plot_force_efficiency(baseline_results, os.path.join(output_dir, 'force_efficiency.png'))

    if delay_results:
        plot_delay_experiment(delay_results,
                             os.path.join(output_dir, '02_delay_experiment.png'))

    if sweep_results:
        plot_slip_sweep(sweep_results,
                       os.path.join(output_dir, '03_slip_sweep.png'))

    if noise_results:
        plot_noise_experiment(noise_results,
                             os.path.join(output_dir, '04_noise_robustness.png'))

    if action_results:
        plot_action_differentiation(action_results,
                                    os.path.join(output_dir, '05_action_differentiation.png'))


def print_summary(baseline_results=None, delay_results=None,
                  sweep_results=None, noise_results=None,
                  action_results=None, snn_module=None):
    """Print comprehensive experiment summary."""
    print("\n" + "=" * 70)
    print("  SNN Reflex Demo v2 — Comprehensive Results")
    print("=" * 70)

    if baseline_results:
        print("\n  [Baseline Comparison]")
        labels = {'no_reflex': 'No Reflex', 'rule_reflex': 'Rule Reflex',
                  'snn_rule_reflex': 'SNN+Rule'}
        for key, label in labels.items():
            if key not in baseline_results: continue
            env = baseline_results[key]
            status = '✓' if env.recovery_success else '✗'
            rt = f'{env.reflex_time:.3f}s' if env.reflex_time else 'N/A'
            print(f"    {label:15s} | Recovery: {status} | Reflex: {rt:>8s} | "
                  f"Max Force: {max(h['gripper_force'] for h in env.history):.1f}N | "
                  f"Force Int: {env.force_integral:.1f}N·s")

    if delay_results:
        print("\n  [Delay Experiment]")
        for key, env in delay_results.items():
            status = '✓' if env.recovery_success else '✗'
            rt = f'{env.reflex_time:.3f}s' if env.reflex_time else 'N/A'
            if env.reflex_time:
                lat = f'{(env.reflex_time - env.cfg.slip_time)*1000:.0f}ms'
            else:
                lat = 'N/A'
            print(f"    {key:25s} | Recovery: {status} | Reflex: {rt:>8s} | Latency: {lat:>6s}")

    if snn_module:
        stats = snn_module.get_sparsity_stats()
        print(f"\n  [SNN Statistics]")
        print(f"    Architecture: 5 → 10 → 3")
        print(f"    Total spikes: {stats['total_spikes']}")
        print(f"    Avg spikes/neuron: {stats['avg_spikes_per_neuron']:.1f}")

    if sweep_results:
        print(f"\n  [Slip Magnitude Sweep]")
        for mag in sorted(sweep_results.keys()):
            entries = []
            for mode in ['no_reflex', 'rule_reflex', 'snn_rule_reflex']:
                if mode in sweep_results[mag]:
                    r = sweep_results[mag][mode]
                    entries.append(f"{'✓' if r['recovered'] else '✗'}")
            print(f"    mag={mag:.0f}: NoReflex={' '.join(entries[:1])} Rule={' '.join(entries[1:2])} SNN={' '.join(entries[2:3])}")

    if action_results:
        print(f"\n  [Action Differentiation]")
        for key, info in action_results.items():
            status = '✓ CORRECT' if info['correct'] else '✗ WRONG'
            print(f"    {key:30s} | Expected: {info['correct_action']:8s} | Got: {str(info['actual_action']):10s} | {status}")

    print("\n" + "=" * 70)
