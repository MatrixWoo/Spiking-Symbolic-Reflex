#!/usr/bin/env python3
"""
Publication-quality architecture figure for the paper/proposal.

"Overall Architecture of Reflexive Neuro-Symbolic Spiking Control"

Layout:
  ┌──────────────┐
  │  Environment │◄──────────────────────────┐
  │ (sim/real)   │                          │
  └──────┬───────┘                          │
         │ state vector (10 dims)           │
         ▼                                  │
  ┌──────────────┐                          │
  │Event Detector│   analog signals [0,1]   │
  └──────┬───────┘                          │
         │                                  │
         ▼                                  │
  ┌──────────────┐                          │
  │Concept Layer │  neural → symbolic       │
  └───┬─────┬────┘                          │
      │     │ event spikes (direct)         │
      │     └──────────────┐                │
      ▼ concepts           ▼                │
  ┌─────────┐    ┌──────────────┐           │
  │  Rule   │    │ SNN Reflex   │           │
  │(symbolic)│   │  (spiking)   │           │
  │ 80ms    │    │  20ms        │           │
  └────┬────┘    └──────┬───────┘           │
       │                │                   │
       └───────┬────────┘                   │
               ▼                            │
       ┌──────────────┐                     │
       │Reflex Action │                     │
       └──────┬───────┘                     │
              │                             │
              └─────────────────────────────┘
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 200, 'savefig.dpi': 200,
})

# Colors
C_OBS = '#5DADE2'; C_EVENT = '#F4D03F'; C_CONCEPT = '#F39C12'
C_RULE = '#3498DB'; C_SNN = '#2ECC71'; C_ACTION = '#E74C3C'
C_ENV = '#7F8C8D'; C_ARROW = '#2C3E50'; C_BG = '#FAFAFA'

def box(ax, x, y, w, h, title, subtitle, color, title_size=11):
    """Draw a rounded box with title + subtitle."""
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25',
                       edgecolor='#2C3E50', facecolor=color, linewidth=2.0, alpha=0.92)
    ax.add_patch(r)
    ax.text(x + w/2, y + h*0.58, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='#1a1a2e')
    ax.text(x + w/2, y + h*0.22, subtitle, ha='center', va='center',
            fontsize=title_size-3, color='#444', style='italic')

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.2, z=1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def carrow(ax, x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=2.2, z=1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))

def main():
    W, H = 18, 10
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)

    # Title
    ax.text(W/2, 9.55, 'Overall Architecture of Reflexive Neuro-Symbolic Spiking Control',
            ha='center', fontsize=20, fontweight='bold')
    ax.text(W/2, 9.10, 'Event → Concept → Rule / SNN → Reflex Action → Environment Feedback',
            ha='center', fontsize=12, style='italic', color='#666')
    ax.plot([2, W-2], [8.95, 8.95], color='#CCC', lw=1)

    # ---- Coordinates ----
    #  Environment:  x=5.5,  y=7.8, w=3.2, h=1.0
    #  Observation:  x=5.5,  y=7.8, w=3.2, h=1.0  (same box)
    #  Event Det:    x=5.8,  y=6.0, w=2.6, h=0.9
    #  Concept:      x=6.0,  y=4.5, w=2.2, h=0.9
    #  Rule (left):  x=2.0,  y=2.5, w=3.8, h=1.3
    #  SNN  (right): x=8.2,  y=2.5, w=3.8, h=1.3
    #  Action:       x=5.0,  y=0.9, w=4.0, h=1.0

    CX = 7.1  # center x of main column

    # --- Environment ---
    box(ax, 5.5, 7.8, 3.2, 1.0, 'Environment', 'gripper + object physics\nslip disturbance at t = 2.5s', C_OBS, 12)

    # Dashed boundary for feedback target
    r = FancyBboxPatch((5.3, 7.55), 3.6, 1.5, boxstyle='round,pad=0.25',
                       edgecolor=C_ENV, facecolor='none', linewidth=2.0, linestyle='--', alpha=0.6)
    ax.add_patch(r)

    # --- Observation label (left of Environment) ---
    ax.text(5.2, 8.3, 'Robot\nObservation', ha='center', fontsize=10,
            fontweight='bold', color=C_OBS, rotation=0)
    ax.annotate('', xy=(5.5, 8.3), xytext=(5.45, 8.3),
                arrowprops=dict(arrowstyle='->', color=C_OBS, lw=2))

    # --- Event Detector ---
    box(ax, 5.8, 6.0, 2.6, 0.9, 'Event Detector',
        '5 analog signals: distance_inc,\nvelocity_anomaly, contact_loss, …', C_EVENT)

    # --- Concept Layer ---
    box(ax, 6.0, 4.5, 2.2, 0.9, 'Concept Layer',
        '4 concepts: slip, grasp_unstable,\nobject_falling, recovery_needed', C_CONCEPT)

    # --- Rule Layer ---
    box(ax, 2.0, 2.5, 3.8, 1.3, 'Rule Layer (Symbolic)',
        'IF slip THEN tighten\nIF falling THEN regrasp\n4 fuzzy rules · 80ms planning delay', C_RULE)

    # --- SNN Reflex ---
    box(ax, 8.2, 2.5, 3.8, 1.3, 'SNN Reflex (Spiking)',
        '5→10→3 LIF network\n305 spikes · 20ms latency\ncontinuous force output', C_SNN)

    # --- Reflex Action ---
    box(ax, 5.0, 0.9, 4.0, 1.0, 'Reflex Action\n→ grip_force, gripper_width',
        'action=6 tighten(18N) | action=8 adaptive(5–18N) | action=7 regrasp', C_ACTION, 12)

    # ---- Arrows (main column) ----
    # Env → Event
    arrow(ax, CX, 7.78, CX, 6.92, lw=2.5)
    ax.text(CX+0.5, 7.25, 'state vector\n(10 dims)', fontsize=8, color='#555', va='center')

    # Event → Concept
    arrow(ax, CX, 5.98, CX, 5.42, lw=2.5)
    ax.text(CX+0.5, 5.65, '5 events\n[0,1]', fontsize=8, color='#555', va='center')

    # Concept → Rule (left branch)
    arrow(ax, 6.8, 4.48, 3.6, 3.82, lw=2.2)
    ax.text(4.6, 3.9, 'concepts', fontsize=9, color='#1a1a2e', rotation=32)

    # Concept → SNN (right branch, direct events)
    carrow(ax, 7.4, 4.48, 9.7, 3.82, rad=-0.25, lw=2.2, color='#27AE60')
    ax.text(8.8, 4.25, 'event spikes\n(direct path)', fontsize=8, color='#27AE60')

    # Rule → Action
    arrow(ax, 3.9, 2.48, 6.4, 1.92, lw=2.0)
    ax.text(5.0, 2.1, 'symbolic', fontsize=8, color='#1a1a2e')

    # SNN → Action
    arrow(ax, 10.1, 2.48, 7.6, 1.92, lw=2.0, color='#27AE60')
    ax.text(8.8, 2.1, 'spike-driven\n20ms', fontsize=8, color='#27AE60')

    # ---- Feedback loop (right side) ----
    carrow(ax, 7.0, 0.88, 7.0, 7.78, rad=-0.55, lw=2.8, color=C_ENV)
    ax.annotate('Environment Feedback\n\ngrip_force (+5~13N)\ngripper_width (tighten/release)\n→ object_z, contact, velocity',
                xy=(14.5, 4.5), fontsize=9, color=C_ENV,
                ha='center', va='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=C_ENV, alpha=0.85))

    # ---- Callout boxes ----
    # SNN latency callout
    ax.annotate('Low-latency bypass\n10-30ms vs 80ms rule',
                xy=(10.5, 3.15), xytext=(14.0, 3.6), fontsize=10,
                color='#27AE60', ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.8,
                                connectionstyle='arc3,rad=-0.15'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#27AE60', alpha=0.85))

    # Concept callout
    ax.annotate('Neural → Symbolic bridge\nEvent-to-concept mapping\ntemporal smoothing (EMA)',
                xy=(8.5, 4.95), xytext=(12.2, 5.8), fontsize=10,
                color=C_CONCEPT, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_CONCEPT, lw=1.8,
                                connectionstyle='arc3,rad=-0.1'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C_CONCEPT, alpha=0.85))

    # Rule callout
    ax.annotate('Interpretable\nsymbolic reasoning\n(what + why)',
                xy=(1.0, 2.8), xytext=(-0.5, 3.3), fontsize=10,
                color=C_RULE, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_RULE, lw=1.8,
                                connectionstyle='arc3,rad=0.15'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C_RULE, alpha=0.85))

    # ---- Legend ----
    legend_el = [
        mpatches.Patch(facecolor=C_OBS, edgecolor='#2C3E50', label='Observation / Environment'),
        mpatches.Patch(facecolor=C_EVENT, edgecolor='#2C3E50', label='Event Detector'),
        mpatches.Patch(facecolor=C_CONCEPT, edgecolor='#2C3E50', label='Concept Layer'),
        mpatches.Patch(facecolor=C_RULE, edgecolor='#2C3E50', label='Rule Layer (slow, symbolic)'),
        mpatches.Patch(facecolor=C_SNN, edgecolor='#2C3E50', label='SNN Reflex (fast, spiking)'),
        mpatches.Patch(facecolor=C_ACTION, edgecolor='#2C3E50', label='Reflex Action'),
    ]
    ax.legend(handles=legend_el, loc='lower left', fontsize=9, ncol=3,
              framealpha=0.9, edgecolor='#CCC', bbox_to_anchor=(-0.05, -0.08))

    # ---- Save ----
    out = os.path.join(os.path.dirname(__file__), 'output', 'architecture_overview.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
