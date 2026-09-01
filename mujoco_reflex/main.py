"""
MuJoCo Reflex Demo — 12 Industrial Scenarios × 4 Recovery Modes.

Usage: conda activate mujoco_reflex && MUJOCO_GL=egl python main.py
"""

import os, sys, csv, numpy as np, cv2
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mujoco_env import MuJoCoReflexEnv, MuJoCoConfig, SCENARIOS
from event_detector import EventDetector
from concept_layer import ConceptLayer
from rule_layer import RuleLayer
from snn_reflex import SNNReflexModule

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'mujoco')


def run_episode(env, mode, ed, cl, rl, snn, writer, record_interval):
    cfg = env.cfg
    env.reset()
    prev_state = env._get_state()
    total_steps = int(cfg.total_time / cfg.dt)
    pending = deque()
    delay_steps = int(cfg.rule_delay_ms / 1000.0 / cfg.dt)

    for si in range(total_steps):
        t = si * cfg.dt
        action = 4  # close gripper
        reflex_action = None; grip_delta = 0.0
        reflex_enabled = env.was_grasped and t > cfg.slip_time - 0.1

        if mode != 'none':
            cs = env._get_state()
            events = ed.detect(env._add_noise(cs), env._add_noise(prev_state), env)
            if cl:
                concepts = cl.activate(events, env)
                if mode in ('rule', 'snn_rule') and rl:
                    imm = rl.decide(concepts, env)
                    if imm and reflex_enabled:
                        if delay_steps > 0:
                            pending.append((t + cfg.rule_delay_ms / 1000.0, imm))
                        else:
                            reflex_action = imm
                while pending and pending[0][0] <= t:
                    _, delayed = pending.popleft()
                    if reflex_action is None: reflex_action = delayed
            if mode in ('snn', 'snn_rule') and snn:
                sig = snn.step(events, env.time)
                if sig:
                    grip_delta = sig.get('force_delta', 0.0)
                    if sig['confidence'] > 0.3 and reflex_action is None:
                        reflex_action = sig['action']
            prev_state = cs

        if not reflex_enabled:
            reflex_action = None; grip_delta = 0.0

        env.step(action, reflex_action, grip_force_delta=grip_delta)
        if writer and si % record_interval == 0:
            writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))
    return env


def main():
    np.random.seed(42)
    ed = EventDetector(MuJoCoConfig())
    cl = ConceptLayer(); rl = RuleLayer(); snn = SNNReflexModule(MuJoCoConfig())
    os.makedirs(OUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps, ri = 30, max(1, int(1.0 / (0.002 * 30)))

    modes = [('none', 'none'), ('rule', 'rule'), ('snn', 'snn'), ('snn_rule', 'snn_rule')]
    csv_rows = []

    for scenario_name in SCENARIOS:
        for mode, _ in modes:
            print(f"  {scenario_name:25s} {mode:8s} ...", end=" ", flush=True)
            ed.reset(); cl.reset(); rl.reset(); snn.reset()
            env = MuJoCoReflexEnv(scenario=scenario_name)
            env.cfg.rule_delay_ms = 80.0

            vp = os.path.join(OUT_DIR, f'{scenario_name}_{mode}.mp4')
            w = cv2.VideoWriter(vp, fourcc, fps, (640, 480))
            run_episode(env, mode, ed, cl, rl, snn, w, ri)
            w.release()

            lat = (env.reflex_time - env.cfg.slip_time) * 1000 if env.reflex_time else ''
            row = {
                'scenario': scenario_name, 'seed': 42, 'mode': mode,
                'success': env.recovery_success, 'dropped': env.object_dropped,
                'false_trigger': (env.reflex_applied and env.reflex_time is not None
                                 and env.reflex_time < env.cfg.slip_time - 0.1),
                'reflex_applied': env.reflex_applied,
                'reflex_latency_ms': round(lat, 1) if lat != '' else '',
                'max_relative_z_m': max(h['relative_distance'] for h in env.history) if env.history else 0,
                'final_box_z_m': env.data.xpos[env.cube_id][2],
            }
            csv_rows.append(row)
            status = 'OK' if env.recovery_success else ('DROP' if env.object_dropped else '---')
            print(f"{status}  lat={lat}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  CSV → {csv_path} ({len(csv_rows)} rows)")

    # Print summary table
    print(f"\n{'Scenario':<25s} {'NoReflex':>8s} {'Rule':>8s} {'SNN':>8s} {'SNN+Rule':>8s}")
    print("-" * 57)
    for s in SCENARIOS:
        row_vals = {}
        for r in csv_rows:
            if r['scenario'] == s:
                row_vals[r['mode']] = 'OK' if r['success'] else ('DROP' if r['dropped'] else '---')
        print(f"{s:<25s} {row_vals.get('none','?'):>8s} {row_vals.get('rule','?'):>8s} "
              f"{row_vals.get('snn','?'):>8s} {row_vals.get('snn_rule','?'):>8s}")


if __name__ == '__main__':
    main()
