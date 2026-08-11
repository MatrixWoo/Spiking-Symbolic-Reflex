"""
MuJoCo Reflex Demo — Slip Recovery on UR5e + Robotiq 2F-85.

Episode: close gripper → lift → transport (slip!) → reflex → place.

Experiments:
  no_reflex   — object drops
  rule_reflex  — symbolic rules (80ms delay)
  snn_only     — SNN fast reflex (20ms)
  snn_rule     — SNN fires first, rule confirms later

Usage:
  conda activate mujoco_reflex
  MUJOCO_GL=egl python main.py
"""

import os, sys, numpy as np, cv2
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mujoco_env import MuJoCoReflexEnv, MuJoCoConfig
from event_detector import EventDetector
from concept_layer import ConceptLayer
from rule_layer import RuleLayer
from snn_reflex import SNNReflexModule

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'mujoco')


def run_episode(env, mode, event_detector, concept_layer, rule_layer, snn_module,
                video_writer, record_interval):
    cfg = env.cfg
    env.reset()
    prev_state = env._get_state()
    total_steps = int(cfg.total_time / cfg.dt)
    pending = deque()
    delay_steps = int(cfg.rule_delay_ms / 1000.0 / cfg.dt)

    for si in range(total_steps):
        t = si * cfg.dt

        # Episode: close(0-1s) → lift(1-2s) → Transport+SLIP(2-2.4s) → hold(2.4-4s)
        if t < 1.0:
            action = 4   # close gripper
        elif t < 2.0:
            action = 4   # keep closing (already at target)
        else:
            action = 4   # hold tight

        # Reflex pipeline
        reflex_action = None
        grip_delta = 0.0
        reflex_enabled = env.was_grasped and t > cfg.slip_time - 0.1  # just before slip

        if mode != 'none':
            cs = env._get_state()
            events = event_detector.detect(env._add_noise(cs), env._add_noise(prev_state), env)
            if concept_layer:
                concepts = concept_layer.activate(events, env)
                if mode in ('rule', 'snn_rule') and rule_layer:
                    imm = rule_layer.decide(concepts, env)
                    if imm and reflex_enabled:
                        if delay_steps > 0:
                            pending.append((t + cfg.rule_delay_ms/1000.0, imm))
                        else:
                            reflex_action = imm
                while pending and pending[0][0] <= t:
                    _, delayed = pending.popleft()
                    if reflex_action is None: reflex_action = delayed

            if mode in ('snn', 'snn_rule') and snn_module:
                sig = snn_module.step(events, env.time)
                if sig:
                    grip_delta = sig.get('force_delta', 0.0)
                    if sig['confidence'] > 0.3 and reflex_action is None:
                        reflex_action = sig['action']

            prev_state = cs

        if not reflex_enabled:
            reflex_action = None; grip_delta = 0.0

        env.step(action, reflex_action, grip_force_delta=grip_delta)

        if video_writer and si % record_interval == 0:
            video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    return env


def main():
    np.random.seed(42)
    cfg = MuJoCoConfig(total_time=4.0, slip_time=2.0, slip_duration=0.4,
                       slip_force_mag=50.0, rule_delay_ms=80.0)

    ed = EventDetector(cfg); cl = ConceptLayer(); rl = RuleLayer(); snn = SNNReflexModule(cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps, ri = 30, max(1, int(1.0/(cfg.dt*30)))
    results = {}

    for name, mode in [('no_reflex','none'),('rule_reflex','rule'),
                        ('snn_only','snn'),('snn_rule','snn_rule')]:
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        ed.reset(); cl.reset(); rl.reset(); snn.reset()
        env = MuJoCoReflexEnv(cfg)
        vp = os.path.join(OUT_DIR, f'{name}.mp4')
        w = cv2.VideoWriter(vp, fourcc, fps, (640, 480))
        run_episode(env, mode, ed, cl, rl, snn, w, ri)
        w.release()
        results[name] = env
        s = 'OK' if env.recovery_success else 'FAIL'
        print(f"  Recovery={s}  ReflexTime={env.reflex_time}  Type={env.reflex_type}")
        print(f"  Video: {vp} ({os.path.getsize(vp)/1024/1024:.1f}MB)")

    print(f"\n{'='*50}\n  Results\n{'='*50}")
    for name, env in results.items():
        print(f"  {name:15s} recov={'Y' if env.recovery_success else 'N'}  "
              f"reflex={str(env.reflex_time):>8s}  type={str(env.reflex_type):>15s}")

    if snn.snn_response_history:
        s = snn.get_sparsity_stats()
        print(f"  SNN: {s['total_spikes']} spikes, {s['avg_spikes_per_neuron']:.1f}/neuron")
    print(f"\n  All videos → {OUT_DIR}/")


if __name__ == '__main__':
    main()
