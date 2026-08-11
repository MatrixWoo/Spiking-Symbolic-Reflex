"""
Toy Simulation Environment: Pick-and-Place with Slip Recovery v2

Key upgrades over v1:
- Always runs event/concept detection (including no_reflex mode)
- Rule delay mechanism (simulates planning latency)
- Adaptive grip force (continuous, not just binary 18N)
- Observation noise support
- Multiple slip magnitudes for robustness testing
- Action differentiation: tighten vs regrasp

State vector (10 dims):
  [gripper_x, gripper_z, gripper_width, object_x, object_z,
   object_vx, object_vz, contact, grasped, gripper_force]

Actions:
  0=mv_right, 1=mv_left, 2=mv_up, 3=mv_down
  4=close_gripper, 5=open_gripper
  6=tighten(reflex, binary), 7=regrasp(reflex)
  8=adaptive_tighten(reflex, continuous delta_force)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque


@dataclass
class EnvConfig:
    dt: float = 0.01
    total_time: float = 5.0
    slip_time: float = 2.5
    slip_duration: float = 0.4
    slip_magnitude: float = 8.0
    gripper_speed: float = 0.3
    gripper_close_force: float = 5.0
    gripper_tighten_force: float = 18.0   # max reflex force (binary mode)
    gripper_adaptive_max: float = 18.0    # max force for adaptive mode
    object_mass: float = 0.5
    gravity: float = 9.81
    friction_coeff: float = 0.55
    grasp_threshold: float = 0.04
    lift_height: float = 0.3
    # New in v2
    rule_delay_ms: float = 0.0            # rule reaction delay (ms)
    noise_std: float = 0.0                # observation noise std
    adaptive_grip: bool = False           # use continuous grip adjustment
    slip_magnitudes: List[float] = field(default_factory=lambda: [4, 6, 8, 10, 12])


class ToyGraspEnvironment:
    """2D pick-and-place with slip disturbance."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.gripper_x = 0.0
        self.gripper_z = 0.25
        self.gripper_width = 0.12
        self.gripper_force = 0.0
        self.object_x = 0.0
        self.object_z = 0.03
        self.object_vx = 0.0
        self.object_vz = 0.0
        self.contact = False
        self.grasped = False
        self.was_grasped = False
        self.object_dropped = False
        self.recovery_success = False
        self.time = 0.0
        self.slip_active = False
        self.slip_progress = 0.0
        self.reflex_applied = False
        self.reflex_type = None
        self.reflex_time = None
        self.grip_force_used = 0.0         # actual force used (for efficiency metric)
        self.force_integral = 0.0          # cumulative force * dt (energy proxy)
        self.history = []
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.gripper_x, self.gripper_z, self.gripper_width,
            self.object_x, self.object_z,
            self.object_vx, self.object_vz,
            float(self.contact), float(self.grasped),
            self.gripper_force
        ])

    def _add_noise(self, state: np.ndarray) -> np.ndarray:
        """Add Gaussian observation noise."""
        if self.cfg.noise_std > 0:
            noise = np.random.randn(*state.shape) * self.cfg.noise_std
            return state + noise
        return state

    def _record_history(self):
        self.force_integral += self.gripper_force * self.cfg.dt
        self.history.append({
            'time': self.time,
            'gripper_x': self.gripper_x, 'gripper_z': self.gripper_z,
            'gripper_width': self.gripper_width, 'gripper_force': self.gripper_force,
            'object_x': self.object_x, 'object_z': self.object_z,
            'object_vx': self.object_vx, 'object_vz': self.object_vz,
            'contact': self.contact, 'grasped': self.grasped,
            'object_dropped': self.object_dropped,
            'slip_active': self.slip_active,
            'relative_distance': self._relative_distance(),
            'reflex_applied': self.reflex_applied,
            'reflex_type': self.reflex_type,
            'force_integral': self.force_integral,
        })

    def _relative_distance(self):
        dx = self.gripper_x - self.object_x
        dz = self.gripper_z - self.object_z
        return np.sqrt(dx**2 + dz**2)

    def step(self, action: int, reflex_action: Optional[int] = None,
             grip_force_delta: float = 0.0):
        """
        Advance simulation by one timestep.

        Args:
            action: Planned action (0-5)
            reflex_action: Reflex override (6=tighten, 7=regrasp, 8=adaptive)
            grip_force_delta: Continuous force adjustment (for adaptive mode)
        """
        cfg = self.cfg
        dt = cfg.dt
        self.time += dt

        # --- Planned action ---
        if action == 0:
            self.gripper_x += cfg.gripper_speed * dt
        elif action == 1:
            self.gripper_x -= cfg.gripper_speed * dt
        elif action == 2:
            self.gripper_z += cfg.gripper_speed * dt
        elif action == 3:
            self.gripper_z = max(self.gripper_z - cfg.gripper_speed * dt, 0.015)
        elif action == 4:
            self.gripper_width = max(self.gripper_width - 0.03, 0.0)
            if self.contact and self.gripper_width < cfg.grasp_threshold:
                self.grasped = True
                self.was_grasped = True
                self.gripper_force = cfg.gripper_close_force
        elif action == 5:
            self.gripper_width = min(self.gripper_width + 0.03, 0.12)
            if self.gripper_width > cfg.grasp_threshold:
                self.gripper_force = 0.0
                self.grasped = False

        # --- Adaptive grip force adjustment ---
        if grip_force_delta != 0.0 and self.grasped:
            new_force = self.gripper_force + grip_force_delta
            self.gripper_force = np.clip(new_force, cfg.gripper_close_force,
                                         cfg.gripper_adaptive_max)
            self.grip_force_used = self.gripper_force

        # --- Reflex action ---
        if reflex_action is not None and not self.reflex_applied:
            self.reflex_applied = True
            self.reflex_time = self.time
            if reflex_action == 6:  # tighten (binary)
                self.reflex_type = 'tighten'
                self.gripper_force = cfg.gripper_tighten_force
                self.gripper_width = max(self.gripper_width - 0.04, 0.0)
                if self.contact:
                    self.grasped = True
            elif reflex_action == 7:  # regrasp
                self.reflex_type = 'regrasp'
                self.gripper_width = 0.12
                self.gripper_force = 0.0
                self.grasped = False
            elif reflex_action == 8:  # adaptive tighten
                self.reflex_type = 'tighten_adaptive'
                self.gripper_force = self.gripper_force + grip_force_delta
                self.gripper_force = np.clip(self.gripper_force,
                                             cfg.gripper_close_force,
                                             cfg.gripper_adaptive_max)
                self.gripper_width = max(self.gripper_width - 0.02, 0.0)
                if self.contact:
                    self.grasped = True

        # --- Regrasp sequence ---
        if self.reflex_type == 'regrasp' and self.reflex_applied:
            elapsed = self.time - self.reflex_time
            if 0.08 < elapsed < 0.25:
                self.gripper_width = max(self.gripper_width - 0.05, 0.0)
                if self.contact and self.gripper_width < cfg.grasp_threshold:
                    self.grasped = True
                    self.gripper_force = cfg.gripper_tighten_force

        # --- Contact check ---
        rel_dist = self._relative_distance()
        gripper_near_object = (abs(self.gripper_x - self.object_x) < 0.06 and
                               abs(self.gripper_z - self.object_z) < 0.06)
        self.contact = gripper_near_object and self.gripper_width < 0.10

        # --- Slip disturbance ---
        self.slip_active = False
        in_slip_window = (cfg.slip_time <= self.time <= cfg.slip_time + cfg.slip_duration)
        if in_slip_window and self.grasped:
            self.slip_active = True
            self.slip_progress = (self.time - cfg.slip_time) / cfg.slip_duration

            max_friction = self.gripper_force * cfg.friction_coeff
            slip_force = cfg.slip_magnitude * cfg.object_mass

            if slip_force > max_friction:
                net_force = slip_force - max_friction
                accel = net_force / cfg.object_mass
                self.object_vz -= accel * dt * 8.0
                self.object_vx += 0.12 * np.sin(self.slip_progress * np.pi) * dt * 15
                self.grasped = False  # temporarily lost grip

        # --- Physics ---
        surface_z = 0.03

        if self.grasped and not self.slip_active:
            self.object_x += (self.gripper_x - self.object_x) * 0.6
            self.object_z += (self.gripper_z - self.object_z) * 0.6
            self.object_vx = (self.gripper_x - self.object_x) * 5.0
            self.object_vz = (self.gripper_z - self.object_z) * 5.0
        else:
            self.object_vz -= cfg.gravity * dt
            self.object_x += self.object_vx * dt
            self.object_z += self.object_vz * dt

            if self.object_z <= surface_z:
                self.object_z = surface_z
                if self.object_vz < -1.0:
                    self.object_vz *= -0.15
                else:
                    self.object_vz = 0.0
                self.object_vx *= 0.7

            if self.object_z > surface_z:
                self.object_vx *= 0.995

        # --- Drop detection ---
        if self.was_grasped and self.object_z <= surface_z + 0.005 and self.gripper_z > surface_z + 0.05:
            if not self.slip_active or self.time > cfg.slip_time + cfg.slip_duration + 0.15:
                self.object_dropped = True
                self.grasped = False

        # --- Recovery check ---
        if self.time > cfg.slip_time + cfg.slip_duration + 0.3:
            if self.grasped and self.object_z > surface_z + 0.02 and not self.object_dropped:
                self.recovery_success = True

        self._record_history()
        return self._get_state()


# ============================================================
# Episode Runner
# ============================================================

def run_episode(env: ToyGraspEnvironment, reflex_mode: str = 'none',
                snn_module=None, event_detector=None, concept_layer=None,
                rule_layer=None, rule_delay_ms: float = 0.0,
                adaptive_grip: bool = False):
    """
    Run a full pick-and-place episode.

    Key fix (v2): Always run event/concept detection even in 'none' mode.
    Only gate the reflex action application.

    Args:
        reflex_mode: 'none', 'rule', 'snn', 'snn_rule', 'adaptive'
        rule_delay_ms: delay before rule can fire (simulates planning latency)
        adaptive_grip: use continuous SNN-driven grip adjustment
    """
    env.reset()
    cfg = env.cfg
    prev_state = env._get_state()
    total_steps = int(cfg.total_time / cfg.dt)

    # Rule delay buffer: queue of (fire_time, action)
    pending_rule_action: deque = deque()
    delay_steps = int(rule_delay_ms / 1000.0 / cfg.dt)

    for step_i in range(total_steps):
        t = step_i * cfg.dt

        # Episode phases
        if t < 0.6:
            action = 3   # move down (approach)
        elif t < 1.2:
            action = 4   # close gripper (grasp)
        elif t < 2.3:
            action = 2   # lift
        elif t < 4.0:
            action = 0   # move right (transport)
        else:
            action = 3   # lower (place)

        # --- Always run event/concept detection ---
        reflex_action = None
        grip_delta = 0.0
        reflex_enabled = env.was_grasped and env.gripper_z > 0.08

        if event_detector is not None:
            curr_state = env._get_state()
            # Add observation noise
            noisy_state = env._add_noise(curr_state)
            noisy_prev = env._add_noise(prev_state)
            events = event_detector.detect(noisy_state, noisy_prev, env)

            if concept_layer is not None:
                concepts = concept_layer.activate(events, env)

                # --- Rule layer (with optional delay) ---
                if reflex_mode in ('rule', 'snn_rule', 'adaptive') and rule_layer is not None:
                    immediate_action = rule_layer.decide(concepts, env)
                    if immediate_action is not None and reflex_enabled:
                        if delay_steps > 0:
                            pending_rule_action.append((t + rule_delay_ms / 1000.0, immediate_action))
                        else:
                            reflex_action = immediate_action

                # Process delayed rule actions
                while pending_rule_action and pending_rule_action[0][0] <= t:
                    _, delayed_action = pending_rule_action.popleft()
                    if reflex_action is None:
                        reflex_action = delayed_action

            # --- SNN ---
            if reflex_mode in ('snn', 'snn_rule', 'adaptive') and snn_module is not None:
                snn_signal = snn_module.step(events, env.time)
                if snn_signal is not None:
                    grip_delta = snn_signal.get('force_delta', 0.0)
                    if snn_signal['confidence'] > 0.3 and reflex_action is None:
                        reflex_action = snn_signal['action']

            prev_state = curr_state

        # Gate: prevent reflex during approach/grasp
        if not reflex_enabled:
            reflex_action = None
            grip_delta = 0.0

        env.step(action, reflex_action, grip_force_delta=grip_delta)

    return env


# ============================================================
# Experiment Runners
# ============================================================

def run_baseline_experiments(config: Optional[EnvConfig] = None,
                             event_detector=None, concept_layer=None,
                             rule_layer=None, snn_module=None):
    """Original 3-way comparison: No Reflex, Rule, SNN+Rule."""
    from event_detector import EventDetector
    from concept_layer import ConceptLayer
    from rule_layer import RuleLayer
    from snn_reflex import SNNReflexModule

    cfg = config or EnvConfig()
    if event_detector is None:
        event_detector = EventDetector(cfg)
    if concept_layer is None:
        concept_layer = ConceptLayer()
    if rule_layer is None:
        rule_layer = RuleLayer()
    if snn_module is None:
        snn_module = SNNReflexModule(cfg)

    results = {}

    # 1. No Reflex (events detected but not acted upon)
    for mod in [event_detector, concept_layer, rule_layer]:
        mod.reset() if mod else None
    env1 = ToyGraspEnvironment(cfg)
    run_episode(env1, reflex_mode='none',
                event_detector=event_detector, concept_layer=concept_layer,
                rule_layer=rule_layer)
    results['no_reflex'] = env1

    # 2. Rule Reflex
    for mod in [event_detector, concept_layer, rule_layer, snn_module]:
        mod.reset() if mod else None
    env2 = ToyGraspEnvironment(cfg)
    run_episode(env2, reflex_mode='rule',
                event_detector=event_detector, concept_layer=concept_layer,
                rule_layer=rule_layer, rule_delay_ms=cfg.rule_delay_ms)
    results['rule_reflex'] = env2

    # 3. SNN + Rule
    for mod in [event_detector, concept_layer, rule_layer, snn_module]:
        mod.reset() if mod else None
    env3 = ToyGraspEnvironment(cfg)
    run_episode(env3, reflex_mode='snn_rule',
                event_detector=event_detector, concept_layer=concept_layer,
                rule_layer=rule_layer, snn_module=snn_module,
                rule_delay_ms=cfg.rule_delay_ms)
    results['snn_rule_reflex'] = env3

    return results, event_detector, concept_layer, rule_layer, snn_module


def run_delay_experiment(config: Optional[EnvConfig] = None,
                         delay_ms: float = 80.0):
    """
    Experiment A: Rule Delay vs SNN Reflex.

    Compares:
    - Rule with 80ms delay (simulating planning latency)
    - SNN reflex (10-20ms)
    - SNN + delayed Rule
    """
    from event_detector import EventDetector
    from concept_layer import ConceptLayer
    from rule_layer import RuleLayer
    from snn_reflex import SNNReflexModule

    cfg = config or EnvConfig()
    ed = EventDetector(cfg)
    cl = ConceptLayer()
    rl = RuleLayer()
    snn = SNNReflexModule(cfg)

    results = {}

    # 1. No Reflex
    for mod in [ed, cl, rl, snn]: mod.reset()
    env = ToyGraspEnvironment(cfg)
    run_episode(env, 'none', event_detector=ed, concept_layer=cl, rule_layer=rl)
    results['no_reflex'] = env

    # 2. Rule with delay
    for mod in [ed, cl, rl, snn]: mod.reset()
    env = ToyGraspEnvironment(cfg)
    run_episode(env, 'rule', event_detector=ed, concept_layer=cl, rule_layer=rl,
                rule_delay_ms=delay_ms)
    results['rule_delayed'] = env

    # 3. SNN only (no rule)
    for mod in [ed, cl, rl, snn]: mod.reset()
    env = ToyGraspEnvironment(cfg)
    run_episode(env, 'snn', event_detector=ed, concept_layer=cl, rule_layer=rl,
                snn_module=snn)
    results['snn_only'] = env

    # 4. SNN + delayed Rule (SNN fires first, Rule confirms later)
    for mod in [ed, cl, rl, snn]: mod.reset()
    env = ToyGraspEnvironment(cfg)
    run_episode(env, 'snn_rule', event_detector=ed, concept_layer=cl, rule_layer=rl,
                snn_module=snn, rule_delay_ms=delay_ms)
    results['snn_plus_delayed_rule'] = env

    return results, ed, cl, rl, snn


def run_slip_magnitude_sweep(config: Optional[EnvConfig] = None):
    """
    Experiment B: Sweep across slip magnitudes.

    Tests recovery rate at slip_magnitude = [4, 6, 8, 10, 12]
    for No Reflex, Rule, and SNN+Rule.
    """
    from event_detector import EventDetector
    from concept_layer import ConceptLayer
    from rule_layer import RuleLayer
    from snn_reflex import SNNReflexModule

    base_cfg = config or EnvConfig()
    magnitudes = base_cfg.slip_magnitudes
    ed = EventDetector(base_cfg)
    cl = ConceptLayer()
    rl = RuleLayer()
    snn = SNNReflexModule(base_cfg)

    sweep_results = {m: {} for m in magnitudes}

    for mag in magnitudes:
        cfg = EnvConfig(
            dt=base_cfg.dt, total_time=base_cfg.total_time,
            slip_time=base_cfg.slip_time, slip_duration=base_cfg.slip_duration,
            slip_magnitude=mag,
            gripper_close_force=base_cfg.gripper_close_force,
            gripper_tighten_force=base_cfg.gripper_tighten_force,
            object_mass=base_cfg.object_mass,
        )

        for mode, label in [('none', 'no_reflex'), ('rule', 'rule_reflex'),
                            ('snn_rule', 'snn_rule_reflex')]:
            for mod in [ed, cl, rl, snn]: mod.reset()
            env = ToyGraspEnvironment(cfg)
            run_episode(env, mode, event_detector=ed, concept_layer=cl,
                        rule_layer=rl, snn_module=snn)
            sweep_results[mag][label] = {
                'recovered': env.recovery_success,
                'dropped': env.object_dropped,
                'reflex_time': env.reflex_time,
                'reflex_type': env.reflex_type,
                'final_obj_z': env.object_z,
                'max_grip_force': max(h['gripper_force'] for h in env.history) if env.history else 0,
            }

    return sweep_results


def run_noise_experiment(config: Optional[EnvConfig] = None,
                         noise_levels: List[float] = None):
    """
    Experiment C: Noise robustness.

    Tests false-positive rate and recovery under observation noise.
    """
    from event_detector import EventDetector
    from concept_layer import ConceptLayer
    from rule_layer import RuleLayer
    from snn_reflex import SNNReflexModule

    cfg = config or EnvConfig()
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05]
    ed = EventDetector(cfg)
    cl = ConceptLayer()
    rl = RuleLayer()
    snn = SNNReflexModule(cfg)

    noise_results = {}

    for noise_std in noise_levels:
        cfg_noise = EnvConfig(
            dt=cfg.dt, total_time=cfg.total_time,
            slip_time=cfg.slip_time, slip_duration=cfg.slip_duration,
            slip_magnitude=cfg.slip_magnitude,
            gripper_close_force=cfg.gripper_close_force,
            gripper_tighten_force=cfg.gripper_tighten_force,
            noise_std=noise_std,
        )

        noise_results[noise_std] = {}

        for mode, label in [('rule', 'rule_reflex'), ('snn', 'snn_only')]:
            for mod in [ed, cl, rl, snn]: mod.reset()
            env = ToyGraspEnvironment(cfg_noise)
            run_episode(env, mode, event_detector=ed, concept_layer=cl,
                        rule_layer=rl, snn_module=snn)

            # Count false positives (reflex triggered before slip)
            false_positive = (env.reflex_time is not None and
                              env.reflex_time < cfg.slip_time - 0.1)

            noise_results[noise_std][label] = {
                'recovered': env.recovery_success,
                'false_positive': false_positive,
                'reflex_time': env.reflex_time,
                'dropped': env.object_dropped,
            }

    return noise_results


def run_action_differentiation_experiment(config: Optional[EnvConfig] = None):
    """
    Experiment D: Action differentiation (tighten vs regrasp).

    Two failure modes:
    - Slip while contact exists → should tighten
    - Object already falling / contact lost → should regrasp

    Test if SNN can differentiate.
    """
    from event_detector import EventDetector
    from concept_layer import ConceptLayer
    from rule_layer import RuleLayer
    from snn_reflex import SNNReflexModule

    cfg = config or EnvConfig()
    ed = EventDetector(cfg)
    cl = ConceptLayer()
    rl = RuleLayer()
    snn = SNNReflexModule(cfg)

    results = {}

    # Scenario 1: Slip with contact preserved (should tighten)
    for mod in [ed, cl, rl, snn]: mod.reset()
    env1 = ToyGraspEnvironment(cfg)
    run_episode(env1, 'snn', event_detector=ed, concept_layer=cl,
                rule_layer=rl, snn_module=snn)
    results['slip_with_contact'] = {
        'env': env1,
        'correct_action': 'tighten',
        'actual_action': env1.reflex_type,
        'correct': env1.reflex_type == 'tighten' or env1.reflex_type == 'tighten_adaptive',
    }

    # Scenario 2: Severe slip causing contact loss (should regrasp)
    cfg_severe = EnvConfig(
        dt=cfg.dt, total_time=cfg.total_time,
        slip_time=cfg.slip_time, slip_duration=cfg.slip_duration,
        slip_magnitude=14.0,  # very severe
        gripper_close_force=cfg.gripper_close_force,
        gripper_tighten_force=cfg.gripper_tighten_force,
    )
    for mod in [ed, cl, rl, snn]: mod.reset()
    env2 = ToyGraspEnvironment(cfg_severe)
    run_episode(env2, 'snn', event_detector=ed, concept_layer=cl,
                rule_layer=rl, snn_module=snn)
    results['slip_with_contact_loss'] = {
        'env': env2,
        'correct_action': 'regrasp',
        'actual_action': env2.reflex_type,
        'correct': env2.reflex_type == 'regrasp',
    }

    return results, snn
