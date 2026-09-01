"""
MuJoCo Slip Recovery Environment v2 — 12 Industrial Scenarios.

Simple gripper (two sliding pads) + cube + gravity + disturbance.

Disturbance types:
  downward / lateral / rotational / vibration / collision / overload
"""

import mujoco
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MuJoCoConfig:
    dt: float = 0.002
    total_time: float = 4.0
    slip_time: float = 2.0
    slip_duration: float = 0.5
    slip_force_mag: float = 5.0
    gripper_close_ctrl: float = 0.025
    gripper_tighten_ctrl: float = 0.035
    rule_delay_ms: float = 80.0
    noise_std: float = 0.0
    scenario: str = "calibration_ramp"
    disturbance_type: str = "downward"
    friction_coeff: float = 1.0          # 1.0=default; <1=oil/water
    object_mass_override: float = 0.0    # 0=default; >0=overload


SCENE_XML = os.path.join(os.path.dirname(__file__), 'assets', 'simple_gripper.xml')

# ── 12 scenarios ──────────────────────────────────────────
SCENARIOS = {
    # Original 6
    "calibration_ramp":    dict(disturbance="downward",   force=5.0,  mass=0.1, friction=1.0, desc="Gradual pull ramp"),
    "transport_fast":      dict(disturbance="downward",   force=8.0,  mass=0.1, friction=1.0, desc="High-speed transport slip"),
    "lateral_impulse":     dict(disturbance="lateral",    force=10.0, mass=0.1, friction=1.0, desc="Side impact during transit"),
    "heavy_low_friction":  dict(disturbance="downward",   force=8.0,  mass=0.3, friction=0.6, desc="Heavy object, low friction"),
    "offset_grasp":        dict(disturbance="downward",   force=5.0,  mass=0.1, friction=1.0, desc="Off-center grasp"),
    "no_disturbance":      dict(disturbance="downward",   force=0.0,  mass=0.1, friction=1.0, desc="Control – no disturbance"),
    # Industrial 6
    "payload_overload":    dict(disturbance="overload",   force=12.0, mass=0.4, friction=1.0, desc="Object heavier than expected"),
    "surface_oil":         dict(disturbance="downward",   force=5.0,  mass=0.1, friction=0.3, desc="Oil/water on surface"),
    "collision_transport": dict(disturbance="collision",  force=15.0, mass=0.1, friction=1.0, desc="Sharp collision during move"),
    "emergency_stop":      dict(disturbance="downward",   force=10.0, mass=0.1, friction=1.0, desc="Inertia from emergency stop"),
    "rotational_slip":     dict(disturbance="rotational", force=8.0,  mass=0.1, friction=1.0, desc="Object twists in grip"),
    "vibration_sustained": dict(disturbance="vibration",  force=6.0,  mass=0.1, friction=1.0, desc="Conveyor/machine vibration"),
}


class MuJoCoReflexEnv:

    def __init__(self, config=None, scenario=None):
        if scenario and scenario in SCENARIOS:
            s = SCENARIOS[scenario]
            config = MuJoCoConfig(
                slip_force_mag=s["force"],
                disturbance_type=s["disturbance"],
                friction_coeff=s["friction"],
                object_mass_override=s["mass"],
                scenario=scenario,
            )
        self.cfg = config or MuJoCoConfig()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = None

        self.gripper_base_id = self.model.body('gripper_base').id
        self.right_pad_id = self.model.body('right_pad').id
        self.left_pad_id = self.model.body('left_pad').id
        self.cube_id = self.model.body('cube').id
        self.right_act, self.left_act = 0, 1
        self.max_ctrl = 0.035
        self._friction_restore = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.time = 0.0
        self.was_grasped = False
        self.object_dropped = False
        self.recovery_success = False
        self.reflex_applied = False
        self.reflex_type = None
        self.reflex_time = None
        self.force_integral = 0.0
        self.history = []
        self.contact = False
        self.object_vx = self.object_vz = 0.0
        self._regrasp_phase = None
        self._regrasp_start_time = 0.0

        # Mass override
        if self.cfg.object_mass_override > 0:
            self.model.body_mass[self.cube_id] = self.cfg.object_mass_override

        # Friction override (only on cube geoms)
        if self.cfg.friction_coeff < 1.0:
            if self._friction_restore is None:
                self._friction_restore = {}
                for i in range(self.model.ngeom):
                    if self.model.geom_bodyid[i] == self.cube_id:
                        self._friction_restore[i] = self.model.geom_friction[i].copy()
            for i in self._friction_restore:
                self.model.geom_friction[i, :] *= self.cfg.friction_coeff

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)

        mujoco.mj_forward(self.model, self.data)
        return self._get_state()

    def _get_state(self):
        gpos = self.data.xpos[self.gripper_base_id]
        cpos = self.data.xpos[self.cube_id]
        rpos = self.data.xpos[self.right_pad_id]
        lpos = self.data.xpos[self.left_pad_id]
        cjnt = self.model.body_jntadr[self.cube_id]
        cv = self.data.qvel[cjnt:cjnt+6] if cjnt >= 0 else np.zeros(6)

        pad_sep = float(np.linalg.norm(rpos - lpos))
        contact = False
        for c in self.data.contact:
            g1, g2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            pads = {self.right_pad_id, self.left_pad_id}
            if (g1 == self.cube_id and g2 in pads) or (g2 == self.cube_id and g1 in pads):
                contact = True; break

        grasped = contact and pad_sep < 0.06
        grip_force = (self.data.ctrl[self.right_act] + self.data.ctrl[self.left_act]) / 0.08 * 10.0
        self.contact = contact
        self.object_vx, self.object_vz = float(cv[0]), float(cv[2])

        return np.array([
            gpos[0], gpos[2], pad_sep, cpos[0], cpos[2],
            float(cv[0]), float(cv[2]), float(contact), float(grasped), grip_force,
        ])

    def _relative_distance(self):
        return float(np.linalg.norm(self.data.xpos[self.gripper_base_id] - self.data.xpos[self.cube_id]))

    def _add_noise(self, s):
        return s + np.random.randn(*s.shape)*self.cfg.noise_std if self.cfg.noise_std>0 else s

    def _apply_disturbance(self):
        cfg = self.cfg
        in_window = cfg.slip_time <= self.time <= cfg.slip_time + cfg.slip_duration
        self.data.xfrc_applied[self.cube_id, :] = 0
        if not in_window:
            return

        f = cfg.slip_force_mag
        t = self.time
        p = (t - cfg.slip_time) / cfg.slip_duration
        d = cfg.disturbance_type

        if d in ("downward", "overload"):
            self.data.xfrc_applied[self.cube_id, :] = [0, 0, -f, 0, 0, 0]
        elif d == "lateral":
            lat = f * np.sin(p * np.pi)
            self.data.xfrc_applied[self.cube_id, :] = [lat, 0, -f*0.3, 0, 0, 0]
        elif d == "rotational":
            self.data.xfrc_applied[self.cube_id, :] = [0, 0, -f*0.5, 0, f*0.02, 0]
        elif d == "collision":
            pulse = f * np.exp(-10 * (p - 0.1)**2)
            self.data.xfrc_applied[self.cube_id, :] = [pulse, 0, -pulse*0.5, 0, 0, 0]
        elif d == "vibration":
            vib = f * np.sin(t * 120)
            self.data.xfrc_applied[self.cube_id, :] = [vib*0.5, vib*0.3, -f*0.5, 0, 0, 0]
        else:
            self.data.xfrc_applied[self.cube_id, :] = [0, 0, -f, 0, 0, 0]

    def step(self, action, reflex_action=None, grip_force_delta=0.0):
        cfg = self.cfg
        self.time += cfg.dt
        ctrl = self.data.ctrl.copy()

        if action == 4:
            ctrl[self.right_act] = min(ctrl[self.right_act]+0.002, cfg.gripper_close_ctrl)
            ctrl[self.left_act] = ctrl[self.right_act]
        elif action == 5:
            ctrl[self.right_act] = max(ctrl[self.right_act]-0.002, 0)
            ctrl[self.left_act] = ctrl[self.right_act]

        if reflex_action is not None and not self.reflex_applied:
            self.reflex_applied = True
            self.reflex_time = self.time
            if reflex_action == 6:
                self.reflex_type = 'tighten'
                ctrl[self.right_act] = ctrl[self.left_act] = cfg.gripper_tighten_ctrl
            elif reflex_action == 7:
                self.reflex_type = 'regrasp'
                self._regrasp_phase = 'open'
                self._regrasp_start_time = self.time
                ctrl[self.right_act] = ctrl[self.left_act] = 0
            elif reflex_action == 8:
                self.reflex_type = 'tighten_adaptive'
                delta = grip_force_delta / 20.0 * self.max_ctrl
                new_f = ctrl[self.right_act] + delta
                ctrl[self.right_act] = ctrl[self.left_act] = np.clip(new_f, 0, self.max_ctrl)

        if self._regrasp_phase:
            elapsed = self.time - self._regrasp_start_time
            if self._regrasp_phase == 'open' and elapsed > 0.15:
                self._regrasp_phase = 'close'
                self._regrasp_start_time = self.time
            if self._regrasp_phase == 'close':
                close_val = min(ctrl[self.right_act]+0.005, 0.04)
                ctrl[self.right_act] = ctrl[self.left_act] = close_val
                if elapsed > 0.3:
                    self._regrasp_phase = None

        self.data.ctrl[:] = ctrl
        self._apply_disturbance()
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        pad_sep = float(np.linalg.norm(self.data.xpos[self.right_pad_id] - self.data.xpos[self.left_pad_id]))
        if pad_sep < 0.06 and ctrl[self.right_act] > 0.02:
            self.was_grasped = True

        cpos = self.data.xpos[self.cube_id]
        if self.was_grasped and cpos[2] < 0.05:
            self.object_dropped = True

        if self.time > cfg.slip_time + cfg.slip_duration + 0.5:
            if self.was_grasped and not self.object_dropped and cpos[2] > 0.10:
                self.recovery_success = True

        self.force_integral += (ctrl[self.right_act] + ctrl[self.left_act]) * cfg.dt
        return self._get_state()

    def render_frame(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data)
        return self.renderer.render()
