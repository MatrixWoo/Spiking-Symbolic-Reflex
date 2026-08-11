"""
Minimal MuJoCo Slip Recovery Environment.

Simple gripper (two sliding pads) + cube + gravity + slip force.

State vector:
  [gripper_x, gripper_z, pad_separation, cube_x, cube_z,
   cube_vx, cube_vz, contact, grasped, grip_force]
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
    slip_force_mag: float = 5.0         # downward force on cube (N)
    gripper_close_ctrl: float = 0.025   # normal grip (pad at cube edge ~0.02)
    gripper_tighten_ctrl: float = 0.035 # max squeeze
    rule_delay_ms: float = 80.0
    noise_std: float = 0.0


SCENE_XML = os.path.join(os.path.dirname(__file__), 'assets', 'simple_gripper.xml')


class MuJoCoReflexEnv:

    def __init__(self, config=None):
        self.cfg = config or MuJoCoConfig()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = None  # lazy init

        self.gripper_base_id = self.model.body('gripper_base').id
        self.right_pad_id = self.model.body('right_pad').id
        self.left_pad_id = self.model.body('left_pad').id
        self.cube_id = self.model.body('cube').id

        self.right_act = 0
        self.left_act = 1
        self.max_ctrl = 0.035  # matches actuator ctrlrange

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
        self.object_vx = 0.0
        self.object_vz = 0.0
        self._regrasp_phase = None
        self._regrasp_start_time = 0.0
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        return self._get_state()

    def _get_state(self):
        gpos = self.data.xpos[self.gripper_base_id]
        cpos = self.data.xpos[self.cube_id]
        rpos = self.data.xpos[self.right_pad_id]
        lpos = self.data.xpos[self.left_pad_id]

        # Velocity from freejoint
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
            gpos[0], gpos[2], pad_sep,        # gripper state
            cpos[0], cpos[2],                  # cube position
            float(cv[0]), float(cv[2]),         # cube velocity
            float(contact), float(grasped),
            grip_force,
        ])

    def _relative_distance(self):
        return float(np.linalg.norm(
            self.data.xpos[self.gripper_base_id] - self.data.xpos[self.cube_id]))

    def _add_noise(self, s):
        return s + np.random.randn(*s.shape)*self.cfg.noise_std if self.cfg.noise_std>0 else s

    def step(self, action, reflex_action=None, grip_force_delta=0.0):
        cfg = self.cfg
        self.time += cfg.dt
        ctrl = self.data.ctrl.copy()

        # Planned actions
        if action == 4:   # close
            ctrl[self.right_act] = min(ctrl[self.right_act]+0.002, cfg.gripper_close_ctrl)
            ctrl[self.left_act] = ctrl[self.right_act]
        elif action == 5: # open
            ctrl[self.right_act] = max(ctrl[self.right_act]-0.002, 0)
            ctrl[self.left_act] = ctrl[self.right_act]

        # Reflex
        if reflex_action is not None and not self.reflex_applied:
            self.reflex_applied = True
            self.reflex_time = self.time
            if reflex_action == 6:  # tighten
                self.reflex_type = 'tighten'
                ctrl[self.right_act] = ctrl[self.left_act] = cfg.gripper_tighten_ctrl
            elif reflex_action == 7:  # regrasp
                self.reflex_type = 'regrasp'
                self._regrasp_phase = 'open'
                self._regrasp_start_time = self.time
                ctrl[self.right_act] = ctrl[self.left_act] = 0
            elif reflex_action == 8:  # adaptive
                self.reflex_type = 'tighten_adaptive'
                delta = grip_force_delta / 20.0 * self.max_ctrl
                new_f = ctrl[self.right_act] + delta
                ctrl[self.right_act] = ctrl[self.left_act] = np.clip(new_f, 0, self.max_ctrl)

        # Regrasp
        if self._regrasp_phase:
            elapsed = self.time - self._regrasp_start_time
            if self._regrasp_phase == 'open' and elapsed > 0.15:
                self._regrasp_phase = 'close'; self._regrasp_start_time = self.time
            if self._regrasp_phase == 'close':
                close_val = min(ctrl[self.right_act]+0.005, 0.04)
                ctrl[self.right_act] = ctrl[self.left_act] = close_val
                if elapsed > 0.3: self._regrasp_phase = None

        self.data.ctrl[:] = ctrl

        # Slip disturbance — downward force on cube
        in_slip = cfg.slip_time <= self.time <= cfg.slip_time + cfg.slip_duration
        if in_slip:
            self.data.xfrc_applied[self.cube_id, :] = [0, 0, -cfg.slip_force_mag, 0, 0, 0]

        mujoco.mj_step(self.model, self.data)
        # mj_forward only needed for contact/kinematics access
        mujoco.mj_forward(self.model, self.data)

        # Update was_grasped
        pad_sep = float(np.linalg.norm(
            self.data.xpos[self.right_pad_id] - self.data.xpos[self.left_pad_id]))
        if pad_sep < 0.06 and ctrl[self.right_act] > 0.02:
            self.was_grasped = True

        # Drop detection
        cpos = self.data.xpos[self.cube_id]
        if self.was_grasped and cpos[2] < 0.05:
            self.object_dropped = True

        # Recovery
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
