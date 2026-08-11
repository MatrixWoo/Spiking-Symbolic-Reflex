"""
Event Detector: monitors state changes and generates event signals.

Detects:
- distance_change: relative distance between gripper and object increases
- velocity_anomaly: object velocity exceeds normal threshold
- contact_loss: contact between gripper and object is lost
- force_insufficient: grip force below threshold while object is heavy
"""

import numpy as np
from typing import Dict, List


class EventDetector:
    """
    Detects anomalous events from state transitions.

    Events are represented as continuous activation values [0, 1],
    which can be thresholded into binary spikes for the SNN.
    """

    def __init__(self, config):
        self.cfg = config
        self.reset()

    def reset(self):
        self.prev_distance = None
        self.prev_contact = False
        self.prev_velocity = 0.0
        self.event_history: List[Dict] = []

    def detect(self, curr_state: np.ndarray, prev_state: np.ndarray,
               env) -> Dict[str, float]:
        """
        Detect events from state transition.

        Args:
            curr_state: current state vector
            prev_state: previous state vector
            env: environment (for additional context)

        Returns:
            dict of event_name -> activation [0, 1]
        """
        # Parse state vectors
        # [gripper_x, gripper_z, gripper_width, object_x, object_z,
        #  object_vx, object_vz, contact, grasped, gripper_force]

        curr_dist = env._relative_distance()
        if self.prev_distance is None:
            self.prev_distance = curr_dist

        curr_contact = env.contact
        curr_vz = env.object_vz
        curr_vx = env.object_vx
        curr_speed = np.sqrt(curr_vx**2 + curr_vz**2)

        events = {}

        # 1. Distance change event
        dist_change = curr_dist - self.prev_distance
        # Normalize: positive change = object moving away
        dist_change_norm = np.clip(dist_change / 0.05, 0, 1)  # 5cm = max activation
        events['distance_increase'] = float(dist_change_norm)

        # 2. Velocity anomaly event
        # Normal object speed during transport < 0.1 m/s
        speed_anomaly = np.clip((curr_speed - 0.05) / 0.3, 0, 1)
        events['velocity_anomaly'] = float(speed_anomaly)

        # 3. Contact loss event
        if self.prev_contact and not curr_contact:
            events['contact_loss'] = 1.0
        elif not self.prev_contact and not curr_contact:
            events['contact_loss'] = 0.0
        else:
            events['contact_loss'] = 0.0
        self.prev_contact = curr_contact

        # 4. Slip-specific: distance increase + velocity anomaly simultaneously
        slip_score = (dist_change_norm * 0.6 + speed_anomaly * 0.4)
        if curr_contact:
            slip_score *= 0.5  # less severe if still in contact
        events['slip_risk'] = float(np.clip(slip_score, 0, 1))

        # 5. Grasp instability: contact but object moving
        if curr_contact and curr_speed > 0.03:
            events['grasp_unstable'] = float(np.clip(curr_speed / 0.2, 0, 1))
        else:
            events['grasp_unstable'] = 0.0

        self.prev_distance = curr_dist
        self.prev_velocity = curr_speed

        self.event_history.append({
            'time': env.time,
            **events
        })

        return events

    def get_event_spikes(self, events: Dict[str, float],
                         threshold: float = 0.5) -> Dict[str, int]:
        """Convert continuous events to binary spikes."""
        return {k: int(v > threshold) for k, v in events.items()}
