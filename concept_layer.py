"""
Concept Layer: maps raw events to symbolic concepts.

Concepts:
- slip: object is slipping from gripper
- grasp_unstable: grasp is not secure
- object_falling: object is in free fall
- recovery_needed: intervention is required
"""

from typing import Dict
import numpy as np


class ConceptLayer:
    """
    Maps event activations to concept activations.

    This is the "neural → symbolic" bridge. Events are continuous/analog,
    concepts are more discrete/symbolic (though we keep them as [0,1]
    activations for compatibility with the rule layer).
    """

    def __init__(self):
        # Concept activation history
        self.concept_history = []
        # Temporal smoothing factors
        self.smoothing = {
            'slip': 0.4,
            'grasp_unstable': 0.4,
            'object_falling': 0.3,
            'recovery_needed': 0.3,
        }
        self.reset()

    def reset(self):
        """Reset internal state between experiments."""
        self.concept_history = []
        self.current_concepts = {
            'slip': 0.0,
            'grasp_unstable': 0.0,
            'object_falling': 0.0,
            'recovery_needed': 0.0,
        }

    def activate(self, events: Dict[str, float], env) -> Dict[str, float]:
        """
        Map events to concepts with temporal smoothing.

        The concept layer implements a simple mapping:

        distance_increase + velocity_anomaly → slip
        grasp_unstable (event) + slip_risk → grasp_unstable (concept)
        contact_loss + velocity_anomaly → object_falling
        slip + grasp_unstable + object_falling → recovery_needed
        """
        # Raw concept activations from events
        raw = {}

        # Slip concept: combination of slip_risk and distance changes
        raw['slip'] = (
            events.get('slip_risk', 0) * 0.5 +
            events.get('distance_increase', 0) * 0.3 +
            events.get('velocity_anomaly', 0) * 0.2
        )

        # Grasp unstable concept
        raw['grasp_unstable'] = (
            events.get('grasp_unstable', 0) * 0.6 +
            events.get('slip_risk', 0) * 0.4
        )

        # Object falling: contact loss + downward velocity
        raw['object_falling'] = (
            events.get('contact_loss', 0) * 0.7 +
            events.get('velocity_anomaly', 0) * 0.3
        )

        # Recovery needed: any of the above is severe
        raw['recovery_needed'] = max(
            raw['slip'] * 0.6,
            raw['grasp_unstable'] * 0.5,
            raw['object_falling'] * 0.8
        )

        # Apply temporal smoothing (exponential moving average)
        concepts = {}
        for concept_name, raw_value in raw.items():
            alpha = self.smoothing.get(concept_name, 0.3)
            prev = self.current_concepts[concept_name]
            smoothed = alpha * raw_value + (1 - alpha) * prev
            self.current_concepts[concept_name] = smoothed
            concepts[concept_name] = float(np.clip(smoothed, 0, 1))

        self.concept_history.append({
            'time': env.time,
            **concepts
        })

        return concepts
