"""
Rule Layer: symbolic reasoning for reflex action selection.

Rules:
  IF slip AND grasp_unstable THEN tighten_gripper
  IF object_falling THEN regrasp
  IF recovery_needed AND NOT slip THEN regrasp (fallback)
  IF contact_loss THEN regrasp
"""

from typing import Dict, Optional


class RuleLayer:
    """
    Symbolic rule engine for reflex action selection.

    Uses fuzzy logic: each rule has a confidence score based on
    antecedent activations. The rule with highest confidence wins.
    """

    def __init__(self):
        self.reset()
        self.rules = [
            {
                'name': 'tighten_for_slip',
                'conditions': {'slip': 0.3, 'grasp_unstable': 0.3},
                'action': 6,  # tighten_grip
                'weight': 1.0,
            },
            {
                'name': 'regrasp_on_fall',
                'conditions': {'object_falling': 0.5},
                'action': 7,  # regrasp
                'weight': 1.2,
            },
            {
                'name': 'regrasp_on_contact_loss',
                'conditions': {'contact_loss': 0.5},
                'action': 7,  # regrasp
                'weight': 1.5,
            },
            {
                'name': 'tighten_on_recovery',
                'conditions': {'recovery_needed': 0.4, 'slip': 0.2},
                'action': 6,  # tighten_grip
                'weight': 0.8,
            },
        ]

    def reset(self):
        self.rule_history = []

    def decide(self, concepts: Dict[str, float], env) -> Optional[int]:
        """
        Evaluate rules and select reflex action.

        Args:
            concepts: current concept activations
            env: environment for additional context

        Returns:
            action_id (6=tighten, 7=regrasp) or None if no rule fires
        """
        best_confidence = 0.0
        best_action = None
        best_rule = None

        for rule in self.rules:
            # Calculate rule match confidence
            matches = []
            for cond_key, threshold in rule['conditions'].items():
                # Get concept value or event value
                if cond_key in concepts:
                    value = concepts[cond_key]
                else:
                    value = 0.0

                # Fuzzy match: how much above threshold
                if value >= threshold:
                    match = min(1.0, (value - threshold) / (1.0 - threshold) + 0.5)
                else:
                    match = value / threshold * 0.5
                matches.append(match)

            # AND combination: use min
            if matches:
                confidence = min(matches) * rule['weight']
            else:
                confidence = 0.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_action = rule['action']
                best_rule = rule['name']

        # Only fire if confidence exceeds threshold
        if best_confidence > 0.3 and best_action is not None:
            self.rule_history.append({
                'time': env.time,
                'rule': best_rule,
                'action': best_action,
                'confidence': best_confidence,
                'concepts': dict(concepts),
            })
            return best_action

        self.rule_history.append({
            'time': env.time,
            'rule': None,
            'action': None,
            'confidence': 0.0,
            'concepts': dict(concepts),
        })
        return None
