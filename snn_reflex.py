"""
SNN Reflex Module v2: Leaky Integrate-and-Fire spiking neural network.

Upgrades over v1:
- Adaptive force output (continuous grip adjustment, not just binary)
- Better action differentiation: tighten (contact preserved) vs regrasp (contact lost)
- Per-neuron tuning curves for event pattern sensitivity

Architecture:
  Input (5 event types) → Hidden (10 LIF neurons) → Output (3 neurons)

Output neurons:
  [0] tighten_confidence  — slip with contact preserved → increase grip
  [1] regrasp_confidence  — contact lost / object falling → release & re-grasp
  [2] force_delta         — continuous grip force adjustment magnitude

The SNN has two key properties:
1. Event-driven: only processes when events occur (sparse computation)
2. Low latency: response within 10-30ms (vs 50-100ms for dense networks)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class LIFNeuron:
    """Leaky Integrate-and-Fire neuron with tunable sensitivity."""
    v_rest: float = -70.0
    v_thresh: float = -55.0
    v_reset: float = -75.0
    tau: float = 10.0
    v: float = -70.0
    refractory: float = 2.0
    last_spike_time: float = -100.0
    spike_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    gain: float = 1.0
    # Tuning: which input pattern this neuron prefers
    tuning_bias: np.ndarray = None  # shape (n_inputs,) — preferred input pattern

    def update(self, current: float, dt: float, time: float) -> int:
        if time - self.last_spike_time < self.refractory:
            self.v = self.v_reset
            return 0

        dv = (-(self.v - self.v_rest) + current * self.gain) / self.tau * dt
        self.v += dv

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.last_spike_time = time
            self.spike_history.append(time)
            return 1
        return 0


class SNNLayer:
    """Layer of LIF neurons with input connections."""

    def __init__(self, n_neurons: int, n_inputs: int, name: str = "layer"):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.name = name
        self.weights = np.random.randn(n_neurons, n_inputs) * 2.0 + 3.0
        inh_mask = np.random.random((n_neurons, n_inputs)) > 0.85
        self.weights[inh_mask] *= -1.0
        self.neurons = [LIFNeuron(tau=8.0 + np.random.random() * 8.0,
                                   gain=8.0 + np.random.random() * 4.0)
                        for _ in range(n_neurons)]
        self.spike_output = np.zeros(n_neurons)

    def forward(self, inputs: np.ndarray, dt: float, time: float) -> np.ndarray:
        currents = self.weights @ inputs
        spikes = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            # Tuning bias: neuron responds more to patterns matching its preference
            if self.neurons[i].tuning_bias is not None:
                pattern_match = np.dot(self.neurons[i].tuning_bias, inputs)
                currents[i] *= (1.0 + 0.5 * pattern_match)
            spikes[i] = self.neurons[i].update(currents[i], dt, time)
        self.spike_output = spikes
        return spikes

    def reset(self):
        for neuron in self.neurons:
            neuron.v = neuron.v_rest
            neuron.last_spike_time = -100.0
            neuron.spike_history.clear()
        self.spike_output = np.zeros(self.n_neurons)


class SNNReflexModule:
    """
    Two-layer SNN for adaptive reflex action generation.

    v2: 3 output neurons for differentiated actions
    - tighten: for slip with contact preserved
    - regrasp: for contact loss / falling
    - force_delta: continuous grip adjustment magnitude
    """

    def __init__(self, config):
        self.cfg = config
        self.dt_ms = config.dt * 1000

        self.input_names = [
            'distance_increase', 'velocity_anomaly', 'contact_loss',
            'slip_risk', 'grasp_unstable'
        ]
        n_inputs = len(self.input_names)

        # Hidden layer with diverse tuning
        self.hidden = SNNLayer(n_neurons=10, n_inputs=n_inputs, name="hidden")
        # Assign tuning biases: half prefer slip+contact, half prefer contact_loss+falling
        for i in range(5):
            self.hidden.neurons[i].tuning_bias = np.array([0.2, 0.2, -0.5, 0.3, 0.3])
        for i in range(5, 10):
            self.hidden.neurons[i].tuning_bias = np.array([-0.3, 0.3, 0.8, -0.2, -0.2])

        # Output layer: 3 neurons
        # [0]=tighten, [1]=regrasp, [2]=force_delta
        self.output = SNNLayer(n_neurons=3, n_inputs=10, name="output")
        self.output.weights = np.random.randn(3, 10) * 3.0 + 5.0
        self.output.neurons = [
            LIFNeuron(tau=8.0, gain=8.0),   # tighten
            LIFNeuron(tau=8.0, gain=8.0),   # regrasp
            LIFNeuron(tau=8.0, gain=10.0),   # force_delta (responsive)
        ]
        self.output_names = ['tighten_grip', 'regrasp', 'force_delta']

        self.output_spike_counts = np.zeros(3)
        self.spike_window = deque(maxlen=20)
        self.snn_response_history = []
        self.total_spikes = 0

    def _events_to_array(self, events: Dict[str, float]) -> np.ndarray:
        arr = np.zeros(len(self.input_names))
        for i, name in enumerate(self.input_names):
            arr[i] = events.get(name, 0.0)
        return arr

    def step(self, events: Dict[str, float], time_s: float) -> Optional[Dict]:
        """
        Process one timestep of events through the SNN.

        Returns dict with differentiated actions and adaptive force.
        """
        time_ms = time_s * 1000
        inputs = self._events_to_array(events)

        hidden_spikes = self.hidden.forward(inputs, self.dt_ms, time_ms)
        output_spikes = self.output.forward(hidden_spikes, self.dt_ms, time_ms)

        self.total_spikes += int(np.sum(output_spikes))
        self.output_spike_counts += output_spikes
        self.spike_window.append(output_spikes.copy())

        avg_spikes = np.mean(np.array(list(self.spike_window)), axis=0) if self.spike_window else output_spikes

        # Differentiated output
        tighten_conf = float(np.clip(avg_spikes[0] * 10.0, 0, 1))
        regrasp_conf = float(np.clip(avg_spikes[1] * 10.0, 0, 1))

        # Force delta: proportional to spike rate
        # Maps spike rate → positive force adjustment
        force_delta = float(np.clip(avg_spikes[2] * 25.0, 0.0, 13.0))

        result = {
            'time': time_s,
            'tighten_confidence': tighten_conf,
            'regrasp_confidence': regrasp_conf,
            'force_delta': force_delta,
            'output_spikes': output_spikes.copy(),
            'total_spikes': self.total_spikes,
        }

        # Decision logic with urgency-based action selection
        contact_loss = events.get('contact_loss', 0.0)
        slip_risk = events.get('slip_risk', 0.0)
        vel_anomaly = events.get('velocity_anomaly', 0.0)

        if regrasp_conf > 0.3 and contact_loss > 0.3:
            # Contact lost → regrasp
            result['action'] = 7
            result['confidence'] = regrasp_conf
            result['decision_reason'] = 'contact_loss'
            result['force_delta'] = 0.0
        elif slip_risk > 0.5 and vel_anomaly > 0.7:
            # Urgent slip → full binary tighten (fast, reliable)
            result['action'] = 6  # binary tighten = 18N
            result['confidence'] = max(tighten_conf, 0.8)
            result['decision_reason'] = 'urgent_slip_binary'
            result['force_delta'] = 13.0
        elif tighten_conf > 0.3 and slip_risk > 0.2:
            # Moderate slip → adaptive tighten (proportional)
            result['action'] = 8  # adaptive tighten
            result['confidence'] = tighten_conf
            result['force_delta'] = max(force_delta, 5.0)  # minimum 5N boost
            result['decision_reason'] = 'slip_adaptive'
        elif tighten_conf > 0.3:
            result['action'] = 8
            result['confidence'] = tighten_conf
            result['force_delta'] = max(force_delta, 3.0)
            result['decision_reason'] = 'tighten_fallback'
        else:
            result['action'] = None
            result['confidence'] = max(tighten_conf, regrasp_conf)
            result['force_delta'] = 0.0

        self.snn_response_history.append(result)

        if result['action'] is not None:
            return result
        return None

    def reset(self):
        self.hidden.reset()
        self.output.reset()
        self.output_spike_counts = np.zeros(3)
        self.spike_window.clear()
        self.total_spikes = 0
        self.snn_response_history = []

    def get_sparsity_stats(self) -> Dict:
        total_neurons = self.hidden.n_neurons + self.output.n_neurons
        total_spikes = sum(
            len(n.spike_history)
            for layer in [self.hidden, self.output]
            for n in layer.neurons
        )
        return {
            'total_spikes': total_spikes,
            'total_neurons': total_neurons,
            'avg_spikes_per_neuron': total_spikes / max(total_neurons, 1),
        }
