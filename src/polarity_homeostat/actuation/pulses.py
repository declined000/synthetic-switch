from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ActuationConfig:
	amplitude_mV: float
	duty: float
	period_s: float
	refractory_s: float
	cap_when_lowE: dict  # { amplitude_mV: -4.0, duty: 0.05 }


class PulseActuator:
	"""
	Bounded hyperpolarizing pulse generator with duty/period and refractory.
	If energy is low (Eok == False), use capped parameters (smaller amplitude/duty).
	"""
	def __init__(self, cfg: ActuationConfig, dt: float):
		self.cfg = cfg
		self.dt = float(dt)
		self.t = 0.0
		self._next_ok_time = 0.0

	def step(self, allow: bool, E_ok: bool, shape: Tuple[int, int]) -> np.ndarray:
		self.t += self.dt
		if not allow:
			return np.zeros(shape, dtype=float)
		if self.t < self._next_ok_time:
			return np.zeros(shape, dtype=float)

		# Choose parameters based on energy
		if E_ok:
			amp = float(self.cfg.amplitude_mV)
			duty = float(self.cfg.duty)
		else:
			amp = float(self.cfg.cap_when_lowE.get("amplitude_mV", 0.0))
			duty = float(self.cfg.cap_when_lowE.get("duty", 0.0))

		if duty <= 0.0 or self.cfg.period_s <= 0.0:
			return np.zeros(shape, dtype=float)

		phase = (self.t % float(self.cfg.period_s)) / float(self.cfg.period_s)
		is_on = phase < duty
		if not is_on:
			return np.zeros(shape, dtype=float)

		# Emit one period of pulses, then enforce refractory after the on-window
		u = np.full(shape, amp, dtype=float)
		# schedule refractory at the end of current on window
		end_of_on = self.t + (duty - phase) * float(self.cfg.period_s)
		self._next_ok_time = max(self._next_ok_time, end_of_on + float(self.cfg.refractory_s))
		return u
