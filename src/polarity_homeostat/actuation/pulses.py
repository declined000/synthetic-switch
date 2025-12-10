from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ActuationConfig:
	# Maximal actuation parameters when fully ON and energy is OK
	amplitude_mV: float
	duty: float
	period_s: float
	refractory_s: float
	cap_when_lowE: dict  # { amplitude_mV: -4.0, duty: 0.05 }

	# Explicit Hill-type nonlinearity for Factor H promoter
	# TF_depol is assumed to be in ~[0,1] (LOW occupancy / chronic depol signal).
	hill_n: float = 1.0           # Hill coefficient n
	hill_K: float = 0.5           # Half-activation (TF_depol giving 50% response)
	hill_input_scale: float = 1.0  # Optional scaling of TF_depol before Hill


class PulseActuator:
	"""
	Hyperpolarizing pulse generator implementing the Factor H arm:

	  TF_depol  --(Hill promoter, n, K_H)-->  Factor H (Kir2.1-like current)

	- When the decoder + safety gates allow REPAIR, we compute a Hill factor
	  H(TF_depol) and scale the pulse amplitude by it.
	- When energy is low (E_ok == False), we use capped parameters
	  (smaller amplitude/duty) *and* still respect the Hill curve.
	"""

	def __init__(self, cfg: ActuationConfig, dt: float):
		self.cfg = cfg
		self.dt = float(dt)
		self.t = 0.0
		self._next_ok_time = 0.0

	def _hill(self, tf_depol: Optional[float]) -> float:
		"""
		Cooperative Hill response H in [0,1] for the depolarization TF.

		H(s) = s^n / (K^n + s^n),  with s >= 0, n >= 1, K > 0.
		If tf_depol is None, return 1.0 (no modulation).
		"""
		if tf_depol is None:
			return 1.0

		n = max(1.0, float(self.cfg.hill_n))
		K = float(self.cfg.hill_K)
		s = max(0.0, float(tf_depol) * float(self.cfg.hill_input_scale))

		if K <= 0.0:
			# Degenerate: treat as always fully ON once s>0
			return 1.0 if s > 0.0 else 0.0

		# Standard Hill form, numerically safe
		num = s ** n
		denom = (K ** n) + num
		if denom <= 0.0:
			return 0.0
		H = num / denom
		# Clamp to [0,1]
		return float(max(0.0, min(1.0, H)))

	def step(
		self,
		allow: bool,
		E_ok: bool,
		shape: Tuple[int, int],
		depol_signal: Optional[float] = None,
	) -> np.ndarray:
		"""
		One step of the actuator.

		Parameters
		----------
		allow : bool
		    Output of the rules-based decoder + safety gates (REPAIR branch).
		E_ok : bool
		    Output of the energy gate (TF_EnergyOK).
		shape : (h, w)
		    Shape of the tissue Vmem grid.
		depol_signal : float in [0,1], optional
		    Domain-level depolarization signal driving TF_depol; in this model
		    we use the mean LOW occupancy fraction from Recorder.

		Returns
		-------
		u_act : np.ndarray
		    Hyperpolarizing current (mV-equivalent) over the grid.
		"""
		self.t += self.dt
		if not allow:
			return np.zeros(shape, dtype=float)
		if self.t < self._next_ok_time:
			return np.zeros(shape, dtype=float)

		# Base parameters (maximal, before Hill scaling)
		if E_ok:
			base_amp = float(self.cfg.amplitude_mV)
			base_duty = float(self.cfg.duty)
		else:
			base_amp = float(self.cfg.cap_when_lowE.get("amplitude_mV", 0.0))
			base_duty = float(self.cfg.cap_when_lowE.get("duty", 0.0))

		if base_duty <= 0.0 or self.cfg.period_s <= 0.0:
			return np.zeros(shape, dtype=float)

		# Phase inside the duty window
		phase = (self.t % float(self.cfg.period_s)) / float(self.cfg.period_s)
		is_on = phase < base_duty
		if not is_on:
			return np.zeros(shape, dtype=float)

		# Hill-modulated amplitude (Factor H output)
		H = self._hill(depol_signal)  # in [0,1]
		if H <= 0.0:
			return np.zeros(shape, dtype=float)

		amp = base_amp * H  # note: base_amp is negative for hyperpolarization

		# Emit one period of pulses, then enforce refractory after the on-window
		u = np.full(shape, amp, dtype=float)
		end_of_on = self.t + (base_duty - phase) * float(self.cfg.period_s)
		self._next_ok_time = max(self._next_ok_time, end_of_on + float(self.cfg.refractory_s))
		return u
