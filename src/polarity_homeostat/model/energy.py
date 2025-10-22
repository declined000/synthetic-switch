from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from ..utils.math_utils import laplacian_2d


@dataclass
class EnergyConfig:
	grid: Tuple[int, int]
	E0: float
	k_oxphos: float
	alpha_actuation_cost: float
	beta_tnt_flux: float
	gamma_decay: float
	Emin: float


class Energy:
	"""
	Grid energy store E(t) in [0,1]-ish units with simple dynamics:
	E' = k_oxphos*(1 - E) - gamma_decay*E - alpha_actuation_cost*|u_act| + beta_tnt_flux*Laplacian(E)
	Explicit Euler integration.
	"""
	def __init__(self, cfg: EnergyConfig):
		self.cfg = cfg
		self.h, self.w = cfg.grid
		self.E = np.full((self.h, self.w), float(cfg.E0), dtype=float)

	def set_initial(self, e0: float | np.ndarray) -> None:
		if isinstance(e0, np.ndarray):
			assert e0.shape == (self.h, self.w)
			self.E = e0.astype(float, copy=True)
		else:
			self.E.fill(float(e0))

	def step(self, dt: float, u_act: Optional[np.ndarray] = None, u_tnt_ev: float = 0.0) -> None:
		E = self.E
		lapE = laplacian_2d(E) if self.cfg.beta_tnt_flux != 0.0 else 0.0
		cost = 0.0
		if u_act is not None:
			cost = self.cfg.alpha_actuation_cost * np.abs(u_act)
		prod = self.cfg.k_oxphos * (1.0 - E)
		decay = self.cfg.gamma_decay * E
		flux = self.cfg.beta_tnt_flux * lapE
		self.E = E + dt * (prod - decay - cost + flux)
		# Clamp to non-negative range
		self.E = np.maximum(self.E, 0.0)
