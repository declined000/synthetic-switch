from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from ..utils.math_utils import laplacian_2d


@dataclass
class TissueConfig:
	grid: Tuple[int, int]
	dt: float
	EL: float
	gL: float
	coupling_D: float
	noise_rms: float = 0.0
	boundary: str = "periodic"


class Tissue:
	"""
	Simple RC grid with leak to EL and diffusive coupling (Laplacian), plus Gaussian noise.
	Explicit Euler integration; periodic boundary via np.roll in Laplacian.
	"""
	def __init__(self, cfg: TissueConfig, seed: Optional[int] = None):
		self.cfg = cfg
		self.h, self.w = cfg.grid
		self.dt = float(cfg.dt)
		self.V = np.full((self.h, self.w), -18.0, dtype=float)  # default initial depolarized state
		self._rng = np.random.default_rng(int(seed) if seed is not None else None)

	def set_initial(self, v0: float | np.ndarray) -> None:
		if isinstance(v0, np.ndarray):
			assert v0.shape == (self.h, self.w)
			self.V = v0.astype(float, copy=True)
		else:
			self.V.fill(float(v0))

	def step(self, u_act: Optional[np.ndarray] = None) -> None:
		V = self.V
		lap = laplacian_2d(V)
		leak = -self.cfg.gL * (V - self.cfg.EL)
		diff = self.cfg.coupling_D * lap
		noise = self._rng.normal(0.0, self.cfg.noise_rms, size=V.shape) if self.cfg.noise_rms > 0 else 0.0
		input_term = 0.0
		if u_act is not None:
			input_term = u_act
		self.V = V + self.dt * (leak + diff + input_term) + noise
