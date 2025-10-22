from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.signal import hilbert


@dataclass
class OscConfig:
	window_seconds: float = 1200.0
	healthy_plv_min: float = 0.5
	downsample: int = 20
	min_bad_duration_s: float = 0.0  # persistence guard
	bands: Optional[list] = None      # placeholder; not used in M3


class OscillationDetector:
	"""
	Maintains a downsampled buffer of mean V and computes a simple PLV
	from its analytic phase via Hilbert transform. Adds a persistence
	threshold so transient dips do not trigger.
	"""
	def __init__(self, cfg: OscConfig, grid: Tuple[int, int], dt: float):
		self.cfg = cfg
		self.h, self.w = grid
		self.dt = float(dt)
		self._ds = max(1, int(cfg.downsample))
		buf_len = max(8, int(round(float(cfg.window_seconds) / (self.dt * self._ds))))
		self._buf = np.zeros(buf_len, dtype=float)
		self._buf_len = buf_len
		self._buf_count = 0
		self._buf_idx = 0
		self._acc_steps = 0
		self._bad_seconds = 0.0
		self._last_plv: Optional[float] = None

	def update(self, V: np.ndarray) -> None:
		"""Downsample and append mean(V) to ring buffer."""
		self._acc_steps += 1
		if self._acc_steps < self._ds:
			return
		self._acc_steps = 0
		val = float(np.mean(V))
		self._buf[self._buf_idx] = val
		self._buf_idx = (self._buf_idx + 1) % self._buf_len
		self._buf_count = min(self._buf_count + 1, self._buf_len)

	def _compute_plv(self) -> Optional[float]:
		if self._buf_count < 8:
			return None
		# Reconstruct buffer in time order
		if self._buf_count == self._buf_len:
			x = np.concatenate((self._buf[self._buf_idx:], self._buf[:self._buf_idx]))
		else:
			x = self._buf[:self._buf_count]
		# Detrend / demean
		x = x - float(np.mean(x))
		if np.allclose(x, 0.0):
			return 0.0
		z = hilbert(x)
		phase = np.angle(z)
		plv = float(np.abs(np.mean(np.exp(1j * phase))))
		return max(0.0, min(1.0, plv))

	def plv_with_persistence(self) -> Tuple[Optional[float], bool]:
		"""
		Returns (plv, is_persistently_bad).
		is_persistently_bad is True only if PLV < healthy_plv_min for >= min_bad_duration_s.
		If PLV is None (insufficient data), returns (None, False).
		"""
		plv = self._compute_plv()
		self._last_plv = plv
		if plv is None:
			return None, False
		if self.cfg.min_bad_duration_s <= 0:
			return plv, (plv < self.cfg.healthy_plv_min)
		# Advance persistence clock by one sample interval (downsampled)
		sample_dt = self.dt * self._ds
		if plv < self.cfg.healthy_plv_min:
			self._bad_seconds += sample_dt
		else:
			self._bad_seconds = max(0.0, self._bad_seconds - sample_dt)
		return plv, (self._bad_seconds >= self.cfg.min_bad_duration_s)

	@property
	def last_plv(self) -> Optional[float]:
		return self._last_plv
