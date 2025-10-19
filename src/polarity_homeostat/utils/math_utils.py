import numpy as np
from typing import Tuple


def laplacian_2d(field: np.ndarray) -> np.ndarray:
	up    = np.roll(field, -1, axis=0)
	down  = np.roll(field, 1, axis=0)
	left  = np.roll(field, 1, axis=1)
	right = np.roll(field, -1, axis=1)
	return up + down + left + right - 4.0 * field


# --- NEW: simple EMA + coupling estimator ---

def ema_update(prev: float, x: float, alpha: float) -> float:
	"""Exponential moving average. alpha in (0,1]; higher = faster."""
	alpha = float(np.clip(alpha, 1e-6, 1.0))
	return (1.0 - alpha) * prev + alpha * x


def estimate_coupling_shortlag(V_window: np.ndarray) -> float:
	"""
	Rough 'coupling' proxy using short-lag spatial correlation of V across the grid,
	averaged over time. V_window shape: [T, H, W]. Returns 0..1-ish.
	"""
	if not isinstance(V_window, np.ndarray) or V_window.ndim != 3 or V_window.shape[0] < 4:
		return 0.0
	T = min(50, V_window.shape[0])
	Vt = V_window[-T:]
	corrs = []
	for frame in Vt:
		up    = np.roll(frame, -1, axis=0)
		down  = np.roll(frame, 1, axis=0)
		left  = np.roll(frame, 1, axis=1)
		right = np.roll(frame, -1, axis=1)
		neigh_mean = (up + down + left + right) / 4.0
		a = frame.flatten() - frame.mean()
		b = neigh_mean.flatten() - neigh_mean.mean()
		denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
		corrs.append(float((a @ b) / denom))
	return float(np.clip((np.mean(corrs) + 1.0) / 2.0, 0.0, 1.0))
