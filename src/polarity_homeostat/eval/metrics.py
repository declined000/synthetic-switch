from __future__ import annotations

from typing import List, Optional
import math
import numpy as np


def compute_recovery_time(domain_low_frac: List[float], dt: float, threshold: float = 0.10, dwell_s: float = 60.0) -> Optional[int]:
	"""
	Return the first step index at which domain_low_fraction falls below `threshold`
	and stays below for at least `dwell_s`. If never satisfied, return None.
	"""
	if not domain_low_frac:
		return None
	dwell_steps = max(1, int(round(dwell_s / max(dt, 1e-9))))
	vals = np.asarray(domain_low_frac, dtype=float)
	below = vals < float(threshold)
	count = 0
	for i, b in enumerate(below):
		count = count + 1 if b else 0
		if count >= dwell_steps:
			return i - dwell_steps + 1
	return None


def compute_flicker_rate(actions: List[int], dt: float, warmup_s: float = 0.0) -> float:
	"""
	Return fraction of time steps with an action switch after warmup.
	If insufficient data, returns 0.0.
	"""
	if not actions:
		return 0.0
	warmup_steps = max(0, int(round(warmup_s / max(dt, 1e-9))))
	acts = np.asarray(actions, dtype=int)
	acts = acts[warmup_steps:] if warmup_steps < len(acts) else acts[-1:]
	if acts.size <= 1:
		return 0.0
	switches = np.sum(acts[1:] != acts[:-1])
	return float(switches) / float(acts.size - 1)


def compute_plv_retention(plv_series: List[Optional[float]], dt: float, window_s: float = 300.0) -> Optional[float]:
	"""
	Compute mean PLV in the first and last `window_s` and return last/first (clipped 0..1).
	If windows cannot be formed or first mean is ~0, return None.
	"""
	if not plv_series:
		return None
	vals = np.array([np.nan if v is None else float(v) for v in plv_series], dtype=float)
	win = max(1, int(round(window_s / max(dt, 1e-9))))
	if vals.size < 2 * win:
		return None
	start_mean = np.nanmean(vals[:win])
	end_mean = np.nanmean(vals[-win:])
	if not np.isfinite(start_mean) or start_mean <= 1e-6 or not np.isfinite(end_mean):
		return None
	ret = float(end_mean / max(start_mean, 1e-6))
	return float(max(0.0, min(1.0, ret)))
