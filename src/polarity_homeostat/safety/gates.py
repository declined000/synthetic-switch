from __future__ import annotations

from typing import Optional


def energy_gate(E_mean: float, Emin_eff: float) -> bool:
	"""Allow actuation only if energy is above the (possibly adaptive) minimum."""
	return float(E_mean) >= float(Emin_eff)


def compute_adaptive_emin(Emin: float, domain_lowE_fraction: float, enabled: bool, k: float, min_floor: float) -> float:
	"""
	Adapt the minimum energy threshold based on the fraction of cells at low energy.
	Emin_eff = max(min_floor, Emin * (1 - k * domain_lowE_fraction)) if enabled else Emin
	"""
	if not enabled:
		return float(Emin)
	return max(float(min_floor), float(Emin) * (1.0 - float(k) * float(domain_lowE_fraction)))


def oscillation_gate(plv_bad: bool) -> bool:
	"""Allow actuation only when oscillations appear degraded (persistently low PLV)."""
	return bool(plv_bad)


def geometry_gate(mismatch_mean: float, max_mismatch: float, D_est: Optional[float], min_coupling: float) -> bool:
	"""
	Require consensus (low mismatch) when coupling is adequate. If coupling is poor
	(D_est < min_coupling or D_est is None/blank), do not block on geometry.
	"""
	if D_est is None:
		return True
	try:
		Dval = float(D_est)
	except Exception:
		return True
	if Dval < float(min_coupling):
		return True
	return float(mismatch_mean) <= float(max_mismatch)
