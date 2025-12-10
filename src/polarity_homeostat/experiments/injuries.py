from __future__ import annotations

from typing import List, Dict, Sequence, Tuple
import numpy as np

DomainSlice = Tuple[slice, slice]


def apply_domain_injuries(
	V: np.ndarray,
	E: np.ndarray,
	domain_slices: Sequence[DomainSlice],
	injury_cfg: List[Dict],
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Apply domain-specific 'injuries' at t = 0.

	Each entry in injury_cfg is a dict with:
	  - id: int
	        Index of the domain (0 .. len(domain_slices)-1).
	  - delta_V_mV: float (optional)
	        Add this many mV to V in that domain (depolarizing if positive).
	  - E0: float (optional)
	        Set E in that domain to this value (e.g., lower ATP).

	Returns updated (V, E) arrays.
	"""
	if not injury_cfg:
		return V, E

	V_new = V.copy()
	E_new = E.copy()

	n_domains = len(domain_slices)

	for inj in injury_cfg:
		if inj is None:
			continue

		k = int(inj.get("id", -1))
		if k < 0 or k >= n_domains:
			# Ignore invalid domain ids
			continue

		sl_i, sl_j = domain_slices[k]

		if "delta_V_mV" in inj:
			dV = float(inj["delta_V_mV"])
			V_new[sl_i, sl_j] += dV

		if "E0" in inj:
			E0 = float(inj["E0"])
			E_new[sl_i, sl_j] = E0

	return V_new, E_new


