from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RulesThresholds:
	low_occ_threshold: float = 0.30
	energy_ok: float = 0.35
	mismatch_ok: float = 0.30
	healthy_plv_min: float = 0.50  # provided from osc config
	global_v_offset_mV: float = 10.0
	domain_low_fraction: float = 0.40
	# Strict PRUNE (demo-only)
	prune_enabled: bool = False
	prune_low_occ_threshold: float = 0.85
	prune_energy_max: float = 0.20
	prune_mismatch_min: float = 0.60
	prune_dwell_steps: int = 200


@dataclass
class DecoderStability:
	hysteresis_margin: float = 0.05
	decision_dwell: int = 50


class RulesDecoder:
	"""
	Rules decoder producing REST(0) / REPAIR(1) / PRUNE(2) decisions from features.
	Stability is enforced by winner-take-all with hysteresis margin and dwell time.
	Dwell uses a hold counter that decrements every step, avoiding stalls when
	proposals keep changing.
	"""
	def __init__(self, thresholds: RulesThresholds, stability: DecoderStability):
		self.th = thresholds
		self.st = stability
		self._last_action: int = 0
		self._hold_steps: int = 0  # countdown; when > 0, block switches
		self._prune_steps: int = 0  # strict dwell for prune evidence

	@staticmethod
	def _clamp01(x: float) -> float:
		return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

	def _scores(
		self,
		low_occ: float,
		mismatch: float,
		E: float,
		plv: Optional[float],
		global_v_offset_mV: float,
		domain_low_fraction: float,
		prune_cond: bool,
	) -> Tuple[float, float, float]:
		"""
		Return soft scores (rest, repair, prune) in [0,1].
		Soft scoring uses fraction of rule conditions satisfied.
		"""
		# Treat missing PLV as healthy for REST bias
		plv_ok = True if plv is None else (plv >= self.th.healthy_plv_min)
		plv_bad = False if plv is None else (plv < self.th.healthy_plv_min)

		# Additional context conditions
		offset_high = (global_v_offset_mV >= self.th.global_v_offset_mV)
		domain_high = (domain_low_fraction >= self.th.domain_low_fraction)

		# REPAIR conditions (5 total when including context signals)
		repair_conds = [
			low_occ >= self.th.low_occ_threshold,
			mismatch <= self.th.mismatch_ok,
			E >= self.th.energy_ok,
			plv_bad,
			offset_high or domain_high,
		]
		repair_score = sum(1.0 for c in repair_conds if c) / float(len(repair_conds))

		# REST conditions
		rest_conds = [
			plv_ok,
			low_occ < self.th.low_occ_threshold,
			mismatch > self.th.mismatch_ok,
		]
		rest_score = sum(1.0 for c in rest_conds if c) / float(len(rest_conds))

		# PRUNE strict: require sustained prune_cond
		if self.th.prune_enabled and prune_cond and self._prune_steps >= max(1, int(self.th.prune_dwell_steps)):
			prune_score = 1.0
		else:
			prune_score = 0.0

		return (
			self._clamp01(rest_score),
			self._clamp01(repair_score),
			self._clamp01(prune_score),
		)

	def decide(
		self,
		low_occ: float,
		mismatch: float,
		E: float,
		plv: Optional[float],
		global_v_offset_mV: float,
		domain_low_fraction: float,
	) -> int:
		# Decrement hold counter each call; never below zero
		if self._hold_steps > 0:
			self._hold_steps -= 1

		# Update prune evidence dwell counter
		if self.th.prune_enabled:
			prune_cond = (
				low_occ >= self.th.prune_low_occ_threshold and
				E <= self.th.prune_energy_max and
				mismatch >= self.th.prune_mismatch_min
			)
			self._prune_steps = self._prune_steps + 1 if prune_cond else 0
		else:
			prune_cond = False
			self._prune_steps = 0

		rest_s, repair_s, prune_s = self._scores(
			low_occ, mismatch, E, plv, global_v_offset_mV, domain_low_fraction, prune_cond
		)
		scores = [rest_s, repair_s, prune_s]
		proposed = int(max(range(3), key=lambda i: scores[i]))

		current = self._last_action
		if proposed != current:
			# Block switches while in dwell
			if self._hold_steps > 0:
				return current
			# Require a margin to switch
			margin = scores[proposed] - scores[current]
			if margin <= self.st.hysteresis_margin:
				return current
			# Accept switch and reset dwell
			self._last_action = proposed
			self._hold_steps = max(0, int(self.st.decision_dwell))
			return proposed
		# No switch
		return current
