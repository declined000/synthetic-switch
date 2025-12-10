import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import csv

from ..sensing.recorder import Recorder, RecorderConfig
from ..sensing.osc import OscillationDetector, OscConfig
from ..model.tissue import Tissue, TissueConfig
from ..model.energy import Energy, EnergyConfig
from ..utils.math_utils import estimate_coupling_shortlag
from ..safety.gates import energy_gate, compute_adaptive_emin, oscillation_gate, geometry_gate
from ..actuation.pulses import PulseActuator, ActuationConfig
from ..decoder.rules import RulesDecoder, RulesThresholds, DecoderStability
from ..eval.metrics import compute_recovery_time, compute_flicker_rate, compute_plv_retention
from .injuries import apply_domain_injuries

"""
Biological mapping (sentinel cell circuit)

- V_mem sensor (NFAT/CREB + Signal V) → TF_depol
    ≈ Recorder.low_occ (time-in-LOW band); tf_depol_k = mean(low_occ in domain k)

- Energy sensor (AMPK-responsive promoter) → TF_EnergyOK
    ≈ Energy.E grid + energy_gate(E_k, Emin_eff)

- Geometry sensor (paracrine/gap-junction comparison) → TF_GeometryOK
    ≈ neighbor_mismatch(V) in domain k + geometry_gate(...)

- Global state sensor (frequency-sensitive CREB / NFAT) → TF_GlobalOK
    ≈ OscillationDetector (PLV with persistence) + global_v_offset override

- Logic / memory (per domain k):
    RulesDecoder (REST / REPAIR / PRUNE) with hysteresis + dwell
    ↔ dCas9 CRISPRi/a logic implementing REST/REPAIR/PRUNE and gates.

- Controller health flag h_k:
    h_k = 1 → controller functional, h_k = 0 → domain can never enter REPAIR.

- Factor H gene / protein (per domain k):
    PulseActuator_k: Hill(TF_depol_k; n, K_H) scales amplitude of Kir2.1-like
    hyperpolarizing pulses in domain k under energy + oscillation + geometry gates.
"""


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to YAML config")
	parser.add_argument("--out", required=True, help="Output directory")
	return parser.parse_args()


def load_config(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def ensure_out_dir(out_dir: str) -> Path:
	p = Path(out_dir)
	p.mkdir(parents=True, exist_ok=True)
	return p


def main():
	args = parse_args()
	cfg = load_config(args.config)
	out_dir = ensure_out_dir(args.out)
	print(f"Starting run → out={out_dir}", flush=True)

	# Build configs
	tcfg = cfg["tissue"]
	ecfg = cfg["energy"]
	rcfg = RecorderConfig(**cfg["recorder"])

	osc_cfg_raw = dict(cfg["osc"])
	# Optional band-pass; only if provided
	if "bandpass" in cfg.get("osc", {}):
		osc_cfg_raw["bandpass"] = tuple(cfg["osc"]["bandpass"])
	ocfg = OscConfig(
		**osc_cfg_raw,
		min_bad_duration_s=float(cfg.get("osc_extras", {}).get("min_bad_duration_s", 0.0)),
	)

	geom_cfg = cfg.get("geometry", {})
	a_cfg = ActuationConfig(**cfg["actuation"])

	dec_cfg = cfg.get("decoder", {})
	stab = DecoderStability(
		hysteresis_margin=float(dec_cfg.get("hysteresis_margin", 0.05)),
		decision_dwell=int(dec_cfg.get("decision_dwell", 50)),
	)
	rules_cfg = dec_cfg.get("rules", {})
	th = RulesThresholds(
		low_occ_threshold=float(rules_cfg.get("low_occ_threshold", 0.30)),
		energy_ok=float(rules_cfg.get("energy_ok", 0.35)),
		mismatch_ok=float(rules_cfg.get("mismatch_ok", 0.30)),
		healthy_plv_min=float(ocfg.healthy_plv_min),
		global_v_offset_mV=float(cfg.get("thresholds", {}).get("global_v_offset_mV", 10.0)),
		domain_low_fraction=float(cfg.get("thresholds", {}).get("domain_low_fraction", 0.40)),
	)

	min_coupling = float(geom_cfg.get("min_coupling_for_consensus", 0.1))
	estimate_coupling = bool(geom_cfg.get("estimate_coupling", True))

	grid_h, grid_w = int(tcfg["grid"][0]), int(tcfg["grid"][1])
	dt = float(tcfg["dt"])
	steps = int(tcfg["steps"])
	atlas_stride = int(cfg.get("logging", {}).get("atlas_stride", 10))

	seed = int(cfg.get("seed", 1337))

	# Multi-domain partition D_k
	domain_cfg = cfg.get("domains", {})
	tile = domain_cfg.get("tile", None)
	if tile is None:
		tile_h, tile_w = grid_h, grid_w  # single domain
	else:
		tile_h, tile_w = int(tile[0]), int(tile[1])

	if grid_h % tile_h != 0 or grid_w % tile_w != 0:
		raise ValueError(
			f"tissue.grid={tcfg['grid']} must be divisible by domains.tile={[tile_h, tile_w]}"
		)

	n_dom_h = grid_h // tile_h
	n_dom_w = grid_w // tile_w
	n_domains = n_dom_h * n_dom_w

	domain_slices = []
	for di in range(n_dom_h):
		for dj in range(n_dom_w):
			i0 = di * tile_h
			i1 = (di + 1) * tile_h
			j0 = dj * tile_w
			j1 = (dj + 1) * tile_w
			domain_slices.append((slice(i0, i1), slice(j0, j1)))

	# Controller health flags h_k (loss-of-function mutations)
	drop_cfg = domain_cfg.get("dropout", {})
	drop_frac = float(drop_cfg.get("frac", 0.0))
	drop_seed = int(drop_cfg.get("seed", seed))
	rng_drop = np.random.default_rng(drop_seed)
	if drop_frac <= 0.0:
		controller_health = np.ones(n_domains, dtype=bool)
	else:
		controller_health = rng_drop.random(n_domains) > drop_frac

	# Core components
	recorder = Recorder(rcfg, grid=(grid_h, grid_w), dt=dt)
	osc = OscillationDetector(ocfg, grid=(grid_h, grid_w), dt=dt)

	tissue = Tissue(
		TissueConfig(
			grid=(grid_h, grid_w),
			dt=dt,
			EL=float(tcfg["EL"]),
			gL=float(tcfg["gL"]),
			coupling_D=float(tcfg["coupling_D"]),
			noise_rms=float(tcfg.get("noise_rms", 0.0)),
			boundary=str(tcfg.get("boundary", "periodic")),
		),
		seed=seed,
	)

	energy = Energy(
		EnergyConfig(
			grid=(grid_h, grid_w),
			E0=float(ecfg["E0"]),
			k_oxphos=float(ecfg["k_oxphos"]),
			alpha_actuation_cost=float(ecfg["alpha_actuation_cost"]),
			beta_tnt_flux=float(ecfg["beta_tnt_flux"]),
			gamma_decay=float(ecfg["gamma_decay"]),
			Emin=float(ecfg["Emin"]),
		)
	)

	# One decoder + actuator per domain
	decoders = [RulesDecoder(thresholds=th, stability=stab) for _ in range(n_domains)]
	actuators = [PulseActuator(a_cfg, dt=dt) for _ in range(n_domains)]

	# Initial state: uniform V/E, then domain-specific injuries
	tissue.set_initial(-18.0)
	healthy_ref = float(tcfg.get("healthy_ref_mV", float(tcfg["EL"]) + 5.0))
	recorder.set_healthy_ref(healthy_ref)

	injuries_cfg = cfg.get("injuries", {}).get("domains", [])
	if injuries_cfg:
		V0 = tissue.V.copy()
		E0_grid = energy.E.copy()
		V0, E0_grid = apply_domain_injuries(V0, E0_grid, domain_slices, injuries_cfg)
		tissue.V[...] = V0
		energy.E[...] = E0_grid

	# Optional window for coupling estimation
	V_window = []
	max_frames = 50

	# Metric series (global)
	actions_series = []
	plv_series = []
	domain_low_series = []

	# Global V offset persistence
	gvo_bad_seconds = 0.0

	# Logging
	atlas_rows = []
	domain_rows = []

	for step in range(steps):
		t = step * dt

		# --- Update oscillation detector with current Vmem ---
		osc.update(tissue.V)

		V = tissue.V
		E_grid = energy.E

		# --- Local sensing: LOW bands, chronic occupancy, mismatch, offset ---
		recorder.update_bands(V)
		recorder.update_low_occupancy()

		mismatch_grid = recorder.neighbor_mismatch(V)
		mismatch_mean = float(mismatch_grid.mean())

		low_state_grid = recorder.low_state
		low_occ_grid = recorder.low_occ
		low_occ_mean = float(low_occ_grid.mean())

		E_mean = float(E_grid.mean())
		plv, plv_bad = osc.plv_with_persistence()
		plv_series.append(plv)

		domain_low_frac_global = float(recorder.domain_low_fraction())
		domain_low_series.append(domain_low_frac_global)

		global_v_off = float(recorder.global_v_offset(V))

		# --- Estimate coupling for geometry gate (global) ---
		D_est = None
		if estimate_coupling:
			V_window.append(V.copy())
			if len(V_window) > max_frames:
				V_window.pop(0)
			if len(V_window) >= 4:
				D_est = float(estimate_coupling_shortlag(np.stack(V_window, axis=0)))

		# --- Energy gate with adaptive Emin (global Emin_eff, domain-local E_k) ---
		domain_lowE_fraction = float(np.mean(E_grid < float(ecfg["Emin"])))
		Emin_eff = compute_adaptive_emin(
			float(ecfg["Emin"]),
			domain_lowE_fraction,
			bool(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("enabled", False)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("k", 0.0)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("min", float(ecfg["Emin"]))),
		)

		# --- Oscillation gate with PLV persistence + global offset override ---
		OscOK = oscillation_gate(bool(plv_bad)) if cfg["safety"].get("enable_osc_gate", True) else True

		gvo_thresh = float(cfg.get("thresholds", {}).get("global_v_offset_mV", 10.0))
		if abs(global_v_off) >= gvo_thresh:
			gvo_bad_seconds += dt
		else:
			gvo_bad_seconds = max(0.0, gvo_bad_seconds - dt)

		# If sustained offset high for same persistence window, treat as bad oscillation
		if gvo_bad_seconds >= float(cfg.get("osc_extras", {}).get("min_bad_duration_s", 0.0)):
			OscOK = True

		# --- Domain-level control loop ---
		u_act = np.zeros_like(V)
		max_action_this_step = 0

		log_this_step = (step % atlas_stride == 0)

		for k, (sl_i, sl_j) in enumerate(domain_slices):
			V_dom = V[sl_i, sl_j]
			E_dom = E_grid[sl_i, sl_j]

			low_occ_dom = float(low_occ_grid[sl_i, sl_j].mean())
			mismatch_dom = float(mismatch_grid[sl_i, sl_j].mean())
			low_frac_dom = float(low_state_grid[sl_i, sl_j].mean())
			E_dom_mean = float(E_dom.mean())

			# TF_depol derived from chronic LOW occupancy in this domain
			tf_depol = max(0.0, min(1.0, low_occ_dom))

			# REST/REPAIR/PRUNE decision for domain k
			if controller_health[k]:
				action_k = decoders[k].decide(
					low_occ=low_occ_dom,
					mismatch=mismatch_dom,
					E=E_dom_mean,
					plv=(plv if plv is not None else None),
					global_v_offset_mV=global_v_off,
					domain_low_fraction=low_frac_dom,
				)
			else:
				# Loss-of-function mutation: controller permanently OFF (stays at REST)
				action_k = 0

			# Domain-level energy & geometry gates
			Eok_dom = energy_gate(E_dom_mean, Emin_eff) if cfg["safety"].get("enable_energy_gate", True) else True
			GeomOK_dom = geometry_gate(
				mismatch_dom,
				float(rcfg.mismatch_threshold),
				D_est,
				min_coupling,
			) if cfg["safety"].get("enable_geometry_gate", True) else True

			allow_pulse = (
				(action_k == 1)  # REPAIR
				and Eok_dom
				and OscOK
				and GeomOK_dom
				and controller_health[k]
			)

			u_dom = actuators[k].step(
				allow=allow_pulse,
				E_ok=Eok_dom,
				shape=V_dom.shape,
				depol_signal=tf_depol,
			)
			u_act[sl_i, sl_j] += u_dom

			max_action_this_step = max(max_action_this_step, int(action_k))

			# Per-domain logging
			if log_this_step:
				domain_rows.append([
					t,
					k,
					int(action_k),
					float(V_dom.mean()),
					low_occ_dom,
					mismatch_dom,
					E_dom_mean,
					low_frac_dom,
					1.0 if controller_health[k] else 0.0,
				])

		# Global "action" summary for flicker metric
		actions_series.append(max_action_this_step)

		# --- Global logging (unchanged schema) ---
		mean_V = float(V.mean())
		if log_this_step:
			atlas_rows.append([
				t,
				max_action_this_step,
				mean_V,
				low_occ_mean,
				mismatch_mean,
				float(E_grid.mean()),
				plv if plv is not None else "",
				global_v_off,
				domain_low_frac_global,
				0.0,  # redox_bit placeholder
				("" if D_est is None else D_est),
			])

		# --- Advance tissue and energy with total actuation ---
		tissue.step(u_act=u_act)
		energy.step(dt=dt, u_act=u_act, u_tnt_ev=0.0)

	# --- Write global event atlas ---
	atlas_cols = [
		"t",
		"action",
		"mean_V",
		"LOW_occ",
		"mismatch",
		"E",
		"PLV",
		"global_v_offset",
		"domain_low_fraction",
		"redox_bit",
		"D_est",
	]
	with open(out_dir / "event_atlas.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(atlas_cols)
		writer.writerows(atlas_rows)

	# --- Write per-domain atlas ---
	domain_atlas_cols = [
		"t",
		"domain_id",
		"action",
		"mean_V_dom",
		"LOW_occ_dom",
		"mismatch_dom",
		"E_dom",
		"domain_low_fraction_dom",
		"controller_health",
	]
	with open(out_dir / "domain_atlas.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(domain_atlas_cols)
		writer.writerows(domain_rows)

	# --- Summary metrics (still global) ---
	recovery_step = compute_recovery_time(domain_low_series, dt=dt, threshold=0.10, dwell_s=60.0)
	flicker = compute_flicker_rate(actions_series, dt=dt, warmup_s=0.0)
	plv_ret = compute_plv_retention(plv_series, dt=dt, window_s=300.0)

	summary = {
		"recovery_time_step": int(recovery_step) if recovery_step is not None else None,
		"flicker_rate": float(flicker),
		"PLV_retention": None if plv_ret is None else float(plv_ret),
		"final_mean_V": float(tissue.V.mean()) if atlas_rows else None,
		"final_mean_E": float(energy.E.mean()) if atlas_rows else None,
		"n_domains": int(n_domains),
		"domain_tile": [tile_h, tile_w],
		"controller_dropout_frac": float(drop_frac),
	}
	with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Wrote {len(atlas_rows)} global atlas rows to {out_dir / 'event_atlas.csv'}", flush=True)
	print(f"Wrote {len(domain_rows)} domain atlas rows to {out_dir / 'domain_atlas.csv'}", flush=True)


if __name__ == "__main__":
	main()


