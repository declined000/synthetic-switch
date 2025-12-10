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

"""
Biological mapping (sentinel cell circuit)

- V_mem sensor (NFAT/CREB + Signal V) → TF_depol
    ≈ Recorder.low_occ (time-in-LOW band); tf_depol = mean(low_occ)

- Energy sensor (AMPK-responsive promoter) → TF_EnergyOK
    ≈ Energy.E grid + energy_gate(E_mean, Emin_eff)

- Geometry sensor (paracrine/gap-junction comparison) → TF_GeometryOK
    ≈ neighbor_mismatch(V) + geometry_gate(...)

- Global state sensor (frequency-sensitive CREB / NFAT) → TF_GlobalOK
    ≈ OscillationDetector (PLV with persistence) + global_v_offset override

- Logic / memory:
    RulesDecoder (REST / REPAIR / PRUNE) with hysteresis + dwell
    ↔ dCas9 CRISPRi/a logic implementing REST/REPAIR/PRUNE and gates.

- Factor H gene / protein:
    PulseActuator: Hill(TF_depol; n, K_H) scales amplitude of Kir2.1-like
    hyperpolarizing pulses under energy + oscillation + geometry gates.
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
	# Light band-pass default (optional): pass bands only if provided in YAML; else leave None
	if "bandpass" in cfg.get("osc", {}):
		osc_cfg_raw["bandpass"] = tuple(cfg["osc"]["bandpass"])  # [lo, hi]
	ocfg = OscConfig(**osc_cfg_raw, min_bad_duration_s=float(cfg.get("osc_extras", {}).get("min_bad_duration_s", 0.0)))
	geom_cfg = cfg.get("geometry", {})
	a_cfg = ActuationConfig(**cfg["actuation"])  # amplitude_mV, duty, period_s, refractory_s, cap_when_lowE

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

	recorder = Recorder(rcfg, grid=(grid_h, grid_w), dt=dt)
	osc = OscillationDetector(ocfg, grid=(grid_h, grid_w), dt=dt)
	tissue = Tissue(TissueConfig(
		grid=(grid_h, grid_w),
		dt=dt,
		EL=float(tcfg["EL"]),
		gL=float(tcfg["gL"]),
		coupling_D=float(tcfg["coupling_D"]),
		noise_rms=float(tcfg.get("noise_rms", 0.0)),
		boundary=str(tcfg.get("boundary", "periodic")),
	), seed=seed)
	energy = Energy(EnergyConfig(
		grid=(grid_h, grid_w),
		E0=float(ecfg["E0"]),
		k_oxphos=float(ecfg["k_oxphos"]),
		alpha_actuation_cost=float(ecfg["alpha_actuation_cost"]),
		beta_tnt_flux=float(ecfg["beta_tnt_flux"]),
		gamma_decay=float(ecfg["gamma_decay"]),
		Emin=float(ecfg["Emin"]),
	))
	actuator = PulseActuator(a_cfg, dt=dt)
	decoder = RulesDecoder(thresholds=th, stability=stab)

	tissue.set_initial(-18.0)
	healthy_ref = float(tcfg.get("healthy_ref_mV", float(tcfg["EL"]) + 5.0))
	recorder.set_healthy_ref(healthy_ref)

	# Optional window for coupling estimation
	V_window = []
	max_frames = 50

	# Metric series
	actions_series = []
	plv_series = []
	domain_low_series = []

	# Global V offset persistence for override
	gvo_bad_seconds = 0.0

	rows = []
	for step in range(steps):
		osc.update(tissue.V)

		V = tissue.V
		recorder.update_bands(V)
		recorder.update_low_occupancy()
		mismatch_grid = recorder.neighbor_mismatch(V)
		mismatch_mean = float(mismatch_grid.mean())
		low_occ_mean = float(recorder.low_occ.mean())
		E_mean = float(energy.E.mean())
		plv, plv_bad = osc.plv_with_persistence()
		plv_series.append(plv)
		domain_low_frac = float(recorder.domain_low_fraction())
		domain_low_series.append(domain_low_frac)
		global_v_off = float(recorder.global_v_offset(V))

		# --- NEW: transcription-factor proxy for depolarization (TF_depol) ---
		# Recorder.low_occ is already a leaky LOW-occupancy fraction in [0,1].
		# We treat its spatial mean as the domain-level TF_depol driving Factor H.
		tf_depol = max(0.0, min(1.0, low_occ_mean))

		# Decoder action with offset/domain context
		action = decoder.decide(
			low_occ=low_occ_mean,
			mismatch=mismatch_mean,
			E=E_mean,
			plv=(plv if plv is not None else None),
			global_v_offset_mV=global_v_off,
			domain_low_fraction=domain_low_frac,
		)
		actions_series.append(action)

		# Coupling estimate for geometry gate
		D_est = None
		if estimate_coupling:
			V_window.append(V.copy())
			if len(V_window) > max_frames:
				V_window.pop(0)
			if len(V_window) >= 4:
				D_est = float(estimate_coupling_shortlag(np.stack(V_window, axis=0)))

		# Energy gate with adaptive Emin
		domain_lowE_fraction = float(np.mean(energy.E < float(ecfg["Emin"])))
		Emin_eff = compute_adaptive_emin(
			float(ecfg["Emin"]),
			domain_lowE_fraction,
			bool(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("enabled", False)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("k", 0.0)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("min", float(ecfg["Emin"]))),
		)
		Eok = energy_gate(E_mean, Emin_eff) if cfg["safety"].get("enable_energy_gate", True) else True

		# Oscillation gate with PLV persistence + global_v_offset override (sustained high offset)
		OscOK = oscillation_gate(bool(plv_bad)) if cfg["safety"].get("enable_osc_gate", True) else True
		gvo_thresh = float(cfg.get("thresholds", {}).get("global_v_offset_mV", 10.0))
		if abs(global_v_off) >= gvo_thresh:
			gvo_bad_seconds += dt
		else:
			gvo_bad_seconds = max(0.0, gvo_bad_seconds - dt)
		# If sustained offset high for same persistence window, treat as bad oscillation
		if gvo_bad_seconds >= float(cfg.get("osc_extras", {}).get("min_bad_duration_s", 0.0)):
			OscOK = True

		# Geometry gate
		GeomOK = geometry_gate(mismatch_mean, float(rcfg.mismatch_threshold), D_est, min_coupling) \
			if cfg["safety"].get("enable_geometry_gate", True) else True

		allow_pulse = (action == 1) and Eok and OscOK and GeomOK

		u_act = actuator.step(
			allow=allow_pulse,
			E_ok=Eok,
			shape=V.shape,
			depol_signal=tf_depol,  # TF_depol → Hill → Factor H → Kir2.1 efflux
		)
		tissue.step(u_act=u_act)
		energy.step(dt=dt, u_act=u_act, u_tnt_ev=0.0)

		V = tissue.V
		mean_V = float(V.mean())

		if step % atlas_stride == 0:
			rows.append([
				step * dt,
				action,
				mean_V,
				low_occ_mean,
				mismatch_mean,
				float(energy.E.mean()),
				plv if plv is not None else "",
				global_v_off,
				domain_low_frac,
				0.0,
				("" if D_est is None else D_est),
			])

	atlas_cols = [
		"t","action","mean_V","LOW_occ","mismatch","E","PLV",
		"global_v_offset","domain_low_fraction","redox_bit","D_est",
	]
	with open(out_dir / "event_atlas.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(atlas_cols)
		writer.writerows(rows)

	recovery_step = compute_recovery_time(domain_low_series, dt=dt, threshold=0.10, dwell_s=60.0)
	flicker = compute_flicker_rate(actions_series, dt=dt, warmup_s=0.0)
	plv_ret = compute_plv_retention(plv_series, dt=dt, window_s=300.0)

	summary = {
		"recovery_time_step": int(recovery_step) if recovery_step is not None else None,
		"flicker_rate": float(flicker),
		"PLV_retention": None if plv_ret is None else float(plv_ret),
		"final_mean_V": float(tissue.V.mean()) if rows else None,
		"final_mean_E": float(energy.E.mean()) if rows else None,
	}
	with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Wrote atlas rows: {len(rows)} to {out_dir / 'event_atlas.csv'}", flush=True)


if __name__ == "__main__":
	main()
