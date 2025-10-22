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

	# Milestone 9: add metrics collection
	tcfg = cfg["tissue"]
	ecfg = cfg["energy"]
	rcfg = RecorderConfig(**cfg["recorder"])
	ocfg = OscConfig(**cfg["osc"], min_bad_duration_s=float(cfg.get("osc_extras", {}).get("min_bad_duration_s", 0.0)))
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
	)

	min_coupling = float(geom_cfg.get("min_coupling_for_consensus", 0.1))
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

	# Metric series
	actions_series = []
	plv_series = []
	domain_low_series = []

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
		domain_low_series.append(float(recorder.domain_low_fraction()))

		action = decoder.decide(low_occ=low_occ_mean, mismatch=mismatch_mean, E=E_mean, plv=(plv if plv is not None else None))
		actions_series.append(action)

		domain_lowE_fraction = float(np.mean(energy.E < float(ecfg["Emin"])))
		Emin_eff = compute_adaptive_emin(
			float(ecfg["Emin"]),
			domain_lowE_fraction,
			bool(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("enabled", False)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("k", 0.0)),
			float(cfg.get("energy_extras", {}).get("adaptive_emin", {}).get("min", float(ecfg["Emin"]))),
		)
		Eok = energy_gate(E_mean, Emin_eff) if cfg["safety"].get("enable_energy_gate", True) else True
		OscOK = oscillation_gate(bool(plv_bad)) if cfg["safety"].get("enable_osc_gate", True) else True
		GeomOK = geometry_gate(mismatch_mean, float(rcfg.mismatch_threshold), None, min_coupling) \
			if cfg["safety"].get("enable_geometry_gate", True) else True
		allow_pulse = (action == 1) and Eok and OscOK and GeomOK

		u_act = actuator.step(allow=allow_pulse, E_ok=Eok, shape=V.shape)
		tissue.step(u_act=u_act)
		energy.step(dt=dt, u_act=u_act, u_tnt_ev=0.0)

		V = tissue.V
		mean_V = float(V.mean())
		global_v_off = float(recorder.global_v_offset(V))
		domain_low_frac = float(recorder.domain_low_fraction())
		D_est = ""

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
				D_est,
			])

	atlas_cols = [
		"t","action","mean_V","LOW_occ","mismatch","E","PLV",
		"global_v_offset","domain_low_fraction","redox_bit","D_est",
	]
	with open(out_dir / "event_atlas.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(atlas_cols)
		writer.writerows(rows)

	# Compute metrics
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
