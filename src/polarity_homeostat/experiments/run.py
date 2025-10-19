import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import csv

from ..sensing.recorder import Recorder, RecorderConfig


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

	# Milestone 1: synthetic voltage field + recorder features
	tcfg = cfg["tissue"]
	rcfg = RecorderConfig(**cfg["recorder"])
	grid_h, grid_w = int(tcfg["grid"][0]), int(tcfg["grid"][1])
	dt = float(tcfg["dt"])
	steps = int(tcfg["steps"])
	noise_rms = float(tcfg.get("noise_rms", 0.0))
	atlas_stride = int(cfg.get("logging", {}).get("atlas_stride", 10))

	# Seeded RNG for reproducibility
	rng = np.random.default_rng(int(cfg.get("seed", 1337)))

	recorder = Recorder(rcfg, grid=(grid_h, grid_w), dt=dt)

	# Synthetic V: alternate between depolarized (-18 mV) and recovered (-5 mV)
	V = np.full((grid_h, grid_w), -18.0, dtype=float)
	# Healthy reference: use provided knob or EL + 5 mV
	healthy_ref = float(tcfg.get("healthy_ref_mV", float(tcfg["EL"]) + 5.0))
	recorder.set_healthy_ref(healthy_ref)

	rows = []
	for step in range(steps):
		# Simple schedule: switch target every 500 steps
		target = -18.0 if (step // 500) % 2 == 0 else -5.0
		V += 0.05 * (target - V)  # gentle relaxation toward target
		V += rng.normal(0.0, noise_rms, size=V.shape)

		# Update recorder states
		recorder.update_bands(V)
		recorder.update_low_occupancy()
		mismatch = recorder.neighbor_mismatch(V)

		# Derived features
		mean_V = float(V.mean())
		low_occ_mean = float(recorder.low_occ.mean())
		mismatch_mean = float(mismatch.mean())
		global_v_off = float(recorder.global_v_offset(V))
		domain_low_frac = float(recorder.domain_low_fraction())

		if step % atlas_stride == 0:
			rows.append([
				step * dt,
				None,                 # action (not yet implemented)
				mean_V,
				low_occ_mean,
				mismatch_mean,
				None,                 # E (not yet implemented)
				None,                 # PLV (not yet implemented)
				global_v_off,
				domain_low_frac,
				0.0,                  # redox_bit (optional feature)
				None,                 # D_est (not yet estimated)
			])

	# Write outputs
	atlas_cols = [
		"t","action","mean_V","LOW_occ","mismatch","E","PLV",
		"global_v_offset","domain_low_fraction","redox_bit","D_est",
	]
	with open(out_dir / "event_atlas.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(atlas_cols)
		writer.writerows(rows)

	summary = {
		"recovery_time_step": None,
		"flicker_rate": None,
		"PLV_retention": None,
		"final_mean_V": float(V.mean()) if rows else None,
		"final_mean_E": None,
	}
	with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Wrote atlas rows: {len(rows)} to {out_dir / 'event_atlas.csv'}")


if __name__ == "__main__":
	main()
