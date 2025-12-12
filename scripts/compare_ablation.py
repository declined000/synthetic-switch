from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load(run_dir: Path):
	s = json.loads((run_dir / "summary.json").read_text())
	ev = pd.read_csv(run_dir / "event_atlas.csv")
	return s, ev


def main() -> None:
	if len(sys.argv) != 3:
		print("Usage: python scripts/compare_ablation.py RUN_ON_DIR RUN_OFF_DIR")
		raise SystemExit(1)

	on_dir = Path(sys.argv[1])
	off_dir = Path(sys.argv[2])
	s_on, ev_on = load(on_dir)
	s_off, ev_off = load(off_dir)

	# Table
	cols = ["recovery_time_step", "final_mean_V", "final_LOW_occ"]
	df = pd.DataFrame([s_on, s_off], index=["controller_ON", "no_actuation_OFF"])[cols]
	print(df.to_string())

	# One “control mattered” figure: mean V over time
	plt.figure()
	plt.plot(ev_on["t"], ev_on["mean_V"], label="controller ON")
	plt.plot(ev_off["t"], ev_off["mean_V"], label="no actuation")
	plt.xlabel("time (s)")
	plt.ylabel("mean V (mV)")
	plt.legend()
	plt.tight_layout()
	plt.savefig("fig_ablation_meanV.png", dpi=200)


if __name__ == "__main__":
	main()


