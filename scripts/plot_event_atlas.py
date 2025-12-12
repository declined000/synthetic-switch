import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
	base = Path("results/exp_baseline")
	df = pd.read_csv(base / "event_atlas.csv")
	with open(base / "summary.json", "r", encoding="utf-8") as f:
		summary = json.load(f)

	t = df["t"].values

	fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 7))

	# 1) Mean Vmem
	axes[0].plot(t, df["mean_V"])
	axes[0].set_ylabel("mean V (mV)")
	axes[0].axhline(-60.0, linestyle=":", linewidth=1)
	axes[0].set_title("Global dynamics (baseline controller)")

	# 2) Fraction of cells in LOW band (global)
	axes[1].plot(t, df["domain_low_fraction"])
	axes[1].set_ylabel("LOW frac\n(global)")
	axes[1].set_ylim(-0.05, 1.05)

	# 3) PLV
	plv = pd.to_numeric(df["PLV"], errors="coerce")
	axes[2].plot(t, plv)
	axes[2].set_ylabel("PLV")

	# 4) Max action across domains (0=REST,1=REPAIR,2=PRUNE)
	axes[3].step(t, df["action"], where="post")
	axes[3].set_ylabel("max action")
	axes[3].set_xlabel("time (s)")

	# Mark recovery time if present
	if summary.get("recovery_time_step") is not None:
		# event_atlas is subsampled at atlas_stride; map step index to nearest logged time
		dt = t[1] - t[0]
		t_rec = summary["recovery_time_step"] * dt
		for ax in axes:
			ax.axvline(t_rec, linestyle="--", linewidth=1)

	fig.tight_layout()
	out = base / "fig_event_atlas.png"
	fig.savefig(out, dpi=300)
	print(f"Saved {out}")


if __name__ == "__main__":
	main()


