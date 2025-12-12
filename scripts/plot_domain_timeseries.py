from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_domain(ax_row, dom_df, label_prefix: str) -> None:
	t = dom_df["t"].values

	# Top: V
	ax_V, ax_low = ax_row

	ax_V.plot(t, dom_df["mean_V_dom"])
	ax_V.axhline(-60.0, linestyle=":", linewidth=1)
	ax_V.set_ylabel(f"{label_prefix}\nV_dom (mV)")

	# Bottom: LOW occupancy & LOW fraction
	ax_low.plot(t, dom_df["LOW_occ_dom"], label="LOW_occ_dom")
	ax_low.plot(
		t,
		dom_df["domain_low_fraction_dom"],
		linestyle="--",
		label="domain_low_frac_dom",
	)
	ax_low.set_ylabel(f"{label_prefix}\nLOW metrics")
	ax_low.set_ylim(-0.05, 1.05)
	# Optionally zoom early time to make LOW decay clearer
	ax_low.set_xlim(0, 50)


def main() -> None:
	base = Path("results/exp_baseline")
	df = pd.read_csv(base / "domain_atlas.csv")

	fig, axes = plt.subplots(
		nrows=4, ncols=2, sharex=False, figsize=(8, 9)
	)

	for dom_id, (row_V, row_low) in enumerate(axes):
		dom_df = df[df["domain_id"] == dom_id]
		label = f"domain {dom_id}"
		plot_domain((row_V, row_low), dom_df, label_prefix=label)

	axes[-1][1].set_xlabel("time (s)")
	axes[-1][1].legend(loc="upper right", fontsize=7)

	fig.tight_layout()
	out = base / "fig_domain_timeseries.png"
	fig.savefig(out, dpi=300)
	print(f"Saved {out}")


if __name__ == "__main__":
	main()


