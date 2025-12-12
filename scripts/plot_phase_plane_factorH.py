import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def tf_depol(V, V_thr: float = -20.0, sigma: float = 5.0):
	"""Map V (mV) to a 0-1 depolarization signal."""
	return 1.0 / (1.0 + np.exp(-(V - V_thr) / sigma))


def hill(x, n: float, K: float = 0.5):
	"""Hill activation with cooperativity n."""
	x_clip = np.clip(x, 0.0, 1.0)
	return x_clip**n / (K**n + x_clip**n)


def phase_plane(n: float = 2.0, out_path: str | Path = "results/phase_plane_n2.png"):
	# Parameters chosen to be qualitatively reasonable, not fit to data
	EL = -60.0   # mV
	gL = 0.05    # leak
	gH = 0.5     # hyperpolarizing strength of H

	k_prod = 0.5
	k_deg = 0.1

	# Grid
	V = np.linspace(-80, 20, 200)
	H = np.linspace(0.0, 1.0, 200)
	VV, HH = np.meshgrid(V, H)

	# Vector field
	dVdt = -gL * (VV - EL) - gH * HH
	TF = tf_depol(VV)
	H_inf = hill(TF, n)
	dHdt = k_prod * H_inf - k_deg * HH

	speed = np.sqrt(dVdt**2 + dHdt**2)
	dVdt_n = dVdt / (speed + 1e-9)
	dHdt_n = dHdt / (speed + 1e-9)

	fig, ax = plt.subplots(figsize=(6, 5))

	# Streamlines / arrows
	ax.streamplot(VV, HH, dVdt_n, dHdt_n, density=1.0, linewidth=0.7)

	# Nullcline dV/dt = 0  -> H = (gL/gH)*(V - EL)
	H_V_null = (gL / gH) * (V - EL)
	ax.plot(V, H_V_null, linestyle="--", label="dV/dt = 0")

	# Nullcline dH/dt = 0 -> H = (k_prod/k_deg) * Hill(TF(V))
	TF_line = tf_depol(V)
	H_H_null = (k_prod / k_deg) * hill(TF_line, n)
	ax.plot(V, H_H_null, linestyle="-", label="dH/dt = 0")

	ax.set_xlabel("V (mV)")
	ax.set_ylabel("H (Factor H activity)")
	ax.set_title(f"Phase plane for V–H subsystem (n={n})")
	ax.set_xlim(-80, 20)
	ax.set_ylim(0.0, 1.0)
	ax.legend()

	fig.tight_layout()
	out = Path(out_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=300)
	print(f"Saved {out}")


def main() -> None:
	# Example: two panels for low vs high cooperativity
	base = Path("results")
	(base / "phase_planes").mkdir(exist_ok=True)

	for n in (1.2, 3.0):
		phase_plane(n=n, out_path=base / "phase_planes" / f"phase_plane_n{n:.1f}.png")


if __name__ == "__main__":
	main()


