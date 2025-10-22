import numpy as np
from polarity_homeostat.model.energy import Energy, EnergyConfig


def test_energy_non_negative_with_actuation_and_flux():
	cfg = EnergyConfig(
		grid=(3,3), E0=0.3, k_oxphos=0.05, alpha_actuation_cost=0.5,
		beta_tnt_flux=0.1, gamma_decay=0.01, Emin=0.2
	)
	energy = Energy(cfg)
	# Strong actuation cost and positive flux; step several times
	for _ in range(100):
		u_act = np.full((3,3), 10.0, dtype=float)  # large cost
		energy.step(dt=0.1, u_act=u_act, u_tnt_ev=0.0)
		assert np.all(energy.E >= 0.0)
