import numpy as np
from polarity_homeostat.sensing.recorder import Recorder, RecorderConfig


def test_low_hysteresis_enter_exit():
	rcfg = RecorderConfig(low_enter=-12.0, low_exit=-18.0, tau_low=10.0)
	rec = Recorder(rcfg, grid=(1,1), dt=1.0)
	V = np.array([[-15.0]])  # between enter and exit → should not enter yet
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is False
	# Cross enter threshold (more depolarized)
	V[...] = -10.0
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is True
	# Come back below exit threshold (more hyperpolarized) to clear LOW
	V[...] = -20.0
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is False
