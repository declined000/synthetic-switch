import numpy as np
from polarity_homeostat.sensing.recorder import Recorder, RecorderConfig


def test_low_hysteresis_enter_exit():
	rcfg = RecorderConfig(low_enter=-15.0, low_exit=-8.0, tau_low=10.0)
	rec = Recorder(rcfg, grid=(1,1), dt=1.0)
	V = np.array([[-10.0]])  # between enter and exit → should not enter yet
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is False
	# Cross enter threshold
	V[...] = -20.0
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is True
	# Come back above exit threshold to clear LOW
	V[...] = -5.0
	rec.update_bands(V)
	assert bool(rec.low_state[0,0]) is False
