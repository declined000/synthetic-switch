import numpy as np
from polarity_homeostat.sensing.osc import OscillationDetector, OscConfig


def test_plv_bounds_with_and_without_bandpass():
	dt = 0.1
	grid = (1,1)
	# Build a sine wave in mean(V)
	t = np.arange(0, 50, dt)
	sig = np.sin(2*np.pi*0.02*t)  # 0.02 Hz
	# Detector without band-pass
	osc1 = OscillationDetector(OscConfig(window_seconds=10.0, healthy_plv_min=0.5, downsample=1), grid=grid, dt=dt)
	for v in sig:
		osc1.update(np.array([[v]], dtype=float))
		plv, _ = osc1.plv_with_persistence()
	if plv is not None:
		assert 0.0 <= plv <= 1.0
	# Detector with band-pass around the sine
	osc2 = OscillationDetector(OscConfig(window_seconds=10.0, healthy_plv_min=0.5, downsample=1, bandpass=(0.005,0.05)), grid=grid, dt=dt)
	for v in sig:
		osc2.update(np.array([[v]], dtype=float))
		plv2, _ = osc2.plv_with_persistence()
	if plv2 is not None:
		assert 0.0 <= plv2 <= 1.0
