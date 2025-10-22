from polarity_homeostat.eval.metrics import compute_recovery_time, compute_plv_retention


def test_metrics_smoke():
	# Recovery: below 0.1 for 3s with dt=1s and dwell_s=3s
	domain_series = [0.5]*5 + [0.08,0.09,0.05,0.04] + [0.2]*5
	rec_step = compute_recovery_time(domain_series, dt=1.0, threshold=0.1, dwell_s=3.0)
	assert rec_step is not None
	# PLV retention: first window vs last window ratio
	plv_series = [0.5]*10 + [0.8]*10
	ret = compute_plv_retention(plv_series, dt=1.0, window_s=5.0)
	assert ret is not None and 0.0 <= ret <= 1.0
