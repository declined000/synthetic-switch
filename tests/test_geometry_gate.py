from polarity_homeostat.safety.gates import geometry_gate


def test_geometry_gate_respects_coupling():
	mismatch_mean = 0.6
	max_mismatch = 0.3
	min_coupling = 0.1
	# High coupling: should enforce consensus (block)
	assert geometry_gate(mismatch_mean, max_mismatch, D_est=0.9, min_coupling=min_coupling) is False
	# Low coupling/isolation: should de-weight consensus (allow)
	assert geometry_gate(mismatch_mean, max_mismatch, D_est=0.05, min_coupling=min_coupling) is True
	# Unknown D_est: do not block on geometry (allow)
	assert geometry_gate(mismatch_mean, max_mismatch, D_est=None, min_coupling=min_coupling) is True
