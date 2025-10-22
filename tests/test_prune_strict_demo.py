from polarity_homeostat.decoder.rules import RulesDecoder, RulesThresholds, DecoderStability


def test_prune_strict_demo():
	th = RulesThresholds(
		low_occ_threshold=0.3, energy_ok=0.3, mismatch_ok=0.3, healthy_plv_min=0.5,
		prune_enabled=True, prune_low_occ_threshold=0.9, prune_energy_max=0.2, prune_mismatch_min=0.7, prune_dwell_steps=5,
	)
	st = DecoderStability(hysteresis_margin=0.05, decision_dwell=1)
	dec = RulesDecoder(thresholds=th, stability=st)
	# Feed frames that meet prune conditions long enough
	acts = []
	for _ in range(6):
		a = dec.decide(low_occ=0.95, mismatch=0.8, E=0.1, plv=0.1, global_v_offset_mV=0.0, domain_low_fraction=0.9)
		acts.append(a)
	assert acts[-1] == 2
