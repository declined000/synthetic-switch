from polarity_homeostat.decoder.rules import RulesDecoder, RulesThresholds, DecoderStability


def test_decoder_dwell_and_hysteresis():
	th = RulesThresholds(low_occ_threshold=0.3, energy_ok=0.3, mismatch_ok=0.3, healthy_plv_min=0.5)
	st = DecoderStability(hysteresis_margin=0.2, decision_dwell=5)
	dec = RulesDecoder(thresholds=th, stability=st)
	# Start REST
	a0 = dec.decide(low_occ=0.1, mismatch=0.9, E=0.5, plv=0.9, global_v_offset_mV=0.0, domain_low_fraction=0.1)
	assert a0 == 0
	# Propose REPAIR but with small margin; keep context neutral so margin < hysteresis
	a1 = dec.decide(low_occ=0.20, mismatch=0.50, E=0.20, plv=0.51, global_v_offset_mV=0.0, domain_low_fraction=0.1)
	assert a1 == 0
	# Increase margin to exceed hysteresis; first switch accepted → dwell starts
	th2 = RulesThresholds(low_occ_threshold=0.3, energy_ok=0.3, mismatch_ok=0.3, healthy_plv_min=0.9)
	dec.th = th2
	a2 = dec.decide(low_occ=0.9, mismatch=0.1, E=0.9, plv=0.0, global_v_offset_mV=20.0, domain_low_fraction=0.7)
	assert a2 == 1
	# While dwell>0, even strong REST proposals should be blocked
	for _ in range(4):
		a = dec.decide(low_occ=0.0, mismatch=1.0, E=1.0, plv=1.0, global_v_offset_mV=0.0, domain_low_fraction=0.0)
		assert a == 1
	# After dwell expires, REST allowed again
	a_after = dec.decide(low_occ=0.0, mismatch=1.0, E=1.0, plv=1.0, global_v_offset_mV=0.0, domain_low_fraction=0.0)
	assert a_after == 0
