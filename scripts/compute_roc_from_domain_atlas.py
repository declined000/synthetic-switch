#!/usr/bin/env python
"""
Compute ROC and PR curves from domain_atlas.csv.

Ground truth:
  - A domain is labeled "chronic" if its mean LOW_occ_dom
    over the *entire* simulation is >= gt_threshold.

Score:
  - For each domain, we compute mean(LOW_occ_dom) over
    the last `late_window` seconds (or full run if shorter). This is
    the detection score; high score → more likely chronic.

Outputs:
  - JSON file with scores, labels, ROC (FPR, TPR), PR (recall, precision),
    and AUCs for both curves.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_roc_pr(scores: np.ndarray, labels: np.ndarray):
	"""
	Compute ROC and PR curves and AUCs from scores and binary labels.

	scores: shape (N,), higher means more likely positive
	labels: shape (N,), 0 or 1
	"""
	scores = np.asarray(scores, dtype=float)
	labels = np.asarray(labels, dtype=int)

	if scores.shape != labels.shape:
		raise ValueError("scores and labels must have the same shape")

	N = scores.shape[0]
	if N == 0:
		raise ValueError("Empty scores/labels")

	P = int(labels.sum())
	N_neg = int(N - P)

	# Sort by score descending
	order = np.argsort(scores)[::-1]
	labels_sorted = labels[order]

	# Cumulative TP / FP as we sweep threshold from +inf -> -inf
	tps = np.cumsum(labels_sorted)
	fps = np.cumsum(1 - labels_sorted)

	# Avoid division by zero
	P_safe = P if P > 0 else 1
	N_safe = N_neg if N_neg > 0 else 1

	tpr = tps / P_safe  # recall
	fpr = fps / N_safe

	# Add (0,0) and (1,1) endpoints for ROC
	fpr_points = np.concatenate([[0.0], fpr, [1.0]])
	tpr_points = np.concatenate([[0.0], tpr, [1.0]])

	auc_roc = float(np.trapz(tpr_points, fpr_points))

	# Precision-Recall curve
	precision = tps / np.maximum(tps + fps, 1)
	recall = tpr

	# Add endpoints: recall=0 → precision=1 (convention), recall=1 → last precision
	if precision.size == 0:
		precision_points = np.array([1.0, 0.0])
		recall_points = np.array([0.0, 1.0])
	else:
		precision_points = np.concatenate([[1.0], precision, [precision[-1]]])
		recall_points = np.concatenate([[0.0], recall, [1.0]])

	auc_pr = float(np.trapz(precision_points, recall_points))

	roc = [
		{"fpr": float(x), "tpr": float(y)}
		for x, y in zip(fpr_points, tpr_points)
	]
	pr = [
		{"recall": float(x), "precision": float(y)}
		for x, y in zip(recall_points, precision_points)
	]

	return roc, pr, auc_roc, auc_pr


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain_atlas", required=True, help="Path to domain_atlas.csv")
	parser.add_argument("--out", required=True, help="Output directory for ROC/PR JSON")
	parser.add_argument("--gt_threshold", type=float, default=0.5,
	                    help="Threshold on mean LOW_occ_dom to label a domain as chronic")
	parser.add_argument("--late_window", type=float, default=300.0,
	                    help="Window [t_max - late_window, t_max] over which to compute detection scores")
	args = parser.parse_args()

	df = pd.read_csv(args.domain_atlas)
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)

	if "domain_id" not in df.columns or "LOW_occ_dom" not in df.columns:
		raise ValueError("domain_atlas.csv must contain 'domain_id' and 'LOW_occ_dom' columns")

	t_max = float(df["t"].max())
	late_start = max(0.0, t_max - float(args.late_window))

	domain_ids = sorted(df["domain_id"].unique().tolist())

	scores = []
	labels = []

	for dom_id in domain_ids:
		d = df[df["domain_id"] == dom_id]

		# Ground truth: chronic if mean LOW occupancy over entire run >= gt_threshold
		mean_low_full = float(d["LOW_occ_dom"].mean())
		labels.append(1 if mean_low_full >= args.gt_threshold else 0)

		# Detection score: mean LOW occupancy over last late_window seconds
		d_late = d[d["t"] >= late_start]
		if len(d_late) == 0:
			mean_low_late = mean_low_full
		else:
			mean_low_late = float(d_late["LOW_occ_dom"].mean())
		scores.append(mean_low_late)

	scores = np.asarray(scores, dtype=float)
	labels = np.asarray(labels, dtype=int)

	roc, pr, auc_roc, auc_pr = compute_roc_pr(scores, labels)

	out_path = out_dir / "roc_pr.json"
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"scores": scores.tolist(),
				"labels": labels.tolist(),
				"gt_threshold": float(args.gt_threshold),
				"late_window": float(args.late_window),
				"roc": roc,
				"pr": pr,
				"auc_roc": auc_roc,
				"auc_pr": auc_pr,
			},
			f,
			indent=2,
		)

	print(f"[ROC/PR] N_domains={len(scores)}, AUC_ROC={auc_roc:.3f}, AUC_PR={auc_pr:.3f}")
	print(f"Wrote {out_path}")


if __name__ == "__main__":
	main()


