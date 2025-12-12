import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
	base = Path("results/exp_baseline")
	with open(base / "roc_pr.json", "r", encoding="utf-8") as f:
		data = json.load(f)

	scores = np.array(data["scores"])
	labels = np.array(data["labels"])
	roc = data["roc"]
	pr = data["pr"]
	auc_roc = data["auc_roc"]
	auc_pr = data["auc_pr"]

	# 1) ROC curve
	fig, ax = plt.subplots(figsize=(4, 4))
	ax.plot([0, 1], [0, 1], linestyle="--")
	ax.plot(
		[p["fpr"] for p in roc],
		[p["tpr"] for p in roc],
		marker="o",
	)
	ax.set_xlabel("False positive rate")
	ax.set_ylabel("True positive rate")
	ax.set_title(f"ROC (AUC = {auc_roc:.2f})")
	fig.tight_layout()
	fig.savefig(base / "fig_detection_roc.png", dpi=300)
	plt.close(fig)

	# 2) PR curve
	fig, ax = plt.subplots(figsize=(4, 4))
	ax.plot(
		[p["recall"] for p in pr],
		[p["precision"] for p in pr],
		marker="o",
	)
	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title(f"PR (AUC = {auc_pr:.2f})")
	fig.tight_layout()
	fig.savefig(base / "fig_detection_pr.png", dpi=300)
	plt.close(fig)

	# 3) Score vs label bar chart (just to show separation)
	fig, ax = plt.subplots(figsize=(4, 3))
	x = np.arange(len(scores))
	ax.bar(x, scores, color=["red" if l == 1 else "gray" for l in labels])
	ax.set_xticks(x)
	ax.set_xticklabels([f"dom{i}" for i in x])
	ax.set_ylabel("late-window LOW_occ_dom")
	ax.set_title("Detection scores (red = chronic label)")
	fig.tight_layout()
	fig.savefig(base / "fig_detection_scores.png", dpi=300)
	plt.close(fig)

	print("Saved ROC/PR + score bar figures in results/exp_baseline")


if __name__ == "__main__":
	main()


