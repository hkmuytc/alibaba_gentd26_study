from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "window_sweep"
FIGURE_PATH = OUTPUT_DIR / "fig_methodology_windowing.png"
LAYOUT_PATH = OUTPUT_DIR / "fig_methodology_windowing_layout.json"

HISTORY_COLOR = "#bfe3cd"
ONE_STEP_COLOR = "#f5bf2a"
MULTI_STEP_COLOR = "#61c3f2"
VAL_COLOR = "#fff1b8"
TEST_COLOR = "#f7d9c4"
EDGE_COLOR = "#333333"
LINE_COLOR = "#222222"


def synthetic_series(n: int = 320) -> tuple[np.ndarray, np.ndarray]:
	x = np.linspace(0, 14, n)
	y = 1.45 + 0.55 * np.sin(0.9 * x) + 0.20 * np.cos(1.9 * x + 0.35)
	return x, y


def style_axis(ax) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_linewidth(1.2)
	ax.spines["bottom"].set_linewidth(1.2)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel("Time", fontsize=10)
	ax.set_ylabel("Power", fontsize=10)


def draw_one_step_panel(ax):
	x, y = synthetic_series()
	history_start = 7.2
	history_end = 8.7
	target_end = 9.1

	ax.plot(x, y, color=LINE_COLOR, linewidth=2.1)
	ax.axvspan(history_start, history_end, color=HISTORY_COLOR, alpha=0.95)
	ax.axvspan(history_end, target_end, color=ONE_STEP_COLOR, alpha=0.95)

	for marker in [history_start, history_end, target_end]:
		ax.axvline(marker, color="#777777", linestyle="--", linewidth=0.9)

	ax.text(0.02, 0.98, "A. One-step prediction", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

	style_axis(ax)
	return {
		"panel": "one_step",
		"history_window": {"start": history_start, "end": history_end},
		"target_window": {"start": history_end, "end": target_end},
	}


def draw_autoregressive_panel(ax):
	x, y = synthetic_series()
	history_start = 7.0
	history_end = 8.3
	step_width = 0.34
	n_steps = 5

	ax.plot(x, y, color=LINE_COLOR, linewidth=2.1)
	ax.axvspan(history_start, history_end, color=HISTORY_COLOR, alpha=0.95)
	ax.axvline(history_start, color="#777777", linestyle="--", linewidth=0.9)
	ax.axvline(history_end, color="#777777", linestyle="--", linewidth=0.9)

	rollout_boxes = []
	for step_idx in range(n_steps):
		x0 = history_end + step_idx * step_width
		rect = Rectangle((x0, 0.42), step_width, 1.92, facecolor=MULTI_STEP_COLOR, edgecolor=EDGE_COLOR, linewidth=0.9)
		ax.add_patch(rect)
		rollout_boxes.append({"start": x0, "end": x0 + step_width, "step": step_idx + 1})

	arrow = FancyArrowPatch(
		(history_end + step_width * 1.2, 0.55),
		(history_end + 0.03, 1.10),
		arrowstyle="->",
		connectionstyle="arc3,rad=-0.18",
		mutation_scale=11,
		linewidth=1.5,
		color="#555555",
	)
	ax.add_patch(arrow)

	ax.text(0.02, 0.98, "B. Autoregressive multi-step rollout", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
	ax.text(0.58, 0.18, "feedback", transform=ax.transAxes, fontsize=9)

	style_axis(ax)
	return {
		"panel": "autoregressive",
		"history_window": {"start": history_start, "end": history_end},
		"rollout_boxes": rollout_boxes,
	}


def draw_rolling_origin_panel(ax):
	ax.axis("off")
	ax.set_xlim(0, 12)
	ax.set_ylim(0, 4.5)

	ax.text(0.0, 4.2, "C. Rolling-origin evaluation", fontsize=12, fontweight="bold", ha="left", va="top")
	ax.text(0.0, 3.75, "each later origin retrains, validates, and tests on a later unseen block", fontsize=9)

	rows = []
	start_x = 1.4
	val_width = 1.2
	test_width = 1.2
	bar_height = 0.58
	train_widths = [4.2, 5.1, 6.0]
	y_positions = [2.8, 1.85, 0.9]

	for idx, (train_width, y) in enumerate(zip(train_widths, y_positions), start=1):
		label = f"Origin {idx}"
		ax.text(0.25, y + bar_height / 2, label, ha="left", va="center", fontsize=10)

		train_rect = Rectangle((start_x, y), train_width, bar_height, facecolor=HISTORY_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2)
		val_rect = Rectangle((start_x + train_width, y), val_width, bar_height, facecolor=VAL_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2)
		test_rect = Rectangle((start_x + train_width + val_width, y), test_width, bar_height, facecolor=TEST_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2)
		ax.add_patch(train_rect)
		ax.add_patch(val_rect)
		ax.add_patch(test_rect)

		ax.text(start_x + train_width / 2, y + bar_height / 2, "Train", ha="center", va="center", fontsize=9)
		ax.text(start_x + train_width + val_width / 2, y + bar_height / 2, "Val", ha="center", va="center", fontsize=9)
		ax.text(start_x + train_width + val_width + test_width / 2, y + bar_height / 2, "Test", ha="center", va="center", fontsize=9)

		rows.append({
			"label": label,
			"train": {"x": start_x, "y": y, "w": train_width, "h": bar_height},
			"validation": {"x": start_x + train_width, "y": y, "w": val_width, "h": bar_height},
			"test": {"x": start_x + train_width + val_width, "y": y, "w": test_width, "h": bar_height},
		})

	return {"panel": "rolling_origin", "rows": rows}


def draw_legend(fig):
	handles = [
		Rectangle((0, 0), 1, 1, facecolor=ONE_STEP_COLOR, edgecolor=EDGE_COLOR),
		Rectangle((0, 0), 1, 1, facecolor=MULTI_STEP_COLOR, edgecolor=EDGE_COLOR),
		Rectangle((0, 0), 1, 1, facecolor=HISTORY_COLOR, edgecolor=EDGE_COLOR),
		Rectangle((0, 0), 1, 1, facecolor=VAL_COLOR, edgecolor=EDGE_COLOR),
		Rectangle((0, 0), 1, 1, facecolor=TEST_COLOR, edgecolor=EDGE_COLOR),
	]
	labels = [
		"One-step target",
		"Recursive future steps",
		"Historical context / train window",
		"Validation block",
		"Test block",
	]
	fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, fontsize=9)


def build_methodology_figure():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	fig = plt.figure(figsize=(12, 8))
	gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.42, wspace=0.28)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])
	ax3 = fig.add_subplot(gs[1, :])

	one_step_layout = draw_one_step_panel(ax1)
	autoregressive_layout = draw_autoregressive_panel(ax2)
	rolling_origin_layout = draw_rolling_origin_panel(ax3)
	draw_legend(fig)

	fig.subplots_adjust(top=0.83, bottom=0.08, left=0.06, right=0.98, hspace=0.42, wspace=0.28)
	fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
	plt.close(fig)

	layout = {
		"figure": FIGURE_PATH.name,
		"panels": [one_step_layout, autoregressive_layout, rolling_origin_layout],
	}
	LAYOUT_PATH.write_text(json.dumps(layout, indent=2), encoding="utf-8")
	print(f"Saved: {FIGURE_PATH}")
	print(f"Saved: {LAYOUT_PATH}")


if __name__ == "__main__":
	build_methodology_figure()
