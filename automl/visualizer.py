#!/usr/bin/env python3
"""visualizer.py – Generate diagnostic figures for the AutoML pipeline

This module is **self‑contained** and depends only on
`matplotlib`, `pandas`, `numpy` and the default Python std‑lib.
It operates directly on the data structures that
`AutoMLPipeline` already stores in `self.results`.

Usage (inside your pipeline code)  
---------------------------------
```python
from automl.visualizer import (
    plot_rl_learning_curve,
    plot_bohb_distribution,
    plot_action_heatmap,
)

plot_dir = Path(results_dir) / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)

plot_rl_learning_curve(pipeline.results, save=plot_dir / "rl_curve.png")
plot_bohb_distribution(pipeline.results, save=plot_dir / "bohb_box.png")
plot_action_heatmap(pipeline.results, save=plot_dir / "action_heatmap.png")
```
Each helper returns the Matplotlib **Axes** so you can further tweak or
display it inline (e.g. in a Jupyter notebook).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _df_from_logs(logs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list‑of‑dicts to a flattened DataFrame (safely)."""
    return pd.json_normalize(list(logs), sep="_")


# -----------------------------------------------------------------------------
# 1) RL learning curve
# -----------------------------------------------------------------------------

def plot_rl_learning_curve(results: Dict[str, Any], *, ax: plt.Axes | None = None, save: str | Path | None = None):
    """Line plot of mean BOHB accuracy per RL iteration."""
    df = _df_from_logs(results.get("rl_training_iterations", []))
    if df.empty:
        raise ValueError("No rl_training_iterations data found in results")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(df["iteration"], df["avg_performance"], marker="o", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean BOHB accuracy")
    ax.set_title("RL learning curve")
    ax.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 2) BOHB score distribution per dataset
# -----------------------------------------------------------------------------

def plot_bohb_distribution(results: Dict[str, Any], *, ax: plt.Axes | None = None, save: str | Path | None = None):
    """Box‑plot of BOHB scores for every dataset collected during training."""
    df = _df_from_logs(results.get("bohb_evaluations", []))
    if df.empty:
        raise ValueError("No bohb_evaluations found in results")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    order = sorted(df["dataset"].unique())
    df.boxplot(column="bohb_score", by="dataset", grid=False, ax=ax)
    ax.set_title("BOHB score distribution by dataset")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    fig.suptitle("")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 3) Action heat‑map (tier selection frequency)
# -----------------------------------------------------------------------------

def plot_action_heatmap(results: Dict[str, Any], *, ax: plt.Axes | None = None, save: str | Path | None = None):
    """Heat‑map of how often the agent picked each action tier per dataset."""
    df = _df_from_logs(results.get("detailed_logs", []))
    if df.empty:
        raise ValueError("No detailed_logs found in results")

    pivot = (
        df.groupby(["dataset", "action"]).size().unstack(fill_value=0)
    )
    for col in [0, 1, 2]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[[0, 1, 2]].sort_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Simple", "Medium", "Complex"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Model tier selections by dataset")

    for (i, j), v in np.ndenumerate(pivot.values):
        ax.text(j, i, str(v), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 4) Convenience batch saver
# -----------------------------------------------------------------------------
def plot_bohb_convergence(
    results: Dict[str, Any],
    dataset: str = "yelp",
    *,
    ax: plt.Axes | None = None,
    save: str | Path | None = None,
):
    """Step‑line of incumbent accuracy over BOHB trials for *dataset*."""
    logs = [
        d for d in results.get("detailed_logs", [])
        if d.get("dataset") == dataset and d.get("bohb", {}).get("incumbent_history")
    ]
    if not logs:
        raise ValueError(f"No BOHB history found for dataset '{dataset}'")

    # Take the last logged history (most complete)
    hist = logs[-1]["bohb"]["incumbent_history"]
    trials  = [h.get("trial", i)  for i, h in enumerate(hist)]
    scores  = [h.get("score")     for h in hist]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    ax.step(trials, scores, where="post")
    ax.set_xlabel("BOHB trial")
    ax.set_ylabel("Incumbent accuracy")
    ax.set_title(f"BOHB convergence – {dataset}")
    ax.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 5) Q‑value heat‑map across training
# -----------------------------------------------------------------------------

def plot_qvalue_heatmap(results: Dict[str, Any], *, ax: plt.Axes | None = None, save: str | Path | None = None):
    """Heat‑map of agent Q‑values (actions × time)."""
    df = _df_from_logs(results.get("detailed_logs", []))
    if df.empty or "q_values" not in df:
        raise ValueError("q_values missing in detailed_logs")

    # Expand list of q‑values into columns q0, q1, q2, …
    q_df = pd.DataFrame(df["q_values"].to_list())
    q_mat = q_df.values.T        # rows = actions

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    im = ax.imshow(q_mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action id")
    ax.set_title("Q‑values over time")
    fig.colorbar(im, ax=ax, label="Q‑value")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 6) PCA of meta‑features, colored by chosen model
# -----------------------------------------------------------------------------

def plot_meta_pca(
    results: Dict[str, Any],
    feature_order: List[str],
    *,
    ax: plt.Axes | None = None,
    save: str | Path | None = None,
):
    """2‑D PCA of meta‑features, point color = model tier selected."""
    meta = results.get("meta_features", {})
    selections = results.get("final_selections", {})

    if not meta or not selections:
        raise ValueError("meta_features or final_selections missing")

    df = pd.DataFrame(meta).T
    X = df[feature_order].values

    from sklearn.decomposition import PCA

    pca_xy = PCA(n_components=2).fit_transform(X)
    labels = [selections.get(ds, {}).get("model_type", "?") for ds in df.index]
    colors = pd.factorize(labels)[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    scatter = ax.scatter(pca_xy[:, 0], pca_xy[:, 1], c=colors, s=70, cmap="tab10")
    for i, txt in enumerate(df.index):
        ax.annotate(txt, (pca_xy[i, 0], pca_xy[i, 1]), fontsize=8, alpha=0.7)
    ax.set_xlabel("PC‑1"); ax.set_ylabel("PC‑2"); ax.set_title("Datasets in PCA space")
    handles, _ = scatter.legend_elements(prop="colors")
    ax.legend(handles, sorted(set(labels)), title="Model tier", loc="best", fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 7) Simple timeline (Gantt‑like) of pipeline stages
# -----------------------------------------------------------------------------

def plot_timeline(results: Dict[str, Any], *, ax: plt.Axes | None = None, save: str | Path | None = None):
    """Horizontal bars showing when each pipeline stage occurred."""
    tl = results.get("timeline", [])
    if not tl:
        raise ValueError("timeline missing in results")

    stages   = [ev["stage"] for ev in tl]
    starts   = [ev["elapsed_minutes"] for ev in tl]
    widths   = [0.5] * len(starts)    # constant bar height
    ypos     = np.arange(len(stages))[::-1]     # top‑down

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.barh(ypos, widths, left=starts, height=0.4, color="skyblue")
    ax.set_yticks(ypos)
    ax.set_yticklabels(stages, fontsize=8)
    ax.set_xlabel("Minutes since start")
    ax.set_title("Pipeline timeline")
    ax.grid(axis="x", ls=":", alpha=0.5)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return ax


# -----------------------------------------------------------------------------
# 8) Convenience batch saver (extended)
# -----------------------------------------------------------------------------

def save_all_figures(results: Dict[str, Any], out_dir: str | Path, feature_order: List[str] | None = None):
    """Generate and save **all** visuals into *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rl_learning_curve(results,      save=out_dir / "rl_learning_curve.png")
    plot_bohb_distribution(results,      save=out_dir / "bohb_scores_box.png")
    plot_action_heatmap(results,         save=out_dir / "action_heatmap.png")
    plot_bohb_convergence(results,       save=out_dir / "bohb_convergence_yelp.png")
    plot_qvalue_heatmap(results,         save=out_dir / "qvalues_heatmap.png")

    if feature_order is not None:
        plot_meta_pca(results, feature_order, save=out_dir / "meta_pca.png")

    plot_timeline(results,               save=out_dir / "pipeline_timeline.png")
    print(f"Figures saved to {out_dir}")

