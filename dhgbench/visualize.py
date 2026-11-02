"""Plot DHG-Bench results saved by lib_utils.exp_agent.save_results.

Run from the ``dhgbench`` directory, e.g.::

    python visualize.py --plot bar --metric test_acc
    python visualize.py --plot line --metric test_acc --x pert_p
    python visualize.py --plot curve --methods HGNN --datasets cora
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")


def as_bool(value):
    return str(value).lower() in ("1", "true", "yes", "t")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flatten_metrics(payload):
    """Normalize the three task layouts into {metric_name: 1D array}."""
    metrics = payload.get("metrics", {})
    task = payload.get("task")
    out = {}

    if task == "edge_pred":
        # {"train": {...}, "val": {...}, "test": {...}}
        for split, split_metrics in metrics.items():
            for key, values in split_metrics.items():
                out[f"{split}_{key}"] = np.asarray(values, dtype=float).ravel()
        return out

    # node_cls / hg_cls: {metric: [[train, valid, test], ...]} per seed
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == len(SPLITS):
            for i, split in enumerate(SPLITS):
                out[f"{split}_{key}"] = arr[:, i]
        else:
            out[key] = arr.ravel()
    return out


def load_results(root):
    records, curves = [], []
    root = Path(root)
    if not root.exists():
        return pd.DataFrame(), []

    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue

        cfg = payload.get("config", {})
        base = {
            "path": str(path),
            "task": payload.get("task"),
            "dataset": payload.get("dataset"),
            "method": payload.get("method"),
            "perturbed": as_bool(cfg.get("is_perturbed", False)),
            "pert_mode": cfg.get("pert_mode"),
            "pert_p": as_float(cfg.get("pert_p")),
        }

        for metric, values in flatten_metrics(payload).items():
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            records.append({
                **base,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n": int(values.size),
            })

        if payload.get("histories"):
            curves.append((base, payload["histories"]))

    return pd.DataFrame(records), curves


def plot_bar(df, metric, out, datasets=None, methods=None):
    sub = df[(df["metric"] == metric) & (~df["perturbed"])].copy()
    if datasets:
        sub = sub[sub["dataset"].isin(datasets)]
    if methods:
        sub = sub[sub["method"].isin(methods)]
    if sub.empty:
        raise SystemExit(
            f"No unperturbed results for metric={metric!r}. "
            f"Available metrics: {sorted(df['metric'].unique())}"
        )

    sub = sub.sort_values("path").drop_duplicates(["method", "dataset"])
    methods_order = methods or sorted(sub["method"].unique())
    datasets_order = datasets or sorted(sub["dataset"].unique())

    x = np.arange(len(datasets_order))
    n = max(len(methods_order), 1)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(datasets_order) + 3), 4.5))

    for i, method in enumerate(methods_order):
        s = sub[sub["method"] == method].set_index("dataset").reindex(datasets_order)
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            s["mean"].to_numpy(dtype=float),
            width,
            yerr=s["std"].to_numpy(dtype=float),
            capsize=3,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (mean ± std)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out}")


def plot_line(df, metric, xcol, out, methods=None):
    sub = df[(df["metric"] == metric) & df["perturbed"] & df[xcol].notna()].copy()
    if methods:
        sub = sub[sub["method"].isin(methods)]
    if sub.empty:
        raise SystemExit(
            f"No perturbed results for metric={metric!r} and x={xcol!r}. "
            "Run experiments with --is_perturbed=True."
        )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, g in sub.groupby("method"):
        g = g.sort_values(xcol)
        ax.errorbar(
            g[xcol].astype(float),
            g["mean"],
            yerr=g["std"],
            marker="o",
            capsize=3,
            label=method,
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} under perturbation")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out}")


def plot_curves(curves, out, method=None, dataset=None):
    selected = [
        (base, histories)
        for base, histories in curves
        if (method is None or base["method"] == method)
        and (dataset is None or base["dataset"] == dataset)
        and any(rec.get("history") for rec in histories)
    ]
    if not selected:
        raise SystemExit(
            "No training curves found. Make sure Trainer.history is recorded "
            "and --epochs is at least --display_step."
        )

    keys = [k for k in selected[0][1][0]["history"][0] if k != "epoch"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)

    for ax, key in zip(axes[0], keys):
        series = []
        for _, histories in selected:
            for rec in histories:
                epochs = [r["epoch"] for r in rec["history"]]
                values = [r.get(key) for r in rec["history"]]
                if all(v is not None for v in values):
                    series.append(pd.Series(values, index=epochs))
        if not series:
            ax.set_visible(False)
            continue

        mat = pd.concat(series, axis=1)
        mean = mat.mean(axis=1)
        std = mat.std(axis=1).fillna(0)
        ax.plot(mat.index, mean)
        ax.fill_between(mat.index, mean - std, mean + std, alpha=0.2)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DHG-Bench results")
    parser.add_argument("--results", default="./results", help="results root directory")
    parser.add_argument("--plot", choices=["bar", "line", "curve"], default="bar")
    parser.add_argument("--metric", default="test_acc")
    parser.add_argument("--x", default="pert_p", help="x-axis column for --plot line")
    parser.add_argument("--out", default=None, help="output image path")
    parser.add_argument("--task", default=None,
                        choices=["node_cls", "edge_pred", "hg_cls"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--include-perturbed", action="store_true",
                        help="also use perturbed runs in the bar plot")
    args = parser.parse_args()

    df, curves = load_results(args.results)
    if args.task:
        df = df[df["task"] == args.task]
        curves = [(b, h) for b, h in curves if b["task"] == args.task]
    if df.empty and not curves:
        raise SystemExit(f"No result JSON found under {args.results!r}")

    out = Path(args.out or f"figures/{args.plot}_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.plot == "bar":
        if args.include_perturbed:
            df["perturbed"] = False
        plot_bar(df, args.metric, out, args.datasets, args.methods)
    elif args.plot == "line":
        plot_line(df, args.metric, args.x, out, args.methods)
    else:
        method = args.methods[0] if args.methods else None
        dataset = args.datasets[0] if args.datasets else None
        plot_curves(curves, out, method, dataset)


if __name__ == "__main__":
    main()
