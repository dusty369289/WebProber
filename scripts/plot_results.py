#!/usr/bin/env python3
"""
Utility script that visualizes probe results stored in the `results/` folder.

Example:
    python scripts/plot_results.py --json-files 4000 4005 --filesizes 16K 1M 2M
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt


Aggregator = Callable[[Sequence[float]], float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot measurement summaries stored in WebProber result files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory that contains JSON files (default: results).",
    )
    mode_help = (
        "Plotting mode: "
        "'summary' compares aggregate stats across filesizes (default). "
        "'packets' graphs per-packet timings for a single filesize across ports."
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "packets", "success"],
        default="summary",
        help=mode_help,
    )
    parser.add_argument(
        "--json-files",
        nargs="+",
        help="Specific JSON basenames (e.g. 4000) or filenames to plot. Defaults to all.",
    )
    parser.add_argument(
        "--filesizes",
        nargs="+",
        help="(summary mode) Subset of payload sizes to include (e.g. 16K 256K 1M). Defaults to all found.",
    )
    parser.add_argument(
        "--filesize",
        help="(packets mode) Single payload size whose per-packet timings are graphed.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median", "min", "max", "p95"],
        default="mean",
        help="Statistic applied to each measurement series (default: mean).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional smoothing window (number of points) applied to series before plotting. "
        "Use values >1 to reduce noise (simple moving average).",
    )
    parser.add_argument(
        "--title",
        default="WebProber Measurements",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--output",
        help="Optional image path. When omitted, the plot opens interactively.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Force showing the plot even when --output is provided.",
    )
    return parser


def resolve_json_paths(results_dir: Path, selection: Iterable[str] | None) -> List[Path]:
    if selection:
        resolved = []
        for item in selection:
            path = Path(item)
            if not path.suffix:
                path = results_dir / f"{item}.json"
            elif not path.is_absolute():
                path = results_dir / path
            resolved.append(path)
    else:
        resolved = sorted(results_dir.glob("*.json"))

    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Could not find: {', '.join(missing)}")
    return resolved


def get_aggregator(name: str) -> Aggregator:
    if name == "mean":
        return fmean
    if name == "median":
        return median
    if name == "min":
        return min
    if name == "max":
        return max
    if name == "p95":
        return lambda values: percentile(values, 95)
    raise ValueError(f"Unsupported aggregate: {name}")


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def determine_filesizes(
    datasets: list[dict],
    override: Sequence[str] | None,
) -> List[str]:
    if override:
        return list(override)

    ordered_sizes: list[str] = []
    seen = set()
    for data in datasets:
        sizes = data.get("filesizes") or sorted(data["measurements"].keys())
        for size in sizes:
            if size not in seen:
                seen.add(size)
                ordered_sizes.append(size)
    return ordered_sizes


def load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_series(
    dataset: dict,
    filesizes: Sequence[str],
    aggregator: Aggregator,
) -> List[float]:
    measurements = dataset.get("measurements", {})
    series: List[float] = []
    for size in filesizes:
        raw_values = measurements.get(size) or []
        # Treat any null/None entries as a fixed sentinel latency of 3.0 seconds
        values = [v if v is not None else 3.0 for v in raw_values]
        series.append(aggregator(values) if values else math.nan)
    return series


def smooth(values: Sequence[float], window: int) -> List[float]:
    """Apply a simple moving-average smoothing with the given window size."""
    if window <= 1 or len(values) <= 1:
        return list(values)
    window = min(window, len(values))
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    # 'same' keeps the output length identical to the input
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def plot_results(
    datasets: Sequence[tuple[str, dict]],
    filesizes: Sequence[str],
    aggregator_name: str,
    title: str,
    output_path: str | None,
    force_show: bool,
    smooth_window: int,
) -> None:
    aggregator = get_aggregator(aggregator_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in datasets:
        series = compute_series(data, filesizes, aggregator)
        series = smooth(series, smooth_window)
        ax.plot(filesizes, series, marker="o", label=label)

    ax.set_ylabel("Seconds")
    ax.set_xlabel("Payload size")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Result file")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if force_show or not output_path:
        plt.show()
    else:
        plt.close(fig)


def plot_packets(
    datasets: Sequence[tuple[str, dict]],
    filesize: str,
    title: str,
    output_path: str | None,
    force_show: bool,
    smooth_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in datasets:
        port_label = str(data.get("port", label))
        raw = data.get("measurements", {}).get(filesize, []) or []
        # Treat any null/None entries as a fixed sentinel latency of 3.0 seconds
        y_values = [v if v is not None else 3.0 for v in raw]
        y_values = smooth(y_values, smooth_window)
        if not y_values:
            continue
        x_values = list(range(1, len(y_values) + 1))
        ax.plot(x_values, y_values, marker="o", label=f"port {port_label}")

    ax.set_xlabel("Packet number")
    ax.set_ylabel("Seconds")
    ax.set_title(title or f"Per-packet timings for {filesize}")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Place legend outside the main plotting area on the right
    ax.legend(
        title="Port",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    # Use a slightly reduced right margin to accommodate the external legend
    # without leaving excessive whitespace.
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    if output_path:
        fig.savefig(output_path, dpi=150)
    if force_show or not output_path:
        plt.show()
    else:
        plt.close(fig)


def compute_success_and_latency(
    dataset: dict,
    filesizes: Sequence[str],
) -> tuple[List[float], List[float]]:
    """
    For each filesize, compute:
    - success rate: percentage of non-null entries
    - average response time (seconds), ignoring null entries
    """
    measurements = dataset.get("measurements", {})
    success_rates: List[float] = []
    avg_latencies: List[float] = []

    for size in filesizes:
        raw_values = measurements.get(size) or []
        if not raw_values:
            success_rates.append(0.0)
            avg_latencies.append(math.nan)
            continue

        non_null = [v for v in raw_values if v is not None]
        total = len(raw_values)
        successes = len(non_null)

        success_rate = (successes / total) * 100 if total > 0 else 0.0
        avg_latency = fmean(non_null) if non_null else math.nan

        success_rates.append(success_rate)
        avg_latencies.append(avg_latency)

    return success_rates, avg_latencies


def plot_success_and_latency(
    dataset_label: str,
    dataset: dict,
    filesizes: Sequence[str],
    title: str,
    output_path: str | None,
    force_show: bool,
) -> None:
    """
    Plot success rate (%) and average response time (s) per filesize
    for a single result dataset.
    """
    success_rates, avg_latencies = compute_success_and_latency(dataset, filesizes)

    x = np.arange(len(filesizes))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.6
    bars = ax1.bar(
        x,
        success_rates,
        bar_width,
        color="tab:green",
        alpha=0.7,
        label="Success rate (%)",
    )

    ax1.set_xlabel("Payload size")
    ax1.set_ylabel("Success rate (%)", color="tab:green")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_xticks(x)
    ax1.set_xticklabels(filesizes)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        avg_latencies,
        color="tab:blue",
        marker="o",
        linewidth=2,
        label="Avg response time (s)",
    )
    ax2.set_ylabel("Average response time (s)", color="tab:blue")
    ax2.set_ylim(0, 3.0)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    full_title = title or f"Success & latency by payload size ({dataset_label})"
    ax1.set_title(full_title)

    # Build a combined legend from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if force_show or not output_path:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' does not exist.")

    json_paths = resolve_json_paths(results_dir, args.json_files)
    datasets = [(path.stem, load_dataset(path)) for path in json_paths]

    if args.mode == "success":
        if not datasets:
            raise RuntimeError("No datasets loaded to plot.")
        # Use a shared set of filesizes across all datasets unless the user
        # explicitly provided a subset via --filesizes.
        filesizes = determine_filesizes([data for _, data in datasets], args.filesizes)
        if not filesizes:
            raise RuntimeError("No payload sizes available to plot.")

        for label, data in datasets:
            # When an explicit --output path is provided and multiple JSON files
            # are requested, derive a per-dataset filename by suffixing the
            # dataset label before the extension (e.g. out.png -> out-4000.png).
            if args.output:
                base = Path(args.output)
                output_path = base.with_name(f"{base.stem}-{label}{base.suffix}")
                output_str: str | None = str(output_path)
            else:
                output_str = None

            plot_success_and_latency(
                label,
                data,
                filesizes,
                args.title,
                output_str,
                args.show,
            )
        return

    if args.mode == "packets":
        if not args.filesize:
            raise ValueError("--filesize is required when --mode=packets")

        plot_packets(
            datasets,
            args.filesize,
            args.title or f"Per-packet timings for {args.filesize}",
            args.output,
            args.show,
            args.smooth_window,
        )
        return

    filesizes = determine_filesizes([data for _, data in datasets], args.filesizes)
    if not filesizes:
        raise RuntimeError("No payload sizes available to plot.")

    plot_results(
        datasets,
        filesizes,
        args.aggregate,
        args.title,
        args.output,
        args.show,
        args.smooth_window,
    )


if __name__ == "__main__":
    main()

