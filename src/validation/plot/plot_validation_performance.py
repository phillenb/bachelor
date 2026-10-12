from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_FILE = "nested_validation_summary.csv"
FOLD_FILE = "nested_fold_panel_summary.csv"
TIMING_FILE = "nested_selection_timing.csv"

PANEL_ORDER = [
    "raw_all_regions",
    "set_based",
    "umap",
    "logreg",
    "union_all_methods",
    "at_least_two_methods",
    "intersection_all_three_coordinates",
    "intersection_all_three_same_condition",
]

PANEL_LABELS = {
    "raw_all_regions": "Raw data",
    "set_based": "Set-based",
    "umap": "UMAP",
    "logreg": "LogReg",
    "union_all_methods": "Union",
    "at_least_two_methods": r"$\geq$2 methods",
    "intersection_all_three_coordinates": "Intersection\n(coordinates)",
    "intersection_all_three_same_condition": "Intersection\n(same condition)",
}

RAW_COLOR = "#4C566A"
SELECTED_COLOR = "#4C78A8"
BEST_COLOR = "#59A14F"


def find_project_root(start: Path) -> Path:
    """Find the nearest project root."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() or (candidate / "src").is_dir():
            return candidate
    return start


def default_input_path() -> Path:
    root = find_project_root(Path.cwd())
    candidates = [
        root
        / "data"
        / "method_comparison"
        / "three_method_nested_balanced_validation",
        root / "data" / "three_method_nested_balanced_validation",
        Path.cwd() / "three_method_nested_balanced_validation",
    ]
    for candidate in candidates:
        if (candidate / SUMMARY_FILE).is_file():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a three-panel comparison of fold-local validation accuracy, "
            "feature count, and runtime."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Validation result directory or ZIP archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input>/plots).",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("classifier", "end-to-end"),
        default="classifier",
        help=(
            "classifier: final scaler/model fit and prediction only; "
            "end-to-end: include required feature-selection steps."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of PNG output (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview after saving.",
    )
    return parser.parse_args()


def locate_result_directory(base: Path) -> Path:
    """Locate one directory with the three result CSVs."""
    if all((base / name).is_file() for name in (SUMMARY_FILE, FOLD_FILE, TIMING_FILE)):
        return base
    matches = [
        path.parent
        for path in base.rglob(SUMMARY_FILE)
        if all((path.parent / name).is_file() for name in (FOLD_FILE, TIMING_FILE))
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one validation result directory below {base}; "
            f"found {len(matches)}."
        )
    return matches[0]


def read_results(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read result tables from a directory or ZIP."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        result_dir = locate_result_directory(input_path)
        return (
            pd.read_csv(result_dir / SUMMARY_FILE, sep=";"),
            pd.read_csv(result_dir / FOLD_FILE, sep=";"),
            pd.read_csv(result_dir / TIMING_FILE, sep=";"),
        )
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="validation_plot_") as temp_dir:
            with zipfile.ZipFile(input_path) as archive:
                archive.extractall(temp_dir)
            result_dir = locate_result_directory(Path(temp_dir))
            return (
                pd.read_csv(result_dir / SUMMARY_FILE, sep=";"),
                pd.read_csv(result_dir / FOLD_FILE, sep=";"),
                pd.read_csv(result_dir / TIMING_FILE, sep=";"),
            )
    raise FileNotFoundError(f"Input is neither a directory nor a ZIP archive: {input_path}")


def validate_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def prepare_plot_data(
    summary: pd.DataFrame,
    fold_data: pd.DataFrame,
    timing: pd.DataFrame,
    runtime_mode: str,
) -> pd.DataFrame:
    validate_columns(summary, {"panel", "accuracy"}, SUMMARY_FILE)
    validate_columns(
        fold_data,
        {"fold", "panel", "n_features", "classification_seconds", "status"},
        FOLD_FILE,
    )
    validate_columns(
        timing,
        {
            "fold",
            "raw_scaler_and_model_fit_seconds",
            "set_selection_seconds",
            "umap_selection_seconds",
            "logreg_coefficient_extraction_seconds",
            "total_selection_seconds",
        },
        TIMING_FILE,
    )

    folds = fold_data.loc[fold_data["status"] == "ok"].copy()
    folds = folds.merge(timing, on="fold", how="left", validate="many_to_one")
    if folds["raw_scaler_and_model_fit_seconds"].isna().any():
        raise ValueError("Could not match all folds to nested_selection_timing.csv.")

    # Add the separately recorded fit time to raw prediction timing.
    folds["classifier_runtime"] = folds["classification_seconds"]
    raw_mask = folds["panel"] == "raw_all_regions"
    folds.loc[raw_mask, "classifier_runtime"] += folds.loc[
        raw_mask, "raw_scaler_and_model_fit_seconds"
    ]

    if runtime_mode == "classifier":
        folds["runtime"] = folds["classifier_runtime"]
    else:
        selection_cost = pd.Series(0.0, index=folds.index)
        selection_cost.loc[folds["panel"] == "set_based"] = folds.loc[
            folds["panel"] == "set_based", "set_selection_seconds"
        ]
        selection_cost.loc[folds["panel"] == "umap"] = folds.loc[
            folds["panel"] == "umap", "umap_selection_seconds"
        ]
        logreg_mask = folds["panel"] == "logreg"
        selection_cost.loc[logreg_mask] = (
            folds.loc[logreg_mask, "raw_scaler_and_model_fit_seconds"]
            + folds.loc[logreg_mask, "logreg_coefficient_extraction_seconds"]
        )
        combined_mask = folds["panel"].isin(
            [
                "union_all_methods",
                "at_least_two_methods",
                "intersection_all_three_coordinates",
                "intersection_all_three_same_condition",
            ]
        )
        selection_cost.loc[combined_mask] = folds.loc[
            combined_mask, "total_selection_seconds"
        ]
        # Raw fitting is already included.
        folds["runtime"] = folds["classifier_runtime"] + selection_cost

    aggregated = (
        folds.groupby("panel", as_index=False)
        .agg(
            features_mean=("n_features", "mean"),
            features_min=("n_features", "min"),
            features_max=("n_features", "max"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
        )
        .merge(summary[["panel", "accuracy"]], on="panel", how="left")
    )
    present_order = [panel for panel in PANEL_ORDER if panel in set(aggregated["panel"])]
    return aggregated.set_index("panel").loc[present_order].reset_index()


def feature_count_text(value: float) -> str:
    """Format feature counts, including fractional means."""
    if np.isclose(value, round(value), rtol=0, atol=1e-8):
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def runtime_text(seconds: float) -> str:
    if seconds >= 10:
        return f"{seconds:.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    return f"{seconds:.3f}s"


def add_bar_labels(ax: plt.Axes, bars, texts: list[str], padding: int = 3) -> None:
    for bar, label in zip(bars, texts):
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def create_figure(data: pd.DataFrame, runtime_mode: str) -> plt.Figure:
    labels = [PANEL_LABELS.get(panel, panel) for panel in data["panel"]]
    x = np.arange(len(data))
    colors = [RAW_COLOR if panel == "raw_all_regions" else SELECTED_COLOR for panel in data["panel"]]

    # Best observed accuracy: green. Raw data: grey.
    best_accuracy = data["accuracy"].max()
    for index, row in data.iterrows():
        if row["panel"] != "raw_all_regions" and np.isclose(row["accuracy"], best_accuracy):
            colors[index] = BEST_COLOR

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)

    raw_accuracy = data.loc[
        data["panel"] == "raw_all_regions", "accuracy"
    ].iloc[0]
    accuracy_bars = axes[0].bar(x, 100 * data["accuracy"], color=colors, width=0.72)
    axes[0].set_title("A  Classification performance", loc="left", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, max(100, 100 * data["accuracy"].max() + 12))
    axes[0].axhline(
        100 * raw_accuracy,
        color=RAW_COLOR,
        linestyle="--",
        linewidth=1,
        alpha=0.75,
    )
    accuracy_labels = []
    for panel, value in zip(data["panel"], data["accuracy"]):
        if panel == "raw_all_regions":
            accuracy_labels.append(f"{100 * value:.1f}%")
        else:
            delta_pp = 100 * (value - raw_accuracy)
            accuracy_labels.append(f"{100 * value:.1f}%\n{delta_pp:+.1f} pp")
    add_bar_labels(axes[0], accuracy_bars, accuracy_labels)

    feature_bars = axes[1].bar(x, data["features_mean"], color=colors, width=0.72)
    axes[1].set_title("B  Feature-space reduction", loc="left", fontweight="bold")
    axes[1].set_ylabel("Mean features per outer fold (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(
        bottom=max(1, data["features_mean"].min() / 3),
        top=data["features_mean"].max() * 2.5,
    )
    raw_features = data.loc[
        data["panel"] == "raw_all_regions", "features_mean"
    ].iloc[0]
    feature_labels = []
    for panel, value in zip(data["panel"], data["features_mean"]):
        if panel == "raw_all_regions":
            feature_labels.append(feature_count_text(value))
        else:
            reduction_factor = raw_features / value
            feature_labels.append(
                f"{feature_count_text(value)}\n{reduction_factor:,.0f}×"
            )
    add_bar_labels(axes[1], feature_bars, feature_labels)

    runtime_bars = axes[2].bar(
        x,
        data["runtime_mean"],
        yerr=data["runtime_std"].fillna(0),
        capsize=3,
        color=colors,
        width=0.72,
        error_kw={"linewidth": 1},
    )
    runtime_title = "Final classifier runtime" if runtime_mode == "classifier" else "End-to-end runtime"
    axes[2].set_title(f"C  {runtime_title}", loc="left", fontweight="bold")
    axes[2].set_ylabel("Mean seconds per outer fold (log scale)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=max(0.001, data["runtime_mean"].min() / 3))

    raw_runtime = data.loc[data["panel"] == "raw_all_regions", "runtime_mean"].iloc[0]
    runtime_labels = []
    for panel, seconds in zip(data["panel"], data["runtime_mean"]):
        if panel == "raw_all_regions":
            runtime_labels.append(runtime_text(seconds))
        else:
            speedup = raw_runtime / seconds
            runtime_labels.append(f"{runtime_text(seconds)}\n{speedup:.1f}x")
    add_bar_labels(axes[2], runtime_bars, runtime_labels)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fully fold-local balanced validation: raw data versus selected region panels", fontweight="bold")
    return fig


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else default_input_path()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif input_path.suffix.lower() == ".zip":
        output_dir = input_path.expanduser().resolve().parent / "validation_performance_plots"
    else:
        output_dir = input_path.expanduser().resolve() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, fold_data, timing = read_results(input_path)
    plot_data = prepare_plot_data(summary, fold_data, timing, args.runtime_mode)
    figure = create_figure(plot_data, args.runtime_mode)

    stem = f"validation_performance_comparison_{args.runtime_mode.replace('-', '_')}"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=args.dpi if suffix == "png" else None,
            bbox_inches="tight",
        )
    plot_data.to_csv(output_dir / f"{stem}_data.csv", sep=";", index=False)

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Ergebnisse gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()