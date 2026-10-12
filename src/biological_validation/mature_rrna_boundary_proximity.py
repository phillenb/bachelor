from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATURE_RRNA_INTERVALS = {
    "18S": (3654, 5524),
    "5.8S": (6601, 6756),
    "28S": (7925, 12990),
}
TOLERANCES = (0, 1, 5, 10)
BAR_COLOR = "#315F7D"


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    if not {"start", "end"}.issubset(frame.columns):
        raise ValueError("The input table must contain 'start' and 'end' columns.")
    frame = frame[["start", "end"]].copy()
    frame["start"] = pd.to_numeric(frame["start"], errors="raise").astype(np.int64)
    frame["end"] = pd.to_numeric(frame["end"], errors="raise").astype(np.int64)
    return frame.drop_duplicates(["start", "end"]).reset_index(drop=True)


def calculate_boundary_proximity(regions: pd.DataFrame) -> pd.DataFrame:
    starts = regions["start"].to_numpy(dtype=np.int64)
    ends = regions["end"].to_numpy(dtype=np.int64)
    mature_starts = np.asarray(
        [interval[0] for interval in MATURE_RRNA_INTERVALS.values()],
        dtype=np.int64,
    )
    mature_ends = np.asarray(
        [interval[1] for interval in MATURE_RRNA_INTERVALS.values()],
        dtype=np.int64,
    )

    rows: list[dict[str, float | int | str]] = []
    for tolerance in TOLERANCES:
        start_match = np.any(
            np.abs(starts[:, np.newaxis] - mature_starts[np.newaxis, :]) <= tolerance,
            axis=1,
        )
        end_match = np.any(
            np.abs(ends[:, np.newaxis] - mature_ends[np.newaxis, :]) <= tolerance,
            axis=1,
        )
        matched = start_match | end_match
        count = int(matched.sum())
        rows.append(
            {
                "tolerance_nt": tolerance,
                "tolerance_label": "Exact" if tolerance == 0 else f"±{tolerance} nt",
                "n_regions": count,
                "percentage": 100.0 * count / len(regions),
            }
        )
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, n_regions: int, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    x = np.arange(len(summary))
    values = summary["percentage"].to_numpy(dtype=float)
    counts = summary["n_regions"].to_numpy(dtype=int)

    fig, axis = plt.subplots(figsize=(5.6, 4.1))
    bars = axis.bar(x, values, width=0.64, color=BAR_COLOR, edgecolor="none")

    axis.set_xticks(x)
    axis.set_xticklabels(summary["tolerance_label"])
    axis.set_xlabel("Endpoint tolerance")
    axis.set_ylabel("Selected regions associated with\na mature rRNA boundary (%)")
    axis.set_ylim(0, 65)
    axis.set_yticks(np.arange(0, 61, 10))
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4A4A4A")
    axis.spines["bottom"].set_color("#4A4A4A")

    for bar, percentage, count in zip(bars, values, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            percentage + 1.2,
            f"{percentage:.2f}%\n({count:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1F1F1F",
        )

    axis.text(
        0.0,
        1.015,
        f"Matching start or end boundary; n = {n_regions:,} selected regions",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "union_csv",
        nargs="?",
        type=Path,
        default=(
            project_root
            / "data"
            / "method_comparison"
            / "three_method_region_sets"
            / "method_union_regions.csv"
        ),
        help="Optional union CSV (default: the project's global method union)",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=(
            project_root
            / "data"
            / "biological validation"
            / "boundary proximity"
            / "mature_rrna_boundary_proximity"
        ),
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = read_table(args.union_csv)
    summary = calculate_boundary_proximity(regions)
    save_plot(summary, len(regions), args.output_stem)
    summary.to_csv(
        args.output_stem.with_name(args.output_stem.name + "_data.csv"),
        sep=";",
        index=False,
    )
    print(f"Ergebnisse gespeichert in: {args.output_stem.parent.resolve()}")


if __name__ == "__main__":
    main()
