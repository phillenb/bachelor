from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()

DEFAULT_INTERMEDIATES_RELATIVE = (
    Path("data") / "literature" / "rna45sn1_intermediates_expanded.csv"
)
DEFAULT_UNION_RELATIVE = (
    Path("data")
    / "method_comparison"
    / "three_method_region_sets"
    / "method_union_regions.csv"
)
DEFAULT_OUTPUT_RELATIVE = (
    Path("data")
    / "biological validation"
    / "expected_intermediate_region_overlays"
)

PLOT_CONDITIONS = ["UTP18", "DIMT1L", "WBSCR22", "LAS1L", "URB1", "RPL3"]
MATURE_28S_END = 12_990
DEFAULT_MRO_THRESHOLD = 0.80

COLOR_ALL_REGIONS = "#9CA3AF"
COLOR_KNOWN_INTERMEDIATES = "#334155"
COLOR_ENRICHMENT = "#C62828"
COLOR_DEPLETION = "#1565C0"
COLOR_ETS = "#7B1FA2"
COLOR_EXPECTED_MARKER = "#F9A825"


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    expectation: str
    patterns: tuple[str, ...] = ()
    mode: str = "named"


EXPECTED_FEATURES: dict[str, tuple[FeatureSpec, ...]] = {
    "UTP18": (
        FeatureSpec("34S", "enrichment", (r"(?<!\d)34S(?!\d)",)),
    ),
    "DIMT1L": (
        FeatureSpec(
            "21S/21S-C",
            "enrichment",
            (r"(?<!\d)21S(?:[-_ ]?C)?(?!\d)",),
        ),
        FeatureSpec("26S", "enrichment", (r"(?<!\d)26S(?!\d)",)),
        FeatureSpec(
            "early precursors (43S/45S/47S)",
            "enrichment",
            (
                r"(?<!\d)43S(?!\d)",
                r"(?<!\d)45S(?!\d)",
                r"(?<!\d)47S(?!\d)",
            ),
        ),
    ),
    "WBSCR22": (
        FeatureSpec(
            "18S-E variants",
            "enrichment",
            (r"(?<!\d)18S[-_ ]?E(?:\+?\d+)?(?!\d)",),
        ),
    ),
    "LAS1L": (
        FeatureSpec("32S", "enrichment", (r"(?<!\d)32S(?!\d)",)),
        FeatureSpec("12S", "depletion", (r"(?<!\d)12S(?!\d)",)),
    ),
    "URB1": (
        FeatureSpec("3'-ETS retention", "enrichment", mode="three_prime_ets"),
    ),
    "RPL3": (
        FeatureSpec("3'-ETS retention", "enrichment", mode="three_prime_ets"),
    ),
}


PANEL_SUBTITLES = {
    "UTP18": "expected: 34S enrichment",
    "DIMT1L": "expected: 21S/21S-C, 26S and early precursor enrichment",
    "WBSCR22": "expected: 18S-E enrichment",
    "LAS1L": "expected: 32S enrichment and 12S depletion",
    "URB1": "expected: 3'-ETS retention",
    "RPL3": "expected: 3'-ETS retention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay condition-specific union regions and expected pre-rRNA "
            "intermediates in start/end intensity-matrix coordinates."
        )
    )
    parser.add_argument(
        "--intermediates",
        type=Path,
        default=None,
        help="Optional override for rna45sn1_intermediates_expanded.csv.",
    )
    parser.add_argument(
        "--union-regions",
        type=Path,
        default=None,
        help="Optional override for method_union_regions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--mro-threshold",
        type=float,
        default=DEFAULT_MRO_THRESHOLD,
        help="Minimum reciprocal overlap for highlighting a match. Default: 0.80.",
    )
    return parser.parse_args()


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(
        f"Projektwurzel konnte ausgehend von {start_path} nicht gefunden werden."
    )


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"File is empty: {path}")
    try:
        return csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
    except csv.Error:
        counts = {delimiter: header.count(delimiter) for delimiter in (",", ";", "\t")}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise ValueError(f"Could not detect a delimiter in {path}")
        return delimiter


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=detect_delimiter(path), encoding="utf-8-sig")


def normalized_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(
    columns: Sequence[object],
    aliases: Sequence[str],
    role: str,
) -> str:
    normalized = {normalized_column_name(column): str(column) for column in columns}
    for alias in aliases:
        key = normalized_column_name(alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not identify the {role} column. Available columns: "
        + ", ".join(map(str, columns))
    )


def find_condition_columns(columns: Sequence[object]) -> list[str]:
    aliases = {
        "condition",
        "selectedcondition",
        "assignedcondition",
        "dominantcondition",
        "targetcondition",
        "highestcondition",
        "conditions",
        "selectedconditions",
    }
    exact = [str(column) for column in columns if normalized_column_name(column) in aliases]
    if exact:
        return exact

    candidates = [
        str(column)
        for column in columns
        if "condition" in normalized_column_name(column)
        and not any(
            excluded in normalized_column_name(column)
            for excluded in ("count", "number", "rank", "score", "effect")
        )
    ]
    if candidates:
        return candidates

    raise ValueError(
        "Could not identify a condition-assignment column in the union file. "
        "Available columns: " + ", ".join(map(str, columns))
    )


def canonical_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text.strip())
    return text.upper()


def value_contains_condition(value: object, condition: str) -> bool:
    if pd.isna(value):
        return False
    tokens = re.split(r"[^A-Za-z0-9]+", str(value).upper())
    return condition.upper() in tokens


def condition_mask(frame: pd.DataFrame, columns: Sequence[str], condition: str) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        mask |= frame[column].map(lambda value: value_contains_condition(value, condition))
    return mask


def prepare_union_regions(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = read_table(path)
    start_col = find_column(
        frame.columns,
        ("start", "region_start", "query_start", "feature_start"),
        "union-region start",
    )
    end_col = find_column(
        frame.columns,
        ("end", "region_end", "query_end", "feature_end"),
        "union-region end",
    )
    condition_cols = find_condition_columns(frame.columns)

    frame = frame.copy()
    frame["plot_start"] = pd.to_numeric(frame[start_col], errors="coerce")
    frame["plot_end"] = pd.to_numeric(frame[end_col], errors="coerce")
    invalid = frame[["plot_start", "plot_end"]].isna().any(axis=1)
    if invalid.any():
        frame = frame.loc[~invalid].copy()
    frame["plot_start"] = frame["plot_start"].astype(int)
    frame["plot_end"] = frame["plot_end"].astype(int)
    frame = frame.loc[frame["plot_end"] >= frame["plot_start"]].copy()
    return frame, condition_cols


def prepare_intermediates(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    start_col = find_column(
        frame.columns,
        ("start", "intermediate_start", "rna_start", "reference_start"),
        "intermediate start",
    )
    end_col = find_column(
        frame.columns,
        ("end", "intermediate_end", "rna_end", "reference_end"),
        "intermediate end",
    )
    name_col = find_column(
        frame.columns,
        (
            "intermediate",
            "name",
            "label",
            "intermediate_name",
            "rna",
            "species",
            "fragment",
            "fragment_name",
            "reference",
        ),
        "intermediate name",
    )

    prepared = pd.DataFrame(
        {
            "name": frame[name_col].astype(str).str.strip(),
            "start": pd.to_numeric(frame[start_col], errors="coerce"),
            "end": pd.to_numeric(frame[end_col], errors="coerce"),
        }
    ).dropna(subset=["start", "end"])
    prepared["start"] = prepared["start"].astype(int)
    prepared["end"] = prepared["end"].astype(int)
    prepared = prepared.loc[prepared["end"] >= prepared["start"]]
    prepared["canonical_name"] = prepared["name"].map(canonical_label)
    return prepared.drop_duplicates(["name", "start", "end"]).reset_index(drop=True)


def select_named_intermediates(
    intermediates: pd.DataFrame,
    patterns: Iterable[str],
) -> pd.DataFrame:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    mask = intermediates["canonical_name"].map(
        lambda name: any(pattern.search(name) for pattern in compiled)
    )
    return intermediates.loc[mask].copy()


def select_reference_intermediates(
    intermediates: pd.DataFrame,
    feature: FeatureSpec,
) -> pd.DataFrame:
    if feature.mode == "three_prime_ets":
        return intermediates.loc[
            (intermediates["start"] <= MATURE_28S_END)
            & (intermediates["end"] > MATURE_28S_END)
        ].copy()
    return select_named_intermediates(intermediates, feature.patterns)


def maximum_reciprocal_overlap(
    regions: pd.DataFrame,
    references: pd.DataFrame,
) -> np.ndarray:
    if regions.empty or references.empty:
        return np.zeros(len(regions), dtype=float)

    query_start = regions["plot_start"].to_numpy(dtype=float)[:, None]
    query_end = regions["plot_end"].to_numpy(dtype=float)[:, None]
    target_start = references["start"].to_numpy(dtype=float)[None, :]
    target_end = references["end"].to_numpy(dtype=float)[None, :]

    overlap = np.maximum(
        0.0,
        np.minimum(query_end, target_end) - np.maximum(query_start, target_start) + 1.0,
    )
    query_length = query_end - query_start + 1.0
    target_length = target_end - target_start + 1.0
    scores = np.minimum(overlap / query_length, overlap / target_length)
    return scores.max(axis=1)


def matching_region_mask(
    regions: pd.DataFrame,
    references: pd.DataFrame,
    feature: FeatureSpec,
    mro_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if feature.mode == "three_prime_ets":
        mask = (
            (regions["plot_start"].to_numpy() <= MATURE_28S_END)
            & (regions["plot_end"].to_numpy() > MATURE_28S_END)
        )
        return mask, mask.astype(float)
    scores = maximum_reciprocal_overlap(regions, references)
    return scores >= mro_threshold, scores


def annotate_reference_points(
    ax: plt.Axes,
    references: pd.DataFrame,
    max_labels: int = 12,
) -> None:
    if references.empty:
        return
    labels = references.sort_values(["start", "end", "name"]).head(max_labels)
    offsets = [(5, 5), (5, -10), (-5, 6), (-5, -11)]
    for index, row in enumerate(labels.itertuples(index=False)):
        dx, dy = offsets[index % len(offsets)]
        ax.annotate(
            row.name,
            (row.start, row.end),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            color="#111827",
            annotation_clip=True,
        )


def format_thousands_as_k(value: float, _position: int) -> str:
    """Format coordinates in thousands (k)."""
    if math.isclose(value, 0.0, abs_tol=1e-9):
        return "0"
    return f"{value / 1000:g}k"


def plot_condition(
    ax: plt.Axes,
    condition: str,
    regions: pd.DataFrame,
    intermediates: pd.DataFrame,
    mro_threshold: float,
    axis_max: float,
) -> list[dict[str, object]]:
    regions = regions.drop_duplicates(["plot_start", "plot_end"]).copy()
    ax.scatter(
        regions["plot_start"],
        regions["plot_end"],
        s=7,
        marker="s",
        color=COLOR_ALL_REGIONS,
        alpha=0.38,
        linewidths=0,
        zorder=1,
    )
    ax.scatter(
        intermediates["start"],
        intermediates["end"],
        s=22,
        marker="D",
        facecolors="none",
        edgecolors=COLOR_KNOWN_INTERMEDIATES,
        linewidths=0.7,
        alpha=0.65,
        zorder=3,
    )

    summary_rows: list[dict[str, object]] = []
    expected_references: list[pd.DataFrame] = []

    for feature in EXPECTED_FEATURES[condition]:
        references = select_reference_intermediates(intermediates, feature)
        mask, scores = matching_region_mask(
            regions, references, feature, mro_threshold
        )
        expected_references.append(references)

        color = COLOR_DEPLETION if feature.expectation == "depletion" else COLOR_ENRICHMENT
        if feature.mode == "three_prime_ets":
            color = COLOR_ETS
            ax.axhspan(
                MATURE_28S_END,
                axis_max,
                color=COLOR_ETS,
                alpha=0.045,
                zorder=0,
            )
            ax.axhline(
                MATURE_28S_END,
                color=COLOR_ETS,
                linewidth=0.9,
                linestyle="--",
                alpha=0.9,
                zorder=2,
            )

        matched = regions.loc[mask]
        ax.scatter(
            matched["plot_start"],
            matched["plot_end"],
            s=15,
            marker="s",
            color=color,
            alpha=0.85,
            linewidths=0,
            zorder=4,
        )

        if not references.empty:
            ax.scatter(
                references["start"],
                references["end"],
                s=80,
                marker="*",
                facecolors=COLOR_EXPECTED_MARKER,
                edgecolors="#111827",
                linewidths=0.55,
                alpha=0.5,
                zorder=6,
            )

        summary_rows.append(
            {
                "condition": condition,
                "expected_feature": feature.label,
                "expectation": feature.expectation,
                "matching_rule": (
                    f"end > {MATURE_28S_END} and start <= {MATURE_28S_END}"
                    if feature.mode == "three_prime_ets"
                    else f"MRO >= {mro_threshold:.2f}"
                ),
                "n_condition_specific_regions": int(len(regions)),
                "n_reference_intermediates": int(len(references)),
                "n_matching_regions": int(mask.sum()),
                "maximum_mro": float(scores.max()) if len(scores) else 0.0,
                "reference_names": " | ".join(references["name"].astype(str)),
            }
        )

    if expected_references:
        combined_references = pd.concat(expected_references, ignore_index=True)
        combined_references = combined_references.drop_duplicates(["name", "start", "end"])
        annotate_reference_points(ax, combined_references)

    ax.plot([0, axis_max], [0, axis_max], color="#CBD5E1", linewidth=0.7, zorder=0)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("start", fontsize=8)
    ax.set_ylabel("end", fontsize=8)
    coordinate_formatter = FuncFormatter(format_thousands_as_k)
    ax.xaxis.set_major_formatter(coordinate_formatter)
    ax.yaxis.set_major_formatter(coordinate_formatter)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(False)
    ax.set_title(
        f"{condition}\n{PANEL_SUBTITLES[condition]}",
        fontsize=9.5,
        fontweight="semibold",
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.7)
    return summary_rows


def shared_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5,
            markerfacecolor=COLOR_ALL_REGIONS, markeredgewidth=0,
            label="Condition-specific union region",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=5,
            markerfacecolor="none", markeredgecolor=COLOR_KNOWN_INTERMEDIATES,
            label="Known intermediate",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markersize=9,
            markerfacecolor=COLOR_EXPECTED_MARKER, markeredgecolor="#111827",
            alpha=0.5,
            label="Expected reference intermediate",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5,
            markerfacecolor=COLOR_ENRICHMENT, markeredgewidth=0,
            label="Region matching expected enrichment",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5,
            markerfacecolor=COLOR_DEPLETION, markeredgewidth=0,
            label="Region matching expected depletion",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5,
            markerfacecolor=COLOR_ETS, markeredgewidth=0,
            label="3'-ETS-retaining candidate",
        ),
    ]


def save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.mro_threshold <= 1.0:
        raise ValueError("--mro-threshold must be in (0, 1]")

    project_root: Optional[Path] = None
    if (
        args.intermediates is None
        or args.union_regions is None
        or args.output_dir is None
    ):
        project_root = find_project_root(CURRENT_FILE.parent)

    intermediates_path = (
        args.intermediates
        if args.intermediates is not None
        else project_root / DEFAULT_INTERMEDIATES_RELATIVE
    )
    union_path = (
        args.union_regions
        if args.union_regions is not None
        else project_root / DEFAULT_UNION_RELATIVE
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else project_root / DEFAULT_OUTPUT_RELATIVE
    )
    output_dir.mkdir(parents=True, exist_ok=True)


    union, condition_columns = prepare_union_regions(union_path)
    intermediates = prepare_intermediates(intermediates_path)

    coordinate_max = max(
        float(union[["plot_start", "plot_end"]].to_numpy().max()),
        float(intermediates[["start", "end"]].to_numpy().max()),
        float(MATURE_28S_END),
    )
    # Space for labels near the 3' end.
    axis_max = math.ceil((coordinate_max + 500.0) / 500.0) * 500.0

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    condition_frames: dict[str, pd.DataFrame] = {}
    for condition in PLOT_CONDITIONS:
        subset = union.loc[condition_mask(union, condition_columns, condition)].copy()
        if subset.empty:
            raise ValueError(
                f"No union regions were assigned to {condition}. "
                f"Checked columns: {condition_columns}"
            )
        condition_frames[condition] = subset

    all_summary_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 10.2),
        sharex=True,
        sharey=True,
    )
    for ax, condition in zip(axes.flat, PLOT_CONDITIONS):
        rows = plot_condition(
            ax,
            condition,
            condition_frames[condition],
            intermediates,
            args.mro_threshold,
            axis_max,
        )
        all_summary_rows.extend(rows)
    fig.suptitle(
        "Condition-specific selected regions and expected pre-rRNA intermediates",
        fontsize=14,
        fontweight="semibold",
        y=0.975,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.89,
        bottom=0.15,
        wspace=0.18,
        hspace=0.30,
    )
    fig.legend(
        handles=shared_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    save_pdf(fig, output_dir / "expected_intermediate_overlays_combined.pdf")

    for condition in PLOT_CONDITIONS:
        fig, ax = plt.subplots(figsize=(7.2, 7.1))
        plot_condition(
            ax,
            condition,
            condition_frames[condition],
            intermediates,
            args.mro_threshold,
            axis_max,
        )
        fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.23)
        fig.legend(
            handles=shared_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=2,
            frameon=False,
            fontsize=7.5,
        )
        save_pdf(fig, output_dir / f"{condition}_expected_intermediates.pdf")

    summary = pd.DataFrame(all_summary_rows)
    summary.to_csv(output_dir / "expected_intermediate_overlap_summary.csv", index=False)

    missing = summary.loc[summary["n_reference_intermediates"] == 0]

    print(f"Ergebnisse gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()
