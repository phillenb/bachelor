from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3, venn3_unweighted

    try:
        from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    except ImportError:
        DefaultLayoutAlgorithm = None
except ImportError as exc:
    raise ImportError(
        "This script requires matplotlib and matplotlib-venn. Install them "
        "with: conda install -c conda-forge matplotlib matplotlib-venn"
    ) from exc


def _find_src_dir(start: Path) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "src":
            return parent
    raise RuntimeError("'src' directory not found.")


SRC_DIR = _find_src_dir(Path(__file__))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import DATA_DIR  # noqa: E402


SEP = ";"
CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]
METHOD_ORDER = ["set_based", "umap", "logreg"]
METHOD_LABELS = {
    "set_based": "Set-based",
    "umap": "UMAP",
    "logreg": "LogReg",
}
METHOD_COLORS = ("#4C78A8", "#F58518", "#54A24B")

DEFAULT_INPUT_DIR = (
    DATA_DIR / "method_comparison" / "three_method_region_sets"
)
DEFAULT_OUTPUT_DIR = DATA_DIR/ "method_comparison" / "venn_diagrams"

INPUT_FILES = {
    "set_based": "set_based_regions.csv",
    "umap": "umap_regions_standardized.csv",
    "logreg": "logreg_regions.csv",
}


def read_method_regions(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the three region files."""
    frames: dict[str, pd.DataFrame] = {}
    required = {"condition", "start", "end"}

    for method in METHOD_ORDER:
        path = input_dir / INPUT_FILES[method]
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        frame = pd.read_csv(path, sep=SEP)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        frame = frame[["condition", "start", "end"]].copy()
        frame["condition"] = frame["condition"].astype(str)
        unknown = sorted(set(frame["condition"]) - set(CONDITION_ORDER))
        if unknown:
            raise ValueError(f"{path}: unknown conditions: {unknown}")

        try:
            frame[["start", "end"]] = frame[["start", "end"]].astype(
                "int64"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: start and end must be integer coordinates."
            ) from exc

        frames[method] = frame.drop_duplicates().reset_index(drop=True)

    return frames


def build_sets(
    frames: dict[str, pd.DataFrame],
    key_columns: list[str],
    condition: str | None = None,
) -> dict[str, set[tuple]]:
    """Build coordinate or condition-coordinate sets."""
    method_sets: dict[str, set[tuple]] = {}
    for method in METHOD_ORDER:
        frame = frames[method]
        if condition is not None:
            frame = frame.loc[frame["condition"].eq(condition)]
        method_sets[method] = set(
            frame[key_columns]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    return method_sets


def calculate_venn_counts(
    method_sets: dict[str, set[tuple]],
) -> dict[str, int]:
    """Count the seven disjoint Venn areas."""
    set_based = method_sets["set_based"]
    umap = method_sets["umap"]
    logreg = method_sets["logreg"]

    return {
        "set_based_only": len(set_based - umap - logreg),
        "umap_only": len(umap - set_based - logreg),
        "logreg_only": len(logreg - set_based - umap),
        "set_based_umap_only": len((set_based & umap) - logreg),
        "set_based_logreg_only": len((set_based & logreg) - umap),
        "umap_logreg_only": len((umap & logreg) - set_based),
        "all_three": len(set_based & umap & logreg),
    }


def draw_venn(
    ax: plt.Axes,
    method_sets: dict[str, set[tuple]],
    title: str,
) -> None:
    """Draw a Venn diagram with exact counts."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    if not set().union(*(method_sets[method] for method in METHOD_ORDER)):
        ax.text(
            0.5,
            0.5,
            "No regions",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    venn_kwargs = {
        "subsets": tuple(method_sets[method] for method in METHOD_ORDER),
        "set_labels": tuple(
            METHOD_LABELS[method] for method in METHOD_ORDER
        ),
        "set_colors": METHOD_COLORS,
        "alpha": 0.55,
        "ax": ax,
    }
    if DefaultLayoutAlgorithm is not None:
        venn_kwargs["layout_algorithm"] = DefaultLayoutAlgorithm(
            fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1)
        )
        diagram = venn3(**venn_kwargs)
    else:
        # Support older matplotlib-venn versions.
        diagram = venn3_unweighted(**venn_kwargs)

    for label in diagram.subset_labels:
        if label is not None:
            label.set_fontsize(10)
            label.set_fontweight("bold")
    for label in diagram.set_labels:
        if label is not None:
            label.set_fontsize(10)
    for patch in diagram.patches:
        if patch is not None:
            patch.set_edgecolor("white")
            patch.set_linewidth(1.2)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Export SVG, PDF and PNG."""
    fig.text(
        0.5,
        0.01,
        "Numbers are exact; circle areas are schematic.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    for suffix in (".svg", ".pdf", ".png"):
        kwargs = {"dpi": 300} if suffix == ".png" else {}
        fig.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def create_diagrams(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Create global and per-condition diagrams and counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count_rows: list[dict[str, str | int]] = []

    coordinate_sets = build_sets(frames, ["start", "end"])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    draw_venn(
        ax,
        coordinate_sets,
        "Method overlap based on exact start-end coordinates",
    )
    save_figure(fig, output_dir / "venn_method_overlap_coordinates")
    count_rows.append(
        {
            "scope": "coordinate",
            "condition": "all",
            **calculate_venn_counts(coordinate_sets),
        }
    )

    condition_coordinate_sets = build_sets(
        frames,
        ["condition", "start", "end"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    draw_venn(
        ax,
        condition_coordinate_sets,
        "Method overlap requiring the same condition assignment",
    )
    save_figure(
        fig,
        output_dir / "venn_method_overlap_condition_start_end",
    )
    count_rows.append(
        {
            "scope": "condition_start_end",
            "condition": "all",
            **calculate_venn_counts(condition_coordinate_sets),
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    for ax, condition in zip(axes.flat, CONDITION_ORDER):
        condition_sets = build_sets(
            frames,
            ["start", "end"],
            condition=condition,
        )
        draw_venn(ax, condition_sets, condition)
        count_rows.append(
            {
                "scope": "within_condition",
                "condition": condition,
                **calculate_venn_counts(condition_sets),
            }
        )

    axes.flat[-1].axis("off")
    fig.suptitle(
        "Overlap of selected regions by condition",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, output_dir / "venn_method_overlap_by_condition")

    counts = pd.DataFrame(count_rows)
    counts.to_csv(output_dir / "venn_overlap_counts.csv", sep=SEP, index=False)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = read_method_regions(args.input_dir)
    counts = create_diagrams(frames, args.output_dir)

    global_coordinates = counts.loc[
        counts["scope"].eq("coordinate")
        & counts["condition"].eq("all")
    ].iloc[0]
    print(f"Ergebnisse gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()