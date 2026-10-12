from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS_TO_RAW_SUBDIRS = {
    "Ctrl": "20240314_Ribolyzer_Ctrl",
    "UTP18": "20240314_Ribolyzer_UTP18",
    "DIMT1L": "20250129_Ribolyzer_DIMT1L",
    "LAS1L": "Las1L_NanoRibolyzer",
    "RPL3": "RPL3_NanoRibolyzer",
    "URB1": "URB1_NanoRibolyzer",
    "WBSCR22": "WBSCR22_NanoRibolyzer",
}

EXPECTED_SAMPLES_PER_CONDITION = 3
DEFAULT_MIN_RAW_READS = 2
SEP = ";"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one UpSet plot per condition comparing the regions in "
            "its three IVPA-nucleus samples."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root containing data and src. By default it is detected "
            "from the location of this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: data/"
            "method1_set_based_and_rule_based_region_selection_data/"
            "upset_plots/within_condition_sample_upsets."
        ),
    )
    parser.add_argument(
        "--min-raw-reads",
        type=int,
        default=DEFAULT_MIN_RAW_READS,
        help=(
            "Minimum aggregated raw reads required for presence in one "
            "sample (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS_TO_RAW_SUBDIRS),
        default=list(CONDITIONS_TO_RAW_SUBDIRS),
        help="Conditions to plot (default: all seven).",
    )
    args = parser.parse_args()

    if args.min_raw_reads < 1:
        parser.error("--min-raw-reads must be at least 1.")

    return args


def find_project_root(script_path: Path) -> Path:
    script_path = script_path.resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate

    raise FileNotFoundError(
        "Projekt-Root konnte nicht automatisch gefunden werden. Erwartet wird "
        "ein übergeordneter Ordner mit den Unterordnern 'data' und 'src'. "
        "Alternativ kann --project-root angegeben werden."
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else find_project_root(Path(__file__))
    )
    raw_data_dir = project_root / "data" / "raw"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            project_root
            / "data"
            / "method1_set_based_and_rule_based_region_selection_data"
            / "upset_plots"
            / "within_condition_sample_upsets"
        )
    )
    return project_root, raw_data_dir, output_dir


def discover_condition_samples(
    raw_data_dir: Path,
    condition: str,
) -> list[Path]:
    condition_dir = raw_data_dir / CONDITIONS_TO_RAW_SUBDIRS[condition]
    files = sorted(
        condition_dir.glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    )

    if len(files) != EXPECTED_SAMPLES_PER_CONDITION:
        raise FileNotFoundError(
            f"{condition}: expected exactly "
            f"{EXPECTED_SAMPLES_PER_CONDITION} IVPA-nucleus samples, found "
            f"{len(files)} below {condition_dir}. Found files: {files}"
        )

    sample_names = [path.parent.parent.name for path in files]
    if len(sample_names) != len(set(sample_names)):
        raise ValueError(
            f"{condition}: duplicate sample directory names: {sample_names}"
        )

    return files


def load_sample_region_set(
    csv_path: Path,
    min_raw_reads: int,
) -> tuple[set[tuple[int, int]], int, int]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep=SEP,
        usecols=["start", "end", "n_reads"],
    )
    required = {"start", "end", "n_reads"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    for column in ("start", "end", "n_reads"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[["start", "end", "n_reads"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{csv_path} contains {int(invalid.sum())} rows with invalid "
            "start, end, or n_reads values."
        )
    if (df["n_reads"] < 0).any():
        raise ValueError(f"Negative read counts found in {csv_path}")

    grouped = df.groupby(["start", "end"], sort=False)["n_reads"].sum()
    present = grouped[grouped.ge(min_raw_reads)]
    region_set = {
        (int(start), int(end)) for start, end in present.index.tolist()
    }
    return region_set, len(grouped), int(grouped.sum())


def load_condition_sample_sets(
    sample_files: list[Path],
    min_raw_reads: int,
) -> tuple[dict[str, set[tuple[int, int]]], pd.DataFrame]:
    sample_sets: dict[str, set[tuple[int, int]]] = {}
    summary_rows: list[dict[str, object]] = []

    for csv_path in sample_files:
        sample_name = csv_path.parent.parent.name
        regions, total_unique_regions, total_reads = load_sample_region_set(
            csv_path,
            min_raw_reads=min_raw_reads,
        )
        sample_sets[sample_name] = regions
        summary_rows.append(
            {
                "sample": sample_name,
                "source_file": str(csv_path),
                "min_raw_reads": min_raw_reads,
                "total_unique_regions_before_presence_filter": (
                    total_unique_regions
                ),
                "regions_present": len(regions),
                "total_raw_reads": total_reads,
            }
        )

    return sample_sets, pd.DataFrame(summary_rows)


def calculate_membership_counts(
    sample_sets: dict[str, set[tuple[int, int]]],
) -> pd.DataFrame:
    sample_names = list(sample_sets)
    union = set().union(*sample_sets.values())
    counts: dict[tuple[bool, ...], int] = {}

    for region in union:
        membership = tuple(region in sample_sets[name] for name in sample_names)
        counts[membership] = counts.get(membership, 0) + 1

    rows: list[dict[str, object]] = []
    for membership, count in counts.items():
        row: dict[str, object] = {
            sample_names[index]: is_present
            for index, is_present in enumerate(membership)
        }
        row["n_samples"] = sum(membership)
        row["intersection_size"] = count
        rows.append(row)

    columns = [*sample_names, "n_samples", "intersection_size"]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["intersection_size", "n_samples"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return result


def plot_upset_for_condition(
    condition: str,
    sample_sets: dict[str, set[tuple[int, int]]],
    output_dir: Path,
    min_raw_reads: int,
) -> None:
    from upsetplot import UpSet, from_contents

    data = from_contents(sample_sets)
    figure = plt.figure(figsize=(15, 8))
    UpSet(
        data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    ).plot(fig=figure)

    figure.suptitle(
        (
            f"{condition}: overlap of detected regions across samples\n"
        ),
        fontsize=13,
        fontweight="bold",
    )

    filename_stem = f"upset_{condition}_samples_min_raw_reads_{min_raw_reads}"
    output_path = output_dir / f"{filename_stem}.svg"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def analyze_condition(
    condition: str,
    raw_data_dir: Path,
    output_base_dir: Path,
    min_raw_reads: int,
) -> dict[str, object]:
    sample_files = discover_condition_samples(raw_data_dir, condition)
    sample_sets, sample_summary = load_condition_sample_sets(
        sample_files,
        min_raw_reads=min_raw_reads,
    )

    condition_output_dir = output_base_dir / condition
    condition_output_dir.mkdir(parents=True, exist_ok=True)

    memberships = calculate_membership_counts(sample_sets)
    three_way_intersection = set.intersection(*sample_sets.values())
    union = set().union(*sample_sets.values())

    sample_summary.insert(0, "condition", condition)
    memberships.insert(0, "condition", condition)

    sample_summary.to_csv(
        condition_output_dir / "sample_region_set_sizes.csv",
        sep=SEP,
        index=False,
    )
    memberships.to_csv(
        condition_output_dir / "upset_membership_counts.csv",
        sep=SEP,
        index=False,
    )

    intersection_df = pd.DataFrame(
        sorted(three_way_intersection),
        columns=["start", "end"],
    )
    intersection_df.to_csv(
        condition_output_dir / "intersection_all_three_samples.csv",
        sep=SEP,
        index=False,
    )

    plot_upset_for_condition(
        condition=condition,
        sample_sets=sample_sets,
        output_dir=condition_output_dir,
        min_raw_reads=min_raw_reads,
    )


    return {
        "condition": condition,
        "min_raw_reads": min_raw_reads,
        "n_samples": len(sample_sets),
        "union_size": len(union),
        "intersection_all_three": len(three_way_intersection),
        "intersection_fraction_of_union": (
            len(three_way_intersection) / len(union) if union else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    project_root, raw_data_dir, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)


    overall_rows = [
        analyze_condition(
            condition=condition,
            raw_data_dir=raw_data_dir,
            output_base_dir=output_dir,
            min_raw_reads=args.min_raw_reads,
        )
        for condition in args.conditions
    ]

    pd.DataFrame(overall_rows).to_csv(
        output_dir / "within_condition_upset_summary.csv",
        sep=SEP,
        index=False,
    )


    print(f"Ergebnis gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()