from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis"
)

CONDITIONS = (
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
)

KD_CONDITIONS = tuple(condition for condition in CONDITIONS if condition != "Ctrl")

RAW_DATA_SUBDIRS = {
    "Ctrl": "20240314_Ribolyzer_Ctrl",
    "UTP18": "20240314_Ribolyzer_UTP18",
    "DIMT1L": "20250129_Ribolyzer_DIMT1L",
    "LAS1L": "Las1L_NanoRibolyzer",
    "RPL3": "RPL3_NanoRibolyzer",
    "URB1": "URB1_NanoRibolyzer",
    "WBSCR22": "WBSCR22_NanoRibolyzer",
}

MATURE_RRNA_INTERVALS = {
    "18S": (3654, 5523),
    "5.8S": (6600, 6757),
    "28S": (7924, 12990),
}

REQUIRED_REGION_COLUMNS = {"start", "end"}
REQUIRED_RAW_COLUMNS = {"start", "end", "n_reads"}
SEP = ";"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shared KD_only regions with mature-rRNA boundaries and "
            "identify highly abundant, low-variance signal profiles."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root containing data/raw (default: %(default)s).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Input regions_in_allen_conditions.csv. By default it is resolved "
            "below the project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: cross_condition_comparison/"
            "mature_rRNA_signal_validation."
        ),
    )
    parser.add_argument(
        "--high-abundance-quantile",
        type=float,
        default=0.90,
        help="Quantile defining high abundance (default: 0.90).",
    )
    parser.add_argument(
        "--low-variance-quantile",
        type=float,
        default=0.10,
        help="Quantile defining low variance (default: 0.10).",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("kd_samples", "all_samples", "condition_means"),
        default="kd_samples",
        help=(
            "Signal profile used for thresholding. Default: kd_samples "
            "(all 18 knockdown samples)."
        ),
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=0,
        help=(
            "Allowed absolute deviation in nucleotides for mature-rRNA "
            "boundary matches (default: 0, exact boundaries)."
        ),
    )
    args = parser.parse_args()

    for name in ("high_abundance_quantile", "low_variance_quantile"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")

    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance must be >= 0.")

    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = args.project_root.expanduser().resolve()
    comparison_dir = (
        project_root
        / "data"
        / "method1_set_based_and_rule_based_region_selection_data"
        / "cross_condition_comparison"
    )
    input_csv = (
        args.input_csv.expanduser().resolve()
        if args.input_csv is not None
        else comparison_dir / "regions_in_allen_conditions.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else comparison_dir / "mature_rRNA_signal_validation"
    )
    raw_data_dir = project_root / "data" / "raw"
    return input_csv, output_dir, raw_data_dir


def read_required_csv(
    path: Path,
    required_columns: set[str],
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep=SEP, usecols=usecols)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def load_candidate_regions(path: Path) -> pd.DataFrame:
    df = read_required_csv(path, REQUIRED_REGION_COLUMNS)
    df = df.copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    invalid = df[["start", "end"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{path} contains {int(invalid.sum())} invalid coordinate rows."
        )

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df = df.drop_duplicates(["start", "end"]).sort_values(
        ["start", "end"]
    )
    df = df.reset_index(drop=True)

    invalid_intervals = df["end"] <= df["start"]
    if invalid_intervals.any():
        raise ValueError(
            f"{path} contains {int(invalid_intervals.sum())} regions with "
            "end <= start."
        )

    if df.empty:
        raise ValueError(f"No valid regions found in {path}")

    return df


def discover_raw_samples(raw_data_dir: Path) -> dict[str, list[Path]]:
    sample_files: dict[str, list[Path]] = {}

    for condition, subdirectory in RAW_DATA_SUBDIRS.items():
        files = sorted(
            (raw_data_dir / subdirectory).glob(
                "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
            )
        )
        if len(files) != 3:
            raise FileNotFoundError(
                f"{condition}: expected exactly 3 IVPA-nucleus intensity "
                f"matrices, found {len(files)} below "
                f"{raw_data_dir / subdirectory}. Files: {files}"
            )
        sample_files[condition] = files

    return sample_files


def load_one_sample_profile(
    csv_path: Path,
    candidate_index: pd.MultiIndex,
) -> tuple[pd.Series, int]:
    df = read_required_csv(
        csv_path,
        REQUIRED_RAW_COLUMNS,
        usecols=["start", "end", "n_reads"],
    )
    for column in ("start", "end", "n_reads"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[["start", "end", "n_reads"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{csv_path} contains {int(invalid.sum())} invalid rows."
        )
    if (df["n_reads"] < 0).any():
        raise ValueError(f"Negative read counts found in {csv_path}")

    grouped = df.groupby(["start", "end"], sort=False)["n_reads"].sum()
    library_size = float(grouped.sum())
    if library_size <= 0:
        raise ValueError(f"Sample contains no reads: {csv_path}")

    candidate_counts = grouped.reindex(candidate_index, fill_value=0.0)
    cpm = candidate_counts / library_size * 1_000_000.0
    log2_cpm = np.log2(cpm + 1.0)
    return log2_cpm, int(library_size)


def build_signal_profiles(
    candidates: pd.DataFrame,
    sample_files: dict[str, list[Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_index = pd.MultiIndex.from_frame(
        candidates[["start", "end"]], names=["start", "end"]
    )
    profiles: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        for csv_path in sample_files[condition]:
            original_name = csv_path.parent.parent.name
            sample_name = original_name
            if sample_name in profiles:
                sample_name = f"{condition}__{original_name}"

            profile, library_size = load_one_sample_profile(
                csv_path, candidate_index
            )
            profiles[sample_name] = profile
            metadata_rows.append(
                {
                    "sample": sample_name,
                    "condition": condition,
                    "is_knockdown": condition != "Ctrl",
                    "library_size": library_size,
                    "source_file": str(csv_path),
                }
            )

    profile_df = pd.DataFrame(profiles, index=candidate_index)
    metadata_df = pd.DataFrame(metadata_rows)
    return profile_df, metadata_df


def add_signal_metrics(
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = candidates.set_index(["start", "end"]).copy()

    all_columns = metadata["sample"].tolist()
    kd_columns = metadata.loc[metadata["is_knockdown"], "sample"].tolist()

    condition_means = pd.DataFrame(
        {
            condition: profiles[
                metadata.loc[
                    metadata["condition"].eq(condition), "sample"
                ].tolist()
            ].mean(axis=1)
            for condition in CONDITIONS
        }
    )

    scopes = {
        "all_samples": profiles[all_columns],
        "kd_samples": profiles[kd_columns],
        "condition_means": condition_means,
    }

    for scope_name, scope_df in scopes.items():
        result[f"mean_log2_cpm_{scope_name}"] = scope_df.mean(axis=1)
        result[f"variance_log2_cpm_{scope_name}"] = scope_df.var(
            axis=1, ddof=1
        )
        result[f"std_log2_cpm_{scope_name}"] = scope_df.std(axis=1, ddof=1)

    return result.reset_index(), condition_means


def compare_with_mature_rrna(
    regions: pd.DataFrame,
    boundary_tolerance: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed_rows: list[dict[str, object]] = []

    for row in regions.itertuples(index=False):
        region_start = int(row.start)
        region_end = int(row.end)
        region_length = region_end - region_start

        for rrna_name, (mature_start, mature_end) in MATURE_RRNA_INTERVALS.items():
            overlap = max(
                0,
                min(region_end, mature_end) - max(region_start, mature_start),
            )
            mature_length = mature_end - mature_start
            start_distance = region_start - mature_start
            end_distance = region_end - mature_end
            start_match = abs(start_distance) <= boundary_tolerance
            end_match = abs(end_distance) <= boundary_tolerance
            exact_match = start_distance == 0 and end_distance == 0
            boundary_match = start_match and end_match
            contains_full_mature = (
                region_start <= mature_start and region_end >= mature_end
            )
            contained_within_mature = (
                region_start >= mature_start and region_end <= mature_end
            )

            if exact_match:
                relationship = "exact_full_mature"
            elif boundary_match:
                relationship = "mature_boundary_match_with_tolerance"
            elif contains_full_mature:
                relationship = "contains_full_mature_rRNA"
            elif contained_within_mature and overlap > 0:
                relationship = "contained_within_mature_rRNA"
            elif overlap > 0:
                relationship = "partial_overlap"
            else:
                relationship = "no_overlap"

            detailed_rows.append(
                {
                    "start": region_start,
                    "end": region_end,
                    "mature_rRNA": rrna_name,
                    "mature_start": mature_start,
                    "mature_end": mature_end,
                    "start_distance_nt": start_distance,
                    "end_distance_nt": end_distance,
                    "start_boundary_match": start_match,
                    "end_boundary_match": end_match,
                    "both_boundaries_match": boundary_match,
                    "exact_full_mature_match": exact_match,
                    "overlap_nt": overlap,
                    "region_overlap_fraction": overlap / region_length,
                    "mature_overlap_fraction": overlap / mature_length,
                    "relationship": relationship,
                }
            )

    detailed = pd.DataFrame(detailed_rows)

    annotation_rows: list[dict[str, object]] = []
    for (start, end), group in detailed.groupby(["start", "end"], sort=False):
        overlapping = group.loc[group["overlap_nt"].gt(0), "mature_rRNA"].tolist()
        exact = group.loc[
            group["exact_full_mature_match"], "mature_rRNA"
        ].tolist()
        boundary = group.loc[
            group["both_boundaries_match"], "mature_rRNA"
        ].tolist()
        start_matches = group.loc[
            group["start_boundary_match"], "mature_rRNA"
        ].tolist()
        end_matches = group.loc[
            group["end_boundary_match"], "mature_rRNA"
        ].tolist()
        best = group.sort_values(
            ["overlap_nt", "mature_overlap_fraction"], ascending=False
        ).iloc[0]

        annotation_rows.append(
            {
                "start": start,
                "end": end,
                "exact_mature_match": bool(exact),
                "exact_mature_rRNA": ",".join(exact),
                "mature_boundary_match": bool(boundary),
                "boundary_matched_mature_rRNA": ",".join(boundary),
                "start_boundary_matches": ",".join(start_matches),
                "end_boundary_matches": ",".join(end_matches),
                "overlapping_mature_rRNAs": ",".join(overlapping),
                "best_overlapping_mature_rRNA": (
                    best["mature_rRNA"] if best["overlap_nt"] > 0 else ""
                ),
                "maximum_mature_overlap_nt": int(best["overlap_nt"]),
                "best_mature_relationship": best["relationship"],
            }
        )

    annotations = pd.DataFrame(annotation_rows)
    annotated = regions.merge(annotations, on=["start", "end"], how="left")
    return annotated, detailed


def apply_abundance_variance_selection(
    annotated: pd.DataFrame,
    scope: str,
    abundance_quantile: float,
    variance_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_column = f"mean_log2_cpm_{scope}"
    variance_column = f"variance_log2_cpm_{scope}"

    abundance_threshold = annotated[mean_column].quantile(abundance_quantile)
    variance_threshold = annotated[variance_column].quantile(variance_quantile)

    result = annotated.copy()
    result["high_abundance"] = result[mean_column].ge(abundance_threshold)
    result["low_variance"] = result[variance_column].le(variance_threshold)
    result["high_abundance_and_low_variance"] = (
        result["high_abundance"] & result["low_variance"]
    )

    thresholds = pd.DataFrame(
        [
            {
                "selection_scope": scope,
                "n_regions": len(result),
                "high_abundance_quantile": abundance_quantile,
                "high_abundance_threshold_mean_log2_cpm": abundance_threshold,
                "low_variance_quantile": variance_quantile,
                "low_variance_threshold": variance_threshold,
                "variance_ddof": 1,
                "n_high_abundance": int(result["high_abundance"].sum()),
                "n_low_variance": int(result["low_variance"].sum()),
                "n_both": int(
                    result["high_abundance_and_low_variance"].sum()
                ),
            }
        ]
    )
    return result, thresholds


def build_mature_summary(
    detailed: pd.DataFrame,
    selected_coordinates: set[tuple[int, int]],
) -> pd.DataFrame:
    detailed = detailed.copy()
    detailed["selected_high_abundance_low_variance"] = [
        (int(row.start), int(row.end)) in selected_coordinates
        for row in detailed.itertuples(index=False)
    ]

    rows: list[dict[str, object]] = []
    for rrna_name, group in detailed.groupby("mature_rRNA", sort=False):
        selected = group[group["selected_high_abundance_low_variance"]]
        rows.append(
            {
                "mature_rRNA": rrna_name,
                "exact_matches_all_candidates": int(
                    group["exact_full_mature_match"].sum()
                ),
                "boundary_matches_all_candidates": int(
                    group["both_boundaries_match"].sum()
                ),
                "overlaps_all_candidates": int(group["overlap_nt"].gt(0).sum()),
                "exact_matches_selected": int(
                    selected["exact_full_mature_match"].sum()
                ),
                "boundary_matches_selected": int(
                    selected["both_boundaries_match"].sum()
                ),
                "overlaps_selected": int(selected["overlap_nt"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_abundance_variance(
    result: pd.DataFrame,
    thresholds: pd.DataFrame,
    scope: str,
    output_dir: Path,
) -> None:
    mean_column = f"mean_log2_cpm_{scope}"
    variance_column = f"variance_log2_cpm_{scope}"
    abundance_threshold = thresholds.loc[
        0, "high_abundance_threshold_mean_log2_cpm"
    ]
    variance_threshold = thresholds.loc[0, "low_variance_threshold"]

    selected = result["high_abundance_and_low_variance"]
    mature_match = result["mature_boundary_match"]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        result.loc[~selected, mean_column],
        result.loc[~selected, variance_column],
        s=25,
        alpha=0.55,
        color="#7f8c8d",
        label="Other shared KD regions",
    )
    axis.scatter(
        result.loc[selected, mean_column],
        result.loc[selected, variance_column],
        s=55,
        alpha=0.9,
        color="#c0392b",
        label="High abundance and low variance",
    )
    axis.scatter(
        result.loc[mature_match, mean_column],
        result.loc[mature_match, variance_column],
        s=110,
        facecolors="none",
        edgecolors="#1f618d",
        linewidths=1.5,
        label="Mature-rRNA boundary match",
    )
    axis.axvline(
        abundance_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Abundance Q={thresholds.loc[0, 'high_abundance_quantile']:.2f}",
    )
    axis.axhline(
        variance_threshold,
        color="black",
        linestyle=":",
        linewidth=1,
        label=f"Variance Q={thresholds.loc[0, 'low_variance_quantile']:.2f}",
    )
    axis.set_xlabel(f"Mean log2(CPM + 1), scope: {scope}")
    axis.set_ylabel(f"Variance of log2(CPM + 1), scope: {scope}")
    axis.set_title("Shared KD regions: abundance and signal variance")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()

    figure.savefig(
        output_dir / f"abundance_vs_variance_{scope}.svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_csv, output_dir, raw_data_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)


    candidates = load_candidate_regions(input_csv)

    sample_files = discover_raw_samples(raw_data_dir)

    profiles, metadata = build_signal_profiles(candidates, sample_files)
    metrics, condition_means = add_signal_metrics(
        candidates, profiles, metadata
    )

    annotated, detailed = compare_with_mature_rrna(
        metrics, args.boundary_tolerance
    )

    result, thresholds = apply_abundance_variance_selection(
        annotated=annotated,
        scope=args.selection_scope,
        abundance_quantile=args.high_abundance_quantile,
        variance_quantile=args.low_variance_quantile,
    )
    selected = result[result["high_abundance_and_low_variance"]].copy()
    selected_coordinates = set(zip(selected["start"], selected["end"]))
    mature_summary = build_mature_summary(detailed, selected_coordinates)

    result.to_csv(
        output_dir / "shared_kd_regions_mature_and_signal_metrics.csv",
        sep=SEP,
        index=False,
    )
    selected.to_csv(
        output_dir / "high_abundance_low_variance_regions.csv",
        sep=SEP,
        index=False,
    )
    detailed.to_csv(
        output_dir / "mature_rRNA_comparison_long.csv",
        sep=SEP,
        index=False,
    )
    mature_summary.to_csv(
        output_dir / "mature_rRNA_summary.csv",
        sep=SEP,
        index=False,
    )
    thresholds.to_csv(
        output_dir / "abundance_variance_thresholds.csv",
        sep=SEP,
        index=False,
    )
    profiles.to_csv(
        output_dir / "log2_cpm_signal_profiles_all_samples.csv",
        sep=SEP,
    )
    condition_means.to_csv(
        output_dir / "log2_cpm_condition_mean_profiles.csv",
        sep=SEP,
    )
    metadata.to_csv(
        output_dir / "sample_metadata_and_library_sizes.csv",
        sep=SEP,
        index=False,
    )

    plot_abundance_variance(
        result=result,
        thresholds=thresholds,
        scope=args.selection_scope,
        output_dir=output_dir,
    )


    print(f"Ergebnis gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()