from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import profile_RNA45SN1_intermediates_across_conditions_expanded as base


SEP = base.SEP

LOG2FC_COLOR_LIMIT = 4.0

DEFAULT_METHOD_UNION_CSV = (
    base.PROJECT_ROOT
    / "data"
    / "method_comparison"
    / "three_method_region_sets"
    / "method_union_regions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path(base.DATA_DIR)
    / "biological validation"
    / "method_union_alignment"
)


def get_fixed_method_union_path() -> Path:
    """Locate the configured global method-union file."""
    path = DEFAULT_METHOD_UNION_CSV.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Die fest konfigurierte Methoden-Union wurde nicht gefunden: "
            f"{path}"
        )
    return path


def load_method_union(path: Path) -> pd.DataFrame:
    """Load unique start/end coordinates from the method union."""
    raw = base.read_delimited(path)
    start_column = base.find_column(
        raw,
        ("start", "region_start", "start_coordinate", "start_position", "x"),
    )
    end_column = base.find_column(
        raw,
        ("end", "region_end", "end_coordinate", "end_position", "y"),
    )
    union = pd.DataFrame(
        {
            "start": pd.to_numeric(raw[start_column], errors="raise").astype(np.int64),
            "end": pd.to_numeric(raw[end_column], errors="raise").astype(np.int64),
        }
    )
    invalid = union["end"] <= union["start"]
    if invalid.any():
        examples = union.loc[invalid].head(5).to_dict("records")
        raise ValueError(
            "Die Methoden-Union enthaelt Regionen mit end <= start. "
            f"Beispiele: {examples}"
        )

    # Collapse provenance rows into unique coordinates.
    union = (
        union.drop_duplicates(["start", "end"])
        .sort_values(["start", "end"])
        .reset_index(drop=True)
    )
    if union.empty:
        raise ValueError("Die Methoden-Union enthaelt keine gueltigen Regionen.")
    return union


def restrict_samples_to_union(
    samples: list[dict],
    meta: pd.DataFrame,
    method_union: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """Filter exact coordinates and retain the raw-read CPM denominators."""
    union_index = pd.MultiIndex.from_frame(method_union[["start", "end"]])
    restricted_samples: list[dict] = []
    qc_rows: list[dict] = []

    for sample in samples:
        raw_regions = sample["regions"]
        raw_index = pd.MultiIndex.from_frame(raw_regions[["start", "end"]])
        keep = raw_index.isin(union_index)
        restricted = raw_regions.loc[keep].copy().reset_index(drop=True)
        union_reads = float(restricted["n_reads"].sum())

        restricted_sample = dict(sample)
        restricted_sample["regions"] = restricted
        # Keep the original raw-read CPM denominator.
        restricted_sample["total_reads"] = float(sample["total_reads"])
        restricted_samples.append(restricted_sample)

        qc_rows.append(
            {
                **base.base_sample_fields(restricted_sample),
                "n_regions_in_global_method_union": len(method_union),
                "n_raw_regions_before_filter": len(raw_regions),
                "n_observed_union_regions": len(restricted),
                "union_raw_reads": union_reads,
                "union_read_fraction_of_all_sample_reads": (
                    union_reads / sample["total_reads"]
                ),
                "union_cpm_sum": union_reads / sample["total_reads"] * 1_000_000.0,
            }
        )

    qc = pd.DataFrame(qc_rows)
    if int(qc["n_observed_union_regions"].sum()) == 0:
        raise ValueError(
            "Keine einzige (start, end)-Region der Methoden-Union wurde in den "
            "Rohdaten gefunden. Bitte Koordinatensystem und Spalten pruefen."
        )

    enriched_meta = meta.merge(
        qc[
            [
                "sample_id",
                "n_regions_in_global_method_union",
                "n_raw_regions_before_filter",
                "n_observed_union_regions",
                "union_raw_reads",
                "union_read_fraction_of_all_sample_reads",
                "union_cpm_sum",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    return restricted_samples, enriched_meta


def save_log2fc_heatmap(
    comparisons: pd.DataFrame,
    analysis: str,
    output_stem: Path,
    title: str,
    tolerance: int | None = None,
) -> None:
    """Export a union heatmap with fixed -4 to +4 limits."""
    selected = comparisons.loc[comparisons["analysis"] == analysis].copy()
    if tolerance is not None:
        selected = selected.loc[selected["tolerance_nt"] == tolerance].copy()
    selected["feature_label"] = base.display_feature_labels(selected, analysis)
    matrix = selected.pivot(
        index="condition",
        columns="feature_label",
        values="log2_fold_change_vs_ctrl",
    ).reindex(index=base.KD_ORDER)

    if analysis in {"mro_intermediate", "intermediate"}:
        available = selected.drop_duplicates("fragment").sort_values(
            ["literature_start", "literature_end"]
        )["fragment"].astype(str).tolist()
        matrix = matrix.reindex(columns=available)
    elif analysis == "boundary_either":
        matrix = matrix.reindex(columns=sorted(matrix.columns, key=int))
    elif analysis == "section_fractional":
        names = {
            "5ETS": "5'ETS",
            "18S": "18S",
            "ITS1": "ITS1",
            "5-8S": "5.8S",
            "ITS2": "ITS2",
            "28S": "28S",
            "3ETS": "3'ETS",
        }
        matrix = matrix.reindex(
            columns=[names[name] for name in base.SECTION_ORDER]
        )

    values = matrix.to_numpy(dtype=float)
    width = max(8.0, 0.58 * len(matrix.columns) + 3.0)
    height = max(4.5, 0.65 * len(matrix.index) + 2.0)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        vmin=-LOG2FC_COLOR_LIMIT,
        vmax=LOG2FC_COLOR_LIMIT,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_xlabel("RNA45SN1 feature")
    axis.set_ylabel("Knockdown vs Ctrl")
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("log2 fold change vs Ctrl")
    colorbar.set_ticks(np.arange(-LOG2FC_COLOR_LIMIT, LOG2FC_COLOR_LIMIT + 1.0, 1.0))
    axis.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def save_mro_barplots_by_condition(
    comparisons: pd.DataFrame,
    literature: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export MRO log2FC bar plots for each knockdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = comparisons.loc[
        comparisons["analysis"] == "mro_intermediate"
    ].copy()
    # Shared fixed colour scale.
    colour_map = plt.get_cmap("RdBu_r")
    colour_norm = matplotlib.colors.Normalize(
        vmin=-LOG2FC_COLOR_LIMIT,
        vmax=LOG2FC_COLOR_LIMIT,
        clip=True,
    )
    fragment_order = literature.sort_values(["start", "end", "fragment"])[
        "fragment"
    ].astype(str).tolist()

    for condition in base.KD_ORDER:
        condition_data = (
            selected.loc[selected["condition"] == condition]
            .set_index("fragment")
            .reindex(fragment_order)
            .reset_index()
        )
        values = condition_data["log2_fold_change_vs_ctrl"].to_numpy(dtype=float)
        consistent = condition_data[
            "direction_consistent_all_three_replicates"
        ].fillna(False).to_numpy(dtype=bool)
        colors = [
            colour_map(colour_norm(value))
            if flag and np.isfinite(value)
            else "#BDBDBD"
            for value, flag in zip(values, consistent)
        ]

        y = np.arange(len(condition_data))
        fig, axis = plt.subplots(
            figsize=(10.5, max(8.0, 0.27 * len(condition_data) + 2.0))
        )
        axis.barh(y, values, color=colors, edgecolor="none")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(y)
        axis.set_yticklabels(condition_data["fragment"])
        axis.invert_yaxis()
        axis.set_xlabel("log2 fold change vs Ctrl (CPM; all-raw-read denominator)")
        axis.set_ylabel("RNA45SN1 literature intermediate")
        axis.set_title(f"{condition} vs Ctrl: MRO-assigned method-union regions")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        fig.tight_layout()

        stem = output_dir / f"{condition}_vs_Ctrl_mro_intermediate_log2fc"
        for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
        plt.close(fig)


def adapt_report_for_union(
    report_path: Path,
    union_path: Path,
    method_union: pd.DataFrame,
    filter_qc: pd.DataFrame,
) -> None:
    """Add union-specific information to the base report."""
    original = report_path.read_text(encoding="utf-8")
    original = original.replace(
        "RNA45SN1 INTERMEDIATE PROFILES ACROSS CONDITIONS",
        "RNA45SN1 METHOD-UNION INTERMEDIATE PROFILES ACROSS CONDITIONS",
        1,
    ).replace(
        "- Eingang: alle beobachteten Rohregionen mit n_reads",
        "- Eingang: exakte (start, end)-Regionen der Drei-Methoden-Union mit "
        "n_reads aus den Rohdaten",
        1,
    )
    header = [
        "Method-union filter",
        f"- Union-Datei: {union_path}",
        f"- Eindeutige Union-Koordinaten: {len(method_union)}",
        "- Filter: exakter Match von start UND end; die Condition-/Methoden-"
        "Provenienz einer Union-Zeile wird fuer das globale Set nicht verwendet.",
        "- CPM-Nenner: Gesamtzahl aller Reads des jeweiligen Rohsamples vor "
        "dem Union-Filter.",
        "- Mittlerer Anteil der Rohreads in Union-Regionen: "
        f"{100 * filter_qc['union_read_fraction_of_all_sample_reads'].mean():.3f}% "
        f"(min={100 * filter_qc['union_read_fraction_of_all_sample_reads'].min():.3f}%, "
        f"max={100 * filter_qc['union_read_fraction_of_all_sample_reads'].max():.3f}%)",
        "",
    ]
    report_path.write_text("\n".join(header) + original, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--literature-csv", type=Path, default=base.DEFAULT_LITERATURE_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tolerances",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Predefined endpoint/boundary tolerances in nt (default: 1 5 10).",
    )
    parser.add_argument("--pseudocount-cpm", type=float, default=0.01)
    parser.add_argument(
        "--minimum-mro",
        type=float,
        default=0.0,
        help="Minimum best MRO for assignment (default: 0.0).",
    )
    parser.add_argument(
        "--minimum-mean-cpm-for-priority", type=float, default=1.0
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    tolerances = tuple(sorted(set(args.tolerances)))
    if not tolerances or any(value < 0 for value in tolerances):
        raise ValueError("Toleranzen muessen nichtnegative Ganzzahlen sein.")
    if args.pseudocount_cpm <= 0:
        raise ValueError("--pseudocount-cpm muss positiv sein.")
    if not 0.0 <= args.minimum_mro <= 1.0:
        raise ValueError("--minimum-mro muss zwischen 0 und 1 liegen.")
    if args.minimum_mean_cpm_for_priority < 0:
        raise ValueError("--minimum-mean-cpm-for-priority darf nicht negativ sein.")
    return tolerances


def save_tables_and_plots(
    args: argparse.Namespace,
    tolerances: tuple[int, ...],
    method_union: pd.DataFrame,
    literature: pd.DataFrame,
    boundaries: pd.DataFrame,
    sections: pd.DataFrame,
    meta: pd.DataFrame,
    mro_intermediate: pd.DataFrame,
    mro_assignments: pd.DataFrame,
    mro_qc: pd.DataFrame,
    mro_condition_summary: pd.DataFrame,
    intermediate: pd.DataFrame,
    assignments: pd.DataFrame,
    intermediate_qc: pd.DataFrame,
    boundary: pd.DataFrame,
    section: pd.DataFrame,
    containment: pd.DataFrame,
    containment_summary: pd.DataFrame,
    exclusive_assignment: pd.DataFrame,
    exclusive_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    top_consistent: pd.DataFrame,
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "method_union_regions_standardized.csv": method_union,
        "literature_intermediates_standardized.csv": literature,
        "literature_boundaries.csv": boundaries,
        "RNA45SN1_basic_sections.csv": sections,
        "sample_metadata_and_union_filter_qc.csv": meta,
        "sample_mro_intermediate_abundance.csv": mro_intermediate,
        "mro_region_assignments.csv": mro_assignments,
        "mro_assignment_qc.csv": mro_qc,
        "condition_mro_intermediate_summary.csv": mro_condition_summary,
        "sample_intermediate_abundance.csv": intermediate,
        "matched_regions_to_intermediates.csv": assignments,
        "intermediate_assignment_qc.csv": intermediate_qc,
        "sample_boundary_usage.csv": boundary,
        "sample_section_signal.csv": section,
        "sample_fragment_containment_percentages.csv": containment,
        "condition_fragment_containment_summary.csv": containment_summary,
        "sample_smallest_containing_fragment_assignment.csv": exclusive_assignment,
        "condition_smallest_fragment_composition.csv": exclusive_summary,
        "condition_vs_ctrl_all_features.csv": comparisons,
        "condition_mro_intermediate_vs_ctrl.csv": comparisons.loc[
            comparisons["analysis"] == "mro_intermediate"
        ],
        "priority_consistent_condition_changes.csv": top_consistent,
    }
    for filename, frame in table_map.items():
        frame.to_csv(output_dir / filename, sep=SEP, index=False)

    for tolerance in tolerances:
        save_log2fc_heatmap(
            comparisons,
            "intermediate",
            output_dir / f"intermediate_log2fc_tolerance_{tolerance}nt",
            f"Intermediate-like method-union regions (boundary tolerance +/-{tolerance} nt)",
            tolerance=tolerance,
        )
        save_log2fc_heatmap(
            comparisons,
            "boundary_either",
            output_dir / f"boundary_usage_log2fc_tolerance_{tolerance}nt",
            f"Method-union boundary usage (tolerance +/-{tolerance} nt)",
            tolerance=tolerance,
        )
    save_log2fc_heatmap(
        comparisons,
        "mro_intermediate",
        output_dir / "mro_intermediate_log2fc",
        "Exclusive MRO-assigned method-union regions",
    )
    save_log2fc_heatmap(
        comparisons,
        "section_fractional",
        output_dir / "section_fractional_signal_log2fc",
        "Overlap-weighted 47S section signal in method-union regions",
    )
    base.save_fragment_containment_heatmap(
        containment_summary,
        literature,
        output_dir / "fragment_containment_percent_heatmap",
    )
    base.save_exclusive_assignment_plot(
        exclusive_summary,
        output_dir / "smallest_fragment_assignment_composition",
    )
    save_mro_barplots_by_condition(
        comparisons,
        literature,
        output_dir / "mro_barplots_by_condition",
    )


def main() -> None:
    args = parse_args()
    tolerances = validate_args(args)
    method_union_path = get_fixed_method_union_path()
    args.method_union_csv = method_union_path


    method_union = load_method_union(method_union_path)

    literature = base.load_literature(args.literature_csv)
    base.validate_expanded_literature(literature, args.literature_csv)
    boundaries = base.build_boundaries(literature)
    sections = base.build_sections(literature)

    raw_samples, raw_meta = base.load_samples(base.CONDITIONS)
    samples, meta = restrict_samples_to_union(raw_samples, raw_meta, method_union)

    mro_intermediate, mro_assignments, mro_qc = (
        base.compute_mro_intermediate_profiles(
            samples, literature, args.minimum_mro
        )
    )
    mro_condition_summary = base.summarize_mro_conditions(
        mro_intermediate, args.pseudocount_cpm
    )

    intermediate, assignments, intermediate_qc = base.compute_intermediate_profiles(
        samples, literature, tolerances
    )

    boundary = base.compute_boundary_profiles(samples, boundaries, tolerances)
    section = base.compute_section_profiles(samples, sections)
    containment, exclusive_assignment = base.compute_fragment_containment_profiles(
        samples, literature
    )
    containment_summary = base.summarize_fragment_containment(containment)
    exclusive_summary = base.summarize_exclusive_assignments(exclusive_assignment)

    comparisons = base.build_comparison_tables(
        mro_intermediate,
        intermediate,
        boundary,
        section,
        args.pseudocount_cpm,
    )
    top_consistent = base.select_top_consistent_changes(
        comparisons, args.minimum_mean_cpm_for_priority
    )

    save_tables_and_plots(
        args,
        tolerances,
        method_union,
        literature,
        boundaries,
        sections,
        meta,
        mro_intermediate,
        mro_assignments,
        mro_qc,
        mro_condition_summary,
        intermediate,
        assignments,
        intermediate_qc,
        boundary,
        section,
        containment,
        containment_summary,
        exclusive_assignment,
        exclusive_summary,
        comparisons,
        top_consistent,
    )
    base.write_report(
        meta,
        mro_qc,
        intermediate_qc,
        comparisons,
        top_consistent,
        containment_summary,
        tolerances,
        args.minimum_mro,
        args.minimum_mean_cpm_for_priority,
        args.output_dir,
    )
    report_path = args.output_dir / "RNA45SN1_condition_profile_report.txt"
    adapt_report_for_union(
        report_path, args.method_union_csv, method_union, meta
    )

    print(f"Ergebnisse gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()