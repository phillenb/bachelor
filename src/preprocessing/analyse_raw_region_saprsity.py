"""Quantify sparsity and prevalence of the 21-sample raw region matrix.

A region is an exact (start, end) coordinate combination. The script reports:

1. the proportion of zero entries in the complete sample-by-region matrix;
2. how many regions remain at different read and sample-prevalence thresholds;
3. within-condition reproducibility across the three replicates; and
4. fold-local zero/constant features in the three balanced 14/7 splits.

No normalized data are required because presence is evaluated from raw counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEP = ";"
REQUIRED_COLUMNS = ("start", "end", "n_reads")
CONDITION_ORDER = (
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
)
EXPECTED_REPLICATES = 3

CONDITION_DIRECTORIES = {
    "Ctrl": "20240314_Ribolyzer_Ctrl",
    "UTP18": "20240314_Ribolyzer_UTP18",
    "DIMT1L": "20250129_Ribolyzer_DIMT1L",
    "LAS1L": "Las1L_NanoRibolyzer",
    "RPL3": "RPL3_NanoRibolyzer",
    "URB1": "URB1_NanoRibolyzer",
    "WBSCR22": "WBSCR22_NanoRibolyzer",
}


def find_project_root(start: Path) -> Path:
    """Find the project directory by locating data/raw."""
    candidates = [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    raise RuntimeError(
        "Projektwurzel nicht gefunden. Bitte --project-root angeben."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysiert Nullwerte und Prevalenz der Rohregionen."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Projektwurzel; wird standardmaessig automatisch erkannt.",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=2,
        help="Mindestzahl Reads, ab der eine Region als vorhanden gilt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Standard: data/method3_supervised_LogReg_RandomForrest_data/"
        "raw_region_sparsity",
    )
    args = parser.parse_args()
    if args.min_reads < 1:
        parser.error("--min-reads muss mindestens 1 sein.")
    return args


def discover_samples(raw_dir: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for condition in CONDITION_ORDER:
        condition_dir = raw_dir / CONDITION_DIRECTORIES[condition]
        paths = sorted(
            condition_dir.glob(
                "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
            )
        )
        if len(paths) != EXPECTED_REPLICATES:
            raise ValueError(
                f"{condition}: {len(paths)} Samples gefunden; erwartet wurden "
                f"{EXPECTED_REPLICATES}. Suchordner: {condition_dir}"
            )
        samples.extend((condition, path) for path in paths)
    return samples


def load_sample(csv_path: Path) -> pd.Series:
    df = pd.read_csv(
        csv_path,
        sep=SEP,
        usecols=list(REQUIRED_COLUMNS),
        low_memory=False,
    )
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"Fehlende Werte in {csv_path}")
    if (df["n_reads"] < 0).any():
        raise ValueError(f"Negative Read Counts in {csv_path}")
    if not np.all(df["start"] == np.floor(df["start"])):
        raise ValueError(f"Nicht-ganzzahlige start-Koordinate in {csv_path}")
    if not np.all(df["end"] == np.floor(df["end"])):
        raise ValueError(f"Nicht-ganzzahlige end-Koordinate in {csv_path}")

    df = df.astype({"start": "int64", "end": "int64"})
    return df.groupby(["start", "end"], sort=False)["n_reads"].sum()


def build_raw_matrix(
    samples: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, list[str]]:
    columns: dict[str, pd.Series] = {}
    labels: list[str] = []
    for condition, path in samples:
        sample_name = path.parent.parent.name
        if sample_name in columns:
            raise ValueError(f"Doppelter Sample-Name: {sample_name}")
        print(f"Lade {condition:<7} | {sample_name}")
        columns[sample_name] = load_sample(path)
        labels.append(condition)

    # Rows are regions and columns are samples. uint32 keeps the dense matrix
    # considerably smaller than float64 while preserving raw read counts.
    matrix = pd.DataFrame(columns).fillna(0)
    max_count = matrix.to_numpy(copy=False).max(initial=0)
    dtype = "uint32" if max_count <= np.iinfo(np.uint32).max else "uint64"
    matrix = matrix.astype(dtype)
    return matrix, labels


def percentage(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else np.nan


def global_sparsity(raw: np.ndarray) -> pd.DataFrame:
    n_regions, n_samples = raw.shape
    n_cells = int(raw.size)
    nonzero = int(np.count_nonzero(raw))
    zeros = n_cells - nonzero
    all_zero_regions = int(np.sum(np.all(raw == 0, axis=1)))
    return pd.DataFrame(
        [
            {
                "n_samples": n_samples,
                "n_regions": n_regions,
                "n_matrix_entries": n_cells,
                "n_nonzero_entries": nonzero,
                "n_zero_entries": zeros,
                "zero_entries_percent": percentage(zeros, n_cells),
                "globally_all_zero_regions": all_zero_regions,
            }
        ]
    )


def prevalence_distribution(raw: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    n_regions, n_samples = raw.shape
    for read_threshold in (1, 2):
        prevalence = np.sum(raw >= read_threshold, axis=1)
        counts = np.bincount(prevalence, minlength=n_samples + 1)
        for n_present, count in enumerate(counts):
            rows.append(
                {
                    "minimum_reads_per_sample": read_threshold,
                    "samples_with_region": n_present,
                    "n_regions": int(count),
                    "percent_of_all_regions": percentage(int(count), n_regions),
                }
            )
    return pd.DataFrame(rows)


def filter_summary(
    raw: np.ndarray,
    labels: np.ndarray,
    min_reads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_regions = raw.shape[0]
    present = raw >= min_reads
    prevalence = present.sum(axis=1)

    rows: list[dict] = []

    def add(name: str, mask: np.ndarray) -> None:
        retained = int(mask.sum())
        rows.append(
            {
                "filter": name,
                "n_regions_retained": retained,
                "n_regions_removed": n_regions - retained,
                "percent_retained": percentage(retained, n_regions),
                "fold_reduction_vs_all": n_regions / retained
                if retained
                else np.inf,
            }
        )

    add("all_observed_regions", np.ones(n_regions, dtype=bool))
    add(f">={min_reads}_reads_in_>=1_sample", prevalence >= 1)
    add(f">={min_reads}_reads_in_>=2_samples", prevalence >= 2)
    add(f">={min_reads}_reads_in_>=3_samples", prevalence >= 3)

    condition_rows: list[dict] = []
    reproducible_two = np.zeros(n_regions, dtype=bool)
    reproducible_three = np.zeros(n_regions, dtype=bool)
    for condition in CONDITION_ORDER:
        condition_mask = labels == condition
        replicate_prevalence = present[:, condition_mask].sum(axis=1)
        for required_replicates in (1, 2, 3):
            retained = int(np.sum(replicate_prevalence >= required_replicates))
            condition_rows.append(
                {
                    "condition": condition,
                    "minimum_reads_per_sample": min_reads,
                    "minimum_replicates": required_replicates,
                    "n_regions": retained,
                }
            )
        reproducible_two |= replicate_prevalence >= 2
        reproducible_three |= replicate_prevalence >= 3

    add(
        f">={min_reads}_reads_in_>=2_replicates_of_any_condition",
        reproducible_two,
    )
    add(
        f">={min_reads}_reads_in_all_3_replicates_of_any_condition",
        reproducible_three,
    )
    return pd.DataFrame(rows), pd.DataFrame(condition_rows)


def balanced_fold_summary(
    raw: np.ndarray,
    labels: np.ndarray,
    sample_names: list[str],
) -> pd.DataFrame:
    per_condition = {
        condition: np.flatnonzero(labels == condition)
        for condition in CONDITION_ORDER
    }
    rows: list[dict] = []
    all_indices = np.arange(raw.shape[1])

    for fold in range(EXPECTED_REPLICATES):
        test_idx = np.asarray(
            [per_condition[condition][fold] for condition in CONDITION_ORDER]
        )
        train_idx = all_indices[~np.isin(all_indices, test_idx)]
        train = raw[:, train_idx]
        test = raw[:, test_idx]

        all_zero_train = np.all(train == 0, axis=1)
        constant_train = np.ptp(train, axis=1) == 0
        test_exclusive = all_zero_train & np.any(test > 0, axis=1)
        rows.append(
            {
                "fold": fold + 1,
                "n_training_samples": len(train_idx),
                "n_test_samples": len(test_idx),
                "test_samples": "|".join(sample_names[i] for i in test_idx),
                "all_zero_in_training": int(all_zero_train.sum()),
                "constant_in_training": int(constant_train.sum()),
                "variable_in_training": int((~constant_train).sum()),
                "test_exclusive_regions": int(test_exclusive.sum()),
                "all_zero_in_training_percent": percentage(
                    int(all_zero_train.sum()), raw.shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path(__file__).parent)
    )
    raw_dir = project_root / "data" / "raw"
    output_dir = args.output_dir or (
        project_root
        / "data"
        / "method3_supervised_LogReg_RandomForrest_data"
        / "raw_region_sparsity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(raw_dir)
    matrix, label_list = build_raw_matrix(samples)
    raw = matrix.to_numpy(copy=False)
    labels = np.asarray(label_list)
    sample_names = matrix.columns.astype(str).tolist()

    sparsity = global_sparsity(raw)
    prevalence = prevalence_distribution(raw)
    filters, conditions = filter_summary(raw, labels, args.min_reads)
    folds = balanced_fold_summary(raw, labels, sample_names)

    sparsity.to_csv(output_dir / "global_matrix_sparsity.csv", sep=SEP, index=False)
    prevalence.to_csv(
        output_dir / "region_prevalence_distribution.csv", sep=SEP, index=False
    )
    filters.to_csv(output_dir / "prevalence_filter_summary.csv", sep=SEP, index=False)
    conditions.to_csv(
        output_dir / "within_condition_prevalence.csv", sep=SEP, index=False
    )
    folds.to_csv(
        output_dir / "balanced_fold_feature_availability.csv", sep=SEP, index=False
    )

    row = sparsity.iloc[0]
    print("\nGlobale Matrix")
    print(f"  Samples:              {int(row.n_samples):,}")
    print(f"  Regionen:             {int(row.n_regions):,}")
    print(f"  Nullwerte:            {int(row.n_zero_entries):,}")
    print(f"  Anteil Nullwerte:     {row.zero_entries_percent:.2f}%")
    print("\nPrevalenzfilter")
    print(filters.to_string(index=False))
    print("\nBalanced folds")
    print(folds.drop(columns="test_samples").to_string(index=False))
    print(f"\nErgebnisse: {output_dir}")


if __name__ == "__main__":
    main()