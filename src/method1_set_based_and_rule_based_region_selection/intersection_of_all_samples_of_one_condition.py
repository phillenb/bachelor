import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate

    raise RuntimeError(
        f"Projektwurzel konnte ausgehend von {start_path} nicht gefunden werden."
    )


PROJECT_ROOT = find_project_root(CURRENT_FILE.parent)
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from method1_set_based_and_rule_based_region_selection.normalization import (
    normalize_log_cpm,
    normalization_summary,
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTENSITY_MATRICES_DATA_DIR = PROJECT_ROOT / "data" / "intensity_matrices"

RAW_DATA_DIR = Path(RAW_DATA_DIR)
INTENSITY_MATRICES_DATA_DIR = Path(INTENSITY_MATRICES_DATA_DIR)

MIN_RAW_READS = 2
EXPECTED_SAMPLES_PER_CONDITION = 3

CONDITIONS: dict[str, list[Path]] = {
    "Ctrl": sorted(
        (RAW_DATA_DIR / "20240314_Ribolyzer_Ctrl").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "UTP18": sorted(
        (RAW_DATA_DIR / "20240314_Ribolyzer_UTP18").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "DIMT1L": sorted(
        (RAW_DATA_DIR / "20250129_Ribolyzer_DIMT1L").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "LAS1L": sorted(
        (RAW_DATA_DIR / "Las1L_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "RPL3": sorted(
        (RAW_DATA_DIR / "RPL3_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "URB1": sorted(
        (RAW_DATA_DIR / "URB1_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "WBSCR22": sorted(
        (RAW_DATA_DIR / "WBSCR22_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
}

BASE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "method1_set_based_and_rule_based_region_selection_data"
    / "shared_regions_across_samples"
)


def get_condition_samples(
    condition: str,
    csv_paths: list[Path],
) -> dict[str, Path]:

    if len(csv_paths) != EXPECTED_SAMPLES_PER_CONDITION:
        raise ValueError(
            f"{condition}: {len(csv_paths)} Samples gefunden; erwartet werden "
            f"genau {EXPECTED_SAMPLES_PER_CONDITION}. Gefundene Dateien: "
            f"{[str(path) for path in csv_paths]}"
        )

    nuc_samples = {}
    for csv_file in csv_paths:
        sample_name = csv_file.parent.parent.name
        if sample_name in nuc_samples:
            raise ValueError(
                f"{condition}: Sample-Name mehrfach vorhanden: {sample_name}"
            )
        nuc_samples[sample_name] = csv_file

    return nuc_samples


def load_intensity_matrix(csv_file_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_file_path, sep=";", header=0)

    required = ["start", "end", "n_reads"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Unerwartete Spaltenstruktur in {csv_file_path}")
    return df[required].copy()


def weighted_mean_and_variance(values, weights):

    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)

    total_weight = weights.sum()
    if total_weight == 0:
        return np.nan, np.nan

    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)

    return mean, variance


def compute_variance_per_sample(nuc_samples):
    rows = []

    for sample_name, csv_file in nuc_samples.items():
        df = load_intensity_matrix(csv_file)

        start_mean, start_var = weighted_mean_and_variance(
            df["start"],
            df["n_reads"],
        )
        end_mean, end_var = weighted_mean_and_variance(
            df["end"],
            df["n_reads"],
        )

        rows.append({
            "sample": sample_name,
            "n_cutting_sites": len(df),
            "total_reads": int(df["n_reads"].sum()),
            "start_mean": start_mean,
            "start_variance": start_var,
            "start_std": np.sqrt(start_var),
            "end_mean": end_mean,
            "end_variance": end_var,
            "end_std": np.sqrt(end_var),
        })

    variance_df = pd.DataFrame(rows).set_index("sample")
    return variance_df


def build_sample_vectors(nuc_samples: dict[str, Path]) -> pd.DataFrame:

    series_per_sample = {}

    for sample_name, csv_file in nuc_samples.items():
        df = load_intensity_matrix(csv_file)
        grouped = df.groupby(["start", "end"])["n_reads"].sum()
        series_per_sample[sample_name] = grouped
        n_present = int((grouped >= MIN_RAW_READS).sum())

    combined = pd.DataFrame(series_per_sample).fillna(0.0)
    return combined


def compute_region_consistency(
    raw_vectors,
    norm_vectors=None,
    min_raw_reads=MIN_RAW_READS,
):
    if norm_vectors is None:
        norm_vectors = raw_vectors.copy()

    if not raw_vectors.index.equals(norm_vectors.index):
        raise ValueError("raw_vectors und norm_vectors haben nicht denselben Index.")

    if not raw_vectors.columns.equals(norm_vectors.columns):
        raise ValueError("raw_vectors und norm_vectors haben nicht dieselben Spalten.")

    n_samples = raw_vectors.shape[1]

    present_mask = raw_vectors >= min_raw_reads
    n_present = present_mask.sum(axis=1)
    prevalence = n_present / n_samples

    raw_total_reads = raw_vectors.sum(axis=1)
    raw_mean_reads = raw_vectors.mean(axis=1)
    raw_median_reads = raw_vectors.median(axis=1)
    raw_std_reads = raw_vectors.std(axis=1, ddof=1)
    raw_cv = raw_std_reads / raw_mean_reads.replace(0.0, np.nan)

    norm_sum = norm_vectors.sum(axis=1)
    norm_mean = norm_vectors.mean(axis=1)
    norm_median = norm_vectors.median(axis=1)
    norm_std = norm_vectors.std(axis=1, ddof=1)
    norm_cv = norm_std / norm_mean.replace(0.0, np.nan)

    region_df = pd.DataFrame({
        "n_samples_present": n_present,
        "prevalence": prevalence,

        "raw_total_reads": raw_total_reads,
        "raw_mean_reads": raw_mean_reads,
        "raw_median_reads": raw_median_reads,
        "raw_std_reads": raw_std_reads,
        "raw_cv": raw_cv,

        "norm_sum": norm_sum,
        "norm_mean": norm_mean,
        "norm_median": norm_median,
        "norm_std": norm_std,
        "norm_cv": norm_cv,
    })

    region_df = region_df.reset_index()

    return region_df


def most_consistent_regions(
    region_df,
    min_prevalence=1.0,
    sort_by="norm_cv",
):
    if sort_by not in ["norm_cv", "raw_cv"]:
        raise ValueError("sort_by muss entweder 'norm_cv' oder 'raw_cv' sein.")

    candidates = region_df[region_df["prevalence"] >= min_prevalence].copy()

    candidates = candidates.sort_values(
        by=["prevalence", sort_by, "norm_mean", "raw_total_reads"],
        ascending=[False, True, False, False],
    )

    return candidates

def compute_region_correlation(norm_vectors, min_total_norm_signal=10, top_regions=200):

    totals = norm_vectors.sum(axis=1)

    keep = totals[totals >= min_total_norm_signal].sort_values(ascending=False)
    keep = keep.head(top_regions).index

    subset = norm_vectors.loc[keep]

    region_corr = subset.T.corr(method="pearson")

    return region_corr


def compute_similarity_matrices(norm_vectors):

    pearson = norm_vectors.corr(method="pearson")

    matrix = norm_vectors.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0

    normalized = matrix / norms

    cosine = pd.DataFrame(
        normalized.T @ normalized,
        index=norm_vectors.columns,
        columns=norm_vectors.columns,
    )

    return pearson, cosine


def most_similar_pairs(similarity_matrix, top_n=3):

    pairs = []

    for s1, s2 in combinations(similarity_matrix.columns, 2):
        pairs.append((s1, s2, similarity_matrix.loc[s1, s2]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs[:top_n]


def analyze_condition(
    condition: str,
    csv_paths: list[Path],
):


    nuc_samples = get_condition_samples(condition, csv_paths)

    variance_df = compute_variance_per_sample(nuc_samples)


    raw_vectors = build_sample_vectors(nuc_samples)


    normalization_summary(raw_vectors)

    norm_vectors = normalize_log_cpm(raw_vectors.copy())
    pearson, cosine = compute_similarity_matrices(norm_vectors)


    region_df = compute_region_consistency(
        raw_vectors=raw_vectors,
        norm_vectors=norm_vectors,
        min_raw_reads=MIN_RAW_READS,
    )
    region_df.insert(0, "condition", condition)

    intersection = (
        region_df.loc[region_df["n_samples_present"].eq(len(nuc_samples))]
        .sort_values(["start", "end"])
        .reset_index(drop=True)
    )


    consistent_norm = most_consistent_regions(
        region_df,
        min_prevalence=1.0,
        sort_by="norm_cv",
    )


    consistent_raw = most_consistent_regions(
        region_df,
        min_prevalence=1.0,
        sort_by="raw_cv",
    )


    region_corr = compute_region_correlation(norm_vectors)


    output_dir = BASE_OUTPUT_DIR / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_df.to_csv(output_dir / "nuc_variance.csv", sep=";")

    pearson.to_csv(output_dir / "nuc_pearson_similarity.csv", sep=";")
    cosine.to_csv(output_dir / "nuc_cosine_similarity.csv", sep=";")

    region_df.to_csv(
        output_dir / "nuc_region_consistency_raw_and_norm.csv",
        sep=";",
        index=False,
    )

    intersection.to_csv(
        output_dir
        / f"intersection_all_samples_min_raw_reads_{MIN_RAW_READS}.csv",
        sep=";",
        index=False,
    )

    consistent_norm.to_csv(
        output_dir / "nuc_all_shared_regions_by_norm_cv.csv",
        sep=";",
        index=False,
    )

    consistent_raw.to_csv(
        output_dir / "nuc_all_shared_regions_by_raw_cv.csv",
        sep=";",
        index=False,
    )

    region_corr.to_csv(output_dir / "nuc_region_correlation.csv", sep=";")


    return (
        variance_df,
        pearson,
        cosine,
        region_df,
        intersection,
        consistent_norm,
        consistent_raw,
    )


def main() -> None:

    for condition, csv_paths in CONDITIONS.items():
        analyze_condition(condition, csv_paths)


    print(f"Ergebnis gespeichert in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()