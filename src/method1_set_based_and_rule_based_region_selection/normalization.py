import numpy as np
import pandas as pd


def normalize_cpm(sample_vectors: pd.DataFrame) -> pd.DataFrame:
    matrix = sample_vectors.to_numpy(dtype=float)
    library_sizes = matrix.sum(axis=0)
    safe_library_sizes = library_sizes.copy()
    safe_library_sizes[safe_library_sizes == 0] = 1.0

    cpm = matrix / safe_library_sizes * 1_000_000.0

    return pd.DataFrame(
        cpm,
        index=sample_vectors.index,
        columns=sample_vectors.columns,
    )


def normalize_log_cpm(
    sample_vectors: pd.DataFrame,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    if pseudocount <= 0:
        raise ValueError("pseudocount must be greater than zero.")

    cpm_df = normalize_cpm(sample_vectors)
    return np.log2(cpm_df + pseudocount)


def normalization_summary(
    sample_vectors: pd.DataFrame,
) -> pd.DataFrame:
    raw_totals = sample_vectors.sum(axis=0)
    cpm_df = normalize_cpm(sample_vectors)
    cpm_totals = cpm_df.sum(axis=0)

    return pd.DataFrame(
        {
            "raw_total_reads": raw_totals.astype(int),
            "cpm_total": cpm_totals.round(1),
        }
    )


def variance_stabilising_transformation(
    sample_vectors: pd.DataFrame,
) -> pd.DataFrame:
    raise NotImplementedError(
        "Variance-stabilising transformation is not implemented."
    )


def relative_frequency_normalization(
    sample_vectors: pd.DataFrame,
) -> pd.DataFrame:
    raise NotImplementedError(
        "Relative-frequency normalization is not implemented."
    )