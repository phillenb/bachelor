from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score


def _find_src_dir(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if path.name == "src":
            return path
    raise RuntimeError("'src'-Verzeichnis nicht gefunden.")


SRC_DIR = _find_src_dir(Path(__file__))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

if not RAW_DATA_DIR.is_dir():
    raise FileNotFoundError(
        "Kein Rohdatenordner gefunden: "
        f"{RAW_DATA_DIR}"
    )


CONDITIONS: list[tuple[str, str]] = [
    ("Ctrl", "20240314_Ribolyzer_Ctrl"),
    ("UTP18", "20240314_Ribolyzer_UTP18"),
    ("DIMT1L", "20250129_Ribolyzer_DIMT1L"),
    ("LAS1L", "Las1L_NanoRibolyzer"),
    ("RPL3", "RPL3_NanoRibolyzer"),
    ("URB1", "URB1_NanoRibolyzer"),
    ("WBSCR22", "WBSCR22_NanoRibolyzer"),
]

RANDOM_STATE = 42

N_NEIGHBORS_VALUES = (3, 5, 8)
MIN_DIST_VALUES = (0.0, 0.1, 0.3)
METRIC = "cosine"

PRIMARY_N_NEIGHBORS = 5
PRIMARY_MIN_DIST = 0.1

TOP_REGIONS_PER_CONDITION = 250
FEATURE_SCORE_CHUNK_SIZE = 50_000
MIN_RAW_READS_FOR_PRESENCE = 2
MIN_REPLICATES_PRESENT_FULL = 2
MIN_BALANCED_SELECTION_FOLDS = 2

OUTPUT_DIR = (
    DATA_DIR
    / "method2_unsupervised_selection_UMAP_PCA_PCoA_data"
    / "umap_across_samples_on_raw_data"
)


CONDITION_COLORS: dict[str, str] = {
    "Ctrl": "#1f77b4",
    "UTP18": "#d62728",
    "DIMT1L": "#2ca02c",
    "LAS1L": "#ff7f0e",
    "RPL3": "#9467bd",
    "URB1": "#8c564b",
    "WBSCR22": "#e377c2",
}


@dataclass(frozen=True)
class UmapConfig:
    n_neighbors: int
    min_dist: float
    metric: str = METRIC

    @property
    def label(self) -> str:
        return (
            f"n_neighbors={self.n_neighbors}, "
            f"min_dist={self.min_dist:g}, metric={self.metric}"
        )


def _is_ivpa_nucleus_sample(sample_name: str) -> bool:
    name = sample_name.lower()
    nucleus_markers = ("nuc", "nucleus", "nucleaus")
    return "ivpa" in name and any(marker in name for marker in nucleus_markers)


def find_samples_for_condition(
    condition_label: str,
    condition_subdir: str,
) -> dict[str, Path]:
    condition_dir = Path(RAW_DATA_DIR) / condition_subdir
    if not condition_dir.is_dir():
        raise FileNotFoundError(
            f"Condition-Verzeichnis nicht gefunden: {condition_dir}"
        )

    result: dict[str, Path] = {}
    for csv_path in condition_dir.glob(
        "*/intensity_matrix/intensity_matrix.csv"
    ):
        sample_name = csv_path.parent.parent.name
        if _is_ivpa_nucleus_sample(sample_name):
            result[sample_name] = csv_path

    result = dict(sorted(result.items()))
    if not result:
        raise RuntimeError(
            f"Keine IVPA-Nucleus-Samples fuer {condition_label} in "
            f"{condition_dir} gefunden."
        )

    if len(result) != 3:
        print(
            f"  WARNUNG: Fuer {condition_label} wurden {len(result)} statt "
            "der erwarteten 3 Replikate gefunden."
        )
    return result


def load_sample_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(
        csv_path,
        sep=";",
        usecols=["start", "end", "n_reads"],
        dtype={"start": np.int32, "end": np.int32, "n_reads": np.float32},
    )
    return df.groupby(["start", "end"], sort=False)["n_reads"].sum()


def build_raw_sample_feature_matrix(
    conditions: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_series: dict[str, pd.Series] = {}
    meta_rows: list[dict[str, str]] = []

    for condition_label, condition_subdir in conditions:
        samples = find_samples_for_condition(condition_label, condition_subdir)
        for sample_name, csv_path in samples.items():
            sample_id = f"{condition_label}__{sample_name}"
            all_series[sample_id] = load_sample_series(csv_path)
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "condition": condition_label,
                    "source_file": str(csv_path),
                }
            )

    if not all_series:
        raise RuntimeError("Keine Samples geladen. Bitte CONDITIONS pruefen.")

    raw_matrix = pd.DataFrame(all_series, dtype=np.float32).T.fillna(0.0)
    raw_matrix.index.name = "sample_id"
    meta = pd.DataFrame(meta_rows).set_index("sample_id").loc[raw_matrix.index]

    return raw_matrix, meta


def normalize_log_cpm(raw_matrix: pd.DataFrame) -> pd.DataFrame:
    counts = raw_matrix.to_numpy(dtype=np.float32, copy=True)
    library_sizes = counts.sum(axis=1, keepdims=True, dtype=np.float64)
    if np.any(library_sizes == 0):
        empty_samples = raw_matrix.index[(library_sizes[:, 0] == 0)].tolist()
        raise ValueError(f"Samples ohne Reads gefunden: {empty_samples}")

    counts /= library_sizes.astype(np.float32)
    counts *= np.float32(1_000_000.0)
    np.log2(counts + np.float32(1.0), out=counts)

    return pd.DataFrame(
        counts,
        index=raw_matrix.index,
        columns=raw_matrix.columns,
        dtype=np.float32,
    )


def make_configs(n_samples: int) -> list[UmapConfig]:
    configs: list[UmapConfig] = []
    seen: set[tuple[int, float]] = set()
    for requested_neighbors in N_NEIGHBORS_VALUES:
        n_neighbors = min(requested_neighbors, n_samples - 1)
        if n_neighbors < 2:
            raise ValueError("UMAP benoetigt mindestens 3 Samples.")
        for min_dist in MIN_DIST_VALUES:
            key = (n_neighbors, min_dist)
            if key not in seen:
                configs.append(UmapConfig(n_neighbors, min_dist))
                seen.add(key)
    return configs


def run_umap_grid(
    X: np.ndarray,
    sample_ids: pd.Index,
) -> tuple[pd.DataFrame, object]:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn ist nicht installiert. Installation: "
            "pip install umap-learn"
        ) from exc

    rows: list[pd.DataFrame] = []
    primary_graph = None
    configs = make_configs(X.shape[0])
    for run_id, config in enumerate(configs, start=1):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            init="spectral",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(X)
        if (
            config.n_neighbors == PRIMARY_N_NEIGHBORS
            and config.min_dist == PRIMARY_MIN_DIST
        ):
            primary_graph = reducer.graph_.tocsr().astype(np.float64).copy()
        run = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "UMAP_1": embedding[:, 0],
                "UMAP_2": embedding[:, 1],
                "n_neighbors": config.n_neighbors,
                "min_dist": config.min_dist,
                "metric": config.metric,
                "random_state": RANDOM_STATE,
                "run_label": config.label,
            }
        )
        rows.append(run)

    if primary_graph is None:
        raise RuntimeError("Primaerer UMAP-Graph wurde nicht erzeugt.")
    return pd.concat(rows, ignore_index=True), primary_graph


def fit_primary_umap(
    X: np.ndarray,
) -> tuple[np.ndarray, object]:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn ist nicht installiert. Installation: "
            "pip install umap-learn"
        ) from exc

    n_neighbors = min(PRIMARY_N_NEIGHBORS, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=PRIMARY_MIN_DIST,
        metric=METRIC,
        init="spectral",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(X)
    graph = reducer.graph_.tocsr().astype(np.float64).copy()
    return embedding, graph


def balanced_replicate_train_splits(meta: pd.DataFrame) -> list[np.ndarray]:
    conditions = meta["condition"].astype(str).to_numpy()
    per_condition = {
        condition: np.flatnonzero(conditions == condition)
        for condition, _ in CONDITIONS
    }
    for condition, indices in per_condition.items():
        if len(indices) != 3:
            raise ValueError(
                f"{condition}: exakt drei Samples fuer Stabilitaet erwartet, "
                f"gefunden wurden {len(indices)}."
            )

    all_indices = np.arange(len(meta))
    train_splits: list[np.ndarray] = []
    for replicate_position in range(3):
        test_indices = np.asarray(
            [
                per_condition[condition][replicate_position]
                for condition, _ in CONDITIONS
            ],
            dtype=int,
        )
        train_splits.append(all_indices[~np.isin(all_indices, test_indices)])
    return train_splits


def _largest_indices(values: np.ndarray, n_keep: int) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return np.asarray([], dtype=int)
    n_keep = min(n_keep, len(valid))
    if n_keep == len(valid):
        candidates = valid
    else:
        relative = np.argpartition(values[valid], -n_keep)[-n_keep:]
        candidates = valid[relative]
    return candidates[np.argsort(values[candidates])[::-1]]


def score_top_regions_for_umap_graph(
    raw_matrix: pd.DataFrame,
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
    sample_indices: np.ndarray,
    graph: object,
    top_regions_per_condition: int,
    chunk_size: int,
    fold_label: str,
    minimum_replicates_present: int,
) -> pd.DataFrame:
    sample_indices = np.asarray(sample_indices, dtype=int)
    if graph.shape != (len(sample_indices), len(sample_indices)):
        raise ValueError(
            f"{fold_label}: Graphform {graph.shape} passt nicht zu "
            f"{len(sample_indices)} Samples."
        )

    weights = np.asarray(graph.toarray(), dtype=np.float64)
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    total_degree = float(degree.sum())
    if total_degree <= 0 or np.any(degree <= 0):
        raise ValueError(f"{fold_label}: ungueltiger oder unverbundener UMAP-Graph.")

    raw_values = raw_matrix.to_numpy(dtype=np.float32, copy=False)
    normalized_values = normalized.to_numpy(dtype=np.float32, copy=False)
    subset_raw = raw_values[sample_indices]
    subset_normalized = normalized_values[sample_indices]
    subset_conditions = meta.iloc[sample_indices]["condition"].astype(str).to_numpy()
    condition_labels = [condition for condition, _ in CONDITIONS]
    condition_positions = [
        np.flatnonzero(subset_conditions == condition)
        for condition in condition_labels
    ]
    for condition, positions in zip(condition_labels, condition_positions):
        if len(positions) < minimum_replicates_present:
            raise ValueError(
                f"{fold_label}: {condition} hat nur {len(positions)} Samples."
            )

    starts = normalized.columns.get_level_values("start").to_numpy(dtype=np.int64)
    ends = normalized.columns.get_level_values("end").to_numpy(dtype=np.int64)
    n_features = normalized.shape[1]
    pools: dict[str, list[pd.DataFrame]] = {
        condition: [] for condition in condition_labels
    }
    epsilon = np.finfo(np.float64).eps

    for chunk_start in range(0, n_features, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_features)
        feature_slice = slice(chunk_start, chunk_end)
        values = subset_normalized[:, feature_slice].astype(np.float64, copy=False)
        counts = subset_raw[:, feature_slice]

        weighted_mean = (degree @ values) / total_degree
        centered = values - weighted_mean[None, :]
        denominator = np.sum(degree[:, None] * centered * centered, axis=0)
        graph_projection = weights @ centered
        numerator = denominator - np.sum(centered * graph_projection, axis=0)
        numerator = np.maximum(numerator, 0.0)
        laplacian = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.inf),
            where=denominator > epsilon,
        )

        ordinary_mean = values.mean(axis=0)
        ss_total = np.sum((values - ordinary_mean[None, :]) ** 2, axis=0)
        condition_means = np.vstack(
            [values[positions].mean(axis=0) for positions in condition_positions]
        )
        ss_between = np.zeros(values.shape[1], dtype=np.float64)
        for positions, condition_mean in zip(
            condition_positions,
            condition_means,
        ):
            ss_between += len(positions) * (condition_mean - ordinary_mean) ** 2
        eta_squared = np.divide(
            ss_between,
            ss_total,
            out=np.zeros_like(ss_between),
            where=ss_total > epsilon,
        )
        eta_squared = np.clip(eta_squared, 0.0, 1.0)

        dominant_index = condition_means.argmax(axis=0)
        sorted_means = np.sort(condition_means, axis=0)
        dominant_mean = sorted_means[-1]
        second_mean = sorted_means[-2]
        signal_margin = dominant_mean - second_mean
        relative_margin = np.divide(
            signal_margin,
            np.abs(dominant_mean) + epsilon,
            out=np.zeros_like(signal_margin),
            where=np.abs(dominant_mean) > epsilon,
        )
        relative_margin = np.clip(relative_margin, 0.0, 1.0)

        prevalence = np.vstack(
            [
                (counts[positions] >= MIN_RAW_READS_FOR_PRESENCE).sum(axis=0)
                for positions in condition_positions
            ]
        )
        dominant_prevalence = prevalence[
            dominant_index,
            np.arange(values.shape[1]),
        ]

        locality_quality = np.divide(
            1.0,
            1.0 + laplacian,
            out=np.zeros_like(laplacian),
            where=np.isfinite(laplacian),
        )
        combined_score = locality_quality * eta_squared * relative_margin
        eligible = (
            np.isfinite(laplacian)
            & (dominant_mean > 0)
            & (eta_squared > 0)
            & (relative_margin > 0)
            & (dominant_prevalence >= minimum_replicates_present)
        )

        for condition_index, condition in enumerate(condition_labels):
            condition_score = np.where(
                eligible & (dominant_index == condition_index),
                combined_score,
                -np.inf,
            )
            local_indices = _largest_indices(
                condition_score,
                top_regions_per_condition,
            )
            if not len(local_indices):
                continue
            global_indices = chunk_start + local_indices
            pools[condition].append(
                pd.DataFrame(
                    {
                        "fold": fold_label,
                        "condition": condition,
                        "start": starts[global_indices],
                        "end": ends[global_indices],
                        "feature_index": global_indices,
                        "combined_umap_characteristic_score": combined_score[
                            local_indices
                        ],
                        "laplacian_score": laplacian[local_indices],
                        "umap_graph_locality_quality": locality_quality[
                            local_indices
                        ],
                        "condition_effect_eta_squared": eta_squared[local_indices],
                        "dominant_mean_log2_cpm": dominant_mean[local_indices],
                        "second_mean_log2_cpm": second_mean[local_indices],
                        "condition_signal_margin": signal_margin[local_indices],
                        "relative_condition_signal_margin": relative_margin[
                            local_indices
                        ],
                        "dominant_condition_replicates_present": (
                            dominant_prevalence[local_indices]
                        ),
                        "n_training_samples": len(sample_indices),
                    }
                )
            )


    selected_frames: list[pd.DataFrame] = []
    for condition in condition_labels:
        if not pools[condition]:
            raise ValueError(
                f"{fold_label}: keine geeigneten Regionen fuer {condition}."
            )
        pool = pd.concat(pools[condition], ignore_index=True)
        pool = pool.sort_values(
            [
                "combined_umap_characteristic_score",
                "laplacian_score",
                "condition_effect_eta_squared",
            ],
            ascending=[False, True, False],
        ).head(top_regions_per_condition).copy()
        pool["rank_within_condition_fold"] = np.arange(1, len(pool) + 1)
        selected_frames.append(pool)

    selected = pd.concat(selected_frames, ignore_index=True)
    condition_rank = {
        condition: rank for rank, condition in enumerate(condition_labels)
    }
    selected["_condition_rank"] = selected["condition"].map(condition_rank)
    return (
        selected.sort_values(
            ["_condition_rank", "rank_within_condition_fold"]
        )
        .drop(columns="_condition_rank")
        .reset_index(drop=True)
    )


def build_umap_region_stability(
    full_top: pd.DataFrame,
    fold_regions: pd.DataFrame,
    minimum_balanced_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = (
        fold_regions.groupby(["condition", "start", "end"], as_index=False)
        .agg(
            balanced_selection_count=("fold", "nunique"),
            mean_balanced_rank=("rank_within_condition_fold", "mean"),
            best_balanced_rank=("rank_within_condition_fold", "min"),
            mean_balanced_combined_score=(
                "combined_umap_characteristic_score",
                "mean",
            ),
            mean_balanced_laplacian_score=("laplacian_score", "mean"),
            mean_balanced_eta_squared=(
                "condition_effect_eta_squared",
                "mean",
            ),
        )
    )
    stability["balanced_selection_rate"] = (
        stability["balanced_selection_count"] / 3.0
    )
    stability["passes_balanced_stability"] = stability[
        "balanced_selection_count"
    ].ge(minimum_balanced_folds)

    stable_condition_counts = (
        stability.loc[stability["passes_balanced_stability"]]
        .groupby(["start", "end"])["condition"]
        .nunique()
        .rename("n_stable_condition_assignments")
        .reset_index()
    )
    stability = stability.merge(
        stable_condition_counts,
        on=["start", "end"],
        how="left",
    )
    stability["n_stable_condition_assignments"] = stability[
        "n_stable_condition_assignments"
    ].fillna(0).astype(int)

    full_top_for_merge = full_top.rename(
        columns={
            "fold": "selection_scope",
            "rank_within_condition_fold": "full_rank_within_condition",
        }
    )
    final = full_top_for_merge.merge(
        stability,
        on=["condition", "start", "end"],
        how="left",
        validate="one_to_one",
        suffixes=("_full", "_stability"),
    )
    final["balanced_selection_count"] = final[
        "balanced_selection_count"
    ].fillna(0).astype(int)
    final["balanced_selection_rate"] = final[
        "balanced_selection_rate"
    ].fillna(0.0)
    final["n_stable_condition_assignments"] = final[
        "n_stable_condition_assignments"
    ].fillna(0).astype(int)
    final["selected_as_umap_characteristic"] = (
        final["balanced_selection_count"].ge(minimum_balanced_folds)
        & final["n_stable_condition_assignments"].eq(1)
    )
    final = final.loc[final["selected_as_umap_characteristic"]].copy()
    final = final.sort_values(
        [
            "condition",
            "balanced_selection_count",
            "full_rank_within_condition",
            "combined_umap_characteristic_score",
        ],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    final["final_rank_within_condition"] = (
        final.groupby("condition").cumcount() + 1
    )
    final.insert(0, "method", "umap_graph_laplacian_stability")
    return stability, final


def nearest_neighbor_same_condition_rate(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> float:
    differences = embedding[:, None, :] - embedding[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    return float(np.mean(labels[nearest] == labels))


def rerun_umap_on_selected_regions(
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
    raw_embeddings: pd.DataFrame,
    selected_regions: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_index = pd.MultiIndex.from_frame(
        selected_regions[["start", "end"]].drop_duplicates(),
        names=["start", "end"],
    )
    missing = selected_index[~selected_index.isin(normalized.columns)]
    if len(missing):
        raise ValueError(f"{len(missing)} finale UMAP-Regionen fehlen in X.")
    X_selected = normalized.loc[:, selected_index].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    selected_embedding, _ = fit_primary_umap(X_selected)

    selected_table = pd.DataFrame(
        {
            "sample_id": normalized.index,
            "UMAP_1": selected_embedding[:, 0],
            "UMAP_2": selected_embedding[:, 1],
            "n_neighbors": min(PRIMARY_N_NEIGHBORS, len(normalized) - 1),
            "min_dist": PRIMARY_MIN_DIST,
            "metric": METRIC,
            "random_state": RANDOM_STATE,
            "run_label": "selected_umap_characteristic_regions",
        }
    ).merge(
        meta.reset_index(),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    raw_primary = raw_embeddings.loc[
        (raw_embeddings["n_neighbors"] == PRIMARY_N_NEIGHBORS)
        & (raw_embeddings["min_dist"] == PRIMARY_MIN_DIST)
    ].set_index("sample_id").loc[normalized.index]
    labels = meta.loc[normalized.index, "condition"].astype(str).to_numpy()
    raw_X = normalized.to_numpy(dtype=np.float32, copy=False)
    raw_embedding = raw_primary[["UMAP_1", "UMAP_2"]].to_numpy()

    metrics = pd.DataFrame(
        [
            {
                "panel": "raw_all_regions",
                "n_features": normalized.shape[1],
                "high_dimensional_cosine_silhouette": silhouette_score(
                    raw_X,
                    labels,
                    metric="cosine",
                ),
                "embedding_euclidean_silhouette": silhouette_score(
                    raw_embedding,
                    labels,
                    metric="euclidean",
                ),
                "embedding_1nn_same_condition_rate": (
                    nearest_neighbor_same_condition_rate(raw_embedding, labels)
                ),
            },
            {
                "panel": "umap_characteristic_regions",
                "n_features": X_selected.shape[1],
                "high_dimensional_cosine_silhouette": silhouette_score(
                    X_selected,
                    labels,
                    metric="cosine",
                ),
                "embedding_euclidean_silhouette": silhouette_score(
                    selected_embedding,
                    labels,
                    metric="euclidean",
                ),
                "embedding_1nn_same_condition_rate": (
                    nearest_neighbor_same_condition_rate(
                        selected_embedding,
                        labels,
                    )
                ),
            },
        ]
    )

    selected_table.to_csv(
        output_dir / "umap_selected_regions_primary_embeddings.csv",
        sep=";",
        index=False,
    )
    metrics.to_csv(
        output_dir / "umap_raw_vs_selected_separation_metrics.csv",
        sep=";",
        index=False,
    )
    plot_primary_embedding(
        selected_table,
        output_dir / "umap_selected_regions_primary.html",
    )
    return selected_table, metrics


def select_umap_characteristic_regions(
    raw_matrix: pd.DataFrame,
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
    primary_graph: object,
    raw_embeddings: pd.DataFrame,
    top_regions_per_condition: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_indices = np.arange(len(normalized))
    full_top = score_top_regions_for_umap_graph(
        raw_matrix,
        normalized,
        meta,
        full_indices,
        primary_graph,
        top_regions_per_condition,
        chunk_size,
        fold_label="full",
        minimum_replicates_present=MIN_REPLICATES_PRESENT_FULL,
    )

    fold_tables: list[pd.DataFrame] = []
    for fold, train_indices in enumerate(
        balanced_replicate_train_splits(meta),
        start=1,
    ):
        _, fold_graph = fit_primary_umap(
            normalized.iloc[train_indices].to_numpy(
                dtype=np.float32,
                copy=False,
            )
        )
        fold_table = score_top_regions_for_umap_graph(
            raw_matrix,
            normalized,
            meta,
            train_indices,
            fold_graph,
            top_regions_per_condition,
            chunk_size,
            fold_label=f"balanced_fold_{fold}",
            minimum_replicates_present=2,
        )
        fold_tables.append(fold_table)

    fold_regions = pd.concat(fold_tables, ignore_index=True)
    stability, final = build_umap_region_stability(
        full_top,
        fold_regions,
        MIN_BALANCED_SELECTION_FOLDS,
    )
    if final.empty:
        raise ValueError(
            "Keine stabile UMAP-charakteristische Region gefunden. "
            "TOP_REGIONS_PER_CONDITION oder Stabilitaet pruefen."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_top.to_csv(
        OUTPUT_DIR / "umap_graph_full_top_regions.csv",
        sep=";",
        index=False,
    )
    fold_regions.to_csv(
        OUTPUT_DIR / "umap_graph_balanced_fold_regions.csv",
        sep=";",
        index=False,
    )
    stability.to_csv(
        OUTPUT_DIR / "umap_graph_region_stability.csv",
        sep=";",
        index=False,
    )
    final.to_csv(
        OUTPUT_DIR / "umap_characteristic_regions.csv",
        sep=";",
        index=False,
    )

    _, metrics = rerun_umap_on_selected_regions(
        normalized,
        meta,
        raw_embeddings,
        final,
        OUTPUT_DIR,
    )
    return final, metrics


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    X64 = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X64, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine-Distanz ist fuer Nullvektoren nicht definiert.")

    unit_vectors = X64 / norms
    similarities = unit_vectors @ unit_vectors.T
    np.clip(similarities, -1.0, 1.0, out=similarities)
    distances = 1.0 - similarities
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def calculate_cosine_distance_results(
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_order = [
        label for label, _ in CONDITIONS if label in set(meta["condition"])
    ]
    unknown_conditions = sorted(set(meta["condition"]) - set(condition_order))
    condition_order.extend(unknown_conditions)

    sample_array = _cosine_distance_matrix(
        normalized.to_numpy(dtype=np.float32, copy=False)
    )
    sample_distances = pd.DataFrame(
        sample_array,
        index=normalized.index,
        columns=normalized.index,
    )
    sample_distances.index.name = "sample_id"
    sample_distances.columns.name = "sample_id"

    condition_distances = pd.DataFrame(
        np.nan,
        index=condition_order,
        columns=condition_order,
        dtype=float,
    )
    condition_distances.index.name = "condition"
    condition_distances.columns.name = "condition"

    summary_rows: list[dict[str, object]] = []
    for i, condition_a in enumerate(condition_order):
        ids_a = meta.index[meta["condition"] == condition_a].tolist()
        for j in range(i, len(condition_order)):
            condition_b = condition_order[j]
            ids_b = meta.index[meta["condition"] == condition_b].tolist()

            if condition_a == condition_b:
                block = sample_distances.loc[ids_a, ids_a].to_numpy()
                values = block[np.triu_indices(len(ids_a), k=1)]
                pair_type = "within_condition"
            else:
                values = sample_distances.loc[ids_a, ids_b].to_numpy().ravel()
                pair_type = "between_conditions"

            if values.size == 0:
                raise ValueError(
                    f"Zu wenige Replikate fuer {condition_a} vs {condition_b}."
                )

            mean_distance = float(np.mean(values))
            condition_distances.loc[condition_a, condition_b] = mean_distance
            condition_distances.loc[condition_b, condition_a] = mean_distance
            summary_rows.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "pair_type": pair_type,
                    "n_sample_pairs": int(values.size),
                    "mean_cosine_distance": mean_distance,
                    "median_cosine_distance": float(np.median(values)),
                    "std_cosine_distance": float(np.std(values, ddof=1))
                    if values.size > 1
                    else np.nan,
                    "min_cosine_distance": float(np.min(values)),
                    "max_cosine_distance": float(np.max(values)),
                }
            )

    pair_summary = pd.DataFrame(summary_rows)
    pair_summary["similarity_rank_between_conditions"] = pd.Series(
        pd.NA,
        index=pair_summary.index,
        dtype="Int64",
    )
    between_mask = pair_summary["pair_type"] == "between_conditions"
    pair_summary.loc[
        between_mask, "similarity_rank_between_conditions"
    ] = (
        pair_summary.loc[between_mask, "mean_cosine_distance"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    pair_summary = pair_summary.sort_values(
        ["pair_type", "mean_cosine_distance"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return sample_distances, condition_distances, pair_summary


def save_plotly_html(fig: go.Figure, html_path: Path) -> None:
    if html_path.suffix.lower() != ".html":
        raise ValueError(f"HTML-Ausgabepfad erwartet, erhalten: {html_path}")

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")


def save_matplotlib_vectors(fig: plt.Figure, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = html_path.with_suffix(".svg")
    pdf_path = html_path.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _condition_order(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    order = [condition for condition, _ in CONDITIONS if condition in present]
    order.extend(sorted(present - set(order)))
    return order


def _legend_handles(conditions: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=CONDITION_COLORS.get(condition, "#999999"),
            markeredgecolor="white",
            label=condition,
        )
        for condition in conditions
    ]


def _plot_parameter_grid_vectors(
    embeddings: pd.DataFrame,
    output_path: Path,
) -> None:
    run_labels = embeddings["run_label"].drop_duplicates().tolist()
    conditions = _condition_order(embeddings["condition"])
    n_cols = 3
    n_rows = int(np.ceil(len(run_labels) / n_cols))
    mpl_fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4.2 * n_rows),
        squeeze=False,
    )

    for index, run_label in enumerate(run_labels):
        ax = axes[index // n_cols, index % n_cols]
        subset = embeddings[embeddings["run_label"] == run_label]
        for condition in conditions:
            condition_data = subset[subset["condition"] == condition]
            ax.scatter(
                condition_data["UMAP_1"],
                condition_data["UMAP_2"],
                s=48,
                color=CONDITION_COLORS.get(condition, "#999999"),
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
            )
        n_neighbors = int(subset["n_neighbors"].iloc[0])
        min_dist = float(subset["min_dist"].iloc[0])
        metric = str(subset["metric"].iloc[0])
        ax.set_title(
            f"n_neighbors={n_neighbors}, min_dist={min_dist:g}, metric={metric}",
            fontsize=9,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for index in range(len(run_labels), n_rows * n_cols):
        axes[index // n_cols, index % n_cols].axis("off")

    mpl_fig.suptitle(
        "UMAP-Parameterraster auf allen Rohdatenregionen\n"
        "Input: log2(CPM + 1); Farben werden nur nachtraeglich verwendet",
        fontsize=14,
    )
    mpl_fig.legend(
        handles=_legend_handles(conditions),
        title="Condition",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
    )
    mpl_fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.94))
    save_matplotlib_vectors(mpl_fig, output_path)


def _plot_primary_embedding_vectors(
    primary: pd.DataFrame,
    output_path: Path,
) -> None:
    conditions = _condition_order(primary["condition"])
    mpl_fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for condition in conditions:
        subset = primary[primary["condition"] == condition]
        ax.scatter(
            subset["UMAP_1"],
            subset["UMAP_2"],
            s=70,
            color=CONDITION_COLORS.get(condition, "#999999"),
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
        )
    ax.set_title(
        "Explorative UMAP-Einbettung der Samples\n"
        f"n_neighbors={PRIMARY_N_NEIGHBORS}, "
        f"min_dist={PRIMARY_MIN_DIST:g}, metric={METRIC}; "
        "Input: log2(CPM + 1)",
        fontsize=13,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=_legend_handles(conditions),
        title="Condition",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    mpl_fig.tight_layout()
    save_matplotlib_vectors(mpl_fig, output_path)


def _plot_sample_distance_vectors(
    ordered: pd.DataFrame,
    labels: list[str],
    output_path: Path,
) -> None:
    mpl_fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(
        ordered.to_numpy(),
        cmap="viridis",
        vmin=0.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title(
        "Cosine-Distanzen zwischen allen Samples\n"
        "Alle log2(CPM + 1)-Rohdatenregionen; kleiner = aehnlicher",
        fontsize=13,
    )
    colorbar = mpl_fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine-Distanz")
    mpl_fig.tight_layout()
    save_matplotlib_vectors(mpl_fig, output_path)


def _plot_condition_distance_vectors(
    condition_distances: pd.DataFrame,
    output_path: Path,
) -> None:
    values = condition_distances.to_numpy(dtype=float)
    vmax = float(np.nanmax(values))
    mpl_fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        values,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    labels = condition_distances.columns.tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(condition_distances.index)))
    ax.set_yticklabels(condition_distances.index.tolist())
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "black" if value > vmax * 0.55 else "white"
            ax.text(
                col,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    ax.set_title(
        "Mittlere Cosine-Distanzen zwischen Conditions\n"
        "Off-diagonal: 3 x 3 Sample-Paare; diagonal: Replikatabstaende",
        fontsize=13,
    )
    colorbar = mpl_fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("mittlere Cosine-Distanz")
    mpl_fig.tight_layout()
    save_matplotlib_vectors(mpl_fig, output_path)


def _add_embedding_traces(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    for condition in sorted(plot_df["condition"].unique()):
        subset = plot_df[plot_df["condition"] == condition]
        fig.add_trace(
            go.Scatter(
                x=subset["UMAP_1"],
                y=subset["UMAP_2"],
                mode="markers",
                name=condition,
                legendgroup=condition,
                showlegend=show_legend,
                marker={
                    "size": 10,
                    "color": CONDITION_COLORS.get(condition, "#999999"),
                    "line": {"color": "white", "width": 0.7},
                    "opacity": 0.9,
                },
                customdata=np.column_stack(
                    [subset["sample_name"], subset["condition"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Condition: %{customdata[1]}<br>"
                    "UMAP 1: %{x:.3f}<br>"
                    "UMAP 2: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def plot_parameter_grid(embeddings: pd.DataFrame, output_path: Path) -> None:
    run_labels = embeddings["run_label"].drop_duplicates().tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(run_labels) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=run_labels,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for index, run_label in enumerate(run_labels):
        row = index // n_cols + 1
        col = index % n_cols + 1
        subset = embeddings[embeddings["run_label"] == run_label]
        _add_embedding_traces(
            fig,
            subset,
            row=row,
            col=col,
            show_legend=(index == 0),
        )

    fig.update_xaxes(showticklabels=False, title_text="UMAP 1")
    fig.update_yaxes(showticklabels=False, title_text="UMAP 2")
    fig.update_layout(
        title=(
            "UMAP-Parameterraster auf allen Rohdatenregionen"
            "<br><sup>Input: log2(CPM + 1); Farben werden nur nachtraeglich "
            "zur Darstellung verwendet</sup>"
        ),
        template="simple_white",
        height=360 * n_rows,
        width=1100,
        legend={
            "title": "Condition",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hovermode="closest",
    )
    save_plotly_html(fig, output_path)
    _plot_parameter_grid_vectors(embeddings, output_path)


def plot_primary_embedding(embeddings: pd.DataFrame, output_path: Path) -> None:
    primary = embeddings[
        (embeddings["n_neighbors"] == PRIMARY_N_NEIGHBORS)
        & (embeddings["min_dist"] == PRIMARY_MIN_DIST)
    ]
    if primary.empty:
        raise RuntimeError("PRIMARY-Konfiguration ist nicht im Parameter-Raster.")

    fig = make_subplots(rows=1, cols=1)
    _add_embedding_traces(fig, primary, row=1, col=1, show_legend=True)
    fig.update_layout(
        title=(
            "Explorative UMAP-Einbettung der Samples"
            f"<br><sup>n_neighbors={PRIMARY_N_NEIGHBORS}, "
            f"min_dist={PRIMARY_MIN_DIST:g}, metric={METRIC}; "
            "Input: log2(CPM + 1)</sup>"
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="simple_white",
        height=720,
        width=920,
        legend={
            "title": "Condition",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hovermode="closest",
    )
    save_plotly_html(fig, output_path)
    _plot_primary_embedding_vectors(primary, output_path)


def plot_sample_distance_heatmap(
    sample_distances: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    condition_rank = {condition: i for i, (condition, _) in enumerate(CONDITIONS)}
    ordered_meta = (
        meta.assign(
            _condition_rank=meta["condition"].map(condition_rank).fillna(999)
        )
        .sort_values(["_condition_rank", "condition", "sample_name"])
    )
    ordered_ids = ordered_meta.index.tolist()
    ordered = sample_distances.loc[ordered_ids, ordered_ids]
    labels = [
        f"{meta.loc[sample_id, 'condition']} | "
        f"{meta.loc[sample_id, 'sample_name']}"
        for sample_id in ordered_ids
    ]

    fig = go.Figure(
        go.Heatmap(
            z=ordered.to_numpy(),
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0.0,
            colorbar={"title": "Cosine-<br>Distanz"},
            hovertemplate=(
                "Sample A: %{y}<br>Sample B: %{x}<br>"
                "Cosine-Distanz: %{z:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            "Cosine-Distanzen zwischen allen Samples"
            "<br><sup>Direkt auf allen log2(CPM + 1)-Rohdatenregionen; "
            "kleiner = aehnlicher</sup>"
        ),
        template="simple_white",
        width=1050,
        height=950,
        xaxis={"tickangle": -55},
        yaxis={"autorange": "reversed"},
    )
    save_plotly_html(fig, output_path)
    _plot_sample_distance_vectors(ordered, labels, output_path)


def plot_condition_distance_heatmap(
    condition_distances: pd.DataFrame,
    output_path: Path,
) -> None:
    rounded = condition_distances.round(4)
    fig = go.Figure(
        go.Heatmap(
            z=condition_distances.to_numpy(),
            x=condition_distances.columns.tolist(),
            y=condition_distances.index.tolist(),
            text=rounded.to_numpy(),
            texttemplate="%{text:.4f}",
            colorscale="Viridis",
            zmin=0.0,
            colorbar={"title": "mittlere<br>Cosine-Distanz"},
            hovertemplate=(
                "Condition A: %{y}<br>Condition B: %{x}<br>"
                "mittlere Cosine-Distanz: %{z:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            "Mittlere Cosine-Distanzen zwischen Conditions"
            "<br><sup>Off-diagonal: alle 3 x 3 Sample-Paare; diagonal: "
            "Abstand der Replikate innerhalb der Condition</sup>"
        ),
        template="simple_white",
        width=850,
        height=780,
        yaxis={"autorange": "reversed"},
    )
    save_plotly_html(fig, output_path)
    _plot_condition_distance_vectors(condition_distances, output_path)


def export_figures_from_saved_tables() -> None:
    embeddings_path = OUTPUT_DIR / "umap_raw_all_embeddings.csv"
    sample_distances_path = OUTPUT_DIR / "cosine_distance_samples.csv"
    condition_distances_path = OUTPUT_DIR / "cosine_distance_conditions.csv"
    required_paths = (
        embeddings_path,
        sample_distances_path,
        condition_distances_path,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Export-only ist nicht moeglich; folgende Ergebnistabellen "
            f"fehlen:\n{missing_text}"
        )

    embeddings = pd.read_csv(embeddings_path, sep=";")
    sample_distances = pd.read_csv(
        sample_distances_path,
        sep=";",
        index_col=0,
    )
    condition_distances = pd.read_csv(
        condition_distances_path,
        sep=";",
        index_col=0,
    )

    meta_columns = ["sample_id", "sample_name", "condition", "source_file"]
    missing_meta = [
        column for column in meta_columns if column not in embeddings.columns
    ]
    if missing_meta:
        raise ValueError(
            "Metadaten fehlen in umap_raw_all_embeddings.csv: "
            f"{missing_meta}"
        )
    meta = (
        embeddings[meta_columns]
        .drop_duplicates(subset="sample_id")
        .set_index("sample_id")
    )

    plot_parameter_grid(
        embeddings,
        OUTPUT_DIR / "umap_raw_parameter_grid.html",
    )
    plot_primary_embedding(
        embeddings,
        OUTPUT_DIR / "umap_raw_primary.html",
    )
    plot_sample_distance_heatmap(
        sample_distances,
        meta,
        OUTPUT_DIR / "cosine_distance_samples_heatmap.html",
    )
    plot_condition_distance_heatmap(
        condition_distances,
        OUTPUT_DIR / "cosine_distance_conditions_heatmap.html",
    )

    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")


def main(
    top_regions_per_condition: int = TOP_REGIONS_PER_CONDITION,
    chunk_size: int = FEATURE_SCORE_CHUNK_SIZE,
) -> None:

    raw_matrix, meta = build_raw_sample_feature_matrix(CONDITIONS)

    normalized = normalize_log_cpm(raw_matrix)
    X = normalized.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Die normalisierte Matrix enthaelt NaN oder Inf.")

    sample_distances, condition_distances, pair_summary = (
        calculate_cosine_distance_results(normalized, meta)
    )

    closest_pairs = pair_summary[
        pair_summary["pair_type"] == "between_conditions"
    ].sort_values("mean_cosine_distance")

    embeddings, primary_graph = run_umap_grid(X, normalized.index)
    embeddings = embeddings.merge(
        meta.reset_index(),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = OUTPUT_DIR / "umap_raw_all_embeddings.csv"
    summary_path = OUTPUT_DIR / "umap_raw_input_summary.csv"
    sample_distances_path = OUTPUT_DIR / "cosine_distance_samples.csv"
    condition_distances_path = OUTPUT_DIR / "cosine_distance_conditions.csv"
    pair_summary_path = OUTPUT_DIR / "cosine_distance_condition_pairs_summary.csv"
    embeddings.to_csv(embeddings_path, sep=";", index=False)
    sample_distances.to_csv(sample_distances_path, sep=";")
    condition_distances.to_csv(condition_distances_path, sep=";")
    pair_summary.to_csv(pair_summary_path, sep=";", index=False)

    input_summary = meta.copy()
    input_summary["library_size_raw_reads"] = raw_matrix.sum(axis=1)
    input_summary["n_observed_regions"] = (raw_matrix > 0).sum(axis=1)
    input_summary.to_csv(summary_path, sep=";")

    grid_path = OUTPUT_DIR / "umap_raw_parameter_grid.html"
    primary_path = OUTPUT_DIR / "umap_raw_primary.html"
    plot_parameter_grid(embeddings, grid_path)
    plot_primary_embedding(embeddings, primary_path)

    sample_heatmap_path = OUTPUT_DIR / "cosine_distance_samples_heatmap.html"
    condition_heatmap_path = OUTPUT_DIR / "cosine_distance_conditions_heatmap.html"
    plot_sample_distance_heatmap(sample_distances, meta, sample_heatmap_path)
    plot_condition_distance_heatmap(
        condition_distances,
        condition_heatmap_path,
    )

    selected_regions, separation_metrics = select_umap_characteristic_regions(
        raw_matrix,
        normalized,
        meta,
        primary_graph,
        embeddings,
        top_regions_per_condition,
        chunk_size,
    )

    selected_summary = (
        selected_regions.groupby("condition", as_index=False)
        .agg(
            n_regions=("start", "size"),
            median_full_rank=("full_rank_within_condition", "median"),
            median_balanced_selection_rate=(
                "balanced_selection_rate",
                "median",
            ),
        )
    )
    selected_summary.to_csv(
        OUTPUT_DIR / "umap_characteristic_regions_summary.csv",
        sep=";",
        index=False,
    )

    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    pio.renderers.default = "browser"
    parser = argparse.ArgumentParser(
        description="UMAP und Cosine-Distanzen ueber alle Rohdatenregionen"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Erzeugt Figures aus vorhandenen Ergebnis-CSVs, ohne Rohdaten "
            "oder UMAP erneut zu berechnen."
        ),
    )
    parser.add_argument(
        "--top-regions-per-condition",
        type=int,
        default=TOP_REGIONS_PER_CONDITION,
        help=(
            "Vorab festgelegtes Top-K je Condition fuer Vollmodell und "
            f"Stabilitaets-Folds (Standard: {TOP_REGIONS_PER_CONDITION})."
        ),
    )
    parser.add_argument(
        "--feature-score-chunk-size",
        type=int,
        default=FEATURE_SCORE_CHUNK_SIZE,
        help=(
            "Anzahl Regionen pro speicherschonendem Scoring-Block "
            f"(Standard: {FEATURE_SCORE_CHUNK_SIZE})."
        ),
    )
    args = parser.parse_args()
    if args.top_regions_per_condition < 1:
        parser.error("--top-regions-per-condition muss mindestens 1 sein.")
    if args.feature_score_chunk_size < 1:
        parser.error("--feature-score-chunk-size muss mindestens 1 sein.")
    if args.export_only:
        export_figures_from_saved_tables()
    else:
        main(
            top_regions_per_condition=args.top_regions_per_condition,
            chunk_size=args.feature_score_chunk_size,
        )