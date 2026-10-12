from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


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
INTENSITY_MATRICES_DATA_DIR = DATA_DIR / "intensity_matrices"
from method1_set_based_and_rule_based_region_selection.normalization import (
    normalize_cpm,
)


SEP = ";"
REQUIRED_COLUMNS = ("start", "end", "n_reads")
RANDOM_STATE = 42
EXPECTED_SAMPLES_PER_CONDITION = 3
MIN_RAW_READS = 2

CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]

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

OUTPUT_DIR = (
    DATA_DIR
    / "method_comparison"
    / "three_method_nested_balanced_validation"
)

LOGREG_C = 0.01
LOGREG_MAX_ITER = 5000
LOGREG_SOLVER = "liblinear"
TOP_K_PER_CONDITION = 250

UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
FEATURE_SCORE_CHUNK_SIZE = 50_000

PANEL_ORDER = [
    "raw_all_regions",
    "set_based",
    "umap",
    "logreg",
    "union_all_methods",
    "at_least_two_methods",
    "intersection_all_three_coordinates",
    "intersection_all_three_same_condition",
]


def load_intensity_matrix(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Intensity-Matrix nicht gefunden: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            sep=SEP,
            usecols=list(REQUIRED_COLUMNS),
            low_memory=False,
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"{csv_path}: Intensity-Matrix konnte nicht gelesen werden. "
            f"Erwartete Spalten: {REQUIRED_COLUMNS}."
        ) from exc

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{csv_path}: Fehlende Werte in benoetigten Spalten.")
    if (df["n_reads"] < 0).any():
        raise ValueError(f"{csv_path}: Negative n_reads-Werte gefunden.")
    if not np.all(np.equal(df["start"], np.floor(df["start"]))) or not np.all(
        np.equal(df["end"], np.floor(df["end"]))
    ):
        raise ValueError(f"{csv_path}: start und end muessen ganzzahlig sein.")

    return df.astype(
        {"start": "int64", "end": "int64", "n_reads": "float64"}
    )

def build_raw_and_normalized_matrices(
    conditions: dict[str, list[Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_by_sample: dict[str, pd.Series] = {}
    cpm_by_sample: dict[str, pd.Series] = {}
    meta_rows: list[dict] = []

    for condition in CONDITION_ORDER:
        paths = list(conditions[condition])
        if len(paths) != EXPECTED_SAMPLES_PER_CONDITION:
            raise ValueError(
                f"{condition}: {len(paths)} Samples; erwartet wurden 3."
            )

        condition_raw: dict[str, pd.Series] = {}
        sample_id_by_name: dict[str, str] = {}
        for replicate_position, csv_path in enumerate(paths):
            sample_name = csv_path.parent.parent.name
            sample_id = f"{condition}__{sample_name}"
            df = load_intensity_matrix(csv_path)
            grouped = df.groupby(["start", "end"])["n_reads"].sum()
            condition_raw[sample_name] = grouped
            sample_id_by_name[sample_name] = sample_id
            raw_by_sample[sample_id] = grouped
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "condition": condition,
                    "replicate_position": replicate_position,
                    "source_file": str(csv_path),
                }
            )

        raw_condition = pd.DataFrame(condition_raw).fillna(0.0)
        cpm_condition = normalize_cpm(raw_condition)
        for sample_name in cpm_condition.columns:
            cpm_by_sample[sample_id_by_name[sample_name]] = cpm_condition[
                sample_name
            ]

    raw = pd.DataFrame(raw_by_sample, dtype=np.float32).T.fillna(0.0)
    cpm = pd.DataFrame(cpm_by_sample, dtype=np.float32).T.fillna(0.0)
    cpm = cpm.reindex(index=raw.index, columns=raw.columns, fill_value=0.0)
    normalized = np.log2(cpm + np.float32(1.0)).astype(np.float32)
    raw.index.name = "sample_id"
    normalized.index.name = "sample_id"
    meta = pd.DataFrame(meta_rows).set_index("sample_id").loc[raw.index]

    if raw.shape != normalized.shape:
        raise AssertionError("Roh- und normalisierte Matrix unterscheiden sich.")
    if raw.shape[0] != 21 or raw.shape[1] == 0:
        raise AssertionError(f"Unerwartete Matrixform: {raw.shape}")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise TypeError("Erwartete (start, end)-MultiIndex-Featureachse.")
    if not np.isfinite(normalized.to_numpy(copy=False)).all():
        raise ValueError("Normalisierte Matrix enthaelt NaN oder Inf.")
    return raw, normalized, meta


def balanced_splits(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=object)
    per_condition = {
        condition: np.flatnonzero(labels == condition)
        for condition in CONDITION_ORDER
    }
    for condition, indices in per_condition.items():
        if len(indices) != 3:
            raise ValueError(
                f"{condition}: fuer balanced CV exakt 3 Samples erforderlich."
            )

    all_indices = np.arange(len(labels))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for position in range(3):
        test_idx = np.asarray(
            [per_condition[c][position] for c in CONDITION_ORDER],
            dtype=int,
        )
        train_idx = all_indices[~np.isin(all_indices, test_idx)]
        if len(train_idx) != 14 or len(test_idx) != 7:
            raise AssertionError("Balanced Split besitzt unerwartete Groesse.")
        splits.append((train_idx, test_idx))
    return splits


def make_model() -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(
            penalty="l2",
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            solver=LOGREG_SOLVER,
        )
    )


def largest_positive_indices(values: np.ndarray, n_keep: int) -> np.ndarray:
    positive = np.flatnonzero(np.isfinite(values) & (values > 0))
    if not len(positive):
        return np.asarray([], dtype=int)
    n_keep = min(n_keep, len(positive))
    if n_keep == len(positive):
        candidates = positive
    else:
        relative = np.argpartition(values[positive], -n_keep)[-n_keep:]
        candidates = positive[relative]
    return candidates[np.argsort(values[candidates])[::-1]]


def select_set_based(
    raw_train: np.ndarray,
    labels_train: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> pd.DataFrame:
    memberships = []
    for condition in CONDITION_ORDER:
        positions = np.flatnonzero(labels_train == condition)
        if len(positions) != 2:
            raise ValueError(f"Set-based: {condition} hat nicht 2 Trainingssamples.")
        memberships.append(
            np.all(raw_train[positions] >= MIN_RAW_READS, axis=0)
        )
    membership = np.vstack(memberships)
    n_condition_memberships = membership.sum(axis=0)

    rows: list[pd.DataFrame] = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        indices = np.flatnonzero(
            membership[condition_index] & (n_condition_memberships == 1)
        )
        if len(indices):
            rows.append(
                pd.DataFrame(
                    {
                        "method": "set_based",
                        "condition": condition,
                        "feature_index": indices,
                        "start": starts[indices],
                        "end": ends[indices],
                        "training_replicates_present": 2,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=["method", "condition", "feature_index", "start", "end"]
        )
    return pd.concat(rows, ignore_index=True)


def fit_training_umap_graph(normalized_train: np.ndarray):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn fehlt. Installation: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, len(normalized_train) - 1),
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        init="spectral",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    reducer.fit(normalized_train)
    return reducer.graph_.tocsr().astype(np.float64).copy()


def select_umap_regions(
    raw_train: np.ndarray,
    normalized_train: np.ndarray,
    labels_train: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_size: int,
    top_k_per_condition: int,
) -> pd.DataFrame:
    graph = fit_training_umap_graph(normalized_train)
    weights = np.asarray(graph.toarray(), dtype=np.float64)
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    total_degree = float(degree.sum())
    if total_degree <= 0 or np.any(degree <= 0):
        raise ValueError("Ungueltiger oder unverbundener Training-UMAP-Graph.")

    condition_positions = []
    for condition in CONDITION_ORDER:
        positions = np.flatnonzero(labels_train == condition)
        if len(positions) != 2:
            raise ValueError(f"UMAP: {condition} hat nicht 2 Trainingssamples.")
        condition_positions.append(positions)

    pools: dict[str, list[pd.DataFrame]] = {
        condition: [] for condition in CONDITION_ORDER
    }
    epsilon = np.finfo(np.float64).eps
    n_features = normalized_train.shape[1]

    for chunk_start in range(0, n_features, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_features)
        values = normalized_train[:, chunk_start:chunk_end].astype(
            np.float64,
            copy=False,
        )
        counts = raw_train[:, chunk_start:chunk_end]

        weighted_mean = (degree @ values) / total_degree
        centered = values - weighted_mean[None, :]
        denominator = np.sum(degree[:, None] * centered * centered, axis=0)
        numerator = denominator - np.sum(centered * (weights @ centered), axis=0)
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
            [values[pos].mean(axis=0) for pos in condition_positions]
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
                (counts[pos] >= MIN_RAW_READS).sum(axis=0)
                for pos in condition_positions
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
            & (dominant_prevalence >= 2)
        )

        for condition_index, condition in enumerate(CONDITION_ORDER):
            scores = np.where(
                eligible & (dominant_index == condition_index),
                combined_score,
                -np.inf,
            )
            local = largest_positive_indices(scores, top_k_per_condition)
            if not len(local):
                continue
            feature_indices = chunk_start + local
            pools[condition].append(
                pd.DataFrame(
                    {
                        "method": "umap",
                        "condition": condition,
                        "feature_index": feature_indices,
                        "start": starts[feature_indices],
                        "end": ends[feature_indices],
                        "combined_score": combined_score[local],
                        "laplacian_score": laplacian[local],
                        "condition_effect_eta_squared": eta_squared[local],
                        "relative_condition_signal_margin": relative_margin[local],
                        "training_replicates_present": dominant_prevalence[local],
                    }
                )
            )

    selected: list[pd.DataFrame] = []
    for condition in CONDITION_ORDER:
        if not pools[condition]:
            continue
        pool = pd.concat(pools[condition], ignore_index=True)
        pool = pool.sort_values(
            ["combined_score", "laplacian_score"],
            ascending=[False, True],
        ).head(top_k_per_condition).copy()
        pool["rank_within_condition"] = np.arange(1, len(pool) + 1)
        selected.append(pool)
    if not selected:
        return pd.DataFrame(
            columns=["method", "condition", "feature_index", "start", "end"]
        )
    return pd.concat(selected, ignore_index=True)


def select_logreg_regions(
    fitted_raw_model: OneVsRestClassifier,
    starts: np.ndarray,
    ends: np.ndarray,
    top_k_per_condition: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if len(fitted_raw_model.estimators_) != len(fitted_raw_model.classes_):
        raise AssertionError("Unerwartete Zahl an OvR-Teilmodellen.")

    for class_row, condition_value in enumerate(fitted_raw_model.classes_):
        condition = str(condition_value)
        coefficient = fitted_raw_model.estimators_[class_row].coef_[0]
        indices = largest_positive_indices(coefficient, top_k_per_condition)
        if not len(indices):
            continue
        rows.append(
            pd.DataFrame(
                {
                    "method": "logreg",
                    "condition": condition,
                    "feature_index": indices,
                    "start": starts[indices],
                    "end": ends[indices],
                    "standardized_coefficient": coefficient[indices],
                    "rank_within_condition": np.arange(1, len(indices) + 1),
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["method", "condition", "feature_index", "start", "end"]
        )
    return pd.concat(rows, ignore_index=True)


def build_fold_panels(
    set_based: pd.DataFrame,
    umap_regions: pd.DataFrame,
    logreg_regions: pd.DataFrame,
    n_features: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    method_frames = {
        "set_based": set_based,
        "umap": umap_regions,
        "logreg": logreg_regions,
    }
    method_indices = {
        method: np.unique(frame["feature_index"].to_numpy(dtype=int))
        if len(frame)
        else np.asarray([], dtype=int)
        for method, frame in method_frames.items()
    }

    all_selected = np.unique(
        np.concatenate(
            [indices for indices in method_indices.values() if len(indices)]
        )
    ) if any(len(v) for v in method_indices.values()) else np.asarray([], dtype=int)

    rows: list[dict] = []
    condition_maps: dict[str, dict[int, set[str]]] = {}
    for method, frame in method_frames.items():
        mapping: dict[int, set[str]] = {}
        for feature_index, group in frame.groupby("feature_index"):
            mapping[int(feature_index)] = set(group["condition"].astype(str))
        condition_maps[method] = mapping

    for feature_index in all_selected:
        present = {
            method: int(feature_index) in condition_maps[method]
            for method in method_frames
        }
        conditions = {
            method: condition_maps[method].get(int(feature_index), set())
            for method in method_frames
        }
        present_condition_sets = [
            values for method, values in conditions.items() if present[method]
        ]
        same_condition = (
            len(present_condition_sets) > 0
            and all(len(values) == 1 for values in present_condition_sets)
            and len({next(iter(values)) for values in present_condition_sets}) == 1
        )
        rows.append(
            {
                "feature_index": int(feature_index),
                "in_set_based": present["set_based"],
                "in_umap": present["umap"],
                "in_logreg": present["logreg"],
                "n_methods": sum(present.values()),
                "set_based_conditions": "|".join(sorted(conditions["set_based"])),
                "umap_conditions": "|".join(sorted(conditions["umap"])),
                "logreg_conditions": "|".join(sorted(conditions["logreg"])),
                "condition_agreement": same_condition,
            }
        )
    membership = pd.DataFrame(rows)

    if membership.empty:
        at_least_two = np.asarray([], dtype=int)
        all_three = np.asarray([], dtype=int)
        all_three_same = np.asarray([], dtype=int)
    else:
        at_least_two = membership.loc[
            membership["n_methods"] >= 2, "feature_index"
        ].to_numpy(dtype=int)
        all_three = membership.loc[
            membership["n_methods"] == 3, "feature_index"
        ].to_numpy(dtype=int)
        all_three_same = membership.loc[
            (membership["n_methods"] == 3)
            & membership["condition_agreement"],
            "feature_index",
        ].to_numpy(dtype=int)

    panels = {
        "raw_all_regions": np.arange(n_features, dtype=int),
        "set_based": method_indices["set_based"],
        "umap": method_indices["umap"],
        "logreg": method_indices["logreg"],
        "union_all_methods": all_selected,
        "at_least_two_methods": np.unique(at_least_two),
        "intersection_all_three_coordinates": np.unique(all_three),
        "intersection_all_three_same_condition": np.unique(all_three_same),
    }
    return panels, membership


def evaluate_one_fold(
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    sample_names: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_size: int,
    top_k: int,
    verbose: bool,
) -> dict:
    raw_train = raw_values[train_idx]
    normalized_train = normalized_values[train_idx]
    labels_train = labels[train_idx]


    raw_started = time.perf_counter()
    raw_scaler = StandardScaler()
    raw_train_scaled = raw_scaler.fit_transform(normalized_train)
    raw_model = make_model()
    raw_model.fit(raw_train_scaled, labels_train)
    raw_selection_fit_seconds = time.perf_counter() - raw_started

    set_started = time.perf_counter()
    set_regions = select_set_based(
        raw_train,
        labels_train,
        starts,
        ends,
    )
    set_seconds = time.perf_counter() - set_started

    umap_started = time.perf_counter()
    umap_regions = select_umap_regions(
        raw_train,
        normalized_train,
        labels_train,
        starts,
        ends,
        chunk_size,
        top_k,
    )
    umap_seconds = time.perf_counter() - umap_started

    logreg_started = time.perf_counter()
    logreg_regions = select_logreg_regions(
        raw_model,
        starts,
        ends,
        top_k,
    )
    logreg_seconds = time.perf_counter() - logreg_started

    panels, membership = build_fold_panels(
        set_regions,
        umap_regions,
        logreg_regions,
        normalized_values.shape[1],
    )


    normalized_test = normalized_values[test_idx]
    labels_test = labels[test_idx]
    predictions: list[dict] = []
    panel_rows: list[dict] = []

    for panel in PANEL_ORDER:
        feature_indices = panels[panel]
        if not len(feature_indices):
            panel_rows.append(
                {
                    "fold": fold,
                    "panel": panel,
                    "n_features": 0,
                    "correct": pd.NA,
                    "n_test": len(test_idx),
                    "accuracy": np.nan,
                    "classification_seconds": 0.0,
                    "status": "empty_panel",
                }
            )
            continue

        started = time.perf_counter()
        if panel == "raw_all_regions":
            test_scaled = raw_scaler.transform(normalized_test)
            predicted = raw_model.predict(test_scaled)
        else:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(
                normalized_train[:, feature_indices]
            )
            test_scaled = scaler.transform(
                normalized_test[:, feature_indices]
            )
            model = make_model()
            model.fit(train_scaled, labels_train)
            predicted = model.predict(test_scaled)
        elapsed = time.perf_counter() - started

        correct = 0
        for position, sample_index in enumerate(test_idx):
            truth = str(labels_test[position])
            prediction = str(predicted[position])
            is_correct = truth == prediction
            correct += int(is_correct)
            predictions.append(
                {
                    "fold": fold,
                    "panel": panel,
                    "sample": str(sample_names[sample_index]),
                    "true": truth,
                    "predicted": prediction,
                    "correct": is_correct,
                    "n_features": len(feature_indices),
                }
            )
        panel_rows.append(
            {
                "fold": fold,
                "panel": panel,
                "n_features": len(feature_indices),
                "correct": correct,
                "n_test": len(test_idx),
                "accuracy": correct / len(test_idx),
                "classification_seconds": elapsed,
                "status": "ok",
            }
        )
        if panel != "raw_all_regions":
            del scaler, train_scaled, model
        del test_scaled, predicted

    timing = {
        "fold": fold,
        "raw_scaler_and_model_fit_seconds": raw_selection_fit_seconds,
        "set_selection_seconds": set_seconds,
        "umap_selection_seconds": umap_seconds,
        "logreg_coefficient_extraction_seconds": logreg_seconds,
        "total_selection_seconds": (
            raw_selection_fit_seconds
            + set_seconds
            + umap_seconds
            + logreg_seconds
        ),
    }
    for frame in (set_regions, umap_regions, logreg_regions, membership):
        frame.insert(0, "fold", fold)
    for panel, indices in panels.items():
        panels[panel] = np.unique(indices)

    del raw_scaler, raw_model, raw_train_scaled
    gc.collect()
    return {
        "predictions": pd.DataFrame(predictions),
        "panel_summary": pd.DataFrame(panel_rows),
        "timing": timing,
        "panels": panels,
        "set_regions": set_regions,
        "umap_regions": umap_regions,
        "logreg_regions": logreg_regions,
        "membership": membership,
        "test_samples": sample_names[test_idx].tolist(),
    }


def run_nested_pipeline(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
    labels: np.ndarray,
    chunk_size: int,
    top_k: int,
    verbose: bool,
) -> dict:
    raw_values = raw.to_numpy(dtype=np.float32, copy=False)
    normalized_values = normalized.to_numpy(dtype=np.float32, copy=False)
    sample_names = meta["sample_name"].astype(str).to_numpy()
    starts = raw.columns.get_level_values("start").to_numpy(dtype=np.int64)
    ends = raw.columns.get_level_values("end").to_numpy(dtype=np.int64)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(
        balanced_splits(labels),
        start=1,
    ):
        fold_results.append(
            evaluate_one_fold(
                raw_values,
                normalized_values,
                sample_names,
                labels,
                train_idx,
                test_idx,
                fold,
                starts,
                ends,
                chunk_size,
                top_k,
                verbose,
            )
        )

    predictions = pd.concat(
        [result["predictions"] for result in fold_results],
        ignore_index=True,
    )
    panel_summary = pd.concat(
        [result["panel_summary"] for result in fold_results],
        ignore_index=True,
    )
    timing = pd.DataFrame([result["timing"] for result in fold_results])
    return {
        "fold_results": fold_results,
        "predictions": predictions,
        "panel_summary": panel_summary,
        "timing": timing,
    }


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in PANEL_ORDER:
        details = predictions.loc[predictions["panel"] == panel]
        if details.empty:
            rows.append(
                {
                    "panel": panel,
                    "n_predictions": 0,
                    "correct": 0,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                }
            )
            continue
        rows.append(
            {
                "panel": panel,
                "n_predictions": len(details),
                "correct": int(details["correct"].sum()),
                "accuracy": accuracy_score(details["true"], details["predicted"]),
                "balanced_accuracy": balanced_accuracy_score(
                    details["true"], details["predicted"]
                ),
                "macro_f1": f1_score(
                    details["true"],
                    details["predicted"],
                    labels=CONDITION_ORDER,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows)


def exact_mcnemar(raw: pd.DataFrame, panel: pd.DataFrame) -> dict:
    left = raw.set_index("sample").sort_index()
    right = panel.set_index("sample").sort_index()
    if not left.index.equals(right.index):
        return {
            "raw_only_correct": pd.NA,
            "panel_only_correct": pd.NA,
            "discordant": pd.NA,
            "exact_two_sided_p": np.nan,
        }
    left_correct = left["correct"].astype(bool).to_numpy()
    right_correct = right["correct"].astype(bool).to_numpy()
    raw_only = int(np.sum(left_correct & ~right_correct))
    panel_only = int(np.sum(~left_correct & right_correct))
    discordant = raw_only + panel_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(raw_only, panel_only)
        lower = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2 * lower)
    return {
        "raw_only_correct": raw_only,
        "panel_only_correct": panel_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def build_raw_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    raw = predictions.loc[predictions["panel"] == "raw_all_regions"]
    rows = []
    for panel in PANEL_ORDER[1:]:
        details = predictions.loc[predictions["panel"] == panel]
        rows.append({"panel": panel, **exact_mcnemar(raw, details)})
    return pd.DataFrame(rows)


def build_panel_jaccards(fold_results: list[dict]) -> pd.DataFrame:
    rows = []
    for panel in PANEL_ORDER[1:]:
        for left_fold in range(3):
            for right_fold in range(left_fold + 1, 3):
                left = set(fold_results[left_fold]["panels"][panel].tolist())
                right = set(fold_results[right_fold]["panels"][panel].tolist())
                union = left | right
                intersection = left & right
                rows.append(
                    {
                        "panel": panel,
                        "fold_a": left_fold + 1,
                        "fold_b": right_fold + 1,
                        "n_a": len(left),
                        "n_b": len(right),
                        "intersection": len(intersection),
                        "union": len(union),
                        "jaccard": len(intersection) / len(union)
                        if union
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_outer_region_stability(
    fold_results: list[dict],
    starts: np.ndarray,
    ends: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for panel in PANEL_ORDER[1:]:
        counts: dict[int, int] = {}
        for result in fold_results:
            for feature_index in result["panels"][panel]:
                counts[int(feature_index)] = counts.get(int(feature_index), 0) + 1
        for feature_index, count in counts.items():
            rows.append(
                {
                    "panel": panel,
                    "feature_index": feature_index,
                    "start": int(starts[feature_index]),
                    "end": int(ends[feature_index]),
                    "outer_selection_count": count,
                    "outer_selection_rate": count / 3.0,
                }
            )
    columns = [
        "panel",
        "feature_index",
        "start",
        "end",
        "outer_selection_count",
        "outer_selection_rate",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["panel", "outer_selection_count", "start", "end"],
        ascending=[True, False, True, True],
    )


def run_label_permutations(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    meta: pd.DataFrame,
    observed_summary: pd.DataFrame,
    n_permutations: int,
    seed: int,
    chunk_size: int,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_permutations <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    original = meta["condition"].astype(str).to_numpy()
    rows = []
    for permutation in range(1, n_permutations + 1):
        permuted = rng.permutation(original)
        result = run_nested_pipeline(
            raw,
            normalized,
            meta,
            permuted,
            chunk_size,
            top_k,
            verbose=False,
        )
        summary = summarize_predictions(result["predictions"])
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "permutation": permutation,
                    "panel": row.panel,
                    "n_predictions": row.n_predictions,
                    "accuracy": row.accuracy,
                }
            )
        del result
        gc.collect()

    null = pd.DataFrame(rows)
    test_rows = []
    for observed in observed_summary.itertuples(index=False):
        values = null.loc[null["panel"] == observed.panel, "accuracy"].dropna()
        if not len(values) or pd.isna(observed.accuracy):
            test_rows.append(
                {
                    "panel": observed.panel,
                    "observed_accuracy": observed.accuracy,
                    "n_valid_permutations": len(values),
                    "null_mean_accuracy": np.nan,
                    "null_std_accuracy": np.nan,
                    "empirical_p_greater_equal": np.nan,
                }
            )
            continue
        p_value = (1 + int(np.sum(values >= observed.accuracy))) / (
            1 + len(values)
        )
        test_rows.append(
            {
                "panel": observed.panel,
                "observed_accuracy": observed.accuracy,
                "n_valid_permutations": len(values),
                "null_mean_accuracy": float(values.mean()),
                "null_std_accuracy": float(values.std(ddof=1))
                if len(values) > 1
                else np.nan,
                "empirical_p_greater_equal": p_value,
            }
        )
    return null, pd.DataFrame(test_rows)


def save_observed_results(
    result: dict,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    jaccards: pd.DataFrame,
    stability: pd.DataFrame,
    raw: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(
        output_dir / "nested_predictions.csv", sep=SEP, index=False
    )
    result["panel_summary"].to_csv(
        output_dir / "nested_fold_panel_summary.csv", sep=SEP, index=False
    )
    result["timing"].to_csv(
        output_dir / "nested_selection_timing.csv", sep=SEP, index=False
    )
    summary.to_csv(
        output_dir / "nested_validation_summary.csv", sep=SEP, index=False
    )
    comparisons.to_csv(
        output_dir / "nested_raw_vs_panel_mcnemar.csv", sep=SEP, index=False
    )
    jaccards.to_csv(
        output_dir / "nested_panel_jaccard.csv", sep=SEP, index=False
    )
    stability.to_csv(
        output_dir / "nested_outer_region_stability.csv", sep=SEP, index=False
    )

    starts = raw.columns.get_level_values("start").to_numpy(dtype=np.int64)
    ends = raw.columns.get_level_values("end").to_numpy(dtype=np.int64)
    for fold_result in result["fold_results"]:
        fold = int(fold_result["timing"]["fold"])
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_result["set_regions"].to_csv(
            fold_dir / "set_based_regions.csv", sep=SEP, index=False
        )
        fold_result["umap_regions"].to_csv(
            fold_dir / "umap_regions.csv", sep=SEP, index=False
        )
        fold_result["logreg_regions"].to_csv(
            fold_dir / "logreg_regions.csv", sep=SEP, index=False
        )
        membership = fold_result["membership"].copy()
        if len(membership):
            feature_indices = membership["feature_index"].to_numpy(dtype=int)
            membership.insert(2, "start", starts[feature_indices])
            membership.insert(3, "end", ends[feature_indices])
        membership.to_csv(
            fold_dir / "method_membership.csv", sep=SEP, index=False
        )
        pd.DataFrame(
            {
                "test_sample": fold_result["test_samples"],
            }
        ).to_csv(fold_dir / "held_out_samples.csv", sep=SEP, index=False)

        panel_rows = []
        for panel, indices in fold_result["panels"].items():
            if panel == "raw_all_regions":
                continue
            for feature_index in indices:
                panel_rows.append(
                    {
                        "panel": panel,
                        "feature_index": int(feature_index),
                        "start": int(starts[feature_index]),
                        "end": int(ends[feature_index]),
                    }
                )
        pd.DataFrame(panel_rows).to_csv(
            fold_dir / "all_fold_local_panels.csv", sep=SEP, index=False
        )


def save_confusion_matrices(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for panel in PANEL_ORDER:
        details = predictions.loc[predictions["panel"] == panel]
        if details.empty:
            continue
        matrix = confusion_matrix(
            details["true"],
            details["predicted"],
            labels=CONDITION_ORDER,
        )
        pd.DataFrame(
            matrix,
            index=CONDITION_ORDER,
            columns=CONDITION_ORDER,
        ).to_csv(confusion_dir / f"{panel}.csv", sep=SEP)


def save_run_configuration(
    output_dir: Path,
    n_samples: int,
    n_features: int,
    top_k: int,
    chunk_size: int,
    n_permutations: int,
    permutation_seed: int,
) -> None:
    rows = [
        ("outer_folds", 3),
        ("training_samples_per_fold", 14),
        ("test_samples_per_fold", 7),
        ("total_samples", n_samples),
        ("total_raw_regions", n_features),
        ("minimum_raw_reads_set_based", MIN_RAW_READS),
        ("umap_n_neighbors", UMAP_N_NEIGHBORS),
        ("umap_min_dist", UMAP_MIN_DIST),
        ("umap_metric", UMAP_METRIC),
        ("umap_random_state", RANDOM_STATE),
        ("top_k_per_condition_umap_and_logreg", top_k),
        ("logreg_C", LOGREG_C),
        ("logreg_solver", LOGREG_SOLVER),
        ("feature_score_chunk_size", chunk_size),
        ("label_permutations", n_permutations),
        ("permutation_seed", permutation_seed),
        ("global_panel_files_loaded", False),
    ]
    pd.DataFrame(rows, columns=["parameter", "value"]).to_csv(
        output_dir / "nested_pipeline_configuration.csv",
        sep=SEP,
        index=False,
    )


def write_report(
    summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    jaccards: pd.DataFrame,
    permutation_tests: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "FULLY NESTED THREE-METHOD BALANCED VALIDATION",
        "=" * 76,
        "",
        "Leakage-Status",
        "- In jedem aeusseren Fold: 14 Training, 7 unangetastete Tests.",
        "- Set-based, UMAP, LogReg und alle kombinierten Panels wurden nur aus",
        "  den jeweiligen Trainingssamples neu erzeugt.",
        "- Globale Regionsdateien wurden nicht verwendet.",
        "- Die globale Koordinatenachse dient nur der Matrixausrichtung;",
        "  test-exklusive Regionen sind im Training konstant null.",
        "- Top-K=250 pro Condition wurde vor der Ergebnisbetrachtung festgelegt.",
        "",
        "Ergebnisse",
    ]
    for row in summary.itertuples(index=False):
        accuracy = "NA" if pd.isna(row.accuracy) else f"{100 * row.accuracy:.1f}%"
        lines.append(
            f"- {row.panel}: {accuracy} ({row.correct}/{row.n_predictions}), "
            f"balanced={row.balanced_accuracy:.4f}, macro-F1={row.macro_f1:.4f}"
        )

    lines.extend(["", "Featurezahlen pro aeusserem Fold"])
    for panel in PANEL_ORDER:
        values = panel_summary.loc[
            panel_summary["panel"] == panel, "n_features"
        ]
        if len(values):
            lines.append(
                f"- {panel}: min={int(values.min())}, "
                f"mean={values.mean():.1f}, max={int(values.max())}"
            )

    lines.extend(["", "Panelstabilitaet (mittlerer paarweiser Jaccard)"])
    for panel, group in jaccards.groupby("panel"):
        lines.append(f"- {panel}: {group['jaccard'].mean():.4f}")

    if len(permutation_tests):
        lines.extend(["", "Full-Pipeline-Label-Permutationstest"])
        for row in permutation_tests.itertuples(index=False):
            lines.append(
                f"- {row.panel}: beobachtet={row.observed_accuracy:.4f}, "
                f"Nullmittel={row.null_mean_accuracy:.4f}, "
                f"p={row.empirical_p_greater_equal:.4f}, "
                f"n={row.n_valid_permutations}"
            )
    else:
        lines.extend(
            [
                "",
                "Kein Permutationstest ausgefuehrt. Fuer die Full-Pipeline-Nullkontrolle:",
                "  --n-permutations 20",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretationsgrenze",
            "- Dies ist die primaere leakage-freie interne Generalisierungsschaetzung",
            "  fuer die vorhandene balancierte 2+1-Replikatstruktur.",
            "- Wegen nur 21 Samples und moeglicher Condition-/Batch-Konfundierung",
            "  ersetzt sie keine Validierung an unabhaengigen Experimenten.",
        ]
    )
    (output_dir / "nested_validation_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--top-k-per-condition",
        type=int,
        default=TOP_K_PER_CONDITION,
        help="Vorab festgelegtes Top-K fuer UMAP und LogReg (Standard: 250).",
    )
    parser.add_argument(
        "--feature-score-chunk-size",
        type=int,
        default=FEATURE_SCORE_CHUNK_SIZE,
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help=(
            "Anzahl kompletter Label-Permutationslaeufe. Zuerst 0 fuer das "
            "beobachtete Ergebnis; danach empfohlen 20."
        ),
    )
    parser.add_argument("--permutation-seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k_per_condition < 1:
        raise ValueError("--top-k-per-condition muss mindestens 1 sein.")
    if args.feature_score_chunk_size < 1:
        raise ValueError("--feature-score-chunk-size muss mindestens 1 sein.")
    if args.n_permutations < 0:
        raise ValueError("--n-permutations darf nicht negativ sein.")


    raw, normalized, meta = build_raw_and_normalized_matrices(CONDITIONS)
    labels = meta["condition"].astype(str).to_numpy()

    observed = run_nested_pipeline(
        raw,
        normalized,
        meta,
        labels,
        args.feature_score_chunk_size,
        args.top_k_per_condition,
        verbose=True,
    )
    summary = summarize_predictions(observed["predictions"])
    comparisons = build_raw_comparisons(observed["predictions"])
    jaccards = build_panel_jaccards(observed["fold_results"])
    starts = raw.columns.get_level_values("start").to_numpy(dtype=np.int64)
    ends = raw.columns.get_level_values("end").to_numpy(dtype=np.int64)
    stability = build_outer_region_stability(
        observed["fold_results"], starts, ends
    )
    save_observed_results(
        observed,
        summary,
        comparisons,
        jaccards,
        stability,
        raw,
        args.output_dir,
    )
    save_confusion_matrices(observed["predictions"], args.output_dir)
    save_run_configuration(
        args.output_dir,
        raw.shape[0],
        raw.shape[1],
        args.top_k_per_condition,
        args.feature_score_chunk_size,
        args.n_permutations,
        args.permutation_seed,
    )


    null, permutation_tests = run_label_permutations(
        raw,
        normalized,
        meta,
        summary,
        args.n_permutations,
        args.permutation_seed,
        args.feature_score_chunk_size,
        args.top_k_per_condition,
    )
    if len(null):
        null.to_csv(
            args.output_dir / "label_permutation_null.csv", sep=SEP, index=False
        )
        permutation_tests.to_csv(
            args.output_dir / "label_permutation_tests.csv", sep=SEP, index=False
        )

    write_report(
        summary,
        observed["panel_summary"],
        jaccards,
        permutation_tests,
        args.output_dir,
    )

    print(f"Ergebnis gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()