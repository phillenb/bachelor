from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def _find_src_dir(start: Path) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "src":
            return parent
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
CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]
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

OUTPUT_DIR = (
    DATA_DIR
    / "method3_supervised_LogReg_RandomForrest_data"
    / "logistic_regression_raw_stability_selection"
)

LOGREG_C = 0.01
LOGREG_MAX_ITER = 5000
LOGREG_SOLVER = "liblinear"

DEFAULT_TOP_K_PER_CONDITION = 250

DEFAULT_LOO_MIN_SELECTION_RATE = 0.80
DEFAULT_BALANCED_MIN_SELECTION_FOLDS = 2


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

def build_sample_matrix(condition_paths: list[Path]) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for csv_path in condition_paths:
        sample_name = csv_path.parent.parent.name
        df = load_intensity_matrix(csv_path)
        grouped = df.groupby(["start", "end"])["n_reads"].sum()
        series[sample_name] = grouped

    if not series:
        raise ValueError("Keine Samples geladen.")
    return pd.DataFrame(series).fillna(0.0)


def build_feature_matrix(
    conditions: dict[str, list[Path]],
) -> tuple[pd.DataFrame, list[str]]:
    all_cpm: dict[str, pd.Series] = {}
    label_by_sample: dict[str, str] = {}

    for condition in CONDITION_ORDER:
        paths = list(conditions[condition])
        if len(paths) != EXPECTED_SAMPLES_PER_CONDITION:
            raise ValueError(
                f"{condition}: {len(paths)} Samples gefunden; erwartet wurden "
                f"{EXPECTED_SAMPLES_PER_CONDITION}."
            )

        raw = build_sample_matrix(paths)
        cpm = normalize_cpm(raw)
        for sample in cpm.columns:
            if sample in all_cpm:
                raise ValueError(f"Doppelter Sample-Name: {sample}")
            all_cpm[sample] = cpm[sample]
            label_by_sample[sample] = condition

    combined = pd.DataFrame(all_cpm, dtype=np.float32).fillna(0.0)
    X = np.log2(combined.T + np.float32(1.0)).astype(np.float32)
    y = [label_by_sample[sample] for sample in X.index]

    if X.shape[0] != 21 or X.shape[1] == 0:
        raise AssertionError(f"Unerwartete Feature-Matrix: {X.shape}")
    if not isinstance(X.columns, pd.MultiIndex):
        raise TypeError("Erwartete (start, end)-MultiIndex-Featureachse.")
    return X, y


def loo_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    return [
        (indices[indices != test], np.asarray([test], dtype=int))
        for test in indices
    ]


def balanced_replicate_block_splits(
    y: list[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    y_array = np.asarray(y)
    per_condition = {
        condition: np.flatnonzero(y_array == condition)
        for condition in CONDITION_ORDER
    }
    for condition, indices in per_condition.items():
        if len(indices) != EXPECTED_SAMPLES_PER_CONDITION:
            raise ValueError(
                f"{condition}: exakt drei Samples fuer balanced CV erforderlich."
            )

    all_indices = np.arange(len(y))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for replicate_position in range(EXPECTED_SAMPLES_PER_CONDITION):
        test_idx = np.asarray(
            [per_condition[c][replicate_position] for c in CONDITION_ORDER],
            dtype=int,
        )
        train_idx = all_indices[~np.isin(all_indices, test_idx)]
        splits.append((train_idx, test_idx))
    return splits


def make_model() -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            random_state=42,
            solver=LOGREG_SOLVER,
        )
    )


def top_positive_coefficients(
    coefficient: np.ndarray,
    top_k: int,
) -> np.ndarray:
    positive = np.flatnonzero(coefficient > 0)
    if not len(positive):
        return np.asarray([], dtype=int)

    n_keep = min(top_k, len(positive))
    if n_keep == len(positive):
        candidates = positive
    else:
        relative = np.argpartition(coefficient[positive], -n_keep)[-n_keep:]
        candidates = positive[relative]
    return candidates[np.argsort(coefficient[candidates])[::-1]]


def select_regions_from_fitted_raw_model(
    model: OneVsRestClassifier,
    feature_starts: np.ndarray,
    feature_ends: np.ndarray,
    design: str,
    fold: int,
    top_k_per_condition: int,
) -> tuple[np.ndarray, list[dict]]:
    selected_parts: list[np.ndarray] = []
    events: list[dict] = []

    if len(model.estimators_) != len(model.classes_):
        raise AssertionError("Unerwartete Anzahl an OvR-Teilmodellen.")

    for class_row, condition in enumerate(model.classes_):
        coefficient = model.estimators_[class_row].coef_[0]
        selected = top_positive_coefficients(
            coefficient,
            top_k_per_condition,
        )
        if not len(selected):
            raise ValueError(
                f"{design}, Fold {fold}, {condition}: kein positiver Koeffizient."
            )
        selected_parts.append(selected)

        for rank, feature_index in enumerate(selected, start=1):
            events.append(
                {
                    "design": design,
                    "fold": fold,
                    "condition": str(condition),
                    "start": int(feature_starts[feature_index]),
                    "end": int(feature_ends[feature_index]),
                    "feature_index": int(feature_index),
                    "rank_within_condition_fold": rank,
                    "standardized_coefficient": float(
                        coefficient[feature_index]
                    ),
                }
            )

    selected_union = np.unique(np.concatenate(selected_parts))
    return selected_union, events


def predictions_to_result(
    details: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    accuracy = float(accuracy_score(details["true"], details["predicted"]))
    matrix = confusion_matrix(
        details["true"],
        details["predicted"],
        labels=CONDITION_ORDER,
    )
    return accuracy, pd.DataFrame(
        matrix,
        index=CONDITION_ORDER,
        columns=CONDITION_ORDER,
    )


def evaluate_design(
    X: pd.DataFrame,
    y: list[str],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    design: str,
    top_k_per_condition: int,
) -> dict:
    X_array = X.to_numpy(dtype=np.float32, copy=False)
    y_array = np.asarray(y)
    sample_names = X.index.to_numpy()
    starts = X.columns.get_level_values("start").to_numpy()
    ends = X.columns.get_level_values("end").to_numpy()

    raw_rows: list[dict] = []
    reduced_rows: list[dict] = []
    fold_rows: list[dict] = []
    all_events: list[dict] = []

    total_raw_seconds = 0.0
    total_reduced_seconds = 0.0

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        raw_started = time.perf_counter()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_array[train_idx])
        X_test_scaled = scaler.transform(X_array[test_idx])

        raw_model = make_model()
        raw_model.fit(X_train_scaled, y_array[train_idx])
        raw_predictions = raw_model.predict(X_test_scaled)
        raw_seconds = time.perf_counter() - raw_started

        selection_started = time.perf_counter()
        selected_indices, events = select_regions_from_fitted_raw_model(
            raw_model,
            starts,
            ends,
            design,
            fold,
            top_k_per_condition,
        )

        reduced_model = make_model()
        reduced_model.fit(
            X_train_scaled[:, selected_indices],
            y_array[train_idx],
        )
        reduced_predictions = reduced_model.predict(
            X_test_scaled[:, selected_indices]
        )
        reduced_seconds = time.perf_counter() - selection_started

        total_raw_seconds += raw_seconds
        total_reduced_seconds += reduced_seconds
        all_events.extend(events)

        raw_correct = 0
        reduced_correct = 0
        for position, sample_idx in enumerate(test_idx):
            truth = str(y_array[sample_idx])
            raw_prediction = str(raw_predictions[position])
            reduced_prediction = str(reduced_predictions[position])
            raw_is_correct = truth == raw_prediction
            reduced_is_correct = truth == reduced_prediction
            raw_correct += int(raw_is_correct)
            reduced_correct += int(reduced_is_correct)

            common = {
                "design": design,
                "fold": fold,
                "sample": str(sample_names[sample_idx]),
                "true": truth,
            }
            raw_rows.append(
                {
                    **common,
                    "predicted": raw_prediction,
                    "correct": raw_is_correct,
                    "n_features": X.shape[1],
                }
            )
            reduced_rows.append(
                {
                    **common,
                    "predicted": reduced_prediction,
                    "correct": reduced_is_correct,
                    "n_features": len(selected_indices),
                }
            )

        fold_rows.append(
            {
                "design": design,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_raw_features": X.shape[1],
                "n_reduced_features": len(selected_indices),
                "raw_correct": raw_correct,
                "reduced_correct": reduced_correct,
                "raw_scaling_fit_predict_seconds": raw_seconds,
                "coefficient_selection_and_reduced_fit_seconds": (
                    reduced_seconds
                ),
            }
        )


        del (
            scaler,
            raw_model,
            reduced_model,
            X_train_scaled,
            X_test_scaled,
            selected_indices,
        )
        gc.collect()

    raw_details = pd.DataFrame(raw_rows)
    reduced_details = pd.DataFrame(reduced_rows)
    fold_summary = pd.DataFrame(fold_rows)
    events = pd.DataFrame(all_events)
    raw_accuracy, raw_cm = predictions_to_result(raw_details)
    reduced_accuracy, reduced_cm = predictions_to_result(reduced_details)

    return {
        "design": design,
        "raw_details": raw_details,
        "reduced_details": reduced_details,
        "fold_summary": fold_summary,
        "events": events,
        "raw_accuracy": raw_accuracy,
        "reduced_accuracy": reduced_accuracy,
        "raw_confusion_matrix": raw_cm,
        "reduced_confusion_matrix": reduced_cm,
        "raw_seconds": total_raw_seconds,
        "reduced_seconds": total_reduced_seconds,
        "n_folds": int(fold_summary.shape[0]),
        "n_raw_features": int(X.shape[1]),
        "mean_reduced_features": float(
            fold_summary["n_reduced_features"].mean()
        ),
        "min_reduced_features": int(
            fold_summary["n_reduced_features"].min()
        ),
        "max_reduced_features": int(
            fold_summary["n_reduced_features"].max()
        ),
    }


def aggregate_region_stability(
    events: pd.DataFrame,
    n_folds: int,
    min_selection_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = (
        events.groupby(["condition", "start", "end"], as_index=False)
        .agg(
            selection_count=("fold", "nunique"),
            mean_standardized_coefficient=(
                "standardized_coefficient",
                "mean",
            ),
            std_standardized_coefficient=(
                "standardized_coefficient",
                "std",
            ),
            min_standardized_coefficient=(
                "standardized_coefficient",
                "min",
            ),
            max_standardized_coefficient=(
                "standardized_coefficient",
                "max",
            ),
            mean_rank=("rank_within_condition_fold", "mean"),
            best_rank=("rank_within_condition_fold", "min"),
            worst_rank=("rank_within_condition_fold", "max"),
        )
    )
    stability["n_folds"] = n_folds
    stability["selection_rate"] = stability["selection_count"] / n_folds
    stability["passes_stability_threshold"] = stability[
        "selection_count"
    ].ge(min_selection_folds)

    stable_condition_counts = (
        stability.loc[stability["passes_stability_threshold"]]
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
    stability["is_condition_specific_stable"] = (
        stability["passes_stability_threshold"]
        & stability["n_stable_condition_assignments"].eq(1)
    )

    condition_specific = stability.loc[
        stability["is_condition_specific_stable"]
    ].copy()
    condition_specific = condition_specific.sort_values(
        [
            "condition",
            "selection_count",
            "mean_rank",
            "mean_standardized_coefficient",
        ],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    condition_specific["stability_rank_within_condition"] = (
        condition_specific.groupby("condition").cumcount() + 1
    )

    stability = stability.sort_values(
        ["condition", "selection_count", "mean_rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return stability, condition_specific


def exact_mcnemar(
    raw_details: pd.DataFrame,
    reduced_details: pd.DataFrame,
) -> dict:
    raw = raw_details.set_index("sample").sort_index()
    reduced = reduced_details.set_index("sample").sort_index()
    if not raw.index.equals(reduced.index):
        raise ValueError("McNemar: Sample-Indizes stimmen nicht ueberein.")

    raw_correct = raw["correct"].astype(bool).to_numpy()
    reduced_correct = reduced["correct"].astype(bool).to_numpy()
    raw_only = int(np.sum(raw_correct & ~reduced_correct))
    reduced_only = int(np.sum(~raw_correct & reduced_correct))
    discordant = raw_only + reduced_only

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(raw_only, reduced_only)
        lower_tail = sum(
            math.comb(discordant, k) for k in range(smaller + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)

    return {
        "raw_only_correct": raw_only,
        "reduced_only_correct": reduced_only,
        "discordant_predictions": discordant,
        "exact_two_sided_p": p_value,
    }


def build_cross_design_consensus(
    loo_specific: pd.DataFrame,
    balanced_specific: pd.DataFrame,
) -> pd.DataFrame:
    loo_columns = [
        "condition",
        "start",
        "end",
        "selection_count",
        "selection_rate",
        "mean_standardized_coefficient",
        "mean_rank",
        "stability_rank_within_condition",
    ]
    balanced_columns = loo_columns.copy()
    consensus = loo_specific[loo_columns].merge(
        balanced_specific[balanced_columns],
        on=["condition", "start", "end"],
        how="inner",
        suffixes=("_loo", "_balanced"),
        validate="one_to_one",
    )
    if consensus.empty:
        return consensus

    consensus["combined_stability_score"] = (
        consensus["selection_rate_loo"]
        * consensus["selection_rate_balanced"]
    )
    consensus = consensus.sort_values(
        [
            "condition",
            "combined_stability_score",
            "mean_rank_loo",
            "mean_rank_balanced",
        ],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    consensus["consensus_rank_within_condition"] = (
        consensus.groupby("condition").cumcount() + 1
    )
    return consensus


def save_design_result(
    result: dict,
    stability: pd.DataFrame,
    condition_specific: pd.DataFrame,
    output_dir: Path,
) -> None:
    design_dir = output_dir / result["design"]
    design_dir.mkdir(parents=True, exist_ok=True)

    result["raw_details"].to_csv(
        design_dir / "raw_predictions.csv", sep=SEP, index=False
    )
    result["reduced_details"].to_csv(
        design_dir / "fold_local_reduced_predictions.csv",
        sep=SEP,
        index=False,
    )
    result["fold_summary"].to_csv(
        design_dir / "fold_summary.csv", sep=SEP, index=False
    )
    result["events"].to_csv(
        design_dir / "top_regions_by_fold.csv", sep=SEP, index=False
    )
    stability.to_csv(
        design_dir / "raw_logreg_region_stability.csv",
        sep=SEP,
        index=False,
    )
    condition_specific.to_csv(
        design_dir / "stable_condition_specific_regions.csv",
        sep=SEP,
        index=False,
    )
    result["raw_confusion_matrix"].to_csv(
        design_dir / "raw_confusion_matrix.csv", sep=SEP
    )
    result["reduced_confusion_matrix"].to_csv(
        design_dir / "fold_local_reduced_confusion_matrix.csv", sep=SEP
    )


def write_summary(
    output_dir: Path,
    results: list[dict],
    stability_thresholds: dict[str, int],
    stable_counts: dict[str, int],
    consensus: pd.DataFrame,
    top_k_per_condition: int,
) -> None:
    rows: list[dict] = []
    comparisons: dict[str, dict] = {}

    for result in results:
        correct_raw = int(result["raw_details"]["correct"].sum())
        correct_reduced = int(result["reduced_details"]["correct"].sum())
        rows.extend(
            [
                {
                    "design": result["design"],
                    "model": "raw_all_regions",
                    "accuracy": result["raw_accuracy"],
                    "correct": correct_raw,
                    "n_predictions": len(result["raw_details"]),
                    "feature_count_mean": result["n_raw_features"],
                    "feature_count_min": result["n_raw_features"],
                    "feature_count_max": result["n_raw_features"],
                },
                {
                    "design": result["design"],
                    "model": "fold_local_raw_logreg_top_k",
                    "accuracy": result["reduced_accuracy"],
                    "correct": correct_reduced,
                    "n_predictions": len(result["reduced_details"]),
                    "feature_count_mean": result["mean_reduced_features"],
                    "feature_count_min": result["min_reduced_features"],
                    "feature_count_max": result["max_reduced_features"],
                },
            ]
        )
        comparisons[result["design"]] = exact_mcnemar(
            result["raw_details"], result["reduced_details"]
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "validation_summary.csv", sep=SEP, index=False)

    lines = [
        "RAW LOGISTIC REGRESSION STABILITY SELECTION",
        "=" * 72,
        "",
        f"Top-K positive Koeffizienten pro Condition/Fold: {top_k_per_condition}",
        "",
        "Leakage-freie Accuracy",
    ]
    for row in rows:
        lines.append(
            f"- {row['design']} | {row['model']}: "
            f"{100 * row['accuracy']:.1f}% "
            f"({row['correct']}/{row['n_predictions']}), "
            f"Features mean={row['feature_count_mean']:.1f}"
        )

    lines.extend(["", "Gepaarter Rohdaten-vs-reduziert-Vergleich"])
    for design, comparison in comparisons.items():
        lines.append(
            f"- {design}: nur Rohdaten korrekt="
            f"{comparison['raw_only_correct']}, nur reduziert korrekt="
            f"{comparison['reduced_only_correct']}, "
            f"McNemar p={comparison['exact_two_sided_p']:.6f}"
        )

    lines.extend(["", "Stabile condition-spezifische Regionen"])
    for design in stability_thresholds:
        lines.append(
            f"- {design}: mindestens {stability_thresholds[design]} Folds, "
            f"n={stable_counts[design]}"
        )
    lines.append(f"- Konsens aus LOO und balanced: n={len(consensus)}")
    lines.extend(
        [
            "",
            "Interpretation",
            "- Die reduzierte Accuracy ist leakage-frei, weil jede Top-K-Liste "
            "nur aus dem jeweiligen aeusseren Trainingsfold stammt.",
            "- Der abschliessende Konsens ist eine deskriptive Kandidatenliste; "
            "seine erneute Bewertung auf denselben 21 Samples waere optimistisch.",
            "- TOP_K darf fuer eine belastbare Accuracy nicht nachtraeglich anhand "
            "der aeusseren Testergebnisse optimiert werden.",
        ]
    )
    (output_dir / "validation_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k-per-condition",
        type=int,
        default=DEFAULT_TOP_K_PER_CONDITION,
    )
    parser.add_argument(
        "--loo-min-selection-rate",
        type=float,
        default=DEFAULT_LOO_MIN_SELECTION_RATE,
    )
    parser.add_argument(
        "--balanced-min-selection-folds",
        type=int,
        default=DEFAULT_BALANCED_MIN_SELECTION_FOLDS,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.top_k_per_condition < 1:
        parser.error("--top-k-per-condition muss mindestens 1 sein.")
    if not 0 < args.loo_min_selection_rate <= 1:
        parser.error("--loo-min-selection-rate muss in (0, 1] liegen.")
    if not 1 <= args.balanced_min_selection_folds <= 3:
        parser.error("--balanced-min-selection-folds muss 1, 2 oder 3 sein.")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = build_feature_matrix(CONDITIONS)

    loo_result = evaluate_design(
        X,
        y,
        loo_splits(len(X)),
        design="loo",
        top_k_per_condition=args.top_k_per_condition,
    )
    loo_min_folds = math.ceil(
        loo_result["n_folds"] * args.loo_min_selection_rate
    )
    loo_stability, loo_specific = aggregate_region_stability(
        loo_result["events"],
        loo_result["n_folds"],
        loo_min_folds,
    )
    save_design_result(
        loo_result,
        loo_stability,
        loo_specific,
        args.output_dir,
    )

    balanced_result = evaluate_design(
        X,
        y,
        balanced_replicate_block_splits(y),
        design="balanced_replicate_block",
        top_k_per_condition=args.top_k_per_condition,
    )
    balanced_stability, balanced_specific = aggregate_region_stability(
        balanced_result["events"],
        balanced_result["n_folds"],
        args.balanced_min_selection_folds,
    )
    save_design_result(
        balanced_result,
        balanced_stability,
        balanced_specific,
        args.output_dir,
    )

    consensus = build_cross_design_consensus(
        loo_specific,
        balanced_specific,
    )
    consensus.to_csv(
        args.output_dir / "raw_logreg_stable_cross_design_consensus.csv",
        sep=SEP,
        index=False,
    )

    stability_thresholds = {
        "loo": loo_min_folds,
        "balanced_replicate_block": args.balanced_min_selection_folds,
    }
    stable_counts = {
        "loo": len(loo_specific),
        "balanced_replicate_block": len(balanced_specific),
    }
    write_summary(
        args.output_dir,
        [loo_result, balanced_result],
        stability_thresholds,
        stable_counts,
        consensus,
        args.top_k_per_condition,
    )


    print(f"Ergebnis gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()