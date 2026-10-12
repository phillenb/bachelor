from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DATA_DIR = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data"
)

INTERSECTION_BASE_DIR = (
    DATA_DIR
    / "method1_set_based_and_rule_based_region_selection_data"
    / "shared_regions_across_samples"
)

UMAP_REGIONS_CSV = (
    DATA_DIR
    / "method2_unsupervised_selection_UMAP_PCA_PCoA_data"
    / "umap_regions"
    / "umap_regions.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "method2_unsupervised_selection_UMAP_PCA_PCoA_data"
    / "condition_intersections_vs_umap"
)

CONDITIONS = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]

MIN_RAW_READS = 2
EXPECTED_SAMPLES_PER_CONDITION = 3
SEP = ";"

CONDITION_COLORS = {
    "Ctrl": "#636EFA",
    "UTP18": "#EF553B",
    "DIMT1L": "#00CC96",
    "LAS1L": "#AB63FA",
    "RPL3": "#FFA15A",
    "URB1": "#19D3F3",
    "WBSCR22": "#FF6692",
}


def normalize_coordinates(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Validiert start/end und entfernt doppelte Koordinaten."""
    required = {"start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: fehlende Spalten: {sorted(missing)}"
        )

    if df[["start", "end"]].isna().any().any():
        raise ValueError(f"{source}: fehlende start/end-Werte gefunden.")

    try:
        df[["start", "end"]] = df[["start", "end"]].astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: start/end müssen ganzzahlige Koordinaten sein."
        ) from exc

    n_duplicates = int(df.duplicated(["start", "end"]).sum())
    if n_duplicates:
        df = df.drop_duplicates(["start", "end"], keep="first")

    return df.reset_index(drop=True)


def load_umap_regions(path: Path) -> pd.DataFrame:
    """Lädt UMAP-Regionen und prüft die benötigten Spalten."""
    if not path.is_file():
        raise FileNotFoundError(f"UMAP-Datei nicht gefunden: {path}")

    df = pd.read_csv(path, sep=SEP)
    required = {"start", "end", "UMAP_1", "UMAP_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: fehlende Spalten: {sorted(missing)}"
        )

    df = normalize_coordinates(df, path)
    return df


def load_condition_intersection(
    condition: str,
) -> tuple[pd.DataFrame, Path]:
    """Lädt vollständige 3-von-3-Schnittmengen direkt oder aus der Consistency-Datei."""
    condition_dir = INTERSECTION_BASE_DIR / condition
    explicit_candidates = [
        condition_dir
        / f"intersection_all_samples_min_raw_reads_{MIN_RAW_READS}.csv",
        condition_dir / f"{condition}_intersection_all_samples.csv",
    ]

    for path in explicit_candidates:
        if path.is_file():
            df = pd.read_csv(path, sep=SEP)
            if "n_samples_present" in df.columns:
                df = df.loc[
                    df["n_samples_present"].eq(
                        EXPECTED_SAMPLES_PER_CONDITION
                    )
                ].copy()
            return normalize_coordinates(df, path), path

    fallback = condition_dir / "nuc_region_consistency_raw_and_norm.csv"
    if fallback.is_file():
        df = pd.read_csv(fallback, sep=SEP)
        if "n_samples_present" in df.columns:
            df = df.loc[
                df["n_samples_present"].eq(
                    EXPECTED_SAMPLES_PER_CONDITION
                )
            ].copy()
        elif "prevalence" in df.columns:
            df = df.loc[np.isclose(df["prevalence"], 1.0)].copy()
        else:
            raise ValueError(
                f"{fallback}: weder n_samples_present noch prevalence vorhanden."
            )
        return normalize_coordinates(df, fallback), fallback

    checked = [*explicit_candidates, fallback]
    raise FileNotFoundError(
        f"{condition}: keine vollständige Schnittmengendatei gefunden. "
        f"Geprüft: {[str(path) for path in checked]}"
    )


def coordinate_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Erzeugt einen Koordinatenindex für exakte Mengenvergleiche."""
    return pd.MultiIndex.from_frame(df[["start", "end"]])


def find_condition_signal_column(
    umap_df: pd.DataFrame,
    condition: str,
) -> str | None:
    expected = f"{condition}_signal".lower()
    for column in umap_df.columns:
        if column.lower() == expected:
            return column
    return None


def assign_umap_condition_by_max_signal(
    umap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ordnet Regionen dem höchsten Condition-Signal zu und markiert Gleichstände."""
    signal_columns: dict[str, str] = {}
    missing_conditions: list[str] = []

    for condition in CONDITIONS:
        signal_col = find_condition_signal_column(umap_df, condition)
        if signal_col is None:
            missing_conditions.append(condition)
        else:
            signal_columns[condition] = signal_col

    if missing_conditions:
        raise ValueError(
            "Für die UMAP-Condition-Zuordnung fehlen Signalspalten für: "
            f"{missing_conditions}. Vorhandene Spalten: {umap_df.columns.tolist()}"
        )

    condition_order = list(signal_columns)
    numeric_signals = umap_df[
        [signal_columns[condition] for condition in condition_order]
    ].apply(pd.to_numeric, errors="coerce")

    all_missing = numeric_signals.isna().all(axis=1)
    if all_missing.any():
        bad_rows = umap_df.index[all_missing].tolist()[:10]
        raise ValueError(
            "Einige UMAP-Regionen besitzen in allen Condition-Signalspalten "
            f"fehlende Werte. Beispielindizes: {bad_rows}"
        )

    values = numeric_signals.to_numpy(dtype=float)
    values_for_max = np.where(np.isnan(values), -np.inf, values)
    max_signal = values_for_max.max(axis=1)
    is_max = np.isclose(
        values_for_max,
        max_signal[:, None],
        rtol=1e-12,
        atol=1e-12,
    )
    n_max_conditions = is_max.sum(axis=1)

    max_signal_conditions = [
        "|".join(
            condition
            for condition, condition_is_max in zip(condition_order, row)
            if condition_is_max
        )
        for row in is_max
    ]
    assigned_condition = [
        max_conditions if n_max == 1 else "TIE"
        for max_conditions, n_max in zip(
            max_signal_conditions,
            n_max_conditions,
        )
    ]

    sorted_signals = np.sort(values_for_max, axis=1)
    second_highest_signal = sorted_signals[:, -2]

    result = umap_df.copy()
    result["umap_assigned_condition"] = assigned_condition
    result["umap_max_signal_conditions"] = max_signal_conditions
    result["umap_assignment_is_tie"] = n_max_conditions > 1
    result["umap_max_signal"] = max_signal
    result["umap_second_highest_signal"] = second_highest_signal
    result["umap_signal_margin"] = max_signal - second_highest_signal

    return result


def compare_condition(
    condition: str,
    intersection_df: pd.DataFrame,
    intersection_path: Path,
    exclusive_key_set: set[tuple[int, int]],
    umap_df: pd.DataFrame,
    umap_keys: pd.MultiIndex,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, pd.DataFrame]:
    """Prüft die Signalzuordnung exklusiver Regionen und vergleicht Schnittmengen."""
    intersection_keys = coordinate_index(intersection_df)
    in_intersection = np.asarray(umap_keys.isin(intersection_keys))
    exclusive_index = pd.MultiIndex.from_tuples(
        sorted(exclusive_key_set),
        names=["start", "end"],
    )
    in_exclusive_intersection = np.asarray(umap_keys.isin(exclusive_index))

    umap_key_set = set(umap_keys.tolist())
    intersection_key_set = set(intersection_keys.tolist())
    overlap_set = umap_key_set & intersection_key_set
    union_set = umap_key_set | intersection_key_set

    umap_for_merge = umap_df.copy()
    signal_col = find_condition_signal_column(umap_df, condition)
    if signal_col is not None:
        rank_col = f"{condition}_signal_rank_desc"
        percentile_col = f"{condition}_signal_percentile"
        umap_for_merge[rank_col] = umap_for_merge[signal_col].rank(
            method="min",
            ascending=False,
        )
        denominator = max(len(umap_for_merge) - 1, 1)
        umap_for_merge[percentile_col] = (
            1.0 - (umap_for_merge[rank_col] - 1.0) / denominator
        )

    matched = intersection_df.merge(
        umap_for_merge,
        on=["start", "end"],
        how="inner",
        suffixes=("_intersection", "_umap"),
    )

    matched_keys = coordinate_index(matched)
    matched["method_a_exclusive_to_condition"] = matched_keys.isin(
        exclusive_index
    )
    matched["same_condition_assignment"] = matched[
        "umap_assigned_condition"
    ].eq(condition)
    matched["condition_among_tied_max_signals"] = matched[
        "umap_max_signal_conditions"
    ].str.split("|").apply(lambda values: condition in values)

    exclusive_matched = matched.loc[
        matched["method_a_exclusive_to_condition"]
    ].copy()
    exclusive_same = exclusive_matched.loc[
        exclusive_matched["same_condition_assignment"]
    ].copy()
    exclusive_tie = exclusive_matched.loc[
        exclusive_matched["umap_assignment_is_tie"]
        & exclusive_matched["condition_among_tied_max_signals"]
    ].copy()
    exclusive_different = exclusive_matched.loc[
        ~exclusive_matched["same_condition_assignment"]
        & ~(
            exclusive_matched["umap_assignment_is_tie"]
            & exclusive_matched["condition_among_tied_max_signals"]
        )
    ].copy()
    shared_matched = matched.loc[
        ~matched["method_a_exclusive_to_condition"]
    ].copy()

    unmatched = intersection_df.loc[
        ~intersection_keys.isin(umap_keys)
    ].copy()

    condition_dir = OUTPUT_DIR / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(
        condition_dir / "intersection_regions_in_umap.csv",
        sep=SEP,
        index=False,
    )
    unmatched.to_csv(
        condition_dir / "intersection_regions_not_in_umap.csv",
        sep=SEP,
        index=False,
    )
    exclusive_matched.to_csv(
        condition_dir / "exclusive_method_a_regions_in_umap.csv",
        sep=SEP,
        index=False,
    )
    exclusive_same.to_csv(
        condition_dir / "exclusive_same_condition_assignment.csv",
        sep=SEP,
        index=False,
    )
    exclusive_tie.to_csv(
        condition_dir / "exclusive_tied_max_including_condition.csv",
        sep=SEP,
        index=False,
    )
    exclusive_different.to_csv(
        condition_dir / "exclusive_different_condition_assignment.csv",
        sep=SEP,
        index=False,
    )
    shared_matched.to_csv(
        condition_dir / "shared_method_a_regions_in_umap.csv",
        sep=SEP,
        index=False,
    )

    n_intersection = len(intersection_key_set)
    n_umap = len(umap_key_set)
    n_overlap = len(overlap_set)
    n_union = len(union_set)
    n_exclusive = len(exclusive_key_set)
    n_exclusive_in_umap = int(in_exclusive_intersection.sum())
    n_exclusive_same = len(exclusive_same)
    n_exclusive_tie = len(exclusive_tie)
    n_exclusive_different = len(exclusive_different)

    summary: dict[str, object] = {
        "condition": condition,
        "intersection_source": str(intersection_path),
        "n_intersection_regions": n_intersection,
        "n_umap_regions": n_umap,
        "n_overlap": n_overlap,
        "n_intersection_not_in_umap": n_intersection - n_overlap,
        "pct_intersection_found_in_umap": (
            100.0 * n_overlap / n_intersection if n_intersection else np.nan
        ),
        "pct_umap_regions_in_intersection": (
            100.0 * n_overlap / n_umap if n_umap else np.nan
        ),
        "jaccard": n_overlap / n_union if n_union else 1.0,
        "condition_signal_column": signal_col or "",
        "n_method_a_exclusive_regions": n_exclusive,
        "n_method_a_shared_regions": n_intersection - n_exclusive,
        "n_exclusive_regions_in_umap": n_exclusive_in_umap,
        "n_exclusive_regions_not_in_umap": (
            n_exclusive - n_exclusive_in_umap
        ),
        "pct_exclusive_regions_found_in_umap": (
            100.0 * n_exclusive_in_umap / n_exclusive
            if n_exclusive
            else np.nan
        ),
        "n_exclusive_same_condition_assignment": n_exclusive_same,
        "n_exclusive_tied_max_including_condition": n_exclusive_tie,
        "n_exclusive_different_condition_assignment": n_exclusive_different,
        "pct_same_condition_of_exclusive_found_in_umap": (
            100.0 * n_exclusive_same / n_exclusive_in_umap
            if n_exclusive_in_umap
            else np.nan
        ),
        "pct_same_or_tied_condition_of_exclusive_found_in_umap": (
            100.0 * (n_exclusive_same + n_exclusive_tie)
            / n_exclusive_in_umap
            if n_exclusive_in_umap
            else np.nan
        ),
    }

    if signal_col is not None:
        selected_signal = umap_df.loc[in_intersection, signal_col]
        other_signal = umap_df.loc[~in_intersection, signal_col]
        summary.update(
            {
                "mean_signal_intersection_in_umap": selected_signal.mean(),
                "median_signal_intersection_in_umap": selected_signal.median(),
                "mean_signal_other_umap_regions": other_signal.mean(),
                "median_signal_other_umap_regions": other_signal.median(),
            }
        )

    exclusive_matched.insert(0, "method_a_condition", condition)
    return (
        summary,
        in_intersection,
        in_exclusive_intersection,
        exclusive_matched,
    )


# Visualisierung

def plot_condition_overlay(
    umap_df: pd.DataFrame,
    condition: str,
    in_intersection: np.ndarray,
    in_exclusive_intersection: np.ndarray,
    output_path: Path,
) -> None:
    """Zeigt Schnittmengenzugehörigkeit und Signalzuordnung im UMAP."""
    assigned_same = umap_df["umap_assigned_condition"].eq(condition).to_numpy()
    assignment_tie = umap_df["umap_assignment_is_tie"].to_numpy(dtype=bool)
    condition_among_max = (
        umap_df["umap_max_signal_conditions"]
        .str.split("|")
        .apply(lambda values: condition in values)
        .to_numpy(dtype=bool)
    )

    exclusive_same_mask = in_exclusive_intersection & assigned_same
    exclusive_tie_mask = (
        in_exclusive_intersection & assignment_tie & condition_among_max
    )
    exclusive_different_mask = (
        in_exclusive_intersection
        & ~exclusive_same_mask
        & ~exclusive_tie_mask
    )
    shared_mask = in_intersection & ~in_exclusive_intersection
    background_mask = ~in_intersection

    background = umap_df.loc[background_mask]
    shared = umap_df.loc[shared_mask]
    exclusive_same = umap_df.loc[exclusive_same_mask]
    exclusive_tie = umap_df.loc[exclusive_tie_mask]
    exclusive_different = umap_df.loc[exclusive_different_mask]
    signal_col = find_condition_signal_column(umap_df, condition)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=background["UMAP_1"],
            y=background["UMAP_2"],
            mode="markers",
            name="Nicht in der Schnittmenge",
            marker={"size": 4, "color": "lightgray", "opacity": 0.35},
            customdata=background[["start", "end"]].to_numpy(),
            hovertemplate=(
                "start=%{customdata[0]}<br>"
                "end=%{customdata[1]}<extra></extra>"
            ),
        )
    )

    def add_assignment_trace(
        subset: pd.DataFrame,
        name: str,
        color: str,
        symbol: str,
        size: int,
    ) -> None:
        if subset.empty:
            return

        hover_columns = [
            "start",
            "end",
            "umap_assigned_condition",
            "umap_max_signal_conditions",
        ]
        if signal_col is not None:
            hover_columns.append(signal_col)

        hovertemplate = (
            "start=%{customdata[0]}<br>"
            "end=%{customdata[1]}<br>"
            "UMAP-Zuordnung=%{customdata[2]}<br>"
            "Max-Signal-Condition(s)=%{customdata[3]}"
        )
        if signal_col is not None:
            hovertemplate += (
                f"<br>{signal_col}=%{{customdata[4]:.4f}}"
            )
        hovertemplate += "<extra></extra>"

        fig.add_trace(
            go.Scattergl(
                x=subset["UMAP_1"],
                y=subset["UMAP_2"],
                mode="markers",
                name=name,
                marker={
                    "size": size,
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.85,
                    "line": {"color": "black", "width": 0.5},
                },
                customdata=subset[hover_columns].to_numpy(),
                hovertemplate=hovertemplate,
            )
        )

    add_assignment_trace(
        shared,
        "Methode A: in mehreren Condition-Schnittmengen",
        "#BDBDBD",
        "circle",
        5,
    )
    add_assignment_trace(
        exclusive_same,
        f"Exklusiv {condition} + gleiches Max-Signal",
        CONDITION_COLORS.get(condition, "#2CA02C"),
        "circle",
        8,
    )
    add_assignment_trace(
        exclusive_tie,
        f"Exklusiv {condition} + Max-Signal-Gleichstand",
        "#FECB52",
        "diamond",
        9,
    )
    add_assignment_trace(
        exclusive_different,
        f"Exklusiv {condition} + andere Max-Signal-Condition",
        "#D62728",
        "x",
        9,
    )

    fig.update_layout(
        title={
            "text": (
                f"Regionen-UMAP: vollständige {condition}-Schnittmenge<br>"
                f"<sup>Exklusive Methode-A-Regionen: "
                f"{int(in_exclusive_intersection.sum())}; gleiche "
                f"Signalzuordnung: {int(exclusive_same_mask.sum())}</sup>"
            )
        },
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="simple_white",
        height=750,
        width=900,
        legend={"orientation": "h", "y": 1.08},
        hovermode="closest",
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


# Hauptablauf

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    umap_df = load_umap_regions(UMAP_REGIONS_CSV)
    umap_df = assign_umap_condition_by_max_signal(umap_df)
    umap_keys = coordinate_index(umap_df)

    # Alle Schnittmengen laden, um Exklusivität zu bestimmen.
    intersections: dict[str, pd.DataFrame] = {}
    intersection_paths: dict[str, Path] = {}
    intersection_sets: dict[str, set[tuple[int, int]]] = {}
    membership_columns: list[str] = []

    for condition in CONDITIONS:
        intersection_df, intersection_path = load_condition_intersection(
            condition
        )
        intersections[condition] = intersection_df
        intersection_paths[condition] = intersection_path
        intersection_sets[condition] = set(
            coordinate_index(intersection_df).tolist()
        )

        membership_col = f"in_{condition}_intersection"
        membership_columns.append(membership_col)
        condition_index = pd.MultiIndex.from_tuples(
            sorted(intersection_sets[condition]),
            names=["start", "end"],
        )
        umap_df[membership_col] = umap_keys.isin(condition_index)

    # Zugehörigkeit zu mehreren Schnittmengen erfassen.
    membership = umap_df[membership_columns]
    umap_df["n_condition_intersections"] = membership.sum(axis=1)
    condition_names = [
        column.removeprefix("in_").removesuffix("_intersection")
        for column in membership_columns
    ]
    membership_values = membership.to_numpy(dtype=bool)
    umap_df["intersection_conditions"] = [
        "|".join(
            condition
            for condition, is_member in zip(condition_names, row)
            if is_member
        )
        for row in membership_values
    ]

    # Exklusiv: nur in der Schnittmenge dieser Condition enthalten.
    exclusive_sets: dict[str, set[tuple[int, int]]] = {}
    for condition in CONDITIONS:
        other_union: set[tuple[int, int]] = set()
        for other_condition in CONDITIONS:
            if other_condition != condition:
                other_union.update(intersection_sets[other_condition])
        exclusive_sets[condition] = (
            intersection_sets[condition] - other_union
        )

    summary_rows: list[dict[str, object]] = []
    exclusive_assignment_frames: list[pd.DataFrame] = []

    for condition in CONDITIONS:
        summary, in_intersection, in_exclusive, exclusive_assignments = (
            compare_condition(
                condition=condition,
                intersection_df=intersections[condition],
                intersection_path=intersection_paths[condition],
                exclusive_key_set=exclusive_sets[condition],
                umap_df=umap_df,
                umap_keys=umap_keys,
            )
        )
        summary_rows.append(summary)
        exclusive_assignment_frames.append(exclusive_assignments)

        plot_path = (
            OUTPUT_DIR
            / condition
            / f"{condition}_exclusive_assignment_umap.html"
        )
        plot_condition_overlay(
            umap_df=umap_df,
            condition=condition,
            in_intersection=in_intersection,
            in_exclusive_intersection=in_exclusive,
            output_path=plot_path,
        )


    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_DIR / "same_condition_assignment_summary.csv",
        sep=SEP,
        index=False,
    )
    umap_df.to_csv(
        OUTPUT_DIR
        / "umap_regions_with_intersections_and_signal_assignment.csv",
        sep=SEP,
        index=False,
    )

    all_exclusive_assignments = pd.concat(
        exclusive_assignment_frames,
        ignore_index=True,
    )
    all_exclusive_assignments.to_csv(
        OUTPUT_DIR / "exclusive_method_a_umap_assignments.csv",
        sep=SEP,
        index=False,
    )

    confusion = pd.crosstab(
        all_exclusive_assignments["method_a_condition"],
        all_exclusive_assignments["umap_assigned_condition"],
        rownames=["method_a_exclusive_condition"],
        colnames=["umap_max_signal_condition"],
        dropna=False,
    ).reindex(
        index=CONDITIONS,
        columns=[*CONDITIONS, "TIE"],
        fill_value=0,
    )
    confusion.to_csv(
        OUTPUT_DIR / "exclusive_assignment_confusion_matrix.csv",
        sep=SEP,
        index=True,
    )

    distribution = (
        umap_df["n_condition_intersections"]
        .value_counts()
        .sort_index()
        .rename_axis("n_condition_intersections")
        .reset_index(name="n_umap_regions")
    )
    distribution.to_csv(
        OUTPUT_DIR / "intersection_membership_distribution.csv",
        sep=SEP,
        index=False,
    )

    print(f"Ergebnisse gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()