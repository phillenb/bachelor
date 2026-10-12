from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


SEP = ";"
CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]
METHOD_ORDER = ["set_based", "umap", "logreg"]

SET_BASED_ROOT = (
    DATA_DIR
    / "method1_set_based_and_rule_based_region_selection_data"
    / "shared_regions_across_samples"
)
SET_BASED_FILENAME = "intersection_all_samples_min_raw_reads_2.csv"

UMAP_REGIONS_CSV = (
    DATA_DIR
    / "method2_unsupervised_selection_UMAP_PCA_PCoA_data"
    / "umap_across_samples_on_raw_data"
    / "umap_characteristic_regions.csv"
)

LOGREG_REGIONS_CSV = (
    DATA_DIR
    / "method3_supervised_LogReg_RandomForrest_data"
    / "logistic_regression_raw_stability_selection"
    / "raw_logreg_stable_cross_design_consensus.csv"
)

OUTPUT_DIR = DATA_DIR / "method_comparison" / "three_method_region_sets"


def read_region_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Regionsdatei nicht gefunden: {path}")
    df = pd.read_csv(path, sep=SEP)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: fehlende Spalten: {sorted(missing)}")
    try:
        df[["start", "end"]] = df[["start", "end"]].astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: start und end muessen ganzzahlig sein."
        ) from exc
    return df


def load_set_based_regions(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for condition in CONDITION_ORDER:
        path = root / condition / SET_BASED_FILENAME
        df = read_region_csv(path, {"start", "end"})
        df = df.drop_duplicates(["start", "end"]).copy()
        if "condition" in df.columns:
            reported = set(df["condition"].dropna().astype(str).unique())
            if reported and reported != {condition}:
                raise ValueError(
                    f"{path}: Condition-Spalte {sorted(reported)}, "
                    f"erwartet wurde '{condition}'."
                )
        frame = df[["start", "end"]].copy()
        frame["condition"] = condition
        frame["source_file"] = str(path)
        rows.append(frame)

    all_memberships = pd.concat(rows, ignore_index=True)
    membership_count = (
        all_memberships.groupby(["start", "end"])["condition"]
        .nunique()
        .rename("n_condition_intersections")
        .reset_index()
    )
    all_memberships = all_memberships.merge(
        membership_count,
        on=["start", "end"],
        how="left",
        validate="many_to_one",
    )
    exclusive = all_memberships.loc[
        all_memberships["n_condition_intersections"].eq(1)
    ].copy()
    exclusive.insert(0, "method", "set_based")
    return exclusive[
        [
            "method",
            "condition",
            "start",
            "end",
            "n_condition_intersections",
            "source_file",
        ]
    ].sort_values(["condition", "start", "end"]).reset_index(drop=True)


def load_umap_regions(path: Path) -> pd.DataFrame:
    df = read_region_csv(path, {"condition", "start", "end"})
    unknown = sorted(set(df["condition"].astype(str)) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"{path}: unbekannte Conditions: {unknown}")
    if df.duplicated(["condition", "start", "end"]).any():
        raise ValueError(f"{path}: doppelte Condition-/Regionszeilen.")
    if "selected_as_umap_characteristic" in df.columns:
        selected = (
            df["selected_as_umap_characteristic"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1"})
        )
        if not selected.all():
            raise ValueError(
                f"{path}: enthaelt nicht-final selektierte UMAP-Regionen."
            )

    out = df.copy()
    if "method" in out.columns:
        out = out.rename(columns={"method": "selection_method"})
    out.insert(0, "method", "umap")
    out["source_file"] = str(path)
    return out.sort_values(["condition", "start", "end"]).reset_index(drop=True)


def load_logreg_regions(path: Path) -> pd.DataFrame:
    df = read_region_csv(path, {"condition", "start", "end"})
    unknown = sorted(set(df["condition"].astype(str)) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"{path}: unbekannte Conditions: {unknown}")
    if df.duplicated(["condition", "start", "end"]).any():
        raise ValueError(f"{path}: doppelte Condition-/Regionszeilen.")

    out = df.copy()
    out.insert(0, "method", "logreg")
    out["source_file"] = str(path)
    preferred = ["method", "condition", "start", "end"]
    remaining = [column for column in out.columns if column not in preferred]
    return out[preferred + remaining].sort_values(
        ["condition", "start", "end"]
    ).reset_index(drop=True)


def aggregate_method_coordinates(
    frame: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    order = {condition: index for index, condition in enumerate(CONDITION_ORDER)}

    def join_conditions(values: pd.Series) -> str:
        unique = sorted(set(values.astype(str)), key=order.__getitem__)
        return "|".join(unique)

    return (
        frame.groupby(["start", "end"], as_index=False)
        .agg(**{f"{method}_conditions": ("condition", join_conditions)})
    )


def build_coordinate_membership(
    set_based: pd.DataFrame,
    umap: pd.DataFrame,
    logreg: pd.DataFrame,
) -> pd.DataFrame:
    by_method = {
        "set_based": set_based,
        "umap": umap,
        "logreg": logreg,
    }
    union: pd.DataFrame | None = None
    for method in METHOD_ORDER:
        aggregated = aggregate_method_coordinates(by_method[method], method)
        union = (
            aggregated
            if union is None
            else union.merge(
                aggregated,
                on=["start", "end"],
                how="outer",
                validate="one_to_one",
            )
        )
    if union is None:
        raise AssertionError("Keine Methoden verarbeitet.")

    condition_columns = [f"{method}_conditions" for method in METHOD_ORDER]
    for method, condition_column in zip(METHOD_ORDER, condition_columns):
        union[condition_column] = union[condition_column].fillna("")
        union[f"in_{method}"] = union[condition_column].ne("")

    membership_columns = [f"in_{method}" for method in METHOD_ORDER]
    union["n_methods"] = union[membership_columns].sum(axis=1)

    def agreement(row: pd.Series) -> tuple[bool, str]:
        present = [row[column] for column in condition_columns if row[column]]
        agree = len(set(present)) == 1
        return agree, present[0] if agree else ""

    agreement_values = union.apply(agreement, axis=1)
    union["condition_agreement_across_present_methods"] = [
        value[0] for value in agreement_values
    ]
    union["agreed_condition"] = [value[1] for value in agreement_values]

    return union.sort_values(["start", "end"]).reset_index(drop=True)


def build_condition_membership(
    set_based: pd.DataFrame,
    umap: pd.DataFrame,
    logreg: pd.DataFrame,
) -> pd.DataFrame:
    core_frames = []
    for method, frame in [
        ("set_based", set_based),
        ("umap", umap),
        ("logreg", logreg),
    ]:
        core = frame[["condition", "start", "end"]].drop_duplicates().copy()
        core[f"in_{method}"] = True
        core_frames.append(core)

    membership = core_frames[0]
    for core in core_frames[1:]:
        membership = membership.merge(
            core,
            on=["condition", "start", "end"],
            how="outer",
            validate="one_to_one",
        )
    method_columns = [f"in_{method}" for method in METHOD_ORDER]
    membership[method_columns] = membership[method_columns].notna()
    membership["n_methods"] = membership[method_columns].sum(axis=1)
    return membership.sort_values(
        ["condition", "start", "end"]
    ).reset_index(drop=True)


def build_overlap_summary(
    coordinate_membership: pd.DataFrame,
    condition_membership: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for scope, membership in [
        ("coordinate", coordinate_membership),
        ("condition_start_end", condition_membership),
    ]:
        for method in METHOD_ORDER:
            rows.append(
                {
                    "scope": scope,
                    "comparison": method,
                    "metric": "n_regions",
                    "value": int(membership[f"in_{method}"].sum()),
                }
            )

        for left_index, left in enumerate(METHOD_ORDER):
            for right in METHOD_ORDER[left_index + 1 :]:
                left_mask = membership[f"in_{left}"]
                right_mask = membership[f"in_{right}"]
                intersection = int((left_mask & right_mask).sum())
                union_count = int((left_mask | right_mask).sum())
                rows.extend(
                    [
                        {
                            "scope": scope,
                            "comparison": f"{left}__{right}",
                            "metric": "intersection",
                            "value": intersection,
                        },
                        {
                            "scope": scope,
                            "comparison": f"{left}__{right}",
                            "metric": "union",
                            "value": union_count,
                        },
                        {
                            "scope": scope,
                            "comparison": f"{left}__{right}",
                            "metric": "jaccard",
                            "value": (
                                intersection / union_count if union_count else np.nan
                            ),
                        },
                    ]
                )

        rows.extend(
            [
                {
                    "scope": scope,
                    "comparison": "all_three",
                    "metric": "intersection",
                    "value": int(membership["n_methods"].eq(3).sum()),
                },
                {
                    "scope": scope,
                    "comparison": "all_methods",
                    "metric": "union",
                    "value": int(len(membership)),
                },
            ]
        )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    set_based: pd.DataFrame,
    umap: pd.DataFrame,
    logreg: pd.DataFrame,
    coordinate_membership: pd.DataFrame,
    condition_membership: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    set_based.to_csv(
        output_dir / "set_based_regions.csv",
        sep=SEP,
        index=False,
    )
    umap.to_csv(
        output_dir / "umap_regions_standardized.csv",
        sep=SEP,
        index=False,
    )
    logreg.to_csv(
        output_dir / "logreg_regions.csv",
        sep=SEP,
        index=False,
    )

    coordinate_membership.to_csv(
        output_dir / "method_union_regions.csv",
        sep=SEP,
        index=False,
    )

    coordinate_membership.loc[
        coordinate_membership["n_methods"].eq(3)
    ].to_csv(
        output_dir / "method_intersection_all_three_coordinates.csv",
        sep=SEP,
        index=False,
    )

    coordinate_membership.loc[
        coordinate_membership["n_methods"].eq(3)
        & coordinate_membership[
            "condition_agreement_across_present_methods"
        ]
    ].to_csv(
        output_dir / "method_intersection_all_three_same_condition.csv",
        sep=SEP,
        index=False,
    )

    coordinate_membership.loc[
        coordinate_membership["n_methods"].ge(2)
    ].to_csv(
        output_dir / "method_regions_found_by_at_least_two_methods.csv",
        sep=SEP,
        index=False,
    )

    condition_membership.to_csv(
        output_dir / "method_membership_by_condition.csv",
        sep=SEP,
        index=False,
    )

    condition_membership.loc[
        condition_membership["n_methods"].ge(2)
    ].to_csv(
        output_dir
        / "method_regions_found_by_at_least_two_methods_same_condition.csv",
        sep=SEP,
        index=False,
    )

    build_overlap_summary(
        coordinate_membership,
        condition_membership,
    ).to_csv(
        output_dir / "method_overlap_summary.csv",
        sep=SEP,
        index=False,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-based-root", type=Path, default=SET_BASED_ROOT)
    parser.add_argument("--umap-regions-csv", type=Path, default=UMAP_REGIONS_CSV)
    parser.add_argument(
        "--logreg-regions-csv", type=Path, default=LOGREG_REGIONS_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_based = load_set_based_regions(args.set_based_root)

    umap = load_umap_regions(args.umap_regions_csv)

    logreg = load_logreg_regions(args.logreg_regions_csv)

    coordinate_membership = build_coordinate_membership(
        set_based,
        umap,
        logreg,
    )
    condition_membership = build_condition_membership(
        set_based,
        umap,
        logreg,
    )
    save_outputs(
        args.output_dir,
        set_based,
        umap,
        logreg,
        coordinate_membership,
        condition_membership,
    )

    coordinate_intersection = coordinate_membership["n_methods"].eq(3)
    same_condition = (
        coordinate_intersection
        & coordinate_membership[
            "condition_agreement_across_present_methods"
        ]
    )
    at_least_two_same_condition = condition_membership["n_methods"].ge(2)

    print(f"Ergebnis gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()