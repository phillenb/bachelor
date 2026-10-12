from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


TARGET_CONDITIONS = ("Ctrl", "URB1", "RPL3")
MIN_RAW_READS = 2
CONSENSUS_FILENAME = (
    f"intersection_all_samples_min_raw_reads_{MIN_RAW_READS}.csv"
)
SEP = ";"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an UpSet plot for Ctrl, URB1, and RPL3."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root containing data and src. By default it is detected "
            "from the location of this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: data/"
            "method1_set_based_and_rule_based_region_selection_data/"
            "upset_plots/ctrl_urb1_rpl3."
        ),
    )
    return parser.parse_args()


def find_project_root(script_path: Path) -> Path:
    script_path = script_path.resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate

    raise FileNotFoundError(
        "Projekt-Root nicht gefunden. Erwartet wird ein übergeordneter "
        "Ordner mit den Unterordnern 'data' und 'src'. Alternativ kann "
        "--project-root angegeben werden."
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else find_project_root(Path(__file__))
    )
    method1_data_dir = (
        project_root
        / "data"
        / "method1_set_based_and_rule_based_region_selection_data"
    )
    consensus_base_dir = method1_data_dir / "shared_regions_across_samples"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else method1_data_dir
        / "upset_plots"
        / "ctrl_urb1_rpl3"
    )
    return project_root, consensus_base_dir, output_dir


def load_region_set(csv_path: Path) -> set[tuple[int, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=SEP, usecols=["start", "end"])
    required = {"start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    invalid = df[["start", "end"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{csv_path} contains {int(invalid.sum())} invalid coordinate rows."
        )
    return set(zip(df["start"].astype(int), df["end"].astype(int)))


def load_target_sets(
    consensus_base_dir: Path,
) -> dict[str, set[tuple[int, int]]]:
    condition_sets: dict[str, set[tuple[int, int]]] = {}
    for condition in TARGET_CONDITIONS:
        path = consensus_base_dir / condition / CONSENSUS_FILENAME
        regions = load_region_set(path)
        condition_sets[condition] = regions
    return condition_sets


def build_region_membership_table(
    condition_sets: dict[str, set[tuple[int, int]]],
) -> pd.DataFrame:
    union = set().union(*condition_sets.values())
    rows: list[dict[str, object]] = []
    for start, end in union:
        membership = {
            condition: (start, end) in regions
            for condition, regions in condition_sets.items()
        }
        present_conditions = [
            condition for condition, present in membership.items() if present
        ]
        rows.append(
            {
                "start": start,
                "end": end,
                **membership,
                "n_conditions": len(present_conditions),
                "conditions": ",".join(present_conditions),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["n_conditions", "start", "end"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def build_membership_counts(membership_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        membership_df.groupby(list(TARGET_CONDITIONS), dropna=False)
        .size()
        .rename("intersection_size")
        .reset_index()
    )
    counts["n_conditions"] = counts[list(TARGET_CONDITIONS)].sum(axis=1)
    counts["conditions"] = counts.apply(
        lambda row: ",".join(
            condition for condition in TARGET_CONDITIONS if row[condition]
        ),
        axis=1,
    )
    return counts.sort_values(
        ["intersection_size", "n_conditions"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_upset(
    condition_sets: dict[str, set[tuple[int, int]]],
    output_dir: Path,
) -> None:
    from upsetplot import UpSet, from_contents

    data = from_contents(condition_sets)
    figure = plt.figure(figsize=(11, 7))
    UpSet(
        data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    ).plot(fig=figure)
    figure.suptitle(
        "Consensus-region overlap: Ctrl, URB1, and RPL3",
        fontsize=14,
        fontweight="bold",
    )
    output_path = output_dir / "upset_ctrl_urb1_rpl3_consensus_regions.svg"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root, consensus_base_dir, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_sets = load_target_sets(consensus_base_dir)
    membership = build_region_membership_table(condition_sets)
    membership_counts = build_membership_counts(membership)

    membership.to_csv(
        output_dir / "region_membership_ctrl_urb1_rpl3.csv",
        sep=SEP,
        index=False,
    )
    membership_counts.to_csv(
        output_dir / "membership_counts_ctrl_urb1_rpl3.csv",
        sep=SEP,
        index=False,
    )
    plot_upset(condition_sets, output_dir)
    print(f"Ergebnis gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()