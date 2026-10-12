from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/"
    "method2_unsupervised_selection_UMAP_PCA_PCoA_data/umap_regions/"
    "umap_regions.csv"
)

EXPECTED_SIGNAL_COLUMNS = [
    "Ctrl_signal",
    "UTP18_signal",
    "DIMT1L_signal",
    "LAS1L_signal",
    "RPL3_signal",
    "URB1_signal",
    "WBSCR22_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select regions that have a positive signal in exactly one of "
            "the seven experimental conditions."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input umap_regions.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. Default: regions_exclusive_to_one_condition.csv "
            "next to the input file."
        ),
    )
    parser.add_argument(
        "--minimum-signal",
        type=float,
        default=0.0,
        help=(
            "A condition counts as having signal when its value is strictly "
            "greater than this threshold (default: 0.0)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_csv.parent / "regions_exclusive_to_one_condition.csv"
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv, sep=";")
    missing = [col for col in EXPECTED_SIGNAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The following expected signal columns are missing: "
            + ", ".join(missing)
        )

    signal_df = df[EXPECTED_SIGNAL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if signal_df.isna().any().any():
        invalid_counts = signal_df.isna().sum()
        invalid_counts = invalid_counts[invalid_counts > 0]
        raise ValueError(
            "Non-numeric or missing values were found in signal columns:\n"
            + invalid_counts.to_string()
        )

    has_signal = signal_df.gt(args.minimum_signal)
    exclusive_mask = has_signal.sum(axis=1).eq(1)
    exclusive = df.loc[exclusive_mask].copy()
    exclusive_has_signal = has_signal.loc[exclusive_mask]

    exclusive["exclusive_condition"] = (
        exclusive_has_signal.idxmax(axis=1).str.removesuffix("_signal")
    )
    exclusive["exclusive_signal"] = signal_df.loc[exclusive_mask].max(axis=1)
    exclusive["n_conditions_with_signal"] = 1

    preferred_columns = [
        col
        for col in ["start", "end", "exclusive_condition", "exclusive_signal"]
        if col in exclusive.columns
    ]
    remaining_columns = [
        col for col in exclusive.columns if col not in preferred_columns
    ]
    exclusive = exclusive[preferred_columns + remaining_columns]
    exclusive = exclusive.sort_values(
        ["exclusive_condition", "exclusive_signal"],
        ascending=[True, False],
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exclusive.to_csv(output_csv, sep=";", index=False)

    summary = (
        exclusive.groupby("exclusive_condition", as_index=False)
        .agg(
            n_exclusive_regions=("exclusive_condition", "size"),
            mean_exclusive_signal=("exclusive_signal", "mean"),
            median_exclusive_signal=("exclusive_signal", "median"),
            max_exclusive_signal=("exclusive_signal", "max"),
        )
        .sort_values("exclusive_condition")
    )
    summary_csv = output_csv.with_name(
        f"{output_csv.stem}_summary{output_csv.suffix}"
    )
    summary.to_csv(summary_csv, sep=";", index=False)
    print(f"Ergebnis gespeichert in: {output_csv.parent}")


if __name__ == "__main__":
    main()