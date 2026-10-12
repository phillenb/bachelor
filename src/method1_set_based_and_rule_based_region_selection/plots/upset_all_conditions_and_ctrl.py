from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent

    raise FileNotFoundError(
        "Projekt-Root nicht gefunden. Erwartet wird ein übergeordneter Ordner "
        "mit den Unterordnern 'data' und 'src'."
    )


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(SCRIPT_PATH)
DATA_DIR = PROJECT_ROOT / "data"


SEP = ";"
MIN_RAW_READS = 2

CONDITIONS = (
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
)

METHOD1_DATA_DIR = (
    DATA_DIR / "method1_set_based_and_rule_based_region_selection_data"
)
CONSENSUS_BASE_DIR = METHOD1_DATA_DIR / "shared_regions_across_samples"
CONSENSUS_FILENAME = (
    f"intersection_all_samples_min_raw_reads_{MIN_RAW_READS}.csv"
)

CONSENSUS_CSV_FILES: dict[str, Path] = {
    condition: CONSENSUS_BASE_DIR / condition / CONSENSUS_FILENAME
    for condition in CONDITIONS
}

OUTPUT_DIR = (
    METHOD1_DATA_DIR / "upset_plots" / "consensus_comparison_all_conditions_with_ctrl"
)


def validate_input_paths() -> None:

    missing: list[Path] = []
    for condition, path in CONSENSUS_CSV_FILES.items():
        exists = path.is_file()
        status = "OK" if exists else "MISSING"
        if not exists:
            missing.append(path)

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Nicht alle sieben Consensus-Dateien wurden gefunden:\n"
            f"{missing_text}\n"
            "Führe zuerst intersection_of_all_samples_of_one_condition_2.py "
            "mit MIN_RAW_READS = 2 aus."
        )


def load_consensus_set(csv_path: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(csv_path, sep=SEP)

    required_columns = {"start", "end"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Fehlende Spalten in {csv_path}: {sorted(missing_columns)}"
        )

    coordinates = df.loc[:, ["start", "end"]].copy()
    coordinates["start"] = pd.to_numeric(
        coordinates["start"], errors="coerce"
    )
    coordinates["end"] = pd.to_numeric(
        coordinates["end"], errors="coerce"
    )

    invalid_rows = coordinates[["start", "end"]].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"{csv_path} enthält {int(invalid_rows.sum())} Zeilen mit "
            "ungültigen start/end-Koordinaten."
        )

    return set(
        zip(
            coordinates["start"].astype(int),
            coordinates["end"].astype(int),
        )
    )


def load_all_conditions() -> dict[str, set[tuple[int, int]]]:
    condition_sets: dict[str, set[tuple[int, int]]] = {}

    for condition, path in CONSENSUS_CSV_FILES.items():
        regions = load_consensus_set(path)
        condition_sets[condition] = regions

    return condition_sets


def plot_upset(
    condition_sets: dict[str, set[tuple[int, int]]],
    filename_stem: str,
    title: str,
) -> None:
    from upsetplot import UpSet, from_contents

    data = from_contents(condition_sets)
    figure = plt.figure(figsize=(14, 7))

    UpSet(
        data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    ).plot(fig=figure)

    plt.suptitle(title, fontsize=13, fontweight="bold")

    output_path = OUTPUT_DIR / f"{filename_stem}.svg"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(figure)






def save_regions_by_condition_count(
    condition_sets: dict[str, set[tuple[int, int]]],
) -> None:
    counter: Counter[tuple[int, int]] = Counter()
    region_to_conditions: dict[tuple[int, int], list[str]] = {}

    for condition, regions in condition_sets.items():
        for region in regions:
            counter[region] += 1
            region_to_conditions.setdefault(region, []).append(condition)

    number_of_conditions = len(condition_sets)

    for minimum_count in sorted({2, 3, number_of_conditions}):
        rows = [
            {
                "start": region[0],
                "end": region[1],
                "n_conditions": count,
                "conditions": ",".join(
                    sorted(region_to_conditions[region])
                ),
            }
            for region, count in counter.items()
            if count >= minimum_count
        ]

        label = (
            "all"
            if minimum_count == number_of_conditions
            else f"min{minimum_count}"
        )
        output_path = OUTPUT_DIR / f"regions_in_{label}_conditions.csv"

        output_df = pd.DataFrame(
            rows,
            columns=["start", "end", "n_conditions", "conditions"],
        )
        if not output_df.empty:
            output_df = output_df.sort_values(
                ["n_conditions", "start", "end"],
                ascending=[False, True, True],
            ).reset_index(drop=True)

        output_df.to_csv(output_path, sep=SEP, index=False)


def main() -> None:

    validate_input_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    condition_sets = load_all_conditions()

    plot_upset(
        condition_sets,
        filename_stem="upset_all_conditions_incl_ctrl",
        title="Consensus regions across all conditions (including Ctrl)",
    )




    save_regions_by_condition_count(condition_sets)


    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()