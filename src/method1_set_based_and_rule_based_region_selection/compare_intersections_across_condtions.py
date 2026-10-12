from pathlib import Path
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CONSENSUS_BASE_DIR = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/"
    "method1_set_based_and_rule_based_region_selection_data/"
    "shared_regions_across_samples"
)
CONSENSUS_FILENAME = "intersection_all_samples_min_raw_reads_2.csv"
CTRL_CONSENSUS_FILE = CONSENSUS_BASE_DIR / "Ctrl" / CONSENSUS_FILENAME

KD_CONSENSUS_FILES = {
    "DIMT1L": CONSENSUS_BASE_DIR / "DIMT1L" / CONSENSUS_FILENAME,
    "UTP18": CONSENSUS_BASE_DIR / "UTP18" / CONSENSUS_FILENAME,
    "LAS1L": CONSENSUS_BASE_DIR / "LAS1L" / CONSENSUS_FILENAME,
    "RPL3": CONSENSUS_BASE_DIR / "RPL3" / CONSENSUS_FILENAME,
    "URB1": CONSENSUS_BASE_DIR / "URB1" / CONSENSUS_FILENAME,
    "WBSCR22": CONSENSUS_BASE_DIR / "WBSCR22" / CONSENSUS_FILENAME,
}

SEP = ";"
OUTPUT_DIR = CONSENSUS_BASE_DIR.parent / "cross_condition_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_region_set(path: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(path, sep=SEP)
    required_columns = {"start", "end"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Fehlende Spalten in {path}: {sorted(missing_columns)}"
        )

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    invalid_rows = df[["start", "end"]].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"{path} enthält {int(invalid_rows.sum())} Zeilen mit "
            "ungültigen start/end-Koordinaten."
        )

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return set(zip(df["start"], df["end"]))


def load_all_conditions() -> dict[str, set[tuple[int, int]]]:
    required_files = {"Ctrl": CTRL_CONSENSUS_FILE, **KD_CONSENSUS_FILES}
    missing_files = [
        (name, path)
        for name, path in required_files.items()
        if not path.is_file()
    ]
    if missing_files:
        missing_text = "\n".join(
            f"    {name}: {path}" for name, path in missing_files
        )
        raise FileNotFoundError(
            "Nicht alle KD_only-Dateien wurden gefunden:\n"
            f"{missing_text}\n\n"
            "Die gemeinsame Schnittmenge wurde deshalb nicht berechnet."
        )

    ctrl_consensus = load_region_set(CTRL_CONSENSUS_FILE)
    kd_only_sets: dict[str, set[tuple[int, int]]] = {}
    for name, path in KD_CONSENSUS_FILES.items():
        kd_consensus = load_region_set(path)
        shared_with_ctrl = kd_consensus & ctrl_consensus
        kd_only = kd_consensus - ctrl_consensus
        kd_only_sets[name] = kd_only
    return kd_only_sets


def pairwise_comparison(condition_sets: dict[str, set]) -> pd.DataFrame:
    rows = []
    for (name_a, set_a), (name_b, set_b) in combinations(condition_sets.items(), 2):
        intersection = set_a & set_b
        union = set_a | set_b
        jaccard = len(intersection) / len(union) if union else 0.0
        rows.append({
            "condition_a": name_a,
            "condition_b": name_b,
            "n_a": len(set_a),
            "n_b": len(set_b),
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": jaccard,
            "only_in_a": len(set_a - set_b),
            "only_in_b": len(set_b - set_a),
        })
    return pd.DataFrame(rows)


def shared_across_all(condition_sets: dict[str, set]) -> set:
    sets = list(condition_sets.values())
    if not sets:
        return set()
    return set.intersection(*sets)


def shared_in_at_least_n(condition_sets: dict[str, set], n: int) -> pd.DataFrame:
    from collections import Counter

    counter: Counter = Counter()
    region_to_conditions: dict[tuple, list] = {}
    for cond_name, regions in condition_sets.items():
        for region in regions:
            counter[region] += 1
            region_to_conditions.setdefault(region, []).append(cond_name)

    rows = []
    for region, count in counter.items():
        if count >= n:
            rows.append({
                "start": region[0],
                "end": region[1],
                "n_conditions": count,
                "conditions": ",".join(sorted(region_to_conditions[region])),
            })
    if not rows:
        return pd.DataFrame(columns=["start", "end", "n_conditions", "conditions"])

    return (
        pd.DataFrame(rows)
        .sort_values(["n_conditions", "start"], ascending=[False, True])
        .reset_index(drop=True)
    )


def export_global_kd_intersection(
    condition_sets: dict[str, set[tuple[int, int]]]
) -> pd.DataFrame:
    all_kd_intersection = set.intersection(*condition_sets.values())
    intersection_df = pd.DataFrame(
        sorted(all_kd_intersection), columns=["start", "end"]
    )
    intersection_df["n_conditions"] = len(condition_sets)
    intersection_df["conditions"] = ",".join(condition_sets.keys())
    out_path = OUTPUT_DIR / "all_KD_only_intersection.csv"
    intersection_df.to_csv(out_path, sep=SEP, index=False)
    return intersection_df


def plot_upset(condition_sets: dict[str, set]):
    from upsetplot import from_contents, UpSet

    data = from_contents(condition_sets)
    fig = plt.figure(figsize=(12, 6))
    UpSet(
        data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    ).plot(fig=fig)
    plt.suptitle(
        "Overlap of condition-specific regions",
        fontsize=13,
        fontweight="bold",
    )
    out_path = OUTPUT_DIR / "upset_condition_specific_regions.svg"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_heatmap(pairwise_df: pd.DataFrame, condition_names: list[str]):
    import numpy as np

    n = len(condition_names)
    idx = {name: i for i, name in enumerate(condition_names)}
    matrix = np.zeros((n, n))
    for _, row in pairwise_df.iterrows():
        i = idx[row["condition_a"]]
        j = idx[row["condition_b"]]
        matrix[i, j] = row["jaccard"]
        matrix[j, i] = row["jaccard"]
    for i in range(n):
        matrix[i, i] = 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Jaccard-Index")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(condition_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(condition_names, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black",
            )
    ax.set_title("Pairwise Jaccard Index\n(condition-specific regions)", fontsize=12)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "jaccard_heatmap_condition_specific.svg"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    condition_sets = load_all_conditions()
    if len(condition_sets) < 2:
        return

    pairwise_df = pairwise_comparison(condition_sets)
    pairwise_df.to_csv(OUTPUT_DIR / "pairwise_comparison.csv", sep=SEP, index=False)
    for min_n in [2, 3, len(condition_sets)]:
        if min_n > len(condition_sets):
            continue
        shared_df = shared_in_at_least_n(condition_sets, min_n)
        label = "allen" if min_n == len(condition_sets) else f"mind_{min_n}"
        out_csv = OUTPUT_DIR / f"regions_in_{label}_conditions.csv"
        shared_df.to_csv(out_csv, sep=SEP, index=False)

    plot_upset(condition_sets)
    plot_jaccard_heatmap(pairwise_df, list(condition_sets.keys()))
    export_global_kd_intersection(condition_sets)
    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()