from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEP = ";"
DEFAULT_PROJECT_ROOT = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis"
)

CONDITION_FOLDERS: dict[str, str] = {
    "Ctrl": "20240314_Ribolyzer_Ctrl",
    "UTP18": "20240314_Ribolyzer_UTP18",
    "DIMT1L": "20250129_Ribolyzer_DIMT1L",
    "LAS1L": "Las1L_NanoRibolyzer",
    "RPL3": "RPL3_NanoRibolyzer",
    "URB1": "URB1_NanoRibolyzer",
    "WBSCR22": "WBSCR22_NanoRibolyzer",
}

CONDITIONS: tuple[str, ...] = tuple(CONDITION_FOLDERS)
INTERSECTION_FILENAME = "intersection_all_samples_min_raw_reads_2.csv"

COLORS: dict[str, str] = {
    "Ctrl": "#7f7f7f",
    "UTP18": "#1f77b4",
    "DIMT1L": "#ff7f0e",
    "LAS1L": "#2ca02c",
    "RPL3": "#d62728",
    "URB1": "#9467bd",
    "WBSCR22": "#8c564b",
}


Region = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PCA auf der Union der Methode-1-Intersections sowie eine "
            "Jaccard-PCoA aller Rohdaten-Replikate."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Projektwurzel. Standard: automatische Suche, danach "
            f"{DEFAULT_PROJECT_ROOT}"
        ),
    )
    parser.add_argument(
        "--intersection-dir",
        type=Path,
        default=None,
        help=(
            "Ordner shared_regions_across_samples mit einem Unterordner pro "
            "Condition."
        ),
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Optionaler data/raw-Ordner.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optionaler Ausgabeordner.",
    )
    parser.add_argument(
        "--presence-cpm",
        type=float,
        default=1.0,
        help=(
            "CPM-Grenze fuer die binaere Anwesenheit einer Region in der "
            "Jaccard-PCoA (Standard: 1.0)."
        ),
    )
    parser.add_argument(
        "--scale-features",
        action="store_true",
        help=(
            "Z-standardisiert jede Region vor der PCA. Standardmaessig wird "
            "log2(CPM+1) nur zentriert; das ist fuer Intensitaetsdaten meist "
            "leichter interpretierbar."
        ),
    )
    parser.add_argument(
        "--top-loadings",
        type=int,
        default=30,
        help="Anzahl exportierter Top-Regionen pro PCA-Achse (Standard: 30).",
    )
    return parser.parse_args()


def find_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Projektwurzel existiert nicht: {root}")
        return root

    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "data" / "raw").is_dir():
            return candidate

    if DEFAULT_PROJECT_ROOT.is_dir():
        return DEFAULT_PROJECT_ROOT

    raise RuntimeError(
        "Projektwurzel nicht gefunden. Bitte mit --project-root angeben."
    )


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline()
    return ";" if header.count(";") >= header.count(",") else ","


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_separator(path))


def normalize_regions(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{source} enthaelt nicht die erforderlichen Spalten: {sorted(missing)}"
        )

    regions = df[["start", "end"]].copy()
    regions["start"] = pd.to_numeric(regions["start"], errors="coerce")
    regions["end"] = pd.to_numeric(regions["end"], errors="coerce")
    regions = regions.dropna().astype({"start": int, "end": int})
    regions = regions.loc[regions["end"] > regions["start"]]
    return regions.drop_duplicates().sort_values(["start", "end"])


def find_intersection_file(intersection_dir: Path, condition: str) -> Path:
    path = intersection_dir / condition / INTERSECTION_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Intersection-Datei fehlt fuer {condition}: {path}"
        )
    return path


def load_intersection_sets(
    intersection_dir: Path,
) -> tuple[dict[str, set[Region]], dict[str, Path]]:
    region_sets: dict[str, set[Region]] = {}
    paths: dict[str, Path] = {}

    for condition in CONDITIONS:
        path = find_intersection_file(intersection_dir, condition)
        regions_df = normalize_regions(read_table(path), path)
        regions = set(map(tuple, regions_df[["start", "end"]].to_numpy()))
        region_sets[condition] = {
            (int(start), int(end)) for start, end in regions
        }
        paths[condition] = path

    if not any(region_sets.values()):
        raise RuntimeError("Alle Intersection-Regionsmengen sind leer.")
    return region_sets, paths


def membership_table(region_sets: dict[str, set[Region]]) -> pd.DataFrame:
    union = sorted(set().union(*region_sets.values()))
    rows: list[dict[str, object]] = []
    for start, end in union:
        memberships = [c for c in CONDITIONS if (start, end) in region_sets[c]]
        row: dict[str, object] = {
            "start": start,
            "end": end,
            "n_conditions": len(memberships),
            "conditions": "+".join(memberships),
        }
        row.update({c: (c in memberships) for c in CONDITIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def jaccard_matrices(
    region_sets: dict[str, set[Region]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = list(region_sets)
    similarity = pd.DataFrame(0.0, index=labels, columns=labels)
    pairwise_rows: list[dict[str, object]] = []

    for i, kd_a in enumerate(labels):
        for j, kd_b in enumerate(labels):
            a = region_sets[kd_a]
            b = region_sets[kd_b]
            union_n = len(a | b)
            intersection_n = len(a & b)
            value = intersection_n / union_n if union_n else 1.0
            similarity.loc[kd_a, kd_b] = value
            if j > i:
                pairwise_rows.append(
                    {
                        "sample_a": kd_a,
                        "sample_b": kd_b,
                        "n_a": len(a),
                        "n_b": len(b),
                        "intersection": intersection_n,
                        "union": union_n,
                        "jaccard_similarity": value,
                        "jaccard_distance": 1.0 - value,
                    }
                )

    distance = 1.0 - similarity
    np.fill_diagonal(distance.values, 0.0)
    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        "jaccard_similarity", ascending=False
    )
    return similarity, distance, pairwise


def classical_pcoa(
    distance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = distance.to_numpy(dtype=float)
    n = d.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    tolerance = max(1.0, abs(eigenvalues[0])) * 1e-12
    positive = eigenvalues > tolerance
    positive_values = eigenvalues[positive]
    positive_vectors = eigenvectors[:, positive]
    coordinates = positive_vectors * np.sqrt(positive_values)

    coord_columns = [f"PCoA{i + 1}" for i in range(coordinates.shape[1])]
    coord_df = pd.DataFrame(
        coordinates, index=distance.index, columns=coord_columns
    )
    coord_df.index.name = "sample_id"

    positive_total = positive_values.sum()
    variance_positive = np.where(
        positive,
        eigenvalues / positive_total if positive_total > 0 else np.nan,
        np.nan,
    )
    eig_df = pd.DataFrame(
        {
            "axis": [f"Axis{i + 1}" for i in range(len(eigenvalues))],
            "eigenvalue": eigenvalues,
            "is_positive": positive,
            "fraction_of_positive_inertia": variance_positive,
        }
    )
    return coord_df, eig_df


def plot_jaccard_heatmap(similarity: pd.DataFrame, output_path: Path) -> None:
    size = max(8.2, min(14.0, 0.48 * len(similarity)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(similarity.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    labels = similarity.index.tolist()
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    if len(labels) <= 10:
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = similarity.iat[row, col]
                color = "white" if value < 0.55 else "black"
                ax.text(
                    col, row, f"{value:.2f}", ha="center", va="center", color=color
                )

    ax.set_title("Jaccard-Aehnlichkeit der Rohdaten-Replikate")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard-Aehnlichkeit")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pcoa(
    coordinates: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    if coordinates.shape[1] < 2:
        print("    WARNUNG: PCoA besitzt weniger als zwei positive Achsen; kein Plot.")
        return

    explained = eigenvalues.loc[
        eigenvalues["is_positive"], "fraction_of_positive_inertia"
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    plot_data = metadata[["condition", "replicate"]].join(coordinates, how="inner")
    for condition in CONDITIONS:
        subset = plot_data.loc[plot_data["condition"] == condition]
        ax.scatter(
            subset["PCoA1"],
            subset["PCoA2"],
            s=90,
            color=COLORS[condition],
            edgecolor="black",
            linewidth=0.8,
            label=condition,
            zorder=3,
        )
        for sample_id, row in subset.iterrows():
            ax.annotate(
                f"R{int(row['replicate'])}",
                (row["PCoA1"], row["PCoA2"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel(f"PCoA1 ({explained[0] * 100:.1f}% positive Inertia)")
    ax.set_ylabel(f"PCoA2 ({explained[1] * 100:.1f}% positive Inertia)")
    ax.set_title("PCoA auf Jaccard-Distanzen der Rohdaten-Replikate")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_raw_files(raw_data_dir: Path, condition: str) -> list[Path]:
    folder = raw_data_dir / CONDITION_FOLDERS[condition]
    if not folder.is_dir():
        raise FileNotFoundError(f"Rohdatenordner fehlt fuer {condition}: {folder}")

    paths = sorted(
        folder.glob("*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv")
    )
    if not paths:
        paths = sorted(
            path
            for path in folder.rglob("intensity_matrix.csv")
            if "ivpa" in path.parent.parent.name.lower()
            and "nuc" in path.parent.parent.name.lower()
        )
    if not paths:
        raise FileNotFoundError(
            f"Keine intensity_matrix.csv-Replikate fuer {condition} unter {folder}."
        )
    return paths


def load_raw_sample(path: Path) -> pd.Series:
    df = read_table(path)
    required = {"start", "end", "n_reads"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} fehlen Spalten: {sorted(missing)}")

    work = df[["start", "end", "n_reads"]].copy()
    for column in ("start", "end", "n_reads"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna()
    work[["start", "end"]] = work[["start", "end"]].astype(int)
    return work.groupby(["start", "end"])["n_reads"].sum()


def build_raw_matrix(raw_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    series: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, object]] = []

    for condition in CONDITION_FOLDERS:
        paths = find_raw_files(raw_data_dir, condition)
        for replicate, path in enumerate(paths, start=1):
            sample_id = path.parent.parent.name
            if sample_id in series:
                raise RuntimeError(f"Doppelte Sample-ID: {sample_id}")
            series[sample_id] = load_raw_sample(path)
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "replicate": replicate,
                    "source_file": str(path),
                }
            )

    raw = pd.DataFrame(series).fillna(0.0)
    raw.index = pd.MultiIndex.from_tuples(raw.index, names=["start", "end"])
    metadata = pd.DataFrame(metadata_rows).set_index("sample_id")
    return raw, metadata


def counts_to_cpm(raw: pd.DataFrame) -> pd.DataFrame:
    library_sizes = raw.sum(axis=0)
    invalid = library_sizes[library_sizes <= 0]
    if not invalid.empty:
        raise ValueError(
            "Samples ohne Reads: " + ", ".join(map(str, invalid.index.tolist()))
        )
    return raw.divide(library_sizes, axis=1) * 1_000_000.0


def raw_sample_region_sets(
    raw: pd.DataFrame, presence_cpm: float
) -> dict[str, set[Region]]:
    if presence_cpm < 0:
        raise ValueError("--presence-cpm darf nicht negativ sein.")

    cpm = counts_to_cpm(raw)
    region_sets: dict[str, set[Region]] = {}
    for sample_id in cpm.columns:
        present = cpm.index[cpm[sample_id] >= presence_cpm]
        region_sets[str(sample_id)] = {
            (int(start), int(end)) for start, end in present.to_list()
        }
    if not all(region_sets.values()):
        empty = [sample for sample, regions in region_sets.items() if not regions]
        raise RuntimeError(
            "Keine praesentierten Regionen bei der gewaehlten CPM-Grenze fuer: "
            + ", ".join(empty)
        )
    return region_sets


def fit_pca(
    matrix_samples_by_features: pd.DataFrame,
    scale_features: bool,
    max_components: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = matrix_samples_by_features.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)

    if scale_features:
        standard_deviation = x.std(axis=0, ddof=1)
        if np.any(standard_deviation <= 0):
            raise ValueError("PCA-Eingabe enthaelt Features ohne Varianz.")
        x = x / standard_deviation

    u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    explained_variance = singular_values**2 / (x.shape[0] - 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance

    n_components = min(max_components, len(singular_values))
    component_names = [f"PC{i + 1}" for i in range(n_components)]

    scores = pd.DataFrame(
        u[:, :n_components] * singular_values[:n_components],
        index=matrix_samples_by_features.index,
        columns=component_names,
    )
    weights = pd.DataFrame(
        vt[:n_components, :].T,
        index=matrix_samples_by_features.columns,
        columns=[f"{pc}_weight" for pc in component_names],
    )
    variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": explained_variance[:n_components],
            "explained_variance_ratio": explained_ratio[:n_components],
            "cumulative_explained_variance_ratio": np.cumsum(
                explained_ratio[:n_components]
            ),
        }
    )
    return scores, weights, variance


def feature_name(region: Region) -> str:
    return f"region_{region[0]}_{region[1]}"


def parse_feature_name(name: str) -> Region:
    _, start, end = name.split("_", maxsplit=2)
    return int(start), int(end)


def prepare_pca_matrix(
    raw: pd.DataFrame,
    candidate_regions: Iterable[Region],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cpm = counts_to_cpm(raw)
    candidates = pd.MultiIndex.from_tuples(
        sorted(candidate_regions), names=["start", "end"]
    )
    candidate_cpm = cpm.reindex(candidates, fill_value=0.0)
    log2cpm = np.log2(candidate_cpm + 1.0).T
    log2cpm.columns = [feature_name((int(s), int(e))) for s, e in log2cpm.columns]
    log2cpm.index.name = "sample_id"

    variance = log2cpm.var(axis=0, ddof=1)
    keep = variance > 0
    excluded_rows = []
    for column in log2cpm.columns[~keep]:
        start, end = parse_feature_name(column)
        excluded_rows.append(
            {"start": start, "end": end, "reason": "zero_variance_log2cpm"}
        )
    excluded = pd.DataFrame(excluded_rows, columns=["start", "end", "reason"])
    return log2cpm.loc[:, keep], excluded


def enrich_pca_weights(
    weights: pd.DataFrame,
    region_sets: dict[str, set[Region]],
) -> pd.DataFrame:
    rows = []
    for name, values in weights.iterrows():
        start, end = parse_feature_name(name)
        memberships = [c for c in CONDITIONS if (start, end) in region_sets[c]]
        row = {
            "feature": name,
            "start": start,
            "end": end,
            "n_conditions": len(memberships),
            "conditions": "+".join(memberships),
        }
        row.update(values.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pca(
    coordinates: pd.DataFrame,
    variance: pd.DataFrame,
    output_path: Path,
) -> None:
    if "PC2" not in coordinates.columns:
        print("    WARNUNG: PCA besitzt weniger als zwei Achsen; kein Plot.")
        return

    pc1_pct = variance.loc[variance["component"] == "PC1", "explained_variance_ratio"].iat[0]
    pc2_pct = variance.loc[variance["component"] == "PC2", "explained_variance_ratio"].iat[0]

    fig, ax = plt.subplots(figsize=(10.2, 7.6))
    for condition in CONDITION_FOLDERS:
        subset = coordinates.loc[coordinates["condition"] == condition]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=72,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=condition,
        )

        centroid_x = subset["PC1"].mean()
        centroid_y = subset["PC2"].mean()
        ax.scatter(
            centroid_x,
            centroid_y,
            marker="X",
            s=165,
            color=COLORS[condition],
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )
        ax.annotate(
            condition,
            (centroid_x, centroid_y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0, color="#c7c7c7", lw=0.8)
    ax.axvline(0, color="#c7c7c7", lw=0.8)
    ax.set_xlabel(f"PC1 ({pc1_pct * 100:.1f}% Varianz)")
    ax.set_ylabel(f"PC2 ({pc2_pct * 100:.1f}% Varianz)")
    ax.set_title("PCA der Proben auf der Union der Methode-1-Intersections")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_top_pca_regions(
    enriched_weights: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    weight_columns = [
        column for column in enriched_weights.columns if column.endswith("_weight")
    ]
    for column in weight_columns:
        pc = column.removesuffix("_weight")
        table = enriched_weights.copy()
        table["abs_weight"] = table[column].abs()
        table = table.sort_values("abs_weight", ascending=False).head(top_n)
        table.to_csv(output_dir / f"pca_top_regions_{pc}.csv", sep=SEP, index=False)


def main() -> None:
    args = parse_args()
    root = find_project_root(args.project_root)
    intersection_dir = (
        args.intersection_dir.expanduser().resolve()
        if args.intersection_dir
        else root
        / "data"
        / "method1_set_based_and_rule_based_region_selection_data"
        / "shared_regions_across_samples"
    )
    raw_data_dir = (
        args.raw_data_dir.expanduser().resolve()
        if args.raw_data_dir
        else root / "data" / "raw"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root
        / "data"
        / "method2_unsupervised_selection_UMAP_PCA_PCoA_data"
        / "PCA_and_PCoA"
        / "raw_sample_pcoa_and_intersection_pca"
    )
    output_dir.mkdir(parents=True, exist_ok=True)


    region_sets, source_paths = load_intersection_sets(intersection_dir)
    membership = membership_table(region_sets)
    membership.to_csv(
        output_dir / "intersection_membership.csv", sep=SEP, index=False
    )
    source_table = pd.DataFrame(
        {
            "condition": list(source_paths),
            "file": [str(source_paths[kd]) for kd in source_paths],
            "n_regions": [len(region_sets[kd]) for kd in source_paths],
        }
    )
    source_table.to_csv(
        output_dir / "intersection_sources.csv", sep=SEP, index=False
    )

    raw, metadata = build_raw_matrix(raw_data_dir)

    sample_region_sets = raw_sample_region_sets(raw, args.presence_cpm)
    presence_rows = [
        {
            "sample_id": sample_id,
            "condition": metadata.loc[sample_id, "condition"],
            "replicate": metadata.loc[sample_id, "replicate"],
            "presence_cpm": args.presence_cpm,
            "n_present_regions": len(regions),
        }
        for sample_id, regions in sample_region_sets.items()
    ]
    pd.DataFrame(presence_rows).to_csv(
        output_dir / "raw_sample_presence_summary.csv", sep=SEP, index=False
    )
    similarity, distance, pairwise = jaccard_matrices(sample_region_sets)
    pcoa_coordinates, pcoa_eigenvalues = classical_pcoa(distance)
    similarity.to_csv(output_dir / "raw_sample_jaccard_similarity.csv", sep=SEP)
    distance.to_csv(output_dir / "raw_sample_jaccard_distance.csv", sep=SEP)
    pairwise.to_csv(
        output_dir / "raw_sample_jaccard_pairwise_overlap.csv", sep=SEP, index=False
    )
    pcoa_coordinates.to_csv(
        output_dir / "raw_sample_jaccard_pcoa_coordinates.csv", sep=SEP
    )
    pcoa_eigenvalues.to_csv(
        output_dir / "raw_sample_jaccard_pcoa_eigenvalues.csv", sep=SEP, index=False
    )
    plot_jaccard_heatmap(
        similarity, output_dir / "raw_sample_jaccard_similarity_heatmap.png"
    )
    plot_pcoa(
        pcoa_coordinates,
        pcoa_eigenvalues,
        metadata,
        output_dir / "raw_sample_jaccard_pcoa.png",
    )
    if not pairwise.empty:
        best = pairwise.iloc[0]

    pca_input, excluded = prepare_pca_matrix(
        raw, set().union(*region_sets.values())
    )
    if pca_input.shape[1] < 2:
        raise RuntimeError("Nach dem Varianzfilter bleiben weniger als zwei Regionen.")

    scores, weights, variance = fit_pca(
        pca_input, scale_features=args.scale_features
    )
    coordinates = metadata.join(scores, how="inner").reset_index()
    condition_order = {condition: i for i, condition in enumerate(CONDITION_FOLDERS)}
    coordinates["_order"] = coordinates["condition"].map(condition_order)
    coordinates = coordinates.sort_values(["_order", "replicate"]).drop(columns="_order")

    centroids = (
        coordinates.groupby("condition", sort=False)[scores.columns.tolist()]
        .mean()
        .reset_index()
    )
    enriched_weights = enrich_pca_weights(weights, region_sets)

    pca_export = metadata[["condition", "replicate"]].join(pca_input)
    pca_export.reset_index().to_csv(
        output_dir / "pca_input_log2cpm.csv", sep=SEP, index=False
    )
    coordinates.to_csv(output_dir / "pca_sample_coordinates.csv", sep=SEP, index=False)
    centroids.to_csv(output_dir / "pca_condition_centroids.csv", sep=SEP, index=False)
    variance.to_csv(output_dir / "pca_explained_variance.csv", sep=SEP, index=False)
    enriched_weights.to_csv(output_dir / "pca_region_weights.csv", sep=SEP, index=False)
    excluded.to_csv(output_dir / "pca_excluded_regions.csv", sep=SEP, index=False)
    export_top_pca_regions(enriched_weights, output_dir, args.top_loadings)
    plot_pca(coordinates, variance, output_dir / "pca_samples.png")

    pc1 = variance.loc[variance["component"] == "PC1", "explained_variance_ratio"].iat[0]
    pc2 = variance.loc[variance["component"] == "PC2", "explained_variance_ratio"].iat[0]

    print(f"Ergebnis gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()