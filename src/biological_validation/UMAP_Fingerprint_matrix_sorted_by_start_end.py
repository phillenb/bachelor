#!/usr/bin/env python3
"""Create fixed-order log2(CPM + 1) fingerprints for 362 UMAP regions.

Absent regions receive zero counts. CPM uses each sample's total raw reads.
Genomic ordering and a shared grid support comparison across samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]

CURRENT_FILE = Path(__file__).resolve()
DEFAULT_REGIONS_RELATIVE_PATH = (
    Path("data")
    / "method_comparison"
    / "three_method_region_sets"
    / "umap_regions_standardized.csv"
)
DEFAULT_OUTPUT_RELATIVE_PATH = (
    Path("data") / "biological validation" / "UMAP_Fingerprint_matrix_sorted_by_start_end"
)
MIN_RAW_READS = 2
EXPECTED_SAMPLES_PER_CONDITION = 3

RAW_DATASET_DIRECTORIES = {
    "Ctrl": "20240314_Ribolyzer_Ctrl",
    "UTP18": "20240314_Ribolyzer_UTP18",
    "DIMT1L": "20250129_Ribolyzer_DIMT1L",
    "LAS1L": "Las1L_NanoRibolyzer",
    "RPL3": "RPL3_NanoRibolyzer",
    "URB1": "URB1_NanoRibolyzer",
    "WBSCR22": "WBSCR22_NanoRibolyzer",
}
RAW_MATRIX_GLOB = "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"

REQUIRED_REGION_COLUMNS = {"start", "end", "condition"}
REQUIRED_RAW_COLUMNS = {"start", "end", "n_reads"}
REQUIRED_SAMPLE_COLUMNS = {"condition", "sample", "path"}


@dataclass(frozen=True)
class SampleSpec:
    condition: str
    sample: str
    replicate: str
    path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.condition}_{self.sample}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed-order dense matrix and fingerprint heatmaps for "
            "the selected UMAP regions."
        )
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help=(
            "Optional override for the selected-region CSV/TSV. By default, "
            "the file is loaded from data/method_comparison/"
            "three_method_region_sets/umap_regions_standardized.csv."
        ),
    )
    parser.add_argument(
        "--sample-sheet",
        type=Path,
        default=None,
        help=(
            "Optional CSV/TSV override with condition, sample, path and "
            "optional replicate. If omitted, all 21 raw matrices are "
            "discovered automatically below data/raw."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output-directory override. Default: "
            "data/biological validation/UMAP_Fingerprint_matrix_sorted_by_start_end."
        ),
    )
    parser.add_argument(
        "--region-order",
        choices=("condition_rank", "genomic"),
        default="genomic",
        help=(
            "Region order shared by every sample. 'genomic' sorts first by "
            "start and then by end; 'condition_rank' groups regions by their "
            "UMAP-assigned condition and final rank. Default: genomic."
        ),
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=EXPECTED_SAMPLES_PER_CONDITION,
        help="Required sample count per condition. Set 0 to disable. Default: 3.",
    )
    parser.add_argument(
        "--min-raw-reads",
        type=int,
        default=MIN_RAW_READS,
        help=(
            "Selected-region counts below this value are set to zero before "
            "CPM normalization. Total reads are still calculated from the "
            "complete raw matrix. Default: 2."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows read per raw-data chunk. Default: 500000.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib heatmap colour map. Default: magma.",
    )
    parser.add_argument(
        "--vmax-quantile",
        type=float,
        default=1.0,
        help=(
            "Shared upper colour limit as a quantile of all values. Use 1.0 "
            "for the observed maximum. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Legacy option retained for command-line compatibility.",
    )
    parser.add_argument(
        "--fingerprint-columns",
        type=int,
        default=20,
        help=(
            "Number of columns in each per-sample fingerprint grid. With 362 "
            "regions, the default creates a 19 x 20 grid with 18 masked "
            "padding cells. Default: 20."
        ),
    )
    return parser.parse_args()


def find_project_root(start_path: Path) -> Path:
    """Find the project containing src/ and data/."""
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(
        f"Projektwurzel konnte ausgehend von {start_path} nicht gefunden werden."
    )


def detect_delimiter(path: Path) -> str:
    """Detect comma, semicolon or tab delimiters."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"File is empty: {path}")
    try:
        dialect = csv.Sniffer().sniff(header, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: header.count(delimiter) for delimiter in (",", ";", "\t")}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise ValueError(f"Could not detect delimiter in: {path}")
        return delimiter


def read_small_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=detect_delimiter(path), encoding="utf-8-sig")


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def infer_replicate_label(path: Path, condition: str, index: int) -> str:
    """Infer a replicate label from the sample path."""
    search_text = " ".join(path.parts[-6:])
    matches = re.findall(
        r"(?<![A-Za-z0-9])([RN])[\s_-]?(\d+)(?![A-Za-z0-9])",
        search_text,
        flags=re.IGNORECASE,
    )
    if matches:
        prefix, number = matches[-1]
        return f"{prefix.upper()}{int(number)}"

    fallback = (
        ["R10", "R11", "R12"]
        if condition in {"Ctrl", "UTP18"}
        else ["N1", "N2", "N3"]
    )
    if index < len(fallback):
        return fallback[index]
    return f"replicate_{index + 1}"


def discover_project_samples(
    raw_data_dir: Path,
    expected_replicates: int,
) -> list[SampleSpec]:
    samples: list[SampleSpec] = []
    for condition in CONDITION_ORDER:
        dataset_dir = raw_data_dir / RAW_DATASET_DIRECTORIES[condition]
        paths = sorted(
            dataset_dir.glob(RAW_MATRIX_GLOB),
            key=lambda path: natural_key(str(path)),
        )
        if expected_replicates > 0 and len(paths) != expected_replicates:
            discovered = "\n".join(f"  - {path}" for path in paths) or "  - none"
            raise RuntimeError(
                f"Für {condition} wurden {len(paths)} statt "
                f"{expected_replicates} Rohmatrizen gefunden.\n"
                f"Suchmuster: {dataset_dir / RAW_MATRIX_GLOB}\n"
                f"Gefunden:\n{discovered}"
            )

        condition_samples: list[SampleSpec] = []
        for index, path in enumerate(paths):
            replicate = infer_replicate_label(path, condition, index)
            condition_samples.append(
                SampleSpec(
                    condition=condition,
                    sample=replicate,
                    replicate=replicate,
                    path=path,
                )
            )
        labels = [sample.sample for sample in condition_samples]
        if len(labels) != len(set(labels)):
            raise RuntimeError(
                f"Replikatnamen für {condition} sind nicht eindeutig: {labels}"
            )
        samples.extend(condition_samples)
    return samples


def load_regions(path: Path, ordering: str) -> pd.DataFrame:
    regions = read_small_table(path)
    missing = REQUIRED_REGION_COLUMNS.difference(regions.columns)
    if missing:
        raise ValueError(f"Region file is missing columns: {sorted(missing)}")

    regions = regions.copy()
    regions["start"] = pd.to_numeric(regions["start"], errors="raise").astype("int64")
    regions["end"] = pd.to_numeric(regions["end"], errors="raise").astype("int64")
    regions["condition"] = regions["condition"].astype(str).str.strip()

    duplicated = regions.duplicated(["start", "end"], keep=False)
    if duplicated.any():
        examples = regions.loc[duplicated, ["start", "end", "condition"]].head(10)
        raise ValueError(
            "Selected regions contain duplicate start/end coordinates. Examples:\n"
            + examples.to_string(index=False)
        )
    unknown = sorted(set(regions["condition"]) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown region conditions: {unknown}")

    regions["_condition_order"] = regions["condition"].map(
        {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    )

    if ordering == "condition_rank":
        if "final_rank_within_condition" not in regions.columns:
            raise ValueError(
                "--region-order condition_rank requires "
                "'final_rank_within_condition' in the region file."
            )
        regions["final_rank_within_condition"] = pd.to_numeric(
            regions["final_rank_within_condition"], errors="raise"
        ).astype("int64")
        regions = regions.sort_values(
            ["_condition_order", "final_rank_within_condition", "start", "end"],
            kind="stable",
        )
    else:
        regions = regions.sort_values(["start", "end"], kind="stable")

    regions = regions.reset_index(drop=True)
    regions.insert(0, "region_order", np.arange(1, len(regions) + 1, dtype=int))
    regions.insert(
        1,
        "region_id",
        [f"{start}_{end}" for start, end in regions[["start", "end"]].itertuples(index=False)],
    )
    return regions.drop(columns="_condition_order")


def load_sample_sheet(path: Path, expected_replicates: int) -> list[SampleSpec]:
    sheet = read_small_table(path)
    missing = REQUIRED_SAMPLE_COLUMNS.difference(sheet.columns)
    if missing:
        raise ValueError(f"Sample sheet is missing columns: {sorted(missing)}")

    sheet = sheet.copy()
    if "replicate" not in sheet.columns:
        sheet["replicate"] = sheet["sample"]
    for column in ("condition", "sample", "replicate", "path"):
        sheet[column] = sheet[column].astype(str).str.strip()

    if (sheet["path"] == "").any() or sheet["path"].str.lower().eq("nan").any():
        rows = (sheet.index[(sheet["path"] == "") | sheet["path"].str.lower().eq("nan")] + 2).tolist()
        raise ValueError(f"Sample sheet contains empty paths in rows: {rows}")

    unknown = sorted(set(sheet["condition"]) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown sample conditions: {unknown}")

    if sheet.duplicated(["condition", "sample"]).any():
        duplicates = sheet.loc[
            sheet.duplicated(["condition", "sample"], keep=False),
            ["condition", "sample"],
        ]
        raise ValueError("Duplicate sample IDs:\n" + duplicates.to_string(index=False))

    counts = sheet.groupby("condition", sort=False).size().reindex(CONDITION_ORDER, fill_value=0)
    if expected_replicates > 0 and not counts.eq(expected_replicates).all():
        raise ValueError(
            f"Expected {expected_replicates} samples per condition, but found:\n"
            + counts.to_string()
        )

    condition_order = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    records = sheet.to_dict("records")
    records.sort(
        key=lambda row: (
            condition_order[row["condition"]],
            natural_key(row["replicate"]),
            natural_key(row["sample"]),
        )
    )

    samples = []
    for row in records:
        raw_path = Path(row["path"]).expanduser()
        if not raw_path.is_absolute():
            raw_path = path.parent / raw_path
        samples.append(
            SampleSpec(
                condition=row["condition"],
                sample=row["sample"],
                replicate=row["replicate"],
                path=raw_path,
            )
        )
    missing_files = [str(sample.path) for sample in samples if not sample.path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Raw intensity-matrix files not found:\n" + "\n".join(missing_files)
        )
    return samples


def coordinate_keys(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if (starts < 0).any() or (ends < 0).any():
        raise ValueError("start/end coordinates must be non-negative")
    return (starts << np.int64(32)) | ends


def extract_sample_counts(
    sample: SampleSpec,
    target_keys: np.ndarray,
    chunksize: int,
    min_raw_reads: int,
) -> tuple[np.ndarray, float]:
    """Extract ordered region counts and the total raw-read count."""
    delimiter = detect_delimiter(sample.path)
    header = pd.read_csv(sample.path, sep=delimiter, nrows=0, encoding="utf-8-sig")
    missing = REQUIRED_RAW_COLUMNS.difference(header.columns)
    if missing:
        raise ValueError(f"{sample.path} is missing columns: {sorted(missing)}")

    key_to_position = {int(key): index for index, key in enumerate(target_keys)}
    selected_counts = np.zeros(len(target_keys), dtype=np.float64)
    total_reads = 0.0

    reader = pd.read_csv(
        sample.path,
        sep=delimiter,
        usecols=["start", "end", "n_reads"],
        dtype={"start": "int64", "end": "int64", "n_reads": "float64"},
        chunksize=chunksize,
        encoding="utf-8-sig",
        engine="c",
    )
    for chunk in reader:
        if chunk[["start", "end", "n_reads"]].isna().any().any():
            raise ValueError(f"Missing start/end/n_reads values in {sample.path}")
        if (chunk["n_reads"] < 0).any():
            raise ValueError(f"Negative n_reads values in {sample.path}")

        reads = chunk["n_reads"].to_numpy(dtype=np.float64, copy=False)
        total_reads += float(reads.sum(dtype=np.float64))
        keys = coordinate_keys(
            chunk["start"].to_numpy(copy=False),
            chunk["end"].to_numpy(copy=False),
        )
        mask = np.isin(keys, target_keys, assume_unique=False)
        if not mask.any():
            continue
        masked_keys = keys[mask]
        masked_reads = reads[mask]
        if len(masked_keys) != len(masked_reads):
            raise ValueError(
                f"Length mismatch between masked keys ({len(masked_keys)}) "
                f"and masked reads ({len(masked_reads)}) in {sample.path}"
            )
        for key, count in zip(masked_keys, masked_reads):
            selected_counts[key_to_position[int(key)]] += float(count)

    if not math.isfinite(total_reads) or total_reads <= 0:
        raise ValueError(f"Total read count must be positive for {sample.path}")
    if min_raw_reads > 1:
        selected_counts[selected_counts < min_raw_reads] = 0.0
    return selected_counts, total_reads


def build_dense_matrices(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    chunksize: int,
    min_raw_reads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_keys = coordinate_keys(
        regions["start"].to_numpy(), regions["end"].to_numpy()
    )
    columns: dict[str, np.ndarray] = {}
    qc_rows: list[dict[str, object]] = []

    for index, sample in enumerate(samples, start=1):
        raw_counts, total_reads = extract_sample_counts(
            sample,
            target_keys,
            chunksize,
            min_raw_reads,
        )
        cpm = raw_counts / total_reads * 1_000_000.0
        log2_cpm = np.log2(cpm + 1.0)
        columns[sample.sample_id] = log2_cpm
        qc_rows.append(
            {
                "condition": sample.condition,
                "sample": sample.sample,
                "replicate": sample.replicate,
                "sample_id": sample.sample_id,
                "source_path": str(sample.path),
                "total_reads": total_reads,
                "selected_regions": len(regions),
                "min_raw_reads": min_raw_reads,
                "selected_regions_nonzero": int(np.count_nonzero(raw_counts)),
                "selected_regions_zero": int(np.count_nonzero(raw_counts == 0)),
                "selected_raw_reads": float(raw_counts.sum()),
                "selected_read_fraction": float(raw_counts.sum() / total_reads),
            }
        )

    matrix = pd.DataFrame(columns, index=regions.index)
    return matrix, pd.DataFrame(qc_rows)


def metadata_columns(regions: pd.DataFrame) -> list[str]:
    preferred = [
        "region_order",
        "region_id",
        "start",
        "end",
        "condition",
        "final_rank_within_condition",
    ]
    return [column for column in preferred if column in regions.columns]


def write_outputs(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    qc: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_dir = output_dir / "condition_matrices"
    condition_dir.mkdir(exist_ok=True)

    region_metadata = regions[metadata_columns(regions)].copy()
    regions.to_csv(output_dir / "region_order.csv", index=False)
    qc.to_csv(output_dir / "sample_qc.csv", index=False)

    wide = pd.concat([region_metadata, matrix], axis=1)
    wide.to_csv(output_dir / "dense_matrix_regions_x_samples.csv", index=False)

    sample_metadata = pd.DataFrame(
        {
            "condition": [sample.condition for sample in samples],
            "sample": [sample.sample for sample in samples],
            "replicate": [sample.replicate for sample in samples],
            "sample_id": [sample.sample_id for sample in samples],
        }
    )
    samples_x_regions = pd.concat(
        [
            sample_metadata.reset_index(drop=True),
            matrix.T.reset_index(drop=True).set_axis(regions["region_id"], axis=1),
        ],
        axis=1,
    )
    samples_x_regions.to_csv(
        output_dir / "dense_matrix_samples_x_regions.csv", index=False
    )

    long_parts: list[pd.DataFrame] = []
    for sample in samples:
        part = region_metadata.copy()
        part.insert(0, "replicate", sample.replicate)
        part.insert(0, "sample", sample.sample)
        part.insert(0, "sample_id", sample.sample_id)
        part.insert(0, "sample_condition", sample.condition)
        part["log2_cpm_plus_1"] = matrix[sample.sample_id].to_numpy()
        long_parts.append(part)
    pd.concat(long_parts, ignore_index=True).to_csv(
        output_dir / "dense_matrix_long.csv", index=False
    )

    for condition in CONDITION_ORDER:
        condition_samples = [
            sample.sample_id for sample in samples if sample.condition == condition
        ]
        condition_matrix = pd.concat(
            [region_metadata, matrix[condition_samples]], axis=1
        )
        condition_matrix.to_csv(
            condition_dir / f"{condition}_dense_matrix.csv", index=False
        )


def fingerprint_layout(
    regions: pd.DataFrame, n_columns: int
) -> tuple[pd.DataFrame, int, int]:
    """Map regions to row-major grid positions with centred padding."""
    if n_columns <= 0:
        raise ValueError("--fingerprint-columns must be positive")
    n_rows = math.ceil(len(regions) / n_columns)
    n_cells = n_rows * n_columns
    n_padding = n_cells - len(regions)
    leading_padding = n_padding // 2
    flat_positions = np.arange(len(regions), dtype=int) + leading_padding

    layout = regions[metadata_columns(regions)].copy()
    layout.insert(0, "grid_column", flat_positions % n_columns + 1)
    layout.insert(0, "grid_row", flat_positions // n_columns + 1)
    return layout, n_rows, n_columns


def vector_to_fingerprint_grid(
    values: np.ndarray,
    n_rows: int,
    n_columns: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_cells = n_rows * n_columns
    n_padding = n_cells - len(values)
    if n_padding < 0:
        raise ValueError("Fingerprint grid is smaller than the region vector")
    leading_padding = n_padding // 2
    grid = np.full(n_cells, np.nan, dtype=float)
    grid[leading_padding : leading_padding + len(values)] = values
    return grid.reshape(n_rows, n_columns)


def write_sample_fingerprint_matrices(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    output_dir: Path,
    n_columns: int,
) -> tuple[int, int]:
    matrix_dir = output_dir / "sample_fingerprint_matrices"
    matrix_dir.mkdir(exist_ok=True)
    layout, n_rows, n_columns = fingerprint_layout(regions, n_columns)
    layout.to_csv(output_dir / "fingerprint_layout.csv", index=False)

    column_names = [f"grid_column_{index}" for index in range(1, n_columns + 1)]
    for sample in samples:
        grid = vector_to_fingerprint_grid(
            matrix[sample.sample_id].to_numpy(), n_rows, n_columns
        )
        grid_table = pd.DataFrame(grid, columns=column_names)
        grid_table.insert(0, "grid_row", np.arange(1, n_rows + 1, dtype=int))
        grid_table.to_csv(
            matrix_dir / f"{sample.sample_id}_fingerprint_matrix.csv", index=False
        )
    return n_rows, n_columns


def condition_boundaries(regions: pd.DataFrame) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    start = 0
    conditions = regions["condition"].tolist()
    for index in range(1, len(conditions) + 1):
        if index == len(conditions) or conditions[index] != conditions[start]:
            blocks.append((conditions[start], start, index))
            start = index
    return blocks


def decorate_region_axis(ax: mpl.axes.Axes, regions: pd.DataFrame) -> None:
    blocks = condition_boundaries(regions)
    # Label condition blocks only when the region order permits.
    if len(blocks) > len(CONDITION_ORDER):
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(regions) - 0.5)
        return
    for condition, start, end in blocks:
        if start:
            ax.axvline(start - 0.5, color="white", linewidth=0.7, alpha=0.9)
        midpoint = (start + end - 1) / 2
        ax.text(
            midpoint,
            1.015,
            condition,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(regions) - 0.5)


def save_figure(fig: mpl.figure.Figure, base_path: Path, dpi: int) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def fingerprint_cmap(cmap_name: str) -> mpl.colors.Colormap:
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad("#D9D9D9")
    return cmap


def show_fingerprint(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    n_rows: int,
    n_columns: int,
    cmap: mpl.colors.Colormap,
    vmax: float,
) -> mpl.image.AxesImage:
    grid = vector_to_fingerprint_grid(values, n_rows, n_columns)
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#4A4A4A")
    return image


def plot_condition_fingerprint_triptychs(
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    output_dir: Path,
    n_rows: int,
    n_columns: int,
    cmap_name: str,
    vmax: float,
    dpi: int,
) -> None:
    plot_dir = output_dir / "condition_fingerprints"
    plot_dir.mkdir(exist_ok=True)
    cmap = fingerprint_cmap(cmap_name)
    for condition in CONDITION_ORDER:
        condition_samples = [
            sample for sample in samples if sample.condition == condition
        ]
        fig, axes = plt.subplots(
            1,
            len(condition_samples),
            figsize=(10.5, 3.7),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        image = None
        if len(axes) != len(condition_samples):
            raise ValueError(
                f"Expected {len(axes)} samples for {condition}, "
                f"found {len(condition_samples)}"
            )
        for ax, sample in zip(axes, condition_samples):
            image = show_fingerprint(
                ax,
                matrix[sample.sample_id].to_numpy(),
                n_rows,
                n_columns,
                cmap,
                vmax,
            )
            ax.set_title(sample.sample, fontsize=10)
        fig.suptitle(f"{condition} replicate fingerprints", fontsize=13)
        assert image is not None
        colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label(r"$\log_2(\mathrm{CPM}+1)$")
        save_figure(fig, plot_dir / f"{condition}_fingerprint_triptych", dpi)


def plot_fingerprint_overview(
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    output_dir: Path,
    n_rows: int,
    n_columns: int,
    cmap_name: str,
    vmax: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(CONDITION_ORDER),
        3,
        figsize=(10.5, 18.5),
        constrained_layout=True,
    )
    cmap = fingerprint_cmap(cmap_name)
    image = None
    for row_index, condition in enumerate(CONDITION_ORDER):
        condition_samples = [
            sample for sample in samples if sample.condition == condition
        ]
        for column_index, sample in enumerate(condition_samples):
            ax = axes[row_index, column_index]
            image = show_fingerprint(
                ax,
                matrix[sample.sample_id].to_numpy(),
                n_rows,
                n_columns,
                cmap,
                vmax,
            )
            ax.set_title(sample.sample, fontsize=9, pad=3)
            if column_index == 0:
                ax.set_ylabel(
                    condition,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=34,
                    fontsize=10,
                )
    fig.suptitle(
        "UMAP-region fingerprints across all samples",
        fontsize=14,
    )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label(r"Signal intensity: $\log_2(\mathrm{CPM}+1)$")
    save_figure(fig, output_dir / "umap_fingerprint_overview_7x3", dpi)


def plot_all_samples(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    vmax: float,
    dpi: int,
) -> None:
    values = matrix[[sample.sample_id for sample in samples]].T.to_numpy()
    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels([sample.sample_id for sample in samples], fontsize=8)
    ax.set_xlabel("362 UMAP-selected regions in fixed order")
    ax.set_ylabel("Sample")
    ax.set_title("UMAP-region signal fingerprints across all samples", pad=28)
    decorate_region_axis(ax, regions)

    for index in range(1, len(samples)):
        if samples[index].condition != samples[index - 1].condition:
            ax.axhline(index - 0.5, color="white", linewidth=1.2)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(r"Signal intensity: $\log_2(\mathrm{CPM}+1)$")
    save_figure(fig, output_dir / "umap_fingerprint_all_samples", dpi)


def plot_condition_panels(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    matrix: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    vmax: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(CONDITION_ORDER),
        ncols=1,
        figsize=(18, 11.5),
        sharex=True,
        constrained_layout=True,
    )
    last_image = None
    if len(axes) != len(CONDITION_ORDER):
        raise ValueError(
            f"Expected {len(CONDITION_ORDER)} condition axes, found {len(axes)}"
        )
    for axis, condition in zip(axes, CONDITION_ORDER):
        condition_samples = [
            sample for sample in samples if sample.condition == condition
        ]
        values = matrix[[sample.sample_id for sample in condition_samples]].T.to_numpy()
        last_image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_yticks(np.arange(len(condition_samples)))
        axis.set_yticklabels(
            [sample.sample for sample in condition_samples], fontsize=8
        )
        axis.set_ylabel(condition, rotation=0, ha="right", va="center", labelpad=45)
        for _, start, _ in condition_boundaries(regions)[1:]:
            axis.axvline(start - 0.5, color="white", linewidth=0.55, alpha=0.9)

    decorate_region_axis(axes[0], regions)
    axes[-1].set_xlabel("362 UMAP-selected regions in fixed order")
    fig.suptitle("Condition-wise UMAP-region signal fingerprints", y=1.015)
    assert last_image is not None
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label(r"Signal intensity: $\log_2(\mathrm{CPM}+1)$")
    save_figure(fig, output_dir / "umap_fingerprint_condition_panels", dpi)


def write_run_summary(
    regions: pd.DataFrame,
    samples: list[SampleSpec],
    qc: pd.DataFrame,
    output_dir: Path,
    region_order: str,
    vmax_quantile: float,
    vmax: float,
    fingerprint_rows: int,
    fingerprint_columns: int,
    min_raw_reads: int,
) -> None:
    summary = {
        "n_regions": int(len(regions)),
        "n_samples": int(len(samples)),
        "conditions": CONDITION_ORDER,
        "samples_per_condition": {
            condition: sum(sample.condition == condition for sample in samples)
            for condition in CONDITION_ORDER
        },
        "normalization": "log2((n_reads / total_reads * 1e6) + 1)",
        "minimum_raw_reads_per_selected_region": min_raw_reads,
        "missing_selected_region_value": 0.0,
        "region_order": region_order,
        "fingerprint_grid": {
            "rows": fingerprint_rows,
            "columns": fingerprint_columns,
            "signal_cells": int(len(regions)),
            "masked_padding_cells": int(
                fingerprint_rows * fingerprint_columns - len(regions)
            ),
            "interpretation": (
                "Fixed visual layout for sample-profile comparison; grid "
                "position is not a two-dimensional genomic coordinate."
            ),
        },
        "colour_scale": {
            "vmin": 0.0,
            "vmax_quantile": vmax_quantile,
            "vmax": vmax,
        },
        "total_reads_min": float(qc["total_reads"].min()),
        "total_reads_max": float(qc["total_reads"].max()),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    if args.min_raw_reads < 1:
        raise ValueError("--min-raw-reads must be at least 1")
    if args.fingerprint_columns <= 0:
        raise ValueError("--fingerprint-columns must be positive")
    if not 0 < args.vmax_quantile <= 1:
        raise ValueError("--vmax-quantile must be in (0, 1]")

    project_root: Path | None = None
    if args.regions is None or args.sample_sheet is None or args.output_dir is None:
        project_root = find_project_root(CURRENT_FILE.parent)

    regions_path = (
        args.regions
        if args.regions is not None
        else project_root / DEFAULT_REGIONS_RELATIVE_PATH
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else project_root / DEFAULT_OUTPUT_RELATIVE_PATH
    )


    regions = load_regions(regions_path, args.region_order)
    if len(regions) != 362:
        raise ValueError(
            f"Expected exactly 362 selected regions, but found {len(regions)}"
        )
    if args.sample_sheet is not None:
        samples = load_sample_sheet(args.sample_sheet, args.expected_replicates)
    else:
        assert project_root is not None
        raw_data_dir = project_root / "data" / "raw"
        intensity_matrices_data_dir = project_root / "data" / "intensity_matrices"
        samples = discover_project_samples(raw_data_dir, args.expected_replicates)
        for condition in CONDITION_ORDER:
            condition_samples = [
                sample for sample in samples if sample.condition == condition
            ]
            labels = ", ".join(sample.sample for sample in condition_samples)

    matrix, qc = build_dense_matrices(
        regions,
        samples,
        args.chunksize,
        args.min_raw_reads,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(regions, samples, matrix, qc, output_dir)
    fingerprint_rows, fingerprint_columns = write_sample_fingerprint_matrices(
        regions,
        samples,
        matrix,
        output_dir,
        args.fingerprint_columns,
    )

    all_values = matrix.to_numpy(dtype=float)
    vmax = float(np.quantile(all_values, args.vmax_quantile))
    if not math.isfinite(vmax) or vmax <= 0:
        raise ValueError("The dense matrix contains no positive signal values")
    plot_all_samples(
        regions, samples, matrix, output_dir, args.cmap, vmax, args.dpi
    )
    plot_condition_panels(
        regions, samples, matrix, output_dir, args.cmap, vmax, args.dpi
    )
    plot_condition_fingerprint_triptychs(
        samples,
        matrix,
        output_dir,
        fingerprint_rows,
        fingerprint_columns,
        args.cmap,
        vmax,
        args.dpi,
    )
    plot_fingerprint_overview(
        samples,
        matrix,
        output_dir,
        fingerprint_rows,
        fingerprint_columns,
        args.cmap,
        vmax,
        args.dpi,
    )
    write_run_summary(
        regions,
        samples,
        qc,
        output_dir,
        args.region_order,
        args.vmax_quantile,
        vmax,
        fingerprint_rows,
        fingerprint_columns,
        args.min_raw_reads,
    )

    print(f"Ergebnisse gespeichert in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()