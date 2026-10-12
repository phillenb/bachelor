from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEP = ";"
EXPECTED_REPLICATES = 3
CONDITION_ORDER = [
    "Ctrl",
    "UTP18",
    "DIMT1L",
    "LAS1L",
    "RPL3",
    "URB1",
    "WBSCR22",
]
KD_ORDER = [condition for condition in CONDITION_ORDER if condition != "Ctrl"]
SECTION_ORDER = ["5ETS", "18S", "ITS1", "5-8S", "ITS2", "28S", "3ETS"]
ANALYSIS_ORDER = [
    "mro_intermediate",
    "intermediate",
    "boundary_either",
    "section_fractional",
]

# Gleiche log2FC-Farbskala fuer Rohdaten- und Union-Profile.
LOG2FC_COLOR_LIMIT = 4.0

EXPANDED_MARKER_FRAGMENTS = {
    "34SL3'",
    "30SL3'",
    "26SL3'/X",
    "21SL3'",
    "18S-E_+9",
    "18S-E_+24",
    "18S-E_+36",
    "18S-E_+40",
    "18S-E_+78",
}


def find_src_dir(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if path.name == "src":
            return path
    raise RuntimeError("Das Skript muss innerhalb des Projektverzeichnisses 'src' liegen.")


SRC_DIR = find_src_dir(Path(__file__))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import RAW_DATA_DIR, INTENSITY_MATRICES_DATA_DIR, DATA_DIR  # noqa: E402
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_LITERATURE_CSV = (
    PROJECT_ROOT
    / "data"
    / "literature"
    / "rna45sn1_intermediates_expanded.csv"
)

DEFAULT_OUTPUT_DIR = Path(DATA_DIR) / "biological validation" / "raw data alignment"


CONDITIONS: dict[str, list[Path]] = {
    "Ctrl": sorted(
        (Path(RAW_DATA_DIR) / "20240314_Ribolyzer_Ctrl").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "UTP18": sorted(
        (Path(RAW_DATA_DIR) / "20240314_Ribolyzer_UTP18").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "DIMT1L": sorted(
        (Path(RAW_DATA_DIR) / "20250129_Ribolyzer_DIMT1L").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "LAS1L": sorted(
        (Path(RAW_DATA_DIR) / "Las1L_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "RPL3": sorted(
        (Path(RAW_DATA_DIR) / "RPL3_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "URB1": sorted(
        (Path(RAW_DATA_DIR) / "URB1_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
    "WBSCR22": sorted(
        (Path(RAW_DATA_DIR) / "WBSCR22_NanoRibolyzer").glob(
            "*IVPA*Nuc*/intensity_matrix/intensity_matrix.csv"
        )
    ),
}


def load_intensity_matrix(path: Path) -> pd.DataFrame:
    """Laedt start, end und n_reads ohne die grosse ids-Spalte."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Intensity-Matrix nicht gefunden: {path}")

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()

    separators = (";", "\t", ",")
    separator = max(separators, key=header.count)
    if header.count(separator) == 0:
        raise ValueError(
            f"Trennzeichen in {path} konnte nicht erkannt werden. "
            "Erwartet werden Semikolon, Tabulator oder Komma."
        )

    required = {"start", "end", "n_reads"}
    frame = pd.read_csv(
        path,
        sep=separator,
        usecols=lambda column: str(column).strip() in required,
        low_memory=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path}: fehlende Spalten {sorted(missing)}; "
            f"gefunden: {frame.columns.tolist()}"
        )
    return frame


def read_delimited(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    mapping = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    raise ValueError(
        f"Keine der erwarteten Spalten {candidates} gefunden; "
        f"vorhanden: {frame.columns.tolist()}"
    )


def load_literature(path: Path) -> pd.DataFrame:
    raw = read_delimited(path)
    reference_column = find_column(raw, ("reference", "reference_name"))
    start_column = find_column(raw, ("start",))
    end_column = find_column(raw, ("end",))
    fragment_column = find_column(raw, ("fragment", "intermediate", "name"))
    literature = pd.DataFrame(
        {
            "reference": raw[reference_column].astype(str).str.strip(),
            "start": pd.to_numeric(raw[start_column], errors="raise").astype(np.int64),
            "end": pd.to_numeric(raw[end_column], errors="raise").astype(np.int64),
            "fragment": raw[fragment_column].astype(str).str.strip(),
        }
    )
    if (literature["end"] <= literature["start"]).any():
        raise ValueError("Literaturdatei enthaelt Intervalle mit End <= Start.")
    if literature["fragment"].duplicated().any():
        duplicates = literature.loc[
            literature["fragment"].duplicated(keep=False), "fragment"
        ].tolist()
        raise ValueError(f"Fragmentnamen muessen eindeutig sein: {duplicates}")
    return literature.sort_values(["start", "end", "fragment"]).reset_index(drop=True)


def validate_expanded_literature(literature: pd.DataFrame, path: Path) -> None:
    """Prueft die erweiterte Annotation mit 42 Fragmenten."""
    available = set(literature["fragment"])
    missing = sorted(EXPANDED_MARKER_FRAGMENTS - available)
    if missing:
        raise ValueError(
            "Die geladene Literaturdatei ist nicht die erwartete expanded Version. "
            f"Es fehlen die charakteristischen Fragmente {missing}. Datei: {path}"
        )
    if len(literature) != 42:
        raise ValueError(
            "Die expanded Literaturdatei sollte genau 42 eindeutige Fragmente "
            f"enthalten, gefunden wurden {len(literature)}. Datei: {path}"
        )


def build_boundaries(literature: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    coordinates = sorted(set(literature["start"]) | set(literature["end"]))
    for coordinate in coordinates:
        starts = literature.loc[literature["start"] == coordinate, "fragment"].tolist()
        ends = literature.loc[literature["end"] == coordinate, "fragment"].tolist()
        labels = [f"start:{name}" for name in starts] + [f"end:{name}" for name in ends]
        rows.append(
            {
                "boundary_coordinate": int(coordinate),
                "fragments_starting": "|".join(sorted(starts)),
                "fragments_ending": "|".join(sorted(ends)),
                "boundary_labels": "|".join(sorted(labels)),
            }
        )
    return pd.DataFrame(rows)


def build_sections(literature: pd.DataFrame) -> pd.DataFrame:
    indexed = literature.set_index("fragment")
    missing = [fragment for fragment in SECTION_ORDER if fragment not in indexed.index]
    if missing:
        raise ValueError(f"Grundlegende 47S-Abschnitte fehlen: {missing}")
    sections = indexed.loc[SECTION_ORDER, ["start", "end"]].reset_index()
    sections["display_name"] = sections["fragment"].replace(
        {"5ETS": "5'ETS", "5-8S": "5.8S", "3ETS": "3'ETS"}
    )
    return sections


def load_samples(
    conditions: dict[str, list[Path]],
) -> tuple[list[dict], pd.DataFrame]:
    samples: list[dict] = []
    meta_rows: list[dict] = []
    for condition in CONDITION_ORDER:
        paths = list(conditions[condition])
        if len(paths) != EXPECTED_REPLICATES:
            raise ValueError(
                f"{condition}: {len(paths)} Samples gefunden; erwartet: 3."
            )
        for replicate_position, path in enumerate(paths, start=1):
            frame = load_intensity_matrix(path)
            required = {"start", "end", "n_reads"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path}: fehlende Spalten {sorted(missing)}")
            grouped = (
                frame.groupby(["start", "end"], as_index=False)["n_reads"]
                .sum()
                .sort_values(["start", "end"])
                .reset_index(drop=True)
            )
            grouped["start"] = pd.to_numeric(grouped["start"], errors="raise").astype(
                np.int64
            )
            grouped["end"] = pd.to_numeric(grouped["end"], errors="raise").astype(
                np.int64
            )
            grouped["n_reads"] = pd.to_numeric(
                grouped["n_reads"], errors="raise"
            ).astype(float)
            grouped = grouped.loc[
                (grouped["end"] > grouped["start"]) & (grouped["n_reads"] > 0)
            ].copy()
            total_reads = float(grouped["n_reads"].sum())
            if total_reads <= 0:
                raise ValueError(f"Keine positiven Reads in {path}")
            sample_name = path.parent.parent.name
            sample_id = f"{condition}__{sample_name}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "condition": condition,
                    "replicate_position": replicate_position,
                    "source_file": str(path),
                    "total_reads": total_reads,
                    "regions": grouped,
                }
            )
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "condition": condition,
                    "replicate_position": replicate_position,
                    "source_file": str(path),
                    "n_observed_regions": len(grouped),
                    "total_reads": total_reads,
                }
            )
    return samples, pd.DataFrame(meta_rows)


def base_sample_fields(sample: dict) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "sample_name": sample["sample_name"],
        "condition": sample["condition"],
        "replicate_position": sample["replicate_position"],
        "total_reads": sample["total_reads"],
    }


def assign_intermediate_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    literature: pd.DataFrame,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordnet nach Grenzpaaren zu. -1 bedeutet kein Treffer."""
    n_regions = len(starts)
    best_index = np.full(n_regions, -1, dtype=np.int32)
    best_distance = np.full(n_regions, np.iinfo(np.int32).max, dtype=np.int32)
    best_start_delta = np.zeros(n_regions, dtype=np.int32)
    best_end_delta = np.zeros(n_regions, dtype=np.int32)
    for index, fragment in enumerate(literature.itertuples(index=False)):
        start_delta = starts - int(fragment.start)
        end_delta = ends - int(fragment.end)
        distance = np.abs(start_delta) + np.abs(end_delta)
        eligible = (
            (np.abs(start_delta) <= tolerance)
            & (np.abs(end_delta) <= tolerance)
            & (distance < best_distance)
        )
        best_index[eligible] = index
        best_distance[eligible] = distance[eligible].astype(np.int32)
        best_start_delta[eligible] = start_delta[eligible].astype(np.int32)
        best_end_delta[eligible] = end_delta[eligible].astype(np.int32)
    return best_index, best_distance, best_start_delta, best_end_delta


def compute_intermediate_profiles(
    samples: list[dict],
    literature: pd.DataFrame,
    tolerances: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict] = []
    assignment_rows: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    fragment_records = literature.to_dict("records")
    for sample in samples:
        frame = sample["regions"]
        starts = frame["start"].to_numpy(dtype=np.int64)
        ends = frame["end"].to_numpy(dtype=np.int64)
        counts = frame["n_reads"].to_numpy(dtype=float)
        for tolerance in tolerances:
            assigned, distances, start_deltas, end_deltas = assign_intermediate_indices(
                starts, ends, literature, tolerance
            )
            matched_mask = assigned >= 0
            matched_reads_total = float(counts[matched_mask].sum())
            qc_rows.append(
                {
                    **base_sample_fields(sample),
                    "tolerance_nt": tolerance,
                    "n_matched_regions": int(matched_mask.sum()),
                    "matched_raw_reads": matched_reads_total,
                    "matched_read_fraction": matched_reads_total / sample["total_reads"],
                }
            )
            for fragment_index, fragment in enumerate(fragment_records):
                mask = assigned == fragment_index
                raw_reads = float(counts[mask].sum())
                profile_rows.append(
                    {
                        **base_sample_fields(sample),
                        "tolerance_nt": tolerance,
                        "fragment": fragment["fragment"],
                        "literature_start": fragment["start"],
                        "literature_end": fragment["end"],
                        "n_matched_regions": int(mask.sum()),
                        "raw_reads": raw_reads,
                        "cpm": raw_reads / sample["total_reads"] * 1_000_000.0,
                    }
                )
            if matched_mask.any():
                matched_positions = np.flatnonzero(matched_mask)
                matched_frame = pd.DataFrame(
                    {
                        **{
                            key: value
                            for key, value in base_sample_fields(sample).items()
                            if key != "total_reads"
                        },
                        "tolerance_nt": tolerance,
                        "start": starts[matched_positions],
                        "end": ends[matched_positions],
                        "n_reads": counts[matched_positions],
                        "assigned_fragment": [
                            fragment_records[index]["fragment"]
                            for index in assigned[matched_positions]
                        ],
                        "boundary_distance_sum": distances[matched_positions],
                        "start_delta": start_deltas[matched_positions],
                        "end_delta": end_deltas[matched_positions],
                    }
                )
                assignment_rows.append(matched_frame)
    assignments = (
        pd.concat(assignment_rows, ignore_index=True)
        if assignment_rows
        else pd.DataFrame()
    )
    return pd.DataFrame(profile_rows), assignments, pd.DataFrame(qc_rows)


def assign_mro_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    literature: pd.DataFrame,
    minimum_mro: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Ordnet Regionen dem Literaturfragment mit dem hoechsten MRO zu.

    MRO(Q, T) = min(overlap / len(Q), overlap / len(T)).
    Sequenzidentitaet und Mapping-Qualitaet werden hier nicht geprueft.
    """
    if len(starts) != len(ends):
        raise ValueError("starts und ends besitzen unterschiedliche Laengen.")
    read_lengths = (ends - starts).astype(float)
    if np.any(read_lengths <= 0):
        raise ValueError("Alle Rohregionen muessen end > start erfuellen.")

    n_regions = len(starts)
    assigned = np.full(n_regions, -1, dtype=np.int32)
    best_mro = np.full(n_regions, -1.0, dtype=float)
    second_best_mro = np.full(n_regions, -1.0, dtype=float)
    best_query_fraction = np.zeros(n_regions, dtype=float)
    best_template_fraction = np.zeros(n_regions, dtype=float)
    n_tied_best = np.zeros(n_regions, dtype=np.int16)
    epsilon = 1e-12

    for fragment_index, fragment in enumerate(literature.itertuples(index=False)):
        fragment_start = int(fragment.start)
        fragment_end = int(fragment.end)
        fragment_length = float(fragment_end - fragment_start)
        if fragment_length <= 0:
            raise ValueError(f"Ungueltige Literaturgrenzen fuer {fragment.fragment}.")

        overlap = np.maximum(
            0,
            np.minimum(ends, fragment_end) - np.maximum(starts, fragment_start),
        ).astype(float)
        query_fraction = overlap / read_lengths
        template_fraction = overlap / fragment_length
        mro = np.minimum(query_fraction, template_fraction)

        better = mro > best_mro + epsilon
        tied = (~better) & (mro > 0) & np.isclose(
            mro, best_mro, rtol=0.0, atol=epsilon
        )

        second_best_mro = np.where(
            better,
            best_mro,
            np.maximum(second_best_mro, mro),
        )
        assigned = np.where(better, fragment_index, assigned)
        best_mro = np.where(better, mro, best_mro)
        best_query_fraction = np.where(
            better, query_fraction, best_query_fraction
        )
        best_template_fraction = np.where(
            better, template_fraction, best_template_fraction
        )
        n_tied_best = np.where(
            better,
            1,
            n_tied_best + tied.astype(np.int16),
        ).astype(np.int16)

    valid = (best_mro > 0) & (best_mro >= minimum_mro)
    assigned[~valid] = -1
    best_mro[~valid] = np.nan
    second_best_mro = np.maximum(second_best_mro, 0.0)
    second_best_mro[~valid] = np.nan
    best_query_fraction[~valid] = np.nan
    best_template_fraction[~valid] = np.nan
    n_tied_best[~valid] = 0
    mro_margin = best_mro - second_best_mro
    return (
        assigned,
        best_mro,
        second_best_mro,
        mro_margin,
        best_query_fraction,
        best_template_fraction,
        n_tied_best,
    )


def compute_mro_intermediate_profiles(
    samples: list[dict],
    literature: pd.DataFrame,
    minimum_mro: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Berechnet exklusive MRO-Zuordnungen und readgewichtete Profile."""
    profile_rows: list[dict] = []
    assignment_frames: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    fragment_records = literature.to_dict("records")
    fragment_names = literature["fragment"].astype(str).to_numpy()
    fragment_starts = literature["start"].to_numpy(dtype=float)
    fragment_ends = literature["end"].to_numpy(dtype=float)

    for sample in samples:
        frame = sample["regions"]
        starts = frame["start"].to_numpy(dtype=np.int64)
        ends = frame["end"].to_numpy(dtype=np.int64)
        counts = frame["n_reads"].to_numpy(dtype=float)
        read_lengths = (ends - starts).astype(float)
        (
            assigned,
            best_mro,
            second_best_mro,
            mro_margin,
            query_fraction,
            template_fraction,
            n_tied_best,
        ) = assign_mro_indices(starts, ends, literature, minimum_mro)

        assigned_mask = assigned >= 0
        ambiguous_mask = assigned_mask & (n_tied_best > 1)
        assigned_reads_total = float(counts[assigned_mask].sum())
        unassigned_reads_total = float(counts[~assigned_mask].sum())
        ambiguous_reads_total = float(counts[ambiguous_mask].sum())
        qc_rows.append(
            {
                **base_sample_fields(sample),
                "minimum_mro": minimum_mro,
                "n_assigned_regions": int(assigned_mask.sum()),
                "assigned_raw_reads": assigned_reads_total,
                "assigned_read_fraction": (
                    assigned_reads_total / sample["total_reads"]
                ),
                "n_unassigned_regions": int((~assigned_mask).sum()),
                "unassigned_raw_reads": unassigned_reads_total,
                "unassigned_read_fraction": (
                    unassigned_reads_total / sample["total_reads"]
                ),
                "n_ambiguous_tied_regions": int(ambiguous_mask.sum()),
                "ambiguous_tied_raw_reads": ambiguous_reads_total,
                "ambiguous_tied_fraction_of_assigned_reads": (
                    ambiguous_reads_total / assigned_reads_total
                    if assigned_reads_total > 0
                    else np.nan
                ),
                "read_weighted_mean_best_mro": (
                    float(np.average(best_mro[assigned_mask], weights=counts[assigned_mask]))
                    if assigned_reads_total > 0
                    else np.nan
                ),
                "read_weighted_mean_mro_margin": (
                    float(np.average(mro_margin[assigned_mask], weights=counts[assigned_mask]))
                    if assigned_reads_total > 0
                    else np.nan
                ),
            }
        )

        for fragment_index, fragment in enumerate(fragment_records):
            mask = assigned == fragment_index
            raw_reads = float(counts[mask].sum())
            profile_rows.append(
                {
                    **base_sample_fields(sample),
                    "minimum_mro": minimum_mro,
                    "fragment": fragment["fragment"],
                    "literature_start": fragment["start"],
                    "literature_end": fragment["end"],
                    "n_assigned_regions": int(mask.sum()),
                    "assigned_raw_reads": raw_reads,
                    "cpm": raw_reads / sample["total_reads"] * 1_000_000.0,
                    "percent_of_all_sample_reads": (
                        100.0 * raw_reads / sample["total_reads"]
                    ),
                    "percent_of_assigned_reads": (
                        100.0 * raw_reads / assigned_reads_total
                        if assigned_reads_total > 0
                        else np.nan
                    ),
                    "read_weighted_mean_mro": (
                        float(np.average(best_mro[mask], weights=counts[mask]))
                        if raw_reads > 0
                        else np.nan
                    ),
                    "read_weighted_mean_mro_margin": (
                        float(np.average(mro_margin[mask], weights=counts[mask]))
                        if raw_reads > 0
                        else np.nan
                    ),
                }
            )

        labels = np.full(len(starts), "UNASSIGNED", dtype=object)
        assigned_starts = np.full(len(starts), np.nan, dtype=float)
        assigned_ends = np.full(len(starts), np.nan, dtype=float)
        labels[assigned_mask] = fragment_names[assigned[assigned_mask]]
        assigned_starts[assigned_mask] = fragment_starts[assigned[assigned_mask]]
        assigned_ends[assigned_mask] = fragment_ends[assigned[assigned_mask]]
        assignment_frames.append(
            pd.DataFrame(
                {
                    **{
                        key: value
                        for key, value in base_sample_fields(sample).items()
                        if key != "total_reads"
                    },
                    "start": starts,
                    "end": ends,
                    "n_reads": counts,
                    "assigned_fragment": labels,
                    "literature_start": assigned_starts,
                    "literature_end": assigned_ends,
                    "overlap_nt_with_assigned_fragment": query_fraction * read_lengths,
                    "query_overlap_fraction": query_fraction,
                    "template_overlap_fraction": template_fraction,
                    "best_mro": best_mro,
                    "second_best_mro": second_best_mro,
                    "mro_margin": mro_margin,
                    "n_tied_best": n_tied_best,
                    "ambiguous_tied_best": n_tied_best > 1,
                    "minimum_mro": minimum_mro,
                }
            )
        )

    assignments = pd.concat(assignment_frames, ignore_index=True)
    return pd.DataFrame(profile_rows), assignments, pd.DataFrame(qc_rows)


def summarize_mro_conditions(
    mro_profile: pd.DataFrame,
    pseudocount_cpm: float,
) -> pd.DataFrame:
    """Fasst MRO-Profile je Condition und Fragment zusammen."""
    rows: list[dict] = []
    for fragment, fragment_group in mro_profile.groupby("fragment", sort=False):
        metadata = fragment_group.iloc[0]
        ctrl = fragment_group.loc[
            fragment_group["condition"] == "Ctrl", "cpm"
        ].to_numpy(dtype=float)
        if len(ctrl) != EXPECTED_REPLICATES:
            raise AssertionError(f"MRO/{fragment}: Ctrl besitzt nicht drei Werte.")

        for condition in CONDITION_ORDER:
            condition_group = fragment_group.loc[
                fragment_group["condition"] == condition
            ]
            values = condition_group["cpm"].to_numpy(dtype=float)
            if len(values) != EXPECTED_REPLICATES:
                raise AssertionError(
                    f"MRO/{fragment}/{condition}: nicht drei Werte vorhanden."
                )
            ctrl_mean = float(ctrl.mean())
            condition_mean = float(values.mean())
            difference = condition_mean - ctrl_mean
            direction = (
                "higher" if difference > 0 else "lower" if difference < 0 else "equal"
            )
            n_above = int(np.sum(values > ctrl_mean))
            n_below = int(np.sum(values < ctrl_mean))
            rows.append(
                {
                    "fragment": fragment,
                    "literature_start": metadata["literature_start"],
                    "literature_end": metadata["literature_end"],
                    "condition": condition,
                    "n_samples": len(values),
                    "mean_cpm": condition_mean,
                    "sd_cpm": float(np.std(values, ddof=1)),
                    "median_cpm": float(np.median(values)),
                    "minimum_cpm": float(np.min(values)),
                    "maximum_cpm": float(np.max(values)),
                    "mean_percent_of_all_sample_reads": float(
                        condition_group["percent_of_all_sample_reads"].mean()
                    ),
                    "mean_percent_of_assigned_reads": float(
                        condition_group["percent_of_assigned_reads"].mean()
                    ),
                    "ctrl_mean_cpm": ctrl_mean,
                    "difference_to_ctrl_cpm": difference,
                    "log2_fold_change_vs_ctrl": (
                        0.0
                        if condition == "Ctrl"
                        else math.log2(
                            (condition_mean + pseudocount_cpm)
                            / (ctrl_mean + pseudocount_cpm)
                        )
                    ),
                    "cliffs_delta_vs_ctrl": (
                        0.0 if condition == "Ctrl" else cliffs_delta(ctrl, values)
                    ),
                    "exact_permutation_p_two_sided": (
                        1.0
                        if condition == "Ctrl"
                        else exact_permutation_p(ctrl, values)
                    ),
                    "direction_vs_ctrl": (
                        "reference" if condition == "Ctrl" else direction
                    ),
                    "direction_consistent_all_three_replicates": (
                        True
                        if condition == "Ctrl"
                        else (
                            (direction == "higher" and n_above == EXPECTED_REPLICATES)
                            or (
                                direction == "lower"
                                and n_below == EXPECTED_REPLICATES
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_boundary_profiles(
    samples: list[dict],
    boundaries: pd.DataFrame,
    tolerances: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in samples:
        frame = sample["regions"]
        starts = frame["start"].to_numpy(dtype=np.int64)
        ends = frame["end"].to_numpy(dtype=np.int64)
        counts = frame["n_reads"].to_numpy(dtype=float)
        for tolerance in tolerances:
            for boundary in boundaries.itertuples(index=False):
                start_mask = np.abs(starts - boundary.boundary_coordinate) <= tolerance
                end_mask = np.abs(ends - boundary.boundary_coordinate) <= tolerance
                either_mask = start_mask | end_mask
                start_reads = float(counts[start_mask].sum())
                end_reads = float(counts[end_mask].sum())
                either_reads = float(counts[either_mask].sum())
                rows.append(
                    {
                        **base_sample_fields(sample),
                        "tolerance_nt": tolerance,
                        "boundary_coordinate": boundary.boundary_coordinate,
                        "boundary_labels": boundary.boundary_labels,
                        "start_raw_reads": start_reads,
                        "end_raw_reads": end_reads,
                        "either_raw_reads": either_reads,
                        "start_cpm": start_reads / sample["total_reads"] * 1_000_000.0,
                        "end_cpm": end_reads / sample["total_reads"] * 1_000_000.0,
                        "either_cpm": either_reads / sample["total_reads"] * 1_000_000.0,
                    }
                )
    return pd.DataFrame(rows)


def compute_section_profiles(
    samples: list[dict], sections: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in samples:
        frame = sample["regions"]
        starts = frame["start"].to_numpy(dtype=np.int64)
        ends = frame["end"].to_numpy(dtype=np.int64)
        counts = frame["n_reads"].to_numpy(dtype=float)
        lengths = (ends - starts).astype(float)
        for section in sections.itertuples(index=False):
            overlap = np.maximum(
                0,
                np.minimum(ends, int(section.end))
                - np.maximum(starts, int(section.start)),
            ).astype(float)
            any_overlap = overlap > 0
            fractional_reads = float(np.sum(counts * overlap / lengths))
            any_overlap_reads = float(counts[any_overlap].sum())
            rows.append(
                {
                    **base_sample_fields(sample),
                    "section": section.fragment,
                    "section_display_name": section.display_name,
                    "section_start": section.start,
                    "section_end": section.end,
                    "fractional_raw_read_equivalents": fractional_reads,
                    "fractional_cpm": (
                        fractional_reads / sample["total_reads"] * 1_000_000.0
                    ),
                    "any_overlap_raw_reads": any_overlap_reads,
                    "any_overlap_cpm": (
                        any_overlap_reads / sample["total_reads"] * 1_000_000.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_fragment_containment_profiles(
    samples: list[dict], literature: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Berechnet nicht-exklusive Containment-Prozente und die Zuordnung
    zum kleinsten enthaltenden Fragment. Die Intervalllage allein belegt
    kein reifes 28S-Molekuel.
    """
    fragment_rows: list[dict] = []
    exclusive_rows: list[dict] = []
    literature_records = literature.to_dict("records")
    fragment_lengths = (literature["end"] - literature["start"]).to_numpy(
        dtype=np.int64
    )
    fragment_47s = literature.loc[literature["fragment"] == "47S"]
    if len(fragment_47s) != 1:
        raise ValueError("Literaturtabelle muss genau ein Fragment '47S' enthalten.")
    start_47s = int(fragment_47s.iloc[0]["start"])
    end_47s = int(fragment_47s.iloc[0]["end"])

    for sample in samples:
        frame = sample["regions"]
        starts = frame["start"].to_numpy(dtype=np.int64)
        ends = frame["end"].to_numpy(dtype=np.int64)
        counts = frame["n_reads"].to_numpy(dtype=float)
        within_47s = (starts >= start_47s) & (ends <= end_47s)
        reads_within_47s = float(counts[within_47s].sum())

        best_index = np.full(len(frame), -1, dtype=np.int32)
        best_length = np.full(
            len(frame), np.iinfo(np.int64).max, dtype=np.int64
        )
        for fragment_index, fragment in enumerate(literature_records):
            contained = (
                (starts >= int(fragment["start"]))
                & (ends <= int(fragment["end"]))
            )
            length = int(fragment_lengths[fragment_index])
            update = contained & (length < best_length)
            best_index[update] = fragment_index
            best_length[update] = length

            contained_reads = float(counts[contained].sum())
            fragment_rows.append(
                {
                    **base_sample_fields(sample),
                    "fragment": fragment["fragment"],
                    "literature_start": fragment["start"],
                    "literature_end": fragment["end"],
                    "n_distinct_regions_fully_contained": int(contained.sum()),
                    "contained_raw_reads": contained_reads,
                    "percent_of_all_sample_reads": (
                        100.0 * contained_reads / sample["total_reads"]
                    ),
                    "reads_fully_contained_in_47S": reads_within_47s,
                    "percent_of_47S_contained_reads": (
                        100.0 * contained_reads / reads_within_47s
                        if reads_within_47s > 0
                        else np.nan
                    ),
                }
            )

        for fragment_index, fragment in enumerate(literature_records):
            assigned = best_index == fragment_index
            assigned_reads = float(counts[assigned].sum())
            exclusive_rows.append(
                {
                    **base_sample_fields(sample),
                    "assigned_smallest_fragment": fragment["fragment"],
                    "literature_start": fragment["start"],
                    "literature_end": fragment["end"],
                    "n_distinct_regions": int(assigned.sum()),
                    "assigned_raw_reads": assigned_reads,
                    "percent_of_all_sample_reads": (
                        100.0 * assigned_reads / sample["total_reads"]
                    ),
                    "percent_of_47S_contained_reads": (
                        100.0 * assigned_reads / reads_within_47s
                        if reads_within_47s > 0
                        else np.nan
                    ),
                }
            )
        unassigned = best_index < 0
        unassigned_reads = float(counts[unassigned].sum())
        exclusive_rows.append(
            {
                **base_sample_fields(sample),
                "assigned_smallest_fragment": "UNASSIGNED_OUTSIDE_47S",
                "literature_start": np.nan,
                "literature_end": np.nan,
                "n_distinct_regions": int(unassigned.sum()),
                "assigned_raw_reads": unassigned_reads,
                "percent_of_all_sample_reads": (
                    100.0 * unassigned_reads / sample["total_reads"]
                ),
                "percent_of_47S_contained_reads": np.nan,
            }
        )
    return pd.DataFrame(fragment_rows), pd.DataFrame(exclusive_rows)


def summarize_fragment_containment(
    containment: pd.DataFrame, pseudocount_percent: float = 0.001
) -> pd.DataFrame:
    rows: list[dict] = []
    for fragment, fragment_group in containment.groupby("fragment", sort=False):
        metadata = fragment_group.iloc[0]
        ctrl = fragment_group.loc[
            fragment_group["condition"] == "Ctrl",
            "percent_of_47S_contained_reads",
        ].to_numpy(dtype=float)
        for condition in CONDITION_ORDER:
            values = fragment_group.loc[
                fragment_group["condition"] == condition,
                "percent_of_47S_contained_reads",
            ].to_numpy(dtype=float)
            row = {
                "fragment": fragment,
                "literature_start": metadata["literature_start"],
                "literature_end": metadata["literature_end"],
                "condition": condition,
                "n_samples": len(values),
                "mean_percent_of_47S_reads": float(np.nanmean(values)),
                "sd_percent_of_47S_reads": float(np.nanstd(values, ddof=1)),
                "median_percent_of_47S_reads": float(np.nanmedian(values)),
                "minimum_percent_of_47S_reads": float(np.nanmin(values)),
                "maximum_percent_of_47S_reads": float(np.nanmax(values)),
            }
            if condition == "Ctrl":
                row.update(
                    {
                        "ctrl_mean_percent": float(np.nanmean(ctrl)),
                        "difference_to_ctrl_percentage_points": 0.0,
                        "log2_ratio_vs_ctrl": 0.0,
                        "cliffs_delta_vs_ctrl": 0.0,
                        "exact_permutation_p_two_sided": 1.0,
                        "direction_vs_ctrl": "reference",
                        "direction_consistent_all_three_replicates": True,
                    }
                )
            else:
                ctrl_mean = float(np.nanmean(ctrl))
                condition_mean = float(np.nanmean(values))
                difference = condition_mean - ctrl_mean
                n_above = int(np.sum(values > ctrl_mean))
                n_below = int(np.sum(values < ctrl_mean))
                direction = (
                    "higher" if difference > 0 else "lower" if difference < 0 else "equal"
                )
                row.update(
                    {
                        "ctrl_mean_percent": ctrl_mean,
                        "difference_to_ctrl_percentage_points": difference,
                        "log2_ratio_vs_ctrl": math.log2(
                            (condition_mean + pseudocount_percent)
                            / (ctrl_mean + pseudocount_percent)
                        ),
                        "cliffs_delta_vs_ctrl": cliffs_delta(ctrl, values),
                        "exact_permutation_p_two_sided": exact_permutation_p(
                            ctrl, values
                        ),
                        "direction_vs_ctrl": direction,
                        "direction_consistent_all_three_replicates": (
                            (direction == "higher" and n_above == EXPECTED_REPLICATES)
                            or (direction == "lower" and n_below == EXPECTED_REPLICATES)
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def exact_permutation_p(ctrl: np.ndarray, kd: np.ndarray) -> float:
    values = np.concatenate([ctrl, kd]).astype(float)
    observed = abs(float(kd.mean() - ctrl.mean()))
    extreme = 0
    total = 0
    indices = np.arange(len(values))
    for kd_indices_tuple in itertools.combinations(indices, len(kd)):
        kd_indices = np.asarray(kd_indices_tuple, dtype=int)
        ctrl_indices = indices[~np.isin(indices, kd_indices)]
        difference = abs(float(values[kd_indices].mean() - values[ctrl_indices].mean()))
        extreme += int(difference >= observed - 1e-12)
        total += 1
    return extreme / total


def hedges_g(ctrl: np.ndarray, kd: np.ndarray) -> float:
    n_ctrl = len(ctrl)
    n_kd = len(kd)
    degrees = n_ctrl + n_kd - 2
    pooled_variance = (
        (n_ctrl - 1) * np.var(ctrl, ddof=1)
        + (n_kd - 1) * np.var(kd, ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        difference = float(kd.mean() - ctrl.mean())
        return 0.0 if difference == 0 else math.copysign(np.inf, difference)
    cohens_d = float(kd.mean() - ctrl.mean()) / math.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
    return correction * cohens_d


def cliffs_delta(ctrl: np.ndarray, kd: np.ndarray) -> float:
    greater = sum(float(kd_value > ctrl_value) for kd_value in kd for ctrl_value in ctrl)
    lower = sum(float(kd_value < ctrl_value) for kd_value in kd for ctrl_value in ctrl)
    return (greater - lower) / (len(kd) * len(ctrl))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def summarize_against_ctrl(
    long_values: pd.DataFrame,
    analysis: str,
    value_column: str,
    feature_columns: list[str],
    tolerance_column: str | None,
    pseudocount_cpm: float,
) -> pd.DataFrame:
    group_columns = feature_columns.copy()
    if tolerance_column is not None:
        group_columns.insert(0, tolerance_column)
    rows: list[dict] = []
    for keys, feature_group in long_values.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        feature_data = dict(zip(group_columns, keys))
        ctrl = feature_group.loc[
            feature_group["condition"] == "Ctrl", value_column
        ].to_numpy(dtype=float)
        if len(ctrl) != EXPECTED_REPLICATES:
            raise AssertionError(f"{analysis}: Ctrl besitzt nicht drei Werte.")
        for condition in KD_ORDER:
            kd = feature_group.loc[
                feature_group["condition"] == condition, value_column
            ].to_numpy(dtype=float)
            if len(kd) != EXPECTED_REPLICATES:
                raise AssertionError(
                    f"{analysis}/{condition}: nicht drei Werte vorhanden."
                )
            ctrl_mean = float(ctrl.mean())
            kd_mean = float(kd.mean())
            difference = kd_mean - ctrl_mean
            n_above = int(np.sum(kd > ctrl_mean))
            n_below = int(np.sum(kd < ctrl_mean))
            direction = "higher" if difference > 0 else "lower" if difference < 0 else "equal"
            rows.append(
                {
                    "analysis": analysis,
                    **feature_data,
                    "condition": condition,
                    "n_ctrl": len(ctrl),
                    "n_kd": len(kd),
                    "ctrl_mean_cpm": ctrl_mean,
                    "ctrl_sd_cpm": float(np.std(ctrl, ddof=1)),
                    "kd_mean_cpm": kd_mean,
                    "kd_sd_cpm": float(np.std(kd, ddof=1)),
                    "mean_difference_cpm": difference,
                    "log2_fold_change_vs_ctrl": math.log2(
                        (kd_mean + pseudocount_cpm)
                        / (ctrl_mean + pseudocount_cpm)
                    ),
                    "pseudocount_cpm": pseudocount_cpm,
                    "hedges_g": hedges_g(ctrl, kd),
                    "cliffs_delta": cliffs_delta(ctrl, kd),
                    "exact_permutation_p_two_sided": exact_permutation_p(ctrl, kd),
                    "n_kd_replicates_above_ctrl_mean": n_above,
                    "n_kd_replicates_below_ctrl_mean": n_below,
                    "direction_vs_ctrl": direction,
                    "direction_consistent_all_three_replicates": (
                        (direction == "higher" and n_above == EXPECTED_REPLICATES)
                        or (direction == "lower" and n_below == EXPECTED_REPLICATES)
                    ),
                    "maximum_group_mean_cpm": max(ctrl_mean, kd_mean),
                }
            )
    result = pd.DataFrame(rows)
    result["bh_adjusted_permutation_p"] = np.nan
    correction_groups = ["analysis"]
    if tolerance_column is not None:
        correction_groups.append(tolerance_column)
    for _, indices in result.groupby(correction_groups).groups.items():
        result.loc[indices, "bh_adjusted_permutation_p"] = benjamini_hochberg(
            result.loc[indices, "exact_permutation_p_two_sided"]
        )
    return result


def build_comparison_tables(
    mro_intermediate: pd.DataFrame,
    intermediate: pd.DataFrame,
    boundary: pd.DataFrame,
    section: pd.DataFrame,
    pseudocount_cpm: float,
) -> pd.DataFrame:
    mro_summary = summarize_against_ctrl(
        mro_intermediate,
        analysis="mro_intermediate",
        value_column="cpm",
        feature_columns=["fragment", "literature_start", "literature_end"],
        tolerance_column=None,
        pseudocount_cpm=pseudocount_cpm,
    )
    intermediate_summary = summarize_against_ctrl(
        intermediate,
        analysis="intermediate",
        value_column="cpm",
        feature_columns=["fragment", "literature_start", "literature_end"],
        tolerance_column="tolerance_nt",
        pseudocount_cpm=pseudocount_cpm,
    )
    boundary_summary = summarize_against_ctrl(
        boundary,
        analysis="boundary_either",
        value_column="either_cpm",
        feature_columns=["boundary_coordinate", "boundary_labels"],
        tolerance_column="tolerance_nt",
        pseudocount_cpm=pseudocount_cpm,
    )
    section_summary = summarize_against_ctrl(
        section,
        analysis="section_fractional",
        value_column="fractional_cpm",
        feature_columns=[
            "section",
            "section_display_name",
            "section_start",
            "section_end",
        ],
        tolerance_column=None,
        pseudocount_cpm=pseudocount_cpm,
    )
    return pd.concat(
        [mro_summary, intermediate_summary, boundary_summary, section_summary],
        ignore_index=True,
        sort=False,
    )


def select_top_consistent_changes(
    comparisons: pd.DataFrame, minimum_mean_cpm: float
) -> pd.DataFrame:
    selected = comparisons.loc[
        comparisons["direction_consistent_all_three_replicates"]
        & (comparisons["maximum_group_mean_cpm"] >= minimum_mean_cpm)
    ].copy()
    selected["absolute_log2_fold_change"] = selected[
        "log2_fold_change_vs_ctrl"
    ].abs()
    return selected.sort_values(
        ["analysis", "condition", "absolute_log2_fold_change"],
        ascending=[True, True, False],
    )


def display_feature_labels(frame: pd.DataFrame, analysis: str) -> pd.Series:
    if analysis in {"mro_intermediate", "intermediate"}:
        return frame["fragment"].astype(str)
    if analysis == "boundary_either":
        return frame["boundary_coordinate"].astype(int).astype(str)
    if analysis == "section_fractional":
        return frame["section_display_name"].astype(str)
    raise ValueError(analysis)


def save_heatmap(
    comparisons: pd.DataFrame,
    analysis: str,
    output_stem: Path,
    title: str,
    tolerance: int | None = None,
) -> None:
    selected = comparisons.loc[comparisons["analysis"] == analysis].copy()
    if tolerance is not None:
        selected = selected.loc[selected["tolerance_nt"] == tolerance].copy()
    selected["feature_label"] = display_feature_labels(selected, analysis)
    matrix = selected.pivot(
        index="condition",
        columns="feature_label",
        values="log2_fold_change_vs_ctrl",
    ).reindex(index=KD_ORDER)
    if analysis in {"mro_intermediate", "intermediate"}:
        available = selected.drop_duplicates("fragment").sort_values(
            ["literature_start", "literature_end"]
        )["fragment"].astype(str).tolist()
        matrix = matrix.reindex(columns=available)
    elif analysis == "boundary_either":
        available = sorted(matrix.columns, key=int)
        matrix = matrix.reindex(columns=available)
    elif analysis == "section_fractional":
        names = {
            "5ETS": "5'ETS",
            "18S": "18S",
            "ITS1": "ITS1",
            "5-8S": "5.8S",
            "ITS2": "ITS2",
            "28S": "28S",
            "3ETS": "3'ETS",
        }
        matrix = matrix.reindex(columns=[names[name] for name in SECTION_ORDER])

    values = matrix.to_numpy(dtype=float)
    width = max(8.0, 0.58 * len(matrix.columns) + 3.0)
    height = max(4.5, 0.65 * len(matrix.index) + 2.0)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        vmin=-LOG2FC_COLOR_LIMIT,
        vmax=LOG2FC_COLOR_LIMIT,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_xlabel("RNA45SN1 feature")
    axis.set_ylabel("Knockdown vs Ctrl")
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("log2 fold change vs Ctrl")
    colorbar.set_ticks(
        np.arange(-LOG2FC_COLOR_LIMIT, LOG2FC_COLOR_LIMIT + 1.0, 1.0)
    )
    axis.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def save_fragment_containment_heatmap(
    containment_summary: pd.DataFrame,
    literature: pd.DataFrame,
    output_stem: Path,
) -> None:
    """Zeigt mittlere Containment-Anteile an den 47S-Reads."""
    matrix = containment_summary.pivot(
        index="condition",
        columns="fragment",
        values="mean_percent_of_47S_reads",
    ).reindex(index=CONDITION_ORDER)
    fragment_order = literature.sort_values(["start", "end"])["fragment"].tolist()
    matrix = matrix.reindex(columns=fragment_order)
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanquantile(values, 0.98)))
    fig, axis = plt.subplots(
        figsize=(max(12.0, 0.6 * len(matrix.columns) + 3.0), 6.5)
    )
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_xlabel("Literature interval (non-exclusive)")
    axis.set_ylabel("Condition")
    axis.set_title("Reads fully contained within RNA45SN1 literature intervals")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean percentage of reads contained in 47S")
    fig.tight_layout()
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def summarize_exclusive_assignments(exclusive: pd.DataFrame) -> pd.DataFrame:
    return (
        exclusive.groupby(["condition", "assigned_smallest_fragment"], as_index=False)
        .agg(
            mean_percent_of_all_reads=("percent_of_all_sample_reads", "mean"),
            sd_percent_of_all_reads=("percent_of_all_sample_reads", "std"),
            mean_percent_of_47S_reads=("percent_of_47S_contained_reads", "mean"),
            sd_percent_of_47S_reads=("percent_of_47S_contained_reads", "std"),
        )
    )


def save_exclusive_assignment_plot(
    exclusive_summary: pd.DataFrame,
    output_stem: Path,
    top_n: int = 12,
) -> None:
    """Zeigt die exklusive Zusammensetzung mit insgesamt 100 %."""
    global_means = (
        exclusive_summary.groupby("assigned_smallest_fragment")[
            "mean_percent_of_all_reads"
        ]
        .mean()
        .sort_values(ascending=False)
    )
    keep = global_means.head(top_n).index.tolist()
    prepared = exclusive_summary.copy()
    prepared["plot_category"] = np.where(
        prepared["assigned_smallest_fragment"].isin(keep),
        prepared["assigned_smallest_fragment"],
        "OTHER",
    )
    prepared = (
        prepared.groupby(["condition", "plot_category"], as_index=False)[
            "mean_percent_of_all_reads"
        ]
        .sum()
    )
    category_order = keep + (["OTHER"] if (prepared["plot_category"] == "OTHER").any() else [])
    matrix = prepared.pivot(
        index="condition",
        columns="plot_category",
        values="mean_percent_of_all_reads",
    ).fillna(0.0).reindex(index=CONDITION_ORDER, columns=category_order, fill_value=0.0)

    fig, axis = plt.subplots(figsize=(11.5, 6.5))
    bottom = np.zeros(len(matrix), dtype=float)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(matrix.columns)))
    x = np.arange(len(matrix.index))
    for color, category in zip(colors, matrix.columns):
        values = matrix[category].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, label=category, color=color, width=0.75)
        bottom += values
    axis.set_xticks(x)
    axis.set_xticklabels(matrix.index, rotation=25, ha="right")
    axis.set_ylabel("Mean percentage of all reads")
    axis.set_xlabel("Condition")
    axis.set_ylim(0, max(100.0, float(bottom.max()) * 1.03))
    axis.set_title("Exclusive assignment to smallest containing literature interval")
    axis.legend(title="Assigned interval", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def feature_description(row: pd.Series) -> str:
    if row["analysis"] in {"mro_intermediate", "intermediate"}:
        return str(row["fragment"])
    if row["analysis"] == "boundary_either":
        return f"boundary {int(row['boundary_coordinate'])}"
    return str(row["section_display_name"])


def write_report(
    meta: pd.DataFrame,
    mro_qc: pd.DataFrame,
    intermediate_qc: pd.DataFrame,
    comparisons: pd.DataFrame,
    top_consistent: pd.DataFrame,
    containment_summary: pd.DataFrame,
    tolerances: tuple[int, ...],
    minimum_mro: float,
    minimum_mean_cpm: float,
    output_dir: Path,
) -> None:
    lines = [
        "RNA45SN1 INTERMEDIATE PROFILES ACROSS CONDITIONS",
        "=" * 76,
        "",
        "Design",
        f"- Samples: {len(meta)} ({len(CONDITION_ORDER)} Conditions x 3 Replikate)",
        "- Eingang: alle beobachteten Rohregionen mit n_reads",
        f"- Grenztoleranzen: {', '.join(str(value) for value in tolerances)} nt",
        "- Vergleich: jeder Knockdown gegen Ctrl (3 vs 3)",
        "",
        "Wichtige methodische Trennung",
        "- MRO intermediate: exklusive Pastore/NanoRibolyzer-nahe Zuordnung",
        "  ueber den maximalen minimalen reziproken Overlap.",
        "- Endpoint intermediate: beide Read-Grenzen passen; eindeutige",
        "  Best-Zuordnung innerhalb der angegebenen Toleranz.",
        "- Boundary: mindestens eine Read-Grenze passt.",
        "- Section: ueberlappungsgewichtetes Signal ueber Grundabschnitte.",
        "- Containment: Anteil der Reads, die vollstaendig innerhalb eines",
        "  Literaturintervalls liegen (nicht exklusiv).",
        "- Smallest-containing: eindeutige geometrische Zuordnung; dies ist",
        "  keine Behauptung, dass der Read ein vollstaendiges Intermediate ist.",
        "",
        "MRO-Zuordnungsqualitaet",
    ]
    lines.append(
        f"- Minimum MRO: {minimum_mro:.3f}; mittlerer zugeordneter Read-Anteil "
        f"{100 * mro_qc['assigned_read_fraction'].mean():.3f}% "
        f"(min={100 * mro_qc['assigned_read_fraction'].min():.3f}%, "
        f"max={100 * mro_qc['assigned_read_fraction'].max():.3f}%)"
    )
    lines.append(
        "- Mittlerer Anteil readgewichteter Gleichstandszuordnungen an allen "
        f"zugeordneten Reads: "
        f"{100 * mro_qc['ambiguous_tied_fraction_of_assigned_reads'].mean():.3f}%"
    )
    lines.extend(["", "Strenge Endpunkt-Zuordnungsrate pro Toleranz"])
    for tolerance, group in intermediate_qc.groupby("tolerance_nt"):
        lines.append(
            f"- +/-{int(tolerance)} nt: mittlerer gematchter Read-Anteil "
            f"{100 * group['matched_read_fraction'].mean():.3f}% "
            f"(min={100 * group['matched_read_fraction'].min():.3f}%, "
            f"max={100 * group['matched_read_fraction'].max():.3f}%)"
        )
    lines.extend(["", "Nicht-exklusive Fragment-Containment-Prozente"])
    for condition in CONDITION_ORDER:
        subset = containment_summary.loc[
            containment_summary["condition"] == condition
        ].sort_values("mean_percent_of_47S_reads", ascending=False)
        top = subset.loc[subset["fragment"] != "47S"].head(8)
        descriptions = [
            f"{row.fragment}={row.mean_percent_of_47S_reads:.2f}%"
            for row in top.itertuples(index=False)
        ]
        lines.append(f"- {condition}: " + "; ".join(descriptions))
    lines.extend(
        [
            "",
            f"Replikatkonsistente Veraenderungen (max. Gruppenmittel >= {minimum_mean_cpm} CPM)",
        ]
    )
    for analysis in ANALYSIS_ORDER:
        subset = top_consistent.loc[top_consistent["analysis"] == analysis]
        lines.append(f"- {analysis}: {len(subset)} Condition-Feature-Paare")
        for condition in KD_ORDER:
            top = subset.loc[subset["condition"] == condition].head(5)
            if top.empty:
                continue
            descriptions = [
                f"{feature_description(row)} ({row['direction_vs_ctrl']}, "
                f"log2FC={row['log2_fold_change_vs_ctrl']:.2f}, "
                f"Cliff={row['cliffs_delta']:.2f})"
                for _, row in top.iterrows()
            ]
            lines.append(f"  {condition}: " + "; ".join(descriptions))
    lines.extend(
        [
            "",
            "Statistische Grenze",
            "- Mit 3 vs 3 Samples besitzt der exakte zweiseitige Permutationstest",
            "  nur 20 moegliche Gruppenzuordnungen und minimal p=0.1.",
            "- Deshalb werden Effektstaerken und konsistente Richtung ueber alle",
            "  drei KD-Replikate berichtet; dies bleibt eine explorative interne Analyse.",
            "- Eine Pastore-basierte Bewertung der erwarteten Condition-Fragment-",
            "  Beziehungen ist nicht Bestandteil dieses Laufs.",
            "- Ein Read innerhalb der 28S-Grenzen wird als 28S-lokalisiert",
            "  beschrieben; ohne passende Endpunkte ist er nicht automatisch",
            "  ein vollstaendiges reifes 28S-Molekuel.",
        ]
    )
    (output_dir / "RNA45SN1_condition_profile_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literature-csv", type=Path, default=DEFAULT_LITERATURE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tolerances",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Vorab festgelegte Grenztoleranzen in nt (Standard: 1 5 10).",
    )
    parser.add_argument("--pseudocount-cpm", type=float, default=0.01)
    parser.add_argument(
        "--minimum-mro",
        type=float,
        default=0.0,
        help=(
            "Minimaler maximaler MRO fuer eine Zuordnung (Standard: 0.0; "
            "entspricht der koordinatenbasierten Argmax-Regel von NanoRibolyzer)."
        ),
    )
    parser.add_argument(
        "--minimum-mean-cpm-for-priority",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tolerances = tuple(sorted(set(args.tolerances)))
    if not tolerances or any(value < 0 for value in tolerances):
        raise ValueError("Toleranzen muessen nichtnegative Ganzzahlen sein.")
    if args.pseudocount_cpm <= 0:
        raise ValueError("--pseudocount-cpm muss positiv sein.")
    if not 0.0 <= args.minimum_mro <= 1.0:
        raise ValueError("--minimum-mro muss zwischen 0 und 1 liegen.")
    if args.minimum_mean_cpm_for_priority < 0:
        raise ValueError("--minimum-mean-cpm-for-priority darf nicht negativ sein.")


    literature = load_literature(args.literature_csv)
    validate_expanded_literature(literature, args.literature_csv)
    boundaries = build_boundaries(literature)
    sections = build_sections(literature)
    fragments = set(literature["fragment"])
    utp18_34s_templates = sorted(
        fragment for fragment in fragments if fragment == "34S" or fragment.startswith("34S")
    )

    samples, meta = load_samples(CONDITIONS)

    mro_intermediate, mro_assignments, mro_qc = compute_mro_intermediate_profiles(
        samples, literature, args.minimum_mro
    )
    mro_condition_summary = summarize_mro_conditions(
        mro_intermediate, args.pseudocount_cpm
    )

    intermediate, assignments, intermediate_qc = compute_intermediate_profiles(
        samples, literature, tolerances
    )

    boundary = compute_boundary_profiles(samples, boundaries, tolerances)
    section = compute_section_profiles(samples, sections)
    containment, exclusive_assignment = compute_fragment_containment_profiles(
        samples, literature
    )
    containment_summary = summarize_fragment_containment(containment)
    exclusive_summary = summarize_exclusive_assignments(exclusive_assignment)

    comparisons = build_comparison_tables(
        mro_intermediate,
        intermediate,
        boundary,
        section,
        args.pseudocount_cpm,
    )
    top_consistent = select_top_consistent_changes(
        comparisons, args.minimum_mean_cpm_for_priority
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    literature.to_csv(
        args.output_dir / "literature_intermediates_standardized.csv",
        sep=SEP,
        index=False,
    )
    boundaries.to_csv(
        args.output_dir / "literature_boundaries.csv", sep=SEP, index=False
    )
    sections.to_csv(
        args.output_dir / "RNA45SN1_basic_sections.csv", sep=SEP, index=False
    )
    meta.to_csv(args.output_dir / "sample_metadata.csv", sep=SEP, index=False)
    mro_intermediate.to_csv(
        args.output_dir / "sample_mro_intermediate_abundance.csv",
        sep=SEP,
        index=False,
    )
    mro_assignments.to_csv(
        args.output_dir / "mro_region_assignments.csv", sep=SEP, index=False
    )
    mro_qc.to_csv(
        args.output_dir / "mro_assignment_qc.csv", sep=SEP, index=False
    )
    mro_condition_summary.to_csv(
        args.output_dir / "condition_mro_intermediate_summary.csv",
        sep=SEP,
        index=False,
    )
    intermediate.to_csv(
        args.output_dir / "sample_intermediate_abundance.csv", sep=SEP, index=False
    )
    assignments.to_csv(
        args.output_dir / "matched_regions_to_intermediates.csv", sep=SEP, index=False
    )
    intermediate_qc.to_csv(
        args.output_dir / "intermediate_assignment_qc.csv", sep=SEP, index=False
    )
    boundary.to_csv(
        args.output_dir / "sample_boundary_usage.csv", sep=SEP, index=False
    )
    section.to_csv(
        args.output_dir / "sample_section_signal.csv", sep=SEP, index=False
    )
    containment.to_csv(
        args.output_dir / "sample_fragment_containment_percentages.csv",
        sep=SEP,
        index=False,
    )
    containment_summary.to_csv(
        args.output_dir / "condition_fragment_containment_summary.csv",
        sep=SEP,
        index=False,
    )
    exclusive_assignment.to_csv(
        args.output_dir / "sample_smallest_containing_fragment_assignment.csv",
        sep=SEP,
        index=False,
    )
    exclusive_summary.to_csv(
        args.output_dir / "condition_smallest_fragment_composition.csv",
        sep=SEP,
        index=False,
    )
    comparisons.to_csv(
        args.output_dir / "condition_vs_ctrl_all_features.csv", sep=SEP, index=False
    )
    comparisons.loc[comparisons["analysis"] == "mro_intermediate"].to_csv(
        args.output_dir / "condition_mro_intermediate_vs_ctrl.csv",
        sep=SEP,
        index=False,
    )
    top_consistent.to_csv(
        args.output_dir / "priority_consistent_condition_changes.csv",
        sep=SEP,
        index=False,
    )

    for tolerance in tolerances:
        save_heatmap(
            comparisons,
            "intermediate",
            args.output_dir / f"intermediate_log2fc_tolerance_{tolerance}nt",
            f"Intermediate-like RNA45SN1 molecules (boundary tolerance +/-{tolerance} nt)",
            tolerance=tolerance,
        )
        save_heatmap(
            comparisons,
            "boundary_either",
            args.output_dir / f"boundary_usage_log2fc_tolerance_{tolerance}nt",
            f"RNA45SN1 boundary usage (tolerance +/-{tolerance} nt)",
            tolerance=tolerance,
        )
    save_heatmap(
        comparisons,
        "mro_intermediate",
        args.output_dir / "mro_intermediate_log2fc",
        "Exclusive MRO-assigned RNA45SN1 intermediates",
    )
    save_heatmap(
        comparisons,
        "section_fractional",
        args.output_dir / "section_fractional_signal_log2fc",
        "Overlap-weighted 47S section signal",
    )
    save_fragment_containment_heatmap(
        containment_summary,
        literature,
        args.output_dir / "fragment_containment_percent_heatmap",
    )
    save_exclusive_assignment_plot(
        exclusive_summary,
        args.output_dir / "smallest_fragment_assignment_composition",
    )
    write_report(
        meta,
        mro_qc,
        intermediate_qc,
        comparisons,
        top_consistent,
        containment_summary,
        tolerances,
        args.minimum_mro,
        args.minimum_mean_cpm_for_priority,
        args.output_dir,
    )

    print(f"Ergebnisse gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    main()
