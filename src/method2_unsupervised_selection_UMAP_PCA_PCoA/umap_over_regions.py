from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler


STATS_DIR = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/"
    "method1_set_based_and_rule_based_region_selection_data/"
    "shared_regions_across_samples"
)

DATA_DIR = Path("/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/method2_unsupervised_selection_UMAP_PCA_PCoA_data")

CONDITIONS: list[tuple[str, str]] = [
    ("Ctrl", "Ctrl"),
    ("UTP18", "UTP18"),
    ("DIMT1L", "DIMT1L"),
    ("LAS1L", "LAS1L"),
    ("RPL3", "RPL3"),
    ("URB1", "URB1"),
    ("WBSCR22", "WBSCR22"),
]

CSV_NAME = "nuc_all_shared_regions_by_norm_cv.csv"
SEP = ";"

SIGNAL_COL = "norm_mean"

OUTPUT_DIR = DATA_DIR / "umap_regions"

N_NEIGHBORS = 20
MIN_DIST = 0.1
RANDOM_STATE = 42

MIN_PREVALENCE: float | None = 1.0
MIN_TOTAL_SIGNAL: float | None = 1.0
SCALE_FEATURES: bool = True


def load_condition(label: str, subdir: str) -> pd.DataFrame:
    csv_path = STATS_DIR / subdir / CSV_NAME

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, sep=SEP)

    required = {"start", "end", SIGNAL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in {csv_path}: {missing}")

    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce").astype("Int64")
    df = df.dropna(subset=["start", "end"])

    keep = {"start", "end", SIGNAL_COL}

    if "prevalence" in df.columns:
        keep.add("prevalence")

    df = df[list(keep)].copy()

    rename: dict[str, str] = {SIGNAL_COL: f"{label}_signal"}
    if "prevalence" in df.columns:
        rename["prevalence"] = f"{label}_prevalence"

    return df.rename(columns=rename)


def build_feature_matrix(conditions: list[tuple[str, str]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for label, subdir in conditions:
        print(f"  Lade Condition: {label}  ({subdir})")
        cdf = load_condition(label, subdir)

        if merged is None:
            merged = cdf
        else:
            merged = merged.merge(cdf, on=["start", "end"], how="outer")

    assert merged is not None, "Keine Conditions geladen."

    signal_cols = [f"{label}_signal" for label, _ in conditions]
    merged[signal_cols] = merged[signal_cols].fillna(0.0)

    return merged.reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    conditions: list[tuple[str, str]],
    min_prevalence: float | None,
    min_total_signal: float | None,
) -> pd.DataFrame:
    prev_cols   = [f"{label}_prevalence" for label, _ in conditions if f"{label}_prevalence" in df.columns]
    signal_cols = [f"{label}_signal"     for label, _ in conditions]

    before = len(df)

    if min_prevalence is not None and prev_cols:
        mask = df[prev_cols].ge(min_prevalence).any(axis=1)
        df = df[mask].copy()
        print(f"  Prevalence-Filter (>= {min_prevalence} in mind. 1 Condition): "
              f"{before} → {len(df)} Regionen")
        before = len(df)

    if min_total_signal is not None:
        total = df[signal_cols].sum(axis=1)
        df = df[total >= min_total_signal].copy()
        print(f"  Signal-Filter (Summe >= {min_total_signal}): "
              f"{before} → {len(df)} Regionen")

    return df.reset_index(drop=True)


def run_umap(X: np.ndarray) -> np.ndarray:
    try:
        import umap as umap_lib
    except ImportError as exc:
        raise ImportError(
            "umap-learn ist nicht installiert.\n"
            "Bitte einmalig ausführen:  pip install umap-learn"
        ) from exc

    reducer = umap_lib.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        random_state=RANDOM_STATE,
        init="random",
        n_jobs=1,
    )
    print(f"  UMAP fit_transform auf Matrix {X.shape} …")
    return reducer.fit_transform(X)


def plot_embedding(
    embedding: np.ndarray,
    df: pd.DataFrame,
    conditions: list[tuple[str, str]],
    outdir: Path,
) -> None:
    signal_cols = [f"{label}_signal" for label, _ in conditions]
    labels      = [label for label, _ in conditions]

    dominant_idx = df[signal_cols].values.argmax(axis=1)
    dominant     = [labels[i] for i in dominant_idx]

    plot_df = pd.DataFrame(
        {
            "UMAP_1":   embedding[:, 0],
            "UMAP_2":   embedding[:, 1],
            "Dominant": dominant,
            "start":    df["start"].astype(str),
            "end":      df["end"].astype(str),
        }
    )

    plot_df["total_signal"] = df[signal_cols].sum(axis=1).round(3)
    for col in signal_cols:
        plot_df[col] = df[col].round(3)

    hover_data = {
        "start":        True,
        "end":          True,
        "total_signal": True,
        **{col: True for col in signal_cols},
    }

    fig = px.scatter(
        plot_df,
        x="UMAP_1",
        y="UMAP_2",
        color="Dominant",
        hover_data=hover_data,
        title="UMAP – Regionen über alle Conditions",
        template="simple_white",
        opacity=0.7,
        size_max=6,
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=700, width=900)

    outfile = outdir / "umap_regions.html"
    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"  Plot gespeichert: {outfile}")

    fig.show()




def main() -> None:
    pio.renderers.default = "browser"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_matrix(CONDITIONS)

    df = apply_filters(df, CONDITIONS, MIN_PREVALENCE, MIN_TOTAL_SIGNAL)

    if len(df) < 5:
        print("  Zu wenige Regionen nach Filtern – Filter lockern (MIN_PREVALENCE, MIN_TOTAL_SIGNAL).")
        return

    signal_cols = [f"{label}_signal" for label, _ in CONDITIONS]
    X = df[signal_cols].values.astype(float)

    if SCALE_FEATURES:
        X = StandardScaler().fit_transform(X)
        print("  StandardScaler angewendet.")
    else:
        print("  Keine Skalierung.")

    embedding = run_umap(X)

    result_df = df[["start", "end"] + signal_cols].copy()
    result_df["UMAP_1"] = embedding[:, 0]
    result_df["UMAP_2"] = embedding[:, 1]

    csv_out = OUTPUT_DIR / "umap_regions.csv"
    result_df.to_csv(csv_out, sep=SEP, index=False)
    print(f"  CSV gespeichert:  {csv_out}")

    plot_embedding(embedding, df, CONDITIONS, OUTPUT_DIR)

    plt.savefig(OUTPUT_DIR/"umap_regions.svg", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(OUTPUT_DIR/"umap_regions.pdf", bbox_inches="tight")

    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()