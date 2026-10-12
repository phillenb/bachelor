from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


REQUIRED_COLUMNS = {"start", "end", "n_reads"}
DEFAULT_COLOR_QUANTILE = 0.995
DEFAULT_MIN_MARKER_SIZE = 1.0
DEFAULT_MAX_MARKER_SIZE = 7.0

READ_COUNT_COLORSCALE = [
    [0.00, "#fff5f0"],
    [0.08, "#fee0d2"],
    [0.25, "#fcbba1"],
    [0.45, "#fc9272"],
    [0.70, "#ef3b2c"],
    [0.88, "#cb181d"],
    [1.00, "#67000d"],
]


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(
        f"Projektwurzel konnte ausgehend von {start_path} nicht gefunden werden. "
        "Lege das Skript innerhalb des Projekts ab."
    )


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(CURRENT_FILE.parent)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "intensity_matrices" / "plots_exact_2D_shared_scale"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeuge interaktive, punktgenaue 2D-Start-End-Matrizen fuer alle "
            "intensity_matrix.csv-Dateien. Es findet kein Binning statt."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ausgabeordner (Standard: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--template-csv",
        type=Path,
        default=None,
        help="Optionale CSV mit den Spalten Fragment, Start und End.",
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=1.0,
        help=(
            "Minimale Read-Zahl einer dargestellten Region. Standard: 1; "
            "Ein-Read-Regionen bleiben damit als heller Hintergrund erhalten."
        ),
    )
    parser.add_argument(
        "--color-quantile",
        type=float,
        default=DEFAULT_COLOR_QUANTILE,
        help=(
            "Gemeinsames oberes Quantil der Farb- und Groessenskala "
            f"(Standard: {DEFAULT_COLOR_QUANTILE})."
        ),
    )
    parser.add_argument(
        "--min-marker-size",
        type=float,
        default=DEFAULT_MIN_MARKER_SIZE,
        help=f"Punktgroesse schwacher Regionen (Standard: {DEFAULT_MIN_MARKER_SIZE}).",
    )
    parser.add_argument(
        "--max-marker-size",
        type=float,
        default=DEFAULT_MAX_MARKER_SIZE,
        help=f"Punktgroesse starker Regionen (Standard: {DEFAULT_MAX_MARKER_SIZE}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help=(
            "Optionales Punktlimit pro Sample. Standard: 0 zeigt alle exakten "
            "Regionen; ein positiver Wert behaelt die staerksten Regionen."
        ),
    )
    return parser.parse_args()


def find_all_intensity_matrix_files() -> list[Path]:
    if not RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Rohdatenordner wurde nicht gefunden: {RAW_DATA_DIR}")

    return sorted(
        path
        for path in RAW_DATA_DIR.rglob("intensity_matrix.csv")
        if path.parent.name == "intensity_matrix"
    )


def load_intensity_matrix(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_file,
        sep=";",
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )
    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
        raise ValueError(f"Erforderliche Spalten fehlen: {missing}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    return df.loc[
        (df["start"] >= 0) & (df["end"] >= 0) & (df["n_reads"] >= 0)
    ].copy()


def determine_global_color_max(
    csv_files: list[Path],
    min_reads: float,
    color_quantile: float,
) -> float:
    values: list[np.ndarray] = []
    for index, csv_file in enumerate(csv_files, start=1):
        df = load_intensity_matrix(csv_file)
        displayed = df.loc[
            df["n_reads"] >= min_reads, "n_reads"
        ].to_numpy(dtype=float)
        if len(displayed):
            values.append(displayed)

    if not values:
        raise ValueError("Keine Regionen fuer die globale Farbskala gefunden.")

    all_values = np.concatenate(values)
    return max(float(np.quantile(all_values, color_quantile)), 1.0)


def build_colorbar(display_max_reads: float) -> dict:
    candidates = np.array(
        [
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1_000,
            2_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
        ],
        dtype=float,
    )
    ticks = candidates[candidates <= display_max_reads]
    if len(ticks) == 0:
        ticks = np.array([1.0])
    if display_max_reads > ticks[-1]:
        ticks = np.append(ticks, display_max_reads)

    return {
        "title": "reads per exact region<br>(log1p colour)",
        "tickvals": np.log1p(ticks),
        "ticktext": [f"{value:g}" for value in ticks],
    }


def marker_sizes(
    reads: np.ndarray,
    display_max_reads: float,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    transformed = np.log1p(reads)
    lower = float(np.log1p(1.0))
    upper = float(np.log1p(max(display_max_reads, 2.0)))
    relative = np.clip((transformed - lower) / (upper - lower), 0.0, 1.0)
    relative = np.power(relative, 0.70)
    return min_size + (max_size - min_size) * relative


def add_template(fig: go.Figure, template_path: Path | None) -> None:
    if template_path is None:
        return
    if not template_path.is_file():
        return

    template = pd.read_csv(template_path, sep=";")
    required = {"Fragment", "Start", "End"}
    if not required.issubset(template.columns):
        return

    fragments = {"18S", "28S", "5-8S", "5ETS", "ITS1", "ITS2", "ITS3", "3ETS"}
    template = template.loc[template["Fragment"].isin(fragments)]
    for row in template.itertuples(index=False):
        start = float(row.Start)
        end = float(row.End)
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=start,
            y1=end,
            line={"color": "red", "width": 1, "dash": "dash"},
        )
        fig.add_annotation(
            x=(start + end) / 2,
            y=start,
            text=str(row.Fragment),
            showarrow=False,
            font={"size": 10, "color": "red"},
        )


def output_path_for_sample(csv_file: Path, output_root: Path) -> Path:
    relative = csv_file.relative_to(RAW_DATA_DIR)
    sample_relative = relative.parent.parent
    destination_dir = output_root / sample_relative.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / f"{sample_relative.name}_exact_2D_intensity_matrix.html"


def plot_single_intensity_matrix(
    csv_file: Path,
    output_root: Path,
    display_max_reads: float,
    min_reads: float,
    min_marker_size: float,
    max_marker_size: float,
    max_points: int,
    template_path: Path | None,
) -> bool:
    df = load_intensity_matrix(csv_file)

    df = df.loc[df["n_reads"] >= min_reads].copy()
    if df.empty:
        return False

    if max_points > 0 and len(df) > max_points:
        original_count = len(df)
        df = df.nlargest(max_points, "n_reads").copy()

    df = df.sort_values("n_reads", kind="stable")
    raw_reads = df["n_reads"].to_numpy(dtype=float)
    color_values = np.log1p(raw_reads)
    sizes = marker_sizes(
        reads=raw_reads,
        display_max_reads=display_max_reads,
        min_size=min_marker_size,
        max_size=max_marker_size,
    )

    fig = go.Figure(
        go.Scattergl(
            x=df["start"],
            y=df["end"],
            mode="markers",
            marker={
                "size": sizes,
                "color": color_values,
                "colorscale": READ_COUNT_COLORSCALE,
                "cmin": float(np.log1p(1.0)),
                "cmax": float(np.log1p(max(display_max_reads, 2.0))),
                "showscale": True,
                "opacity": 0.72,
                "colorbar": build_colorbar(display_max_reads),
                "line": {"width": 0},
            },
            customdata=df[["n_reads"]].to_numpy(),
            hovertemplate=(
                "start=%{x}<br>"
                "end=%{y}<br>"
                "n_reads=%{customdata[0]:g}<extra></extra>"
            ),
            name="Exact regions",
        )
    )

    add_template(fig, template_path)

    sample_relative = csv_file.relative_to(RAW_DATA_DIR).parent.parent
    fig.update_layout(
        title=str(sample_relative),
        xaxis_title="Start",
        yaxis_title="End",
        template="simple_white",
        height=800,
        width=900,
        dragmode="zoom",
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    output_file = output_path_for_sample(csv_file, output_root)
    fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)

    return True


def main() -> None:
    args = parse_args()
    if args.min_reads < 0:
        raise ValueError("--min-reads muss groesser oder gleich 0 sein.")
    if not 0.0 < args.color_quantile <= 1.0:
        raise ValueError("--color-quantile muss im Intervall (0, 1] liegen.")
    if args.min_marker_size <= 0:
        raise ValueError("--min-marker-size muss groesser als 0 sein.")
    if args.max_marker_size < args.min_marker_size:
        raise ValueError("--max-marker-size darf nicht kleiner als --min-marker-size sein.")
    if args.max_points < 0:
        raise ValueError("--max-points muss groesser oder gleich 0 sein.")

    output_root = args.output_dir.expanduser().resolve()
    template_path = args.template_csv.expanduser().resolve() if args.template_csv else None


    csv_files = find_all_intensity_matrix_files()
    if not csv_files:
        return

    conditions = sorted({path.relative_to(RAW_DATA_DIR).parts[0] for path in csv_files})
    display_max_reads = determine_global_color_max(
        csv_files=csv_files,
        min_reads=args.min_reads,
        color_quantile=args.color_quantile,
    )

    successful = 0
    for index, csv_file in enumerate(csv_files, start=1):
        successful += plot_single_intensity_matrix(
            csv_file=csv_file,
            output_root=output_root,
            display_max_reads=display_max_reads,
            min_reads=args.min_reads,
            min_marker_size=args.min_marker_size,
            max_marker_size=args.max_marker_size,
            max_points=args.max_points,
            template_path=template_path,
        )


    print("Ergebnisse gespeichert in: ", output_root)


if __name__ == "__main__":
    main()