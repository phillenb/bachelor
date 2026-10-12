from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


PROJECT_DIR = Path("/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis")
DEFAULT_INPUT = PROJECT_DIR / "data" / "literature" / "rna45sn1_intermediates_expanded.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "Intensity_matrices" / "plots_intermediate_radial_hills"
DEFAULT_EXTENT_NT = 50.0
DEFAULT_SIGMA_NT = 25.0
DEFAULT_LAYERS = 36

HILL_COLORSCALE = [
    [0.00, "#f7fbff"],
    [0.12, "#deebf7"],
    [0.30, "#c6dbef"],
    [0.50, "#6baed6"],
    [0.72, "#2171b5"],
    [0.90, "#08519c"],
    [1.00, "#08306b"],
]

START_COLUMN_CANDIDATES = (
    "start",
    "region_start",
    "start_coordinate",
    "start_coord",
    "start_position",
    "start_site",
    "start_nt",
    "expected_start",
    "canonical_start",
    "reference_start",
    "ref_start",
    "five_prime",
    "five_prime_end",
    "5_prime",
    "5prime",
    "5p",
)
END_COLUMN_CANDIDATES = (
    "end",
    "region_end",
    "end_coordinate",
    "end_coord",
    "end_position",
    "end_site",
    "end_nt",
    "expected_end",
    "canonical_end",
    "reference_end",
    "ref_end",
    "three_prime",
    "three_prime_end",
    "3_prime",
    "3prime",
    "3p",
)
LABEL_COLUMN_CANDIDATES = (
    "intermediate",
    "intermediate_name",
    "intermediate_id",
    "name",
    "label",
    "fragment",
    "region",
    "rna",
    "id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot literature-derived RNA45SN1 intermediate coordinates as "
            "radial Gaussian hills in the two-dimensional start/end space."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Intermediate CSV (Standard: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output HTML (Standard: {DEFAULT_OUTPUT_DIR / 'rna45sn1_intermediate_hills.html'}).",
    )
    parser.add_argument(
        "--start-column",
        default=None,
        help="Name der Start-Spalte; wird normalerweise automatisch erkannt.",
    )
    parser.add_argument(
        "--end-column",
        default=None,
        help="Name der End-Spalte; wird normalerweise automatisch erkannt.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optionale Spalte mit dem Namen des Intermediates.",
    )
    parser.add_argument(
        "--extent-nt",
        type=float,
        default=DEFAULT_EXTENT_NT,
        help=(
            "Sichtbare Ausdehnung um die exakte Koordinate in jede Richtung. "
            f"Standard: +/-{DEFAULT_EXTENT_NT:g} nt."
        ),
    )
    parser.add_argument(
        "--sigma-nt",
        type=float,
        default=DEFAULT_SIGMA_NT,
        help=(
            "Radius/Standardabweichung der Gauß-Glocke. "
            f"Standard: {DEFAULT_SIGMA_NT:g} nt."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=DEFAULT_LAYERS,
        help=(
            "Anzahl konzentrischer Farbstufen pro Intermediate. "
            f"Standard: {DEFAULT_LAYERS}."
        ),
    )
    parser.add_argument(
        "--peak-marker-size",
        type=float,
        default=9.0,
        help="Bildschirmgroesse der exakten Zentren; bleibt beim Herauszoomen sichtbar.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Namen der Intermediates nicht direkt im Plot anzeigen.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Zusaetzlich eine PNG-Datei speichern (benoetigt Kaleido).",
    )
    return parser.parse_args()


def normalize_column_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace("′", "")
    )


def resolve_column(
    frame: pd.DataFrame,
    explicit_name: str | None,
    candidates: tuple[str, ...],
    role: str,
    required: bool,
) -> str | None:
    normalized_to_original = {
        normalize_column_name(column): str(column) for column in frame.columns
    }

    if explicit_name is not None:
        if explicit_name in frame.columns:
            return explicit_name
        normalized = normalize_column_name(explicit_name)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]
        raise ValueError(
            f"Die angegebene {role}-Spalte '{explicit_name}' fehlt. "
            f"Vorhanden: {list(frame.columns)}"
        )

    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    if required:
        raise ValueError(
            f"Keine {role}-Spalte erkannt. Vorhanden: {list(frame.columns)}. "
            f"Nutze --{role}-column, um sie explizit anzugeben."
        )
    return None


def load_intermediates(
    csv_path: Path,
    start_column: str | None,
    end_column: str | None,
    label_column: str | None,
) -> tuple[pd.DataFrame, str, str, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Intermediate-Datei nicht gefunden: {csv_path}")

    frame = pd.read_csv(csv_path, sep=None, engine="python")
    if frame.empty:
        raise ValueError(f"Intermediate-Datei ist leer: {csv_path}")

    resolved_start = resolve_column(
        frame,
        start_column,
        START_COLUMN_CANDIDATES,
        role="start",
        required=True,
    )
    resolved_end = resolve_column(
        frame,
        end_column,
        END_COLUMN_CANDIDATES,
        role="end",
        required=True,
    )
    resolved_label = resolve_column(
        frame,
        label_column,
        LABEL_COLUMN_CANDIDATES,
        role="label",
        required=False,
    )

    standardized = pd.DataFrame(
        {
            "start": pd.to_numeric(frame[resolved_start], errors="coerce"),
            "end": pd.to_numeric(frame[resolved_end], errors="coerce"),
        }
    )
    if resolved_label is None:
        standardized["label"] = [
            f"Intermediate {index + 1}" for index in range(len(frame))
        ]
        label_description = "generated labels"
    else:
        standardized["label"] = frame[resolved_label].astype(str)
        label_description = resolved_label

    standardized = standardized.dropna(subset=["start", "end"]).copy()
    if standardized.empty:
        raise ValueError("Start- und End-Spalten enthalten keine gueltigen Zahlen.")

    return standardized, resolved_start, resolved_end, label_description


def normalized_gaussian_at_radius(
    radius: float,
    sigma_nt: float,
    extent_nt: float,
) -> float:
    value = np.exp(-0.5 * (radius / sigma_nt) ** 2)
    edge_value = np.exp(-0.5 * (extent_nt / sigma_nt) ** 2)
    return float(np.clip((value - edge_value) / (1.0 - edge_value), 0.0, 1.0))


def add_radial_hill(
    figure: go.Figure,
    start: float,
    end: float,
    extent_nt: float,
    sigma_nt: float,
    layers: int,
) -> None:
    radii = np.linspace(extent_nt, max(extent_nt / layers, 0.25), layers)
    for radius in radii:
        intensity = normalized_gaussian_at_radius(
            radius=float(radius),
            sigma_nt=sigma_nt,
            extent_nt=extent_nt,
        )
        color = sample_colorscale(HILL_COLORSCALE, [intensity])[0]
        figure.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=start - radius,
            x1=start + radius,
            y0=end - radius,
            y1=end + radius,
            fillcolor=color,
            line={"width": 0},
            layer="below",
        )


def create_figure(
    intermediates: pd.DataFrame,
    extent_nt: float,
    sigma_nt: float,
    layers: int,
    peak_marker_size: float,
    show_labels: bool,
) -> go.Figure:
    figure = go.Figure()
    for row in intermediates.itertuples(index=False):
        add_radial_hill(
            figure=figure,
            start=float(row.start),
            end=float(row.end),
            extent_nt=extent_nt,
            sigma_nt=sigma_nt,
            layers=layers,
        )

    hover_data = np.column_stack(
        [
            intermediates["label"].to_numpy(dtype=object),
            np.full(len(intermediates), extent_nt),
            np.full(len(intermediates), sigma_nt),
        ]
    )
    figure.add_trace(
        go.Scatter(
            x=intermediates["start"],
            y=intermediates["end"],
            mode="markers+text" if show_labels else "markers",
            text=intermediates["label"] if show_labels else None,

            textposition=[
                [
                    "top center",
                    "bottom center",
                    "middle right",
                    "middle left",
                    "top right",
                    "top left",
                    "bottom right",
                    "bottom left",
                ][i % 8]
                for i in range(len(intermediates))
            ] if show_labels else None,

            textfont={"size": 10, "color": "#08306b"},
            marker={
                "size": peak_marker_size,
                "color": "#08306b",
                "line": {"color": "white", "width": 1},
            },
            customdata=hover_data,
            hovertemplate=(
                "Intermediate: %{customdata[0]}<br>"
                "exact start: %{x:g}<br>"
                "exact end: %{y:g}<br>"
                "visible extent: +/- %{customdata[1]:g} nt<br>"
                "Gaussian radius (sigma): %{customdata[2]:g} nt"
                "<extra></extra>"
            ),
            name="Exact intermediate coordinates",
        )
    )
    padding = extent_nt * 1.4
    x_min = float(intermediates["start"].min() - padding)
    x_max = float(intermediates["start"].max() + padding)
    y_min = float(intermediates["end"].min() - padding)
    y_max = float(intermediates["end"].max() + padding)

    figure.update_layout(
        title=(
            "RNA45SN1 intermediates: exact coordinates with radial "
            f"Gaussian hills (+/-{extent_nt:g} nt, sigma={sigma_nt:g} nt)"
        ),
        xaxis={"title": "Start coordinate", "range": [x_min, x_max]},
        yaxis={
            "title": "End coordinate",
            "range": [y_min, y_max],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        template="simple_white",
        height=850,
        width=950,
        dragmode="zoom",
        margin={"l": 80, "r": 40, "b": 70, "t": 90},
    )
    return figure


def main() -> None:
    args = parse_args()
    if args.extent_nt <= 0:
        raise ValueError("--extent-nt muss groesser als 0 sein.")
    if args.sigma_nt <= 0:
        raise ValueError("--sigma-nt muss groesser als 0 sein.")
    if args.layers < 3:
        raise ValueError("--layers muss mindestens 3 sein.")
    if args.peak_marker_size <= 0:
        raise ValueError("--peak-marker-size muss groesser als 0 sein.")

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else DEFAULT_OUTPUT_DIR / "rna45sn1_intermediate_hills.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intermediates, start_column, end_column, label_column = load_intermediates(
        csv_path=input_path,
        start_column=args.start_column,
        end_column=args.end_column,
        label_column=args.label_column,
    )

    figure = create_figure(
        intermediates=intermediates,
        extent_nt=args.extent_nt,
        sigma_nt=args.sigma_nt,
        layers=args.layers,
        peak_marker_size=args.peak_marker_size,
        show_labels=not args.no_labels,
    )
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)

    if args.png:
        png_path = output_path.with_suffix(".png")
        figure.write_image(png_path, width=1900, height=1700, scale=2)

    print("Ergebnis gespeichert in: ", output_path.parent)


if __name__ == "__main__":
    main()