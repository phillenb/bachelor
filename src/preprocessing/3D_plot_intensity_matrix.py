from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DEFAULT_INPUT = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/raw/"
    "20240314_Ribolyzer_Ctrl/20240314_Ribolyzer_Ctrl_R10_IVPA_Nuc"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis/data/intensity_matrices/"
    "intensity_matrices_plots_3D"
)
REQUIRED_COLUMNS = ("start", "end", "n_reads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot start and end coordinates on the x/y axes and n_reads on the "
            "z axis and as a color gradient."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input table or directory containing a CSV/TSV/TXT table.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated HTML plot.",
    )
    parser.add_argument(
        "--max-points", type=int, default=150_000,
        help=(
            "Maximum number of displayed regions. If exceeded, regions with the "
            "largest n_reads are retained. Use 0 to display every region."
        ),
    )
    parser.add_argument(
        "--min-reads", type=float, default=2,
        help=(
            "Only display regions with at least this many reads. Default: 2, "
            "which removes the large number of single-read regions."
        ),
    )
    parser.add_argument(
        "--linear-scale", action="store_true",
        help=(
            "Use raw n_reads for height and color. By default, both are "
            "log10(n_reads + 1)-transformed to make intermediate signals visible."
        ),
    )
    parser.add_argument(
        "--marker-size", type=float, default=2.5,
        help="Size of the points in the 3D scatter plot.",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Also save a PNG (requires the optional kaleido package).",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {path}")

    supported_suffixes = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")
    candidates = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.name.lower().endswith(supported_suffixes)
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV, TSV, or TXT table found below: {path}")
    if len(candidates) > 1:
        names = "\n  ".join(str(file) for file in candidates)
        raise RuntimeError(
            "Several possible input tables were found. Pass the desired file "
            f"explicitly:\n  {names}"
        )
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=";",
        usecols=list(REQUIRED_COLUMNS),
        dtype={"start": np.int64, "end": np.int64, "n_reads": np.int64},
        engine="c",
    )
    frame = frame[(frame["start"] >= 0) & (frame["end"] >= 0) & (frame["n_reads"] >= 0)]
    return frame


def logarithmic_ticks(max_reads: float) -> tuple[np.ndarray, list[str]]:
    candidates = np.array(
        [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000,
         10_000, 20_000, 50_000, 100_000],
        dtype=float,
    )
    raw_ticks = candidates[candidates <= max_reads]
    if raw_ticks.size == 0 or raw_ticks[-1] != max_reads:
        raw_ticks = np.append(raw_ticks, max_reads)

    if raw_ticks.size > 9:
        indices = np.unique(np.linspace(0, raw_ticks.size - 1, 9).round().astype(int))
        raw_ticks = raw_ticks[indices]
        if raw_ticks[-1] != max_reads:
            raw_ticks = np.append(raw_ticks, max_reads)

    positions = np.log10(raw_ticks + 1)
    labels = [f"{value:,.0f}" for value in raw_ticks]
    return positions, labels


def create_figure(
    frame: pd.DataFrame,
    title: str,
    linear_scale: bool,
    marker_size: float,
) -> go.Figure:
    raw_reads = frame["n_reads"].to_numpy(dtype=float)
    max_reads = float(raw_reads.max())

    if linear_scale:
        plotted_reads = raw_reads
        z_axis_title = "Number of reads (n_reads)"
        tick_values = np.linspace(0, max_reads, 6)
        tick_labels = [f"{value:,.0f}" for value in tick_values]
    else:
        plotted_reads = np.log10(raw_reads + 1)
        z_axis_title = "Read intensity: log10(n_reads + 1)"
        tick_values, tick_labels = logarithmic_ticks(max_reads)


    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=frame["start"],
                y=frame["end"],
                z=plotted_reads,
                mode="markers",
                customdata=frame[["n_reads"]].to_numpy(),
                marker={
                    "size": marker_size,
                    "color": plotted_reads,
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": float(plotted_reads.max()),
                    "colorbar": {
                        "title": "n_reads",
                        "tickmode": "array",
                        "tickvals": tick_values,
                        "ticktext": tick_labels,
                    },
                    "opacity": 0.78,
                },
                hovertemplate=(
                    "Start: %{x:,.0f}<br>End: %{y:,.0f}<br>"
                    "n_reads: %{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": "Start coordinate",
            "yaxis_title": "End coordinate",
            "zaxis_title": z_axis_title,
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
    )
    return figure


def main() -> None:
    args = parse_args()
    input_file = resolve_input(args.input)
    frame = read_table(input_file)
    frame = frame[frame["n_reads"] >= args.min_reads]
    if frame.empty:
        raise ValueError("No regions remain after applying --min-reads.")

    original_count = len(frame)
    if args.max_points > 0 and original_count > args.max_points:
        frame = frame.nlargest(args.max_points, "n_reads").sort_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_linear" if args.linear_scale else "_log_scaled"
    output_base = args.output_dir / f"{input_file.stem}_3D_intensity_matrix{suffix}"
    figure = create_figure(
        frame,
        title=f"3D intensity matrix: {input_file.stem}",
        linear_scale=args.linear_scale,
        marker_size=args.marker_size,
    )
    html_path = output_base.with_suffix(".html")
    figure.write_html(html_path, include_plotlyjs="cdn")

    if args.png:
        png_path = output_base.with_suffix(".png")
        figure.write_image(png_path, width=1800, height=1300, scale=2)

    print("Ergebnis gespeichert in: ", output_base.parent)


if __name__ == "__main__":
    main()