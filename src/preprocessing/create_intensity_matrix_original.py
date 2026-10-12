from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


PROJECT_DIR = Path("/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis")
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" /"intensity_matrices"/"intensity_matrices_original"


def find_all_intensity_matrix_files(raw_data_path=RAW_DATA_PATH):
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.is_dir():
        raise FileNotFoundError(f"Raw-Datenordner nicht gefunden: {raw_data_path}")
    return sorted(raw_data_path.rglob("intensity_matrix.csv"))


def plot_single_intensity_matrix_plotly(
    csv_file_path,
    output_dir=OUTPUT_DIR,
    template_df_name=None,
    with_template=True,
    plot_title=None,
):
    csv_file_path = Path(csv_file_path)

    df = pd.read_csv(csv_file_path, sep=";")
    required = {"start", "end", "n_reads"}
    missing = required.difference(df.columns)
    if missing:
        return None

    for column in ("start", "end", "n_reads"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["start", "end", "n_reads"])

    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["start"],
            y=df["end"],
            mode="markers",
            marker=dict(
                size=1,
                color=df["n_reads"],
                colorscale="Viridis",
                cmin=0,
                cmax=500,
                showscale=True,
                opacity=0.4,
                colorbar=dict(title="n_reads"),
            ),
            text=df["n_reads"],
            hovertemplate=(
                "start=%{x}<br>"
                "end=%{y}<br>"
                "n_reads=%{text}<extra></extra>"
            ),
        )
    )

    if with_template and template_df_name:
        template = pd.read_csv(template_df_name, sep=";")
        template = template[
            template["Fragment"].isin(
                ["18S", "28S", "5-8S", "5ETS", "ITS1", "ITS2", "ITS3", "3ETS"]
            )
        ]
        for _, row in template.iterrows():
            start, end = row["Start"], row["End"]
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=start,
                y1=end,
                line=dict(color="red", width=1, dash="dash"),
            )
            fig.add_annotation(
                x=(start + end) / 2,
                y=start,
                text=row["Fragment"],
                showarrow=False,
                font=dict(size=10, color="red"),
            )

    sample_dir = (
        csv_file_path.parent.parent
        if csv_file_path.parent.name == "intensity_matrix"
        else csv_file_path.parent
    )
    sample_name = sample_dir.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{sample_name}_intensity_matrix.html"

    fig.update_layout(
        title=plot_title or sample_name,
        xaxis_title="Start",
        yaxis_title="End",
        template="simple_white",
        height=800,
        width=900,
        dragmode="zoom",
    )
    fig.write_html(outfile, include_plotlyjs="cdn")
    return outfile


def create_all_intensity_plots(
    template_df_name=None,
    raw_data_path=RAW_DATA_PATH,
    output_dir=OUTPUT_DIR,
):
    raw_data_path = Path(raw_data_path)
    output_dir = Path(output_dir)
    files = find_all_intensity_matrix_files(raw_data_path)
    if not files:
        return []

    saved = []
    for csv_file_path in files:
        relative_parent = csv_file_path.parent.relative_to(raw_data_path)
        relative_sample = (
            relative_parent.parent
            if relative_parent.name == "intensity_matrix"
            else relative_parent
        )
        outfile = plot_single_intensity_matrix_plotly(
            csv_file_path,
            output_dir=output_dir / relative_parent,
            template_df_name=template_df_name,
            plot_title=str(relative_sample),
        )
        if outfile is not None:
            saved.append(outfile)

    print("Ergebnisse gespeicher in: ", output_dir)
    return saved


if __name__ == "__main__":
    create_all_intensity_plots()