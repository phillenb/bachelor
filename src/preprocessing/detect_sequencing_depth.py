import glob
import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc


PROJECT_DIR = Path("/mnt/data01/phillenb/RNA45SN1_rRNA_intermediate_analysis")
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "supplements" / "sequencing_depth_ivpa_nucleus"

SAMPLE_GLOBS = [
    "*_IVPA_Nuc",
    "*_IVPA_Nuc*",
    "IVPA_Nucleus_*",
    "IVPA*_*Nucleus*",
    "*IVPA_Nucleaus_*",
    "*IVPA_*_Nucleus_*",
    "*IVPA_Nuc_*",
    "IVPA_*_Nucleus*",
]


def matches_sample_pattern(sample_name: str) -> bool:
    return any(fnmatch(sample_name, pattern) for pattern in SAMPLE_GLOBS)


def extract_condition(sample_name: str) -> str:
    cond = sample_name
    cond = re.sub(r"^\d{8}_", "", cond)
    cond = cond.replace("Ribolyzer_", "")
    cond = re.sub(r"_R\d+", "", cond)
    cond = re.sub(r"_\d+$", "", cond)
    cond = re.sub(r"Nucleus\d+$", "", cond)
    cond = cond.replace("_IVPA_Nucleus", "")
    cond = cond.replace("_IVPA_Nuc", "")
    cond = re.sub(r"^IVPA_", "", cond)
    return cond.strip("_")


def find_all_intensity_matrix_files():
    pattern = RAW_DATA_PATH / "**" / "intensity_matrix" / "intensity_matrix.csv"
    return glob.glob(str(pattern), recursive=True)


def collect_sequencing_depths():
    files = find_all_intensity_matrix_files()
    records = []
    for csv_path in sorted(files):
        sample_name = Path(csv_path).parent.parent.name
        if not matches_sample_pattern(sample_name):
            continue
        df = pd.read_csv(csv_path, sep=";")
        n_rows = len(df)
        condition = extract_condition(sample_name)
        records.append(
            {
                "sample": sample_name,
                "condition": condition,
                "n_rows": n_rows,
            }
        )
    return pd.DataFrame(records)


def plot_sequencing_depth(depth_df):
    depth_df = depth_df.sort_values(["condition", "sample"])
    palette = pc.qualitative.Plotly
    conditions = sorted(depth_df["condition"].unique())
    colour_map = {
        cond: palette[i % len(palette)]
        for i, cond in enumerate(conditions)
    }
    bar_colors = [colour_map[c] for c in depth_df["condition"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=depth_df["sample"],
            y=depth_df["n_rows"],
            marker_color=bar_colors,
            customdata=depth_df["condition"],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Condition: %{customdata}<br>"
                "Rows: %{y:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Sequencing Depth per Sample",
        template="simple_white",
        xaxis_title="Sample",
        yaxis_title="Number of reads",
        xaxis_tickangle=-45,
        width=max(1200, len(depth_df) * 55),
        height=650,
        margin=dict(b=220),
    )
    return fig


def main():
    depth_df = collect_sequencing_depths()
    if depth_df.empty:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "sequencing_depth.csv"
    depth_df.to_csv(out_csv, sep=";", index=False)

    fig = plot_sequencing_depth(depth_df)
    out_html = OUTPUT_DIR / "sequencing_depth_barplot.html"
    fig.write_html(out_html, include_plotlyjs="cdn")

    print(f"Ergebnis gespeichert in: {OUTPUT_DIR}")
    fig.show()


if __name__ == "__main__":
    main()