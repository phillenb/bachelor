# RNA45SN1 Region Selection and Validation

This project compares three approaches to selecting rRNA start–end regions: set-based selection, UMAP graph-based selection, and supervised logistic regression. The resulting panels are evaluated by classification and comparison with known RNA45SN1 intermediates.

## Project structure

Place this README in the project root, alongside `src/` and `data/`. The tree below shows the main pipeline scripts; package initialization files, legacy scripts, and supplementary analyses are omitted. Filenames match the supplied source code, including existing spelling.

```text
project/
├── README.md
├── src/
│   ├── paths.py                                      # Shared project and data paths
│   ├── preprocessing/
│   │   ├── create_intensity_matrix_original.py        # Original intensity plots
│   │   ├── 3D_plot_intensity_matrix.py                # 3D start–end/read-count plots
│   │   ├── updated_intensity_matirces_2D.py            # Improved 2D intensity plots
│   │   ├── Hill_plot_intermediates.py                 # Optional literature-coordinate plot
│   │   └── detect_sequencing_depth.py                 # Sample-level overview
│   ├── method1_set_based_and_rule_based_region_selection/
│   │   ├── normalization.py                          # CPM and log-CPM helper functions
│   │   ├── intersection_of_all_samples_of_one_condition.py
│   │   ├── compare_intersections_across_condtions.py
│   │   └── plots/
│   │       ├── upset_between_samples_of_one_condition.py
│   │       ├── upset_all_conditions_and_ctrl.py
│   │       └── upset_ctrl_rpl3_urb1.py
│   ├── method2_unsupervised_selection_UMAP_PCA_PCoA/
│   │   ├── umap_over_regions.py                      # Region-level UMAP overview
│   │   ├── PCoA_and_jaccard.py                        # PCA and Jaccard-based PCoA
│   │   ├── method2_region_selection.py                # UMAP graph-based feature selection
│   │   └── sanity check/
│   │       └── umap_regions_check_signal_single_cond.py
│   ├── method3_supervised_LogReg_RandomForrest/
│   │   └── logistic_regression_raw_stability_selection.py
│   ├── validation/
│   │   ├── build_three_method_region_sets.py          # Individual and combined panels
│   │   ├── validate_three_nested_balanced.py          # Classification-based validation
│   │   ├── compare_condition_intersections_with_umap.py
│   │   └── plot/
│   │       ├── venn_diagram_method_comparison.py
│   │       └── plot_validation_performance.py
│   └── biological_validation/
│       ├── profile_RNA45SN1_intermediates_across_conditions_expanded.py
│       ├── biological_validation_using_methods_union.py
│       ├── intnsity_matrix_with_expected_intermediates.py
│       └── UMAP_Fingerprint_matrix_sorted_by_start_end.py
└── data/
    ├── raw/                                          # Input intensity tables
    │   └── <dataset>/<sample>/intensity_matrix/
    │       └── intensity_matrix.csv
    ├── literature/                                   # Reference annotation
    │   └── rna45sn1_intermediates_expanded.csv
    ├── intensity_matrices/                           # Preprocessing plots
    ├── supplements/
    │   └── sequencing_depth_ivpa_nucleus/
    ├── method1_set_based_and_rule_based_region_selection_data/
    │   ├── shared_regions_across_samples/<condition>/
    │   │   └── intersection_all_samples_min_raw_reads_2.csv
    │   ├── cross_condition_comparison/
    │   └── upset_plots/
    ├── method2_unsupervised_selection_UMAP_PCA_PCoA_data/
    │   ├── umap_regions/
    │   │   └── umap_regions.csv
    │   ├── umap_across_samples_on_raw_data/
    │   │   └── umap_characteristic_regions.csv
    │   └── condition_intersections_vs_umap/
    ├── method3_supervised_LogReg_RandomForrest_data/
    │   └── logistic_regression_raw_stability_selection/
    │       └── raw_logreg_stable_cross_design_consensus.csv
    ├── method_comparison/
    │   ├── three_method_region_sets/
    │   │   ├── method_union_regions.csv
    │   │   └── umap_regions_standardized.csv
    │   ├── three_method_nested_balanced_validation/
    │   └── venn_diagrams/
    └── biological validation/
        ├── raw data alignment/
        ├── method_union_alignment/
        ├── expected_intermediate_region_overlays/
        └── UMAP_Fingerprint_matrix_sorted_by_start_end/
```

The `data/` tree shows required inputs and selected result files, rather than an exhaustive output listing. Result folders are created by the corresponding scripts. The optional Hill plot defaults to `data/Intensity_matrices/` with a capital I; adjust its output path to use the shared lowercase folder.

## Pipeline overview

1. **Preprocessing:** visualize the existing intensity tables as original, improved 2D, and 3D plots. The Hill plot optionally displays known intermediate coordinates. The sequencing-depth script reports the number of table rows per sample, rather than the sum of `n_reads`.

2. **Method 1 — set-based selection:** use `normalization.py` as an imported helper, identify regions with at least two reads in all three replicates of a condition, and compare these consensus sets across conditions. UpSet plots summarize overlaps within and between conditions.

3. **Method 2 — unsupervised selection:** generate the region-level UMAP overview and check for signals exclusive to one condition. PCA and Jaccard-based PCoA assess sample relationships. `method2_region_selection.py` independently uses raw data to select characteristic regions from the UMAP graph and assess their stability across replicate splits.

4. **Method 3 — supervised selection:** select regions using logistic regression and stability across leave-one-out and balanced replicate splits. The supplied code contains no Random Forest implementation despite the folder name.

5. **Panel validation:** build standardized panels from the three methods, including their union and intersections. Compare classification performance using logistic regression, then summarize overlap and performance with Venn diagrams and validation plots. Feature selection is repeated using training samples only within each validation fold; the exported global panels support descriptive comparisons.

6. **Biological validation:** compare raw regions and the method-union panel with known RNA45SN1 intermediates using minimum reciprocal overlap (MRO). Overlay expected intermediates and display sample fingerprints of UMAP-selected regions sorted by start and end. The fingerprint panel size is determined by the selected-region file; 362 regions is not hard-coded.

## Reproducing the results

Provide the input files under `data/raw/` and `data/literature/`, then follow steps 1–6. The analysis expects three IVPA-Nucleus replicates for each condition: Ctrl, UTP18, DIMT1L, LAS1L, RPL3, URB1, and WBSCR22. Raw tables use semicolon separators and contain `start`, `end`, and `n_reads`. The expanded literature annotation contains 42 named fragments with `reference`, `start`, `end`, and `fragment`, using the same coordinate system as the raw data.

Python dependencies include NumPy, pandas, Matplotlib, matplotlib-venn, Plotly, scikit-learn, and umap-learn. Adjust hard-coded project paths and sample-folder mappings to the local data layout. Keep the same data, parameters, and package versions when comparing repeated runs.

The supplied archive contains source code only. Sequencing alignment and generation of the input intensity tables are upstream steps; a complete data run was not verified because the input tables and literature annotation were not included.