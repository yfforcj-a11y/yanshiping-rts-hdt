# Yanshiping RTS hybrid digital twin reproducibility code

This repository contains the custom analysis code used for the manuscript **“An Earth observation-enabled hybrid digital twin framework for event-scale thermo-erosional evolution of a retrogressive thaw slump.”** It covers uncertainty-aware DoD analysis, Siamese change detection, borehole/meteorological processing, PINN thermal reconstruction, GNSS and crack-meter event detection, geomorphic feature extraction, and hybrid digital twin evidence fusion.

The version archived as `v1.0.0` preserves the scientific parameters, thresholds, random seeds, model structure, and result definitions used in the manuscript. Public-release edits are limited to path configuration, English/ASCII source text, filenames, privacy removal, and repository packaging.

## Data

Data DOI: **[10.5281/zenodo.22210857](https://doi.org/10.5281/zenodo.22210857)**

Extract the dataset beside this repository using the following layout:

```text
workspace/
├── data/
│   ├── dod_0p1m/
│   ├── borehole_temperature/
│   ├── gnss/
│   └── crack_meter/
├── outputs/
└── yanshiping-rts-hdt/
    ├── scripts/
    ├── DATA_MANIFEST.yaml
    └── config.example.yaml
```

Some downstream scripts also require manuscript-derived products that are not part of the raw public dataset, including four epoch DEMs, LoD95 rasters, SCM products, a monitoring-site catalog, PINN intermediate fields, or optional DoD LAS files. Their expected relative locations are documented in `config.example.yaml` and `DATA_MANIFEST.yaml`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can be adjusted for the available CPU/CUDA platform following the official PyTorch instructions.

## Main workflow

The core script order is:

1. `02_lod95_uncertainty.py` — DoD uncertainty propagation, LoD95 filtering, zones, and volume summaries.
2. `03_siamese_prepare.py` — paired DEM/slope tile and label preparation.
3. `04_siamese_train.py` — Siamese U-Net training with the manuscript settings.
4. `05_siamese_infer.py` — tile-level change-probability inference.
5. `06_siamese_merge.py` — full-grid SCM probability/binary-map fusion.
6. `07_patch_postprocess.py` — active-patch filtering, vectorization, and statistics.
7. `08_n_factor.py` — ERA5/borehole daily forcing, degree days, and n-factors.
8. `09_kappa_tinit.py` — effective thermal properties and initial profiles.
9. `10_pinn_thermal.py` — borehole-constrained one-dimensional thermal PINN.
10. `11_geomorph_features.py` — site-scale DoD/LoD95/SCM feature extraction.
11. `12_joint_optimizer.py` — GNSS, crack-meter, thermal, and geomorphic joint interpretation.
12. `13_event_detection.py` — monitoring cleaning, rate calculation, and event detection.
13. `14_ice_weakening.py` — subsurface ice-weakening indicators.
14. `16_geomorph_budget.py` — geomorphic class and budget integration.
15. `19_results_fusion.py` — paper-level multi-source fusion.

`28_dod_volume_statistics.py`, `61_event_catalog_table.py`, `62_site_evidence_table.py`, and `69_hdt_attribution_tables.py` generate supporting volume and manuscript-table products from the corresponding existing inputs.

Scripts intentionally retain their manuscript scientific settings near the top of each file. Review `config.example.yaml` and `DATA_MANIFEST.yaml` before running. Each production stage writes to `../outputs/`; no manuscript computation is performed during installation or import.

## Reproducibility scope

The released data allow direct use of the six DoD rasters and the raw borehole, GNSS, and crack-meter time series. Steps requiring non-deposited derived products are included for methodological transparency and can be run when those inputs are available. DoD values always follow the `later - earlier` convention.

## Software archive

Software DOI: **[10.5281/zenodo.22210859](https://doi.org/10.5281/zenodo.22210859)**

## License and citation

Code is licensed under the MIT License. Cite the software using `CITATION.cff` and the archived software DOI.
