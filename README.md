# AHP-Attention for NTD skin diagnosis — supplementary material

Code, trained artifacts, figures and result tables for an **AHP-weighted attention
classifier** applied to skin signs of Neglected Tropical Diseases (NTDs), together
with a Streamlit demonstration app.

> Research / decision-support prototype only. It is **not** a medical device and
> must not be used for clinical diagnosis.

## Contents

| Path | Description |
|---|---|
| `app/streamlit_app.py` | Streamlit diagnostic assistant (entry point) |
| `app/main.py` | Same application (legacy duplicate) |
| `app/models/ahp_transformer_model.pth` | Trained `AHPTransformer` state dict |
| `app/models/ahp_weights.pt` | AHP priority weights (tensor registered as a buffer) |
| `app/models/feature_columns.pkl` | Ordered list of the 23 one-hot feature names |
| `app/models/label_encoder.pkl` | `LabelEncoder` over the 4 output classes |
| `Untitled.ipynb` | Dataset generation + training / evaluation notebook |
| `ntd_clinical_dataset.csv` | 600 records × 19 columns (synthetic, see below) |
| `synthetic_ntds_dataset.csv` | 650 records × 9 columns (reduced schema used by the app) |
| `stability_results.csv`, `stability_results/` | Stability experiment outputs |
| `figures/` | Confusion matrices, loss/training curves, inference time, t-SNE, stability |
| `fig*.png`, `tsne_comparison.pdf` | Root-level figure exports |

## Data provenance

`ntd_clinical_dataset.csv` is **synthetic**. It is generated in the first cell of
`Untitled.ipynb` from a hand-written disease/sign table (`random.seed = 42`,
`N_SAMPLES = 600`, 11 diseases). It contains no real patient records and no
identifiers.

`synthetic_ntds_dataset.csv` holds the reduced 8-field schema that the app
actually consumes:

`location, lesion_number, lesion_type, base, bilateral, pain, edema, varicose_veins`

One-hot encoding of these 8 fields gives the 23 features in `feature_columns.pkl`;
the target has 4 classes: `Nodular`, `Parasitic_Itchy`, `Swelling`, `Ulcerative`.

## Reported results

Both tables below are reproduced verbatim from the bundled CSV files — they were
produced by different evaluation runs and are not directly comparable.

`figures/results_table.csv`

| Model | Mean accuracy | Std (stability) | Inference time (s) |
|---|---|---|---|
| Baseline Attention | 0.8246 | 0.0511 | 0.00033 |
| AHP-Enhanced Attention | 0.8000 | 0.0617 | 0.00033 |
| Random-Weighted Attention | 0.7108 | 0.0609 | 0.00025 |

`stability_results.csv`

| Model | Accuracy mean | Accuracy std | F1 mean | Attention variance | Train time (s) |
|---|---|---|---|---|---|
| Baseline Attention | 0.4833 | 0.0000 | 0.1303 | 0.00336 | 0.0648 |
| AHP-Enhanced Attention | 0.4817 | 0.0033 | 0.1300 | 0.00925 | 0.0513 |
| AHP + Noise (20%) | 0.4800 | 0.0041 | 0.1299 | 0.01391 | 0.0485 |

**Note on results.** In these runs the AHP-enhanced variant does **not** improve
accuracy over the baseline attention model: it is slightly lower in both tables
(−0.025 and −0.002 respectively), while its attention variance is roughly 2.8×
higher. The AHP weighting does clearly separate from random weighting
(0.800 vs 0.711). These figures are reported as produced; no value has been
adjusted or removed.

## Running the app

```bash
pip install -r requirements.txt
cd app
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

The login screen is a **hard-coded demonstration stub**:

- user: `clinician`
- password: `securepass`

Replace it with a real authentication mechanism before exposing the app anywhere.

Note: the app resolves model paths relative to the working directory
(`MODEL_DIR = "models"`), so it must be started from inside `app/`.
`main.py` and `streamlit_app.py` are identical duplicates.

## Reproducing the results

1. Open `Untitled.ipynb` from the repository root (it reads
   `ntd_clinical_dataset.csv` with a relative path).
2. Run the cells in order: dataset generation → AHP weight derivation → model
   training/evaluation → figures.

The bundled `app/models/` artifacts were produced by that notebook; the exact
versions of `torch` and `streamlit` used at training time were not recorded.

## License

No license file is included in this material. Add one before redistribution if
required by the venue.
