# Accuracy inflation in public sickle cell imaging datasets

Analysis code for "Quantifying accuracy inflation across public sickle cell
imaging datasets, with an acceptance test for screening deployment"
(A. Philomina Jenifer, P. Rishi).

## What this does

Measures how much of the detection accuracy reported on public sickle cell
imaging datasets comes from how the datasets were built rather than from the
erythrocytes: duplicated images, repeated fields, participant dependence,
file-level acquisition traces, drawn annotations and label conflicts.

## Datasets

None are redistributed here. All four are public:

- RedTell (Sadafi et al., 2023)
- Tushabe (Tushabe et al., 2024)
- erythrocytesIDB (Gonzalez-Hidalgo et al., 2015)
- erythroSight (Shrestha et al., 2024)

## Code

| File | Purpose |
|---|---|
| `o5_final_experiment.py` | source grouping, calibration, inflation across all corpora |
| `o5_remaining.py` | verified grouping, estimator recovery, leave-one-phone-out, optimism |
| `o5_closing.py` | balanced confounding index, resolution confound, annotation sheets |
| `o5_tushabe_final.py` | Tushabe inflation at human-anchored thresholds |
| `tushabe_copies.py` | locates drawn annotations by differencing two public copies |
| `o5_tushabe_order_probe.py` | tests whether capture order is recoverable from filenames |
| `o5_annotate_large.py` | builds the image pairs for the human reference study |
| `o5_make_final_figures.py` | all figures in the manuscript |

## Data

`o5_annotation_scored.csv` — the reference-study labels: 60 image pairs rated
independently by two raters, with each pair's similarity score.

## Running

Written for Kaggle notebooks with the four datasets attached. Requires Python
3.10+, numpy, pandas, scipy, scikit-learn, scikit-image, Pillow and matplotlib.
Each script is standalone and prints its own progress.

## Licence

MIT
