# Domain adaptation for fresh and thawed meat classification

RSDE classification with lightweight domain adaptation on raw NIR spectra.
Source: on-meat (OM). Targets: through-package top (TP) and through-bottom (TB).

Open **`notebooks/experiments.ipynb`** for ten numbered tables with titles,
metric definitions, calculations, and CSV filenames. `results/example_run/`
contains a previously completed experiment, including the actual per-split
scores. The export retains those calculations; it does not retrain models.

The notebook starts in `MODE = "saved"` so all ten tables can be inspected
immediately. This mode checks hashes and regenerates summaries from the included
scores. Set `MODE = "recompute"` to train and evaluate all methods again.
Recomputation writes to `results/local_run/` by default.

```powershell
python -m pip install -r requirements.txt
python run_experiments.py --mode saved
python run_experiments.py --mode recompute --jobs 2
```

For the notebook, install the display/kernel dependencies as well, then select
that Python environment in your notebook editor:

```powershell
python -m pip install -r requirements-notebook.txt
```

`requirements.txt` pins the numerical dependencies. `requirements-notebook.txt`
includes those packages plus the IPython and kernel versions used for validation.
Python 3.10.18 was used for the included run.

To continue an interrupted recomputation with identical data/code/software:

```powershell
python run_experiments.py --mode recompute --jobs 2 --resume
```

## Protocol

- 4,053 retained spectra, 153 fillets, and 125 wavelength features.
- Raw spectra; SNV is a separate method variant.
- Tables 5–10 use 25 repeated 70/30 splits of biological fillet IDs, seeds 0–24.
- Replicates from a fillet stay in the same outer partition.
- Target fractions are 0%, 5%, 10%, 20%, 50%, 80%, and 100%, nested by fillet.
- Predictions are averaged within each fillet–condition unit before macro-F1 scoring.
- Confidence intervals and paired comparisons use corrected repeated-resampling variance.
- AUC integrates macro-F1 against target fraction, including 0%; it is not ROC-AUC.

Table 2 uses a separate deterministic DUPLEX holdout. It shows scan accuracy,
scan macro-F1, and fillet macro-F1 explicitly; these metrics are different.
Fresh-spectrum signatures from all three acquisition modes determine this split.
Median imputation applies only to missing signature blocks for split selection.

Tables 1, 3, and 4 describe the fixed evaluation configuration. No hyperparameter
search is included. The parameter-selection history is not established by this
repository. After a transform to p dimensions, RSDE uses
`max(2, round(60 * p / 125))` features per estimator, capped by available dimensions.

## Traceability

`table_index.csv` maps IDs 1–10 to titles and CSV files. The result folder also
contains full-precision summaries, all per-split scores, split manifests,
holdout predictions, and `provenance.json` with input/result hashes and run origin.
The source dataset citation is in `data/README.md`.

The included calculations used Python 3.10.18, NumPy 2.2.6, SciPy 1.15.2,
scikit-learn 1.7.2, pandas 2.3.3, and Joblib 1.5.3 on Windows with MKL.
Different numerical backends can affect results; backend details are retained
in `results/example_run/numerical_environment.json`.

```powershell
python -m unittest discover -s tests -v
```
