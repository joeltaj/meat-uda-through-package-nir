"""Run domain adaptation experiments and produce ten numbered result tables."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "Data_Parastar_corrected.csv"
EXAMPLE_RUN = ROOT / "results" / "example_run"
LOCAL_RUN = ROOT / "results" / "local_run"
_DLL_HANDLES = []


def configure_runtime():
    """Configure only this Python process, including IDE-launched Conda Python."""
    if os.name == "nt":
        dll_dir = Path(sys.prefix) / "Library" / "bin"
        if dll_dir.is_dir():
            normalized = {os.path.normcase(p) for p in os.get_exec_path()}
            if os.path.normcase(str(dll_dir)) not in normalized:
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if not _DLL_HANDLES:
                _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


configure_runtime()
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from src import models as R

MAIN_METHODS = ["CORAL", "SA", "TCA", "GFK", "H1-GFK-DTCA", "H2",
                "SNV+CORAL", "SNV+SA", "SNV+TCA", "SNV+GFK",
                "Pseudo-Label", "KMM", "DS", "TrAdaBoost"]
FRACS = (.05, .10, .20, .50, .80, 1.00)
VERSIONS = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _validate_split(frame, seed):
    expected = len(R.TARGET_MODES) * (len(R.UDA_FRACS) * len(R.ALL_METHODS) + 3)
    if len(frame) != expected or not (frame["split"] == seed).all():
        raise ValueError(f"Incomplete split {seed}: {len(frame)} rows, expected {expected}")
    if frame.duplicated(["split", "mode", "frac", "method", "kind"]).any():
        raise ValueError(f"Duplicate records in split {seed}")
    bad = frame[~np.isfinite(frame.f1) | (frame.f1 < 0) | (frame.f1 > 1)]
    if len(bad):
        raise ValueError(f"Method failure in split {seed}; no partial averages allowed:\n{bad}")

def _compute_split(data_path, seed, directory):
    start = time.perf_counter()
    _, X, sid, mode, y, _ = R.load_data(Path(data_path))
    with threadpool_limits(limits=1):
        frame = pd.DataFrame(R.evaluate_one_split(seed, X, sid, mode, y,
                                                 preprocess="raw", include_slow=True))
    _validate_split(frame, seed)
    path = Path(directory) / f"split_{seed:02d}.csv"
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return seed, time.perf_counter() - start

def compute_experiments(data_path, outdir, jobs=1, resume=False):
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
    checkpoint = outdir / "splits"
    checkpoint.mkdir(exist_ok=True)
    identity = {"data_sha256": sha256(data_path), "implementation_sha256": sha256(R.__file__),
                "runner_sha256": sha256(__file__), "versions": VERSIONS,
                "n_splits": R.DEFAULT_N_SPLITS, "preprocessing": "raw",
                "include_slow": True, "target_mode_index": R.MODE_INDEX,
                "fractions": R.UDA_FRACS.tolist(), "model_config": R.MODEL_CONFIG}
    provenance_path = outdir / "computation_provenance.json"
    if resume:
        if not provenance_path.exists() or json.loads(provenance_path.read_text()) != identity:
            raise ValueError("Cannot resume: data/code/software/configuration identity differs or is missing")
    provenance_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    pending, reused = [], []
    for seed in range(R.DEFAULT_N_SPLITS):
        path = checkpoint / f"split_{seed:02d}.csv"
        if resume and path.exists():
            _validate_split(pd.read_csv(path), seed)
            reused.append(seed)
        else:
            pending.append(seed)
    start = time.perf_counter()
    timings = []
    if jobs == 1:
        for seed in pending:
            print(f"Computing split {seed + 1}/25 (all methods, including TCA/H1)...", flush=True)
            timings.append(_compute_split(str(data_path), seed, str(checkpoint)))
            print(f"Split {seed + 1}/25 completed in {timings[-1][1]:.1f}s", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_compute_split, str(data_path), s, str(checkpoint)) for s in pending]
            for future in as_completed(futures):
                seed, seconds = future.result()
                timings.append((seed, seconds))
                print(f"Split {seed + 1}/25 completed in {seconds:.1f}s; "
                      f"{len(reused) + len(timings)}/25 available", flush=True)
    frames = [pd.read_csv(checkpoint / f"split_{s:02d}.csv") for s in range(R.DEFAULT_N_SPLITS)]
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(outdir / "all_methods_per_split.csv", index=False)
    (outdir / "execution.json").write_text(json.dumps({
        "jobs": jobs, "resumed_splits": reused, "newly_calculated_splits": pending,
        "elapsed_seconds": time.perf_counter() - start, "split_seconds": dict(timings),
        "mode": "recompute",
    }, indent=2), encoding="utf-8")
    return full

def duplex_split(X, sid, mode, freshness):
    """Construct fresh-spectrum signatures and a deterministic DUPLEX split."""
    unique = np.array(sorted(np.unique(sid)))
    signatures = []
    for sample in unique:
        parts = []
        for acquisition in ("OM", "TP", "TB"):
            mask = (sid == sample) & (mode == acquisition) & (freshness == "FR")
            parts.append(X[mask].mean(0) if mask.any() else np.full(X.shape[1], np.nan))
        signatures.append(np.concatenate(parts))
    Z = np.asarray(signatures, float)
    med = np.nanmedian(Z, axis=0)
    rr, cc = np.where(np.isnan(Z))
    Z[rr, cc] = med[cc]
    sd = Z.std(0, ddof=1)
    sd[sd == 0] = 1
    Z = (Z - Z.mean(0)) / sd
    n_test = round(R.TEST_FRACTION * len(unique))
    n_train = len(unique) - n_test
    D = squareform(pdist(Z))
    np.fill_diagonal(D, -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    train = [int(i), int(j)]
    remaining = set(range(len(unique))) - set(train)
    rem = sorted(remaining)
    a, b = np.unravel_index(np.argmax(D[np.ix_(rem, rem)]), (len(rem), len(rem)))
    test = [rem[a], rem[b]]
    remaining -= set(test)
    while remaining:
        if len(train) >= n_train:
            target = test
        elif len(test) >= n_test:
            target = train
        else:
            target = train if ((n_train - len(train)) / n_train >=
                               (n_test - len(test)) / n_test) else test
        candidates = sorted(remaining)
        scores = [min(D[c, s] for s in target) for c in candidates]
        chosen = candidates[int(np.argmax(scores))]
        target.append(chosen)
        remaining.remove(chosen)
    train_ids, test_ids = set(unique[sorted(train)]), set(unique[sorted(test)])
    assert train_ids.isdisjoint(test_ids)
    return train_ids, test_ids

def _default(function, name):
    return inspect.signature(function).parameters[name].default

def configuration_tables():
    model = R.RSDE()
    t1 = [
        ("Classifier", type(model).__name__, "Random Subspace Discriminant Ensemble"),
        ("Base classifier", "LDA-style binary classifier", "Uses pooled covariance and class priors"),
        ("n_estimators", str(model.n_estimators), "Number of subspace classifiers constructed"),
        ("max_features", str(model.max_features), "Wavelengths randomly selected per subspace"),
        ("Feature sampling", "Without replacement", "Avoids wavelength repetition within a subspace"),
        ("Probability aggregation", "Mean probability", "Probabilities averaged across all estimators"),
        ("Covariance regularization", f"{model.regularization:g}", "Stabilizes covariance-matrix inversion"),
        ("Output", "Predicted class", "Class with the highest mean probability"),
    ]
    # Numeric settings come from the implementation's callable defaults.
    t3 = [
        ("CORAL", "reg", _default(R.coral, "reg")),
        ("SA", "var", _default(R.sa, "var")),
        ("TCA", "n_comp", _default(R.tca, "n_comp")),
        ("TCA", "mu", _default(R.tca, "mu")),
        ("TCA", "gamma", "1 / n_features"),
        ("GFK", "var", _default(R._gfk_full, "var")),
        ("SNV", "epsilon", next(c for c in R.snv.__code__.co_consts if c == 1e-10)),
        ("Pseudo-Label", "n_iter", _default(R.pseudo_label, "n_iter")),
        ("Pseudo-Label", "conf_thresh", _default(R.pseudo_label, "conf_thresh")),
        ("KMM", "B", _default(R.kmm_weights, "B")),
        ("KMM", "n_ref", "min(200, n_s)"),
        ("DS", "reg", _default(R.direct_standardize, "reg")),
        ("TrAdaBoost", "T", _default(R.tradaboost_uda, "T")),
        ("TrAdaBoost", "conf_thresh", _default(R.tradaboost_uda, "conf_thresh")),
    ]
    t4 = [
        ("H1-GFK-DTCA", "GFK + Discriminative TCA", "var", _default(R.hybrid1, "var")),
        ("H1-GFK-DTCA", "Discriminative TCA", "n_comp", _default(R.hybrid1, "n_comp")),
        ("H1-GFK-DTCA", "Regularization", "mu_reg", _default(R.hybrid1, "mu_reg")),
        ("H1-GFK-DTCA", "Discriminative weight", "lambda (alpha)", _default(R.hybrid1, "alpha")),
        ("H2-Ensemble", "Source-only expert", "RSDE", "OM model without transformation"),
        ("H2-Ensemble", "GFK expert", "GFK + RSDE", "Uses the current unlabeled target subset"),
        ("H2-Ensemble", "CORAL expert", "CORAL + RSDE", "Uses the current unlabeled target subset"),
        ("H2-Ensemble", "Target information", "Current target only", "No access to the alternate packaged domain"),
        ("H2-Ensemble", "Gating", "Source/target centroid distance", "Centroids from source + current unlabeled target subset"),
        ("H2-Ensemble", "Gating scale (sigma)", "Median positive squared distance", "Computed from source/current-target adaptation data only"),
        ("H2-Ensemble", "Adapted block / final rule", "Equal GFK-CORAL average + two-way gate", "psource and padapt combined by source/target proximity"),
    ]
    return {1: pd.DataFrame(t1, columns=["Parameter", "Value", "Role in the model"]),
            3: pd.DataFrame(t3, columns=["Method", "Key parameter", "Value"]),
            4: pd.DataFrame(t4, columns=["Hybrid method", "Component", "Parameter", "Value / mechanism"])}

def format_ci(mean, low, high, percent=False):
    scale, precision = (100, 1) if percent else (1, 4)
    return f"{scale * mean:.{precision}f} [{scale * low:.{precision}f}, {scale * high:.{precision}f}]"

def build_tables(full, outdir):
    # Retain the original full-precision summaries, then render at table display precision.
    R.make_summaries(full, outdir)
    summary = pd.read_csv(outdir / "summary_f1.csv")
    references = pd.read_csv(outdir / "references.csv")
    generated = configuration_tables()
    names = {"source_only": "Source-only (OM -> target)",
             "in_domain_supervised": "In-domain supervised",
             "supervised_pooled_reference": "Supervised pooled-domain reference"}
    table5 = []
    for internal, display in names.items():
        row = {"Scenario": display}
        for mode in ("OM", "TP", "TB"):
            q = references[(references.reference == internal) & (references["mode"] == mode)].iloc[0]
            if q.n_splits != 25:
                raise ValueError("Incomplete reference distribution")
            row[mode] = format_ci(q.mean_f1, q.ci95_lo, q.ci95_hi, percent=True)
        table5.append(row)
    generated[5] = pd.DataFrame(table5)
    for mode, number in (("OM", 6), ("TP", 7), ("TB", 8)):
        rows = []
        for method in MAIN_METHODS:
            row = {"Method": method}
            for frac in FRACS:
                q = summary[(summary["mode"] == mode) & (summary.method == method)
                            & np.isclose(summary.frac, frac)].iloc[0]
                if q.n_splits != 25:
                    raise ValueError(f"Incomplete distribution: {mode}, {method}, {frac}")
                row[f"{round(100 * frac)}%"] = format_ci(q.mean_f1, q.ci95_lo, q.ci95_hi, percent=True)
            rows.append(row)
        generated[number] = pd.DataFrame(rows)
    ablation = full[(full.kind == "method") & (full["mode"].isin(["TP", "TB"]))
                    & full.method.isin(R.ABLATION_METHODS)].copy()
    auc = R.ablation_auc_from_long(ablation)
    auc.to_csv(outdir / "ablation_auc_per_split.csv", index=False)
    t9 = R.ablation_summary(auc)
    generated[9] = t9[["Aggregation / expert", "TP_AUC", "TB_AUC"]].rename(
        columns={"TP_AUC": "TP AUC", "TB_AUC": "TB AUC"})
    stats = R.paired_ablation_statistics(auc)
    stats.to_csv(outdir / "ablation_paired_statistics_full_precision.csv", index=False)
    rows = []
    for comparison in ["Equal probability average", "Majority vote", "Confidence weighting"]:
        row = {"Comparison": "H2 - " + comparison}
        for mode in ("TP", "TB"):
            q = stats[(stats["mode"] == mode) & (stats.comparison == "H2 - " + comparison)].iloc[0]
            row[f"{mode} delta AUC [95% CI]"] = format_ci(q.mean_diff_auc, q.ci95_lo, q.ci95_hi)
        rows.append(row)
    generated[10] = pd.DataFrame(rows)
    return generated


TABLE_TITLES = {
    1: "RSDE model configuration",
    2: "In-domain DUPLEX holdout: accuracy and macro-F1",
    3: "Domain adaptation parameter settings",
    4: "Hybrid H1 and H2 configurations",
    5: "Source-only and supervised reference performance",
    6: "Domain adaptation performance on OM",
    7: "Domain adaptation performance on TP",
    8: "Domain adaptation performance on TB",
    9: "H2 ablation: normalized performance-versus-fraction AUC",
    10: "Paired H2 AUC differences and corrected confidence intervals",
}


def validate_full_results(full):
    if set(full["split"]) != set(range(25)):
        raise ValueError("All 25 split seeds (0--24) are required")
    for seed, frame in full.groupby("split"):
        _validate_split(frame, seed)
        expected = {
            (mode, float(frac), method, "method")
            for mode in R.TARGET_MODES for frac in R.UDA_FRACS for method in R.ALL_METHODS
        }
        expected |= {
            (mode, 0.0, reference, "reference") for mode in R.TARGET_MODES
            for reference in ("source_only", "in_domain_supervised", "supervised_pooled_reference")
        }
        actual = set(frame[["mode", "frac", "method", "kind"]].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(f"Incorrect domain/fraction/method grid in split {seed}")


def compute_holdout(data_path, outdir):
    """Fit in-domain RSDE models on the deterministic DUPLEX holdout."""
    _, X, sid, mode, y, _ = R.load_data(data_path)
    train_ids, test_ids = duplex_split(X, sid, mode, y)
    manifest = [{"Sample_number": sample, "partition": partition}
                for partition, ids in (("train", train_ids), ("test", test_ids)) for sample in sorted(ids)]
    pd.DataFrame(manifest).to_csv(outdir / "duplex_manifest.csv", index=False)
    rows, predictions = [], []
    with threadpool_limits(limits=1):
        for acquisition in ("OM", "TP", "TB"):
            train = (mode == acquisition) & np.isin(sid, list(train_ids))
            test = (mode == acquisition) & np.isin(sid, list(test_ids))
            model = R.rsde_for(X.shape[1], X.shape[1]).fit(X[train], y[train])
            proba = model.predict_proba(X[test])
            pred = model.classes_[proba.argmax(1)]
            fillet_f1, _ = R.fillet_probability_f1(y[test], model.classes_, proba, sid[test])
            rows.append({"Mode": acquisition, "n_train_spectra": int(train.sum()),
                         "n_test_spectra": int(test.sum()),
                         "accuracy_pct": 100 * accuracy_score(y[test], pred),
                         "scan_macro_f1_pct": 100 * f1_score(y[test], pred, average="macro", labels=["FR", "TH"]),
                         "fillet_macro_f1_pct": 100 * fillet_f1})
            for index, truth, prediction, p in zip(np.flatnonzero(test), y[test], pred, proba):
                predictions.append({"row_index": int(index), "mode": acquisition,
                                    "sample_id": sid[index], "truth": truth, "prediction": prediction,
                                    "p_FR": p[0], "p_TH": p[1]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "holdout_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(outdir / "holdout_predictions.csv", index=False)
    return metrics


RUN_INPUT_FILES = ["all_methods_per_split.csv", "holdout_metrics.csv", "holdout_predictions.csv",
                   "duplex_manifest.csv", "outer_split_manifest.csv", "leakage_audit.csv"]


def record_provenance(data_path, outdir, origin):
    provenance = {"data_file": Path(data_path).name, "data_sha256": sha256(data_path),
                  "public_model_code_sha256": sha256(R.__file__),
                  "model_config": R.MODEL_CONFIG, "versions": VERSIONS,
                  "origin": origin,
                  "artifacts": {name: sha256(outdir / name) for name in RUN_INPUT_FILES}}
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def load_saved_results(data_path, outdir):
    provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    if provenance["data_sha256"] != sha256(data_path):
        raise ValueError("Saved results belong to a different dataset")
    if provenance["public_model_code_sha256"] != sha256(R.__file__):
        raise ValueError("Model code changed; recompute the experiment")
    for name, expected in provenance["artifacts"].items():
        if sha256(outdir / name) != expected:
            raise ValueError(f"Saved result artifact changed: {name}")
    full = pd.read_csv(outdir / "all_methods_per_split.csv")
    validate_full_results(full)
    metrics = pd.read_csv(outdir / "holdout_metrics.csv")
    if set(metrics.Mode) != {"OM", "TP", "TB"} or len(metrics) != 3:
        raise ValueError("Incomplete in-domain holdout results")
    return full, metrics


def run(mode="saved", data_path=DEFAULT_DATA, outdir=None, jobs=2, resume=False):
    if mode not in {"saved", "recompute"}:
        raise ValueError("mode must be saved or recompute")
    data_path = Path(data_path)
    outdir = Path(outdir) if outdir is not None else (EXAMPLE_RUN if mode == "saved" else LOCAL_RUN)
    outdir.mkdir(parents=True, exist_ok=True)
    df, X, sid, scan_type, y, columns = R.load_data(data_path)
    R.validate_dataset(df, X, sid, columns)
    print(f"Execution mode: {mode}", flush=True)
    if mode == "saved":
        print("Loading the included calculated run; models are not retrained in this mode.", flush=True)
        full, holdout = load_saved_results(data_path, outdir)
    else:
        R.leakage_audit(sid, scan_type).to_csv(outdir / "leakage_audit.csv", index=False)
        R.make_split_manifest(sid).to_csv(outdir / "outer_split_manifest.csv", index=False)
        holdout = compute_holdout(data_path, outdir)
        full = compute_experiments(data_path, outdir, jobs=jobs, resume=resume)
        validate_full_results(full)
        execution = json.loads((outdir / "execution.json").read_text())
        record_provenance(data_path, outdir, {"type": "computed_run", "execution": execution})
    tables = build_tables(full, outdir)
    tables[2] = holdout
    tracker = []
    for number in range(1, 11):
        filename = f"table{number:02d}.csv"
        tables[number].to_csv(outdir / filename, index=False)
        tracker.append({"Table": number, "Title": TABLE_TITLES[number],
                        "Type": "Configuration" if number in (1, 3, 4) else "Experiment",
                        "CSV": filename})
    tracking = pd.DataFrame(tracker)
    tracking.to_csv(outdir / "table_index.csv", index=False)
    R.dataset_accounting(df).to_csv(outdir / "dataset_accounting.csv", index=False)
    display_dir = outdir.relative_to(ROOT) if outdir.is_relative_to(ROOT) else outdir
    print(f"Tables 1--10 ready; {len(full):,} per-split records. Output: {display_dir}", flush=True)
    return {"tables": tables, "full": full, "tracking": tracking, "mode": mode, "outdir": outdir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["saved", "recompute"], default="recompute")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.data, args.outdir, args.jobs, args.resume)


if __name__ == "__main__":
    main()
