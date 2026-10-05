from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSIFICATION_SCHEMA_VERSION = 1


@dataclass
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    epoch_ids: np.ndarray
    states: np.ndarray
    folds: np.ndarray
    feature_names: list[str]
    epochs: pd.DataFrame
    subject_folds: pd.DataFrame
    configuration: dict
    summary: dict


_CI_GLOBAL_KEYS = {"X", "y", "subjects", "epoch_ids", "states", "folds", "sample_indices", "feature_names"}
_CI_PARTITION_KEYS = {"X", "y", "subjects", "epoch_ids", "sample_indices", "feature_names"}
_CI_LABEL_MAPPING = {"HC": 0, "ADHD": 1}
_CI_EPOCH_COLUMNS = {
    "sample_index", "subject", "label", "label_name", "fold", "epoch_id", "epoch_index", "state", "segment",
    "concatenated_start_sample", "concatenated_stop_sample_exclusive", "original_start_sample",
    "original_stop_sample_exclusive", "onset_seconds", "offset_seconds", "duration_seconds",
}


METRIC_NAMES = ("BAcc", "AUC", "Sens", "Spec", "F1")


def _ci_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _ci_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return int(value)


def _ci_integers(values: Any, name: str, minimum: int, maximum: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.size == 0:
        raise ValueError(f"{name} must be a nonempty integer vector")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{name} contains values outside {minimum} through {maximum}")
    return array.astype(np.int64, copy=False)


def _ci_strings(values: Any, name: str, unique: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "UO":
        raise ValueError(f"{name} must be a nonempty text vector")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in array):
        raise ValueError(f"{name} contains empty, nontext, or whitespace-padded values")
    result = array.astype(np.str_)
    if unique and len(set(result.tolist())) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _ci_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys or len(archive.files) != len(expected_keys):
            raise ValueError(f"NPZ keys do not match the saved feature schema: {path}")
        arrays = {name: archive[name] for name in expected_keys}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError(f"Object arrays are not supported: {path}")
    return arrays


def _ci_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, float_precision="round_trip",
        dtype={"subject": str, "epoch_id": str, "state": str, "label_name": str, "age": str, "sex": str},
    )
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError(f"Expected a nonempty CSV with unique columns: {path}")
    return frame.reset_index(drop=True)


def _ci_subject_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if set(frame.columns) != {"subject", "label", "label_name", "fold"}:
        raise ValueError(f"{name} does not have the saved subject-fold columns")
    result = frame.copy()
    result["subject"] = _ci_strings(result["subject"].to_numpy(), f"{name} subjects", unique=True)
    result["label"] = _ci_integers(result["label"].to_numpy(), f"{name} labels", 0, 1)
    result["fold"] = _ci_integers(result["fold"].to_numpy(), f"{name} folds", 1, 10)
    expected_names = result["label"].map({0: "HC", 1: "ADHD"})
    if not result["label_name"].equals(expected_names.rename("label_name")):
        raise ValueError(f"{name} label names do not match 0=HC and 1=ADHD")
    return result.loc[:, ["subject", "label", "label_name", "fold"]].sort_values("subject").reset_index(drop=True)


def _ci_equal_frame(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    if list(actual.columns) != list(expected.columns) or not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError(f"{name} differs from the saved global dataset")


def _ci_epochs(frame: pd.DataFrame, arrays: dict[str, np.ndarray], expected_indices: np.ndarray, name: str) -> None:
    if not _CI_EPOCH_COLUMNS.issubset(frame.columns) or len(frame) != len(expected_indices):
        raise ValueError(f"{name} has incomplete epoch metadata or the wrong row count")
    columns = {"sample_index": "sample_indices", "label": "y", "subject": "subjects", "epoch_id": "epoch_ids"}
    for column, key in columns.items():
        if not np.array_equal(frame[column].to_numpy(), arrays[key]):
            raise ValueError(f"{name} column {column} differs from its NPZ array")
    if not np.array_equal(arrays["sample_indices"], expected_indices):
        raise ValueError(f"{name} does not preserve the required global sample indices")
    for column in ("sample_index", "label", "fold", "epoch_index", "segment", "concatenated_start_sample", "concatenated_stop_sample_exclusive", "original_start_sample", "original_stop_sample_exclusive"):
        _ci_integers(frame[column].to_numpy(), f"{name} {column}", 0, np.iinfo(np.int64).max)
    for column in ("onset_seconds", "offset_seconds", "duration_seconds"):
        values = frame[column].to_numpy()
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name} contains invalid {column}")
    if not frame["label_name"].equals(frame["label"].map({0: "HC", 1: "ADHD"}).rename("label_name")):
        raise ValueError(f"{name} contains inconsistent label names")


def _ci_validate_saved_folds(directory: Path, dataset: FeatureDataset) -> None:
    sample_count = len(dataset.labels)
    metadata_name = "epochs"
    global_indices = np.arange(sample_count, dtype=np.int64)
    held_out_counts = np.zeros(sample_count, dtype=np.int64)
    manifest = dataset.configuration.get("subject_fold_manifest")
    if not isinstance(manifest, dict) or manifest.get("seed") != dataset.configuration["seed"]:
        raise ValueError("The configuration lacks matching saved subject-fold metadata")
    cohort_hash = manifest.get("cohort_hash")
    if not isinstance(cohort_hash, str) or len(cohort_hash) != 64 or any(character not in "0123456789abcdef" for character in cohort_hash):
        raise ValueError("The saved fold cohort hash is invalid")
    fold_root = directory / "folds"
    expected_names = {f"fold_{fold:02d}" for fold in range(1, 11)}
    if {path.name for path in fold_root.iterdir() if path.is_dir() and path.name.startswith("fold_")} != expected_names:
        raise ValueError("Exactly the ten saved fold directories must be available")
    for fold in range(1, 11):
        fold_directory = fold_root / f"fold_{fold:02d}"
        split = _ci_json(fold_directory / "split.json")
        expected_metadata = {
            "fold": fold, "n_splits": 10, "split_unit": "subject", "feature_count": len(dataset.feature_names),
            "label_mapping": _CI_LABEL_MAPPING, "cohort_hash": cohort_hash, "seed": dataset.configuration["seed"],
            "fold_manifest": manifest.get("fold_manifest"), "feature_scaling": "none", "subject_overlap": False, "held_out_once": True,
        }
        if any(split.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"Saved fold {fold} metadata disagrees with the feature configuration")
        if split.get("subject_overlap") is not False or split.get("held_out_once") is not True:
            raise ValueError(f"Saved fold {fold} does not guarantee subject-disjoint held-out evaluation")
        partition_metadata = split.get("partitions")
        if not isinstance(partition_metadata, dict) or set(partition_metadata) != {"train", "test"}:
            raise ValueError(f"Saved fold {fold} lacks train/test partition metadata")
        partition_subject_sets = []
        for partition in ("train", "test"):
            selected = dataset.folds == fold if partition == "test" else dataset.folds != fold
            indices = global_indices[selected]
            if not indices.size or set(dataset.labels[indices].tolist()) != {0, 1}:
                raise ValueError(f"Saved fold {fold} {partition} must contain both classes")
            arrays = _ci_npz(fold_directory / f"{partition}.npz", _CI_PARTITION_KEYS)
            expected_arrays = {
                "X": dataset.features[indices], "y": dataset.labels[indices], "sample_indices": indices,
                "subjects": dataset.subjects[indices],
                "feature_names": np.asarray(dataset.feature_names, dtype=np.str_),
            }
            expected_arrays["epoch_ids"] = dataset.epoch_ids[indices]
            if any(not np.array_equal(arrays[key], value) for key, value in expected_arrays.items()):
                raise ValueError(f"Saved fold {fold} {partition} NPZ differs from its global rows or feature order")
            partition_samples = _ci_csv(fold_directory / f"{partition}_{metadata_name}.csv")
            _ci_epochs(partition_samples, arrays, indices, f"Fold {fold} {partition} {metadata_name}")
            _ci_equal_frame(partition_samples, dataset.epochs.iloc[indices], f"Fold {fold} {partition} {metadata_name}")
            subject_set = set(dataset.subjects[indices].tolist())
            expected_subjects = dataset.subject_folds.loc[dataset.subject_folds["subject"].isin(subject_set)].reset_index(drop=True)
            partition_subjects = _ci_subject_table(_ci_csv(fold_directory / f"{partition}_subjects.csv"), f"Fold {fold} {partition}")
            _ci_equal_frame(partition_subjects, expected_subjects, f"Fold {fold} {partition} subjects")
            counts = partition_metadata[partition]
            expected_counts = {
                "epoch_count": len(indices), "subject_count": len(expected_subjects),
                "class_subject_counts": {"HC": int(expected_subjects["label"].eq(0).sum()), "ADHD": int(expected_subjects["label"].eq(1).sum())},
                "subjects": expected_subjects["subject"].tolist(),
            }
            if not isinstance(counts, dict) or any(counts.get(key) != value for key, value in expected_counts.items()):
                raise ValueError(f"Saved fold {fold} {partition} counts or subject identities are inconsistent")
            partition_subject_sets.append(subject_set)
            if partition == "test":
                held_out_counts[indices] += 1
        if partition_subject_sets[0].intersection(partition_subject_sets[1]):
            raise ValueError(f"Subject leakage in saved fold {fold}")
    if not np.all(held_out_counts == 1):
        raise ValueError("Every epoch must appear in exactly one saved test fold")


def load_feature_dataset(feature_directory: Path) -> FeatureDataset:
    directory = Path(feature_directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("The feature input must be one completed SE configuration directory")
    configuration = _ci_json(directory / "configuration.json")
    if configuration.get("feature_family", "spectral_entropy") != "spectral_entropy" or configuration.get("sample_unit", "epoch") != "epoch":
        raise ValueError("This script only classifies epoch-level SE features from extract_fs_sfc_se_feature.py. Use classify_fs_mean_sfc_feature.py for subject-level mean SFC features.")
    summary = _ci_json(directory / "summary.json")
    schema = _ci_json(directory / "feature_schema.json")
    if schema.get("sample_unit") != "one retained epoch":
        raise ValueError("SE feature metadata must identify one retained epoch per sample")
    if summary.get("feature_family", "spectral_entropy") != "spectral_entropy" or summary.get("sample_unit", "epoch") != "epoch":
        raise ValueError("The SE summary must identify epoch-level spectral entropy features")
    if configuration.get("schema_version") != 2 or configuration.get("n_splits") != 10:
        raise ValueError("The classifier requires SE schema version 2 with ten saved subject folds")
    _ci_integer(configuration.get("seed"), "Saved split seed", 0, 2**32 - 1)
    if summary.get("status") != "complete" or summary.get("subject_disjoint_folds") is not True or summary.get("n_splits") != 10:
        raise ValueError("The SE dataset must report complete subject-disjoint ten-fold export")
    if (directory / "run_status.json").is_file() and _ci_json(directory / "run_status.json").get("status") != "complete":
        raise ValueError("The SE run status is not complete")
    for key, value in {"features_file": "features.npz", "epoch_metadata": "epochs.csv", "subject_folds": "subject_folds.csv", "fold_directory": "folds"}.items():
        if summary.get(key) != value:
            raise ValueError(f"Unexpected SE summary file mapping: {key}")
    condition = configuration.get("condition")
    if condition not in ("EO", "EC", "ALL") or summary.get("condition") != condition:
        raise ValueError("SE configuration and summary must identify the same valid eye condition")
    if configuration.get("graph") not in ("SC", "FC", "SFC") or summary.get("graph") != configuration["graph"]:
        raise ValueError("SE configuration and summary graph types disagree")
    if any(record.get("label_mapping") != _CI_LABEL_MAPPING for record in (configuration, schema)):
        raise ValueError("The saved label mapping must be HC=0 and ADHD=1")
    bands = configuration.get("bands")
    if not isinstance(bands, list) or not bands or any(band not in ("delta", "theta", "alpha", "beta", "gamma") for band in bands) or len(set(bands)) != len(bands):
        raise ValueError("The SE configuration contains invalid or duplicate frequency bands")
    arrays = _ci_npz(directory / "features.npz", _CI_GLOBAL_KEYS)
    features = arrays["X"]
    if features.ndim != 2 or features.dtype != np.dtype("float64") or min(features.shape) < 1 or not np.isfinite(features).all():
        raise ValueError("Saved SE features must be a nonempty finite float64 matrix")
    if np.any(features < -1e-10) or np.any(features > 1.0 + 1e-10):
        raise ValueError("Normalized spectral entropy features must lie between zero and one")
    sample_count, feature_count = features.shape
    labels = _ci_integers(arrays["y"], "Epoch labels", 0, 1)
    folds = _ci_integers(arrays["folds"], "Epoch folds", 1, 10)
    indices = _ci_integers(arrays["sample_indices"], "Sample indices", 0, sample_count - 1)
    subjects = _ci_strings(arrays["subjects"], "Epoch subjects")
    epoch_ids = _ci_strings(arrays["epoch_ids"], "Epoch identifiers", unique=True)
    states = _ci_strings(arrays["states"], "Epoch states")
    names = _ci_strings(arrays["feature_names"], "Feature names", unique=True).tolist()
    if any(len(values) != sample_count for values in (labels, folds, indices, subjects, epoch_ids, states)) or not np.array_equal(indices, np.arange(sample_count)):
        raise ValueError("Global feature arrays do not share consecutive sample rows")
    if names != [f"SE_{band}" for band in bands] or len(names) != feature_count or summary.get("feature_names") != names or schema.get("feature_names") != names:
        raise ValueError("Saved feature names or frequency-band order disagree")
    if summary.get("feature_shape") != list(features.shape) or schema.get("X_shape") != list(features.shape) or summary.get("epoch_count") != sample_count:
        raise ValueError("Saved global feature dimensions disagree with the metadata")
    if schema.get("X_dtype") != "float64" or set(schema.get("npz_keys", [])) != _CI_GLOBAL_KEYS or schema.get("epoch_metadata") != "epochs.csv":
        raise ValueError("The feature schema does not match the saved dataset representation")
    expected_states = {"EO", "EC"} if condition == "ALL" else {condition}
    if set(states.tolist()) != expected_states or set(labels.tolist()) != {0, 1}:
        raise ValueError("Saved epoch conditions or classes disagree with the requested dataset")
    npy_path = directory / "features.npy"
    if npy_path.is_file() and not np.array_equal(np.load(npy_path, allow_pickle=False), features):
        raise ValueError("features.npy differs from the authoritative features.npz matrix")
    epochs = _ci_csv(directory / "epochs.csv")
    _ci_epochs(epochs, arrays, indices, "Global epochs")
    if not np.array_equal(epochs["state"].to_numpy(), states) or not np.array_equal(epochs["fold"].to_numpy(), folds):
        raise ValueError("Global epoch states or fold assignments disagree with features.npz")
    subject_folds = _ci_subject_table(_ci_csv(directory / "subject_folds.csv"), "Global subject folds")
    if set(subject_folds["subject"]) != set(subjects.tolist()) or summary.get("subject_count") != len(subject_folds):
        raise ValueError("The complete saved subject cohort must contribute feature epochs")
    by_subject = subject_folds.set_index("subject")
    if not np.array_equal(by_subject.loc[subjects, "label"].to_numpy(), labels) or not np.array_equal(by_subject.loc[subjects, "fold"].to_numpy(), folds):
        raise ValueError("Epoch labels or folds disagree with their subject assignments")
    for fold in range(1, 11):
        if set(subject_folds.loc[subject_folds["fold"].eq(fold), "label"]) != {0, 1}:
            raise ValueError(f"Saved test fold {fold} must contain both HC and ADHD subjects")
    cohort = configuration.get("cohort")
    if not isinstance(cohort, list) or sorted(cohort, key=lambda row: row["subject"]) != subject_folds[["subject", "label"]].to_dict(orient="records"):
        raise ValueError("The configuration cohort differs from the saved subject-fold cohort")
    if condition == "ALL" and any(set(group["state"]) != {"EO", "EC"} for _, group in epochs.groupby("subject", sort=False)):
        raise ValueError("Every ALL-condition subject must contribute both EO and EC epochs")
    dataset = FeatureDataset(features, labels, subjects, epoch_ids, states, folds, names, epochs, subject_folds, configuration, summary)
    _ci_validate_saved_folds(directory, dataset)
    return dataset


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    scores = np.asarray(decision_scores)
    if truth.ndim != 1 or predictions.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels, predictions and decision scores must be one-dimensional")
    if truth.size < 2 or predictions.shape != truth.shape or scores.shape != truth.shape:
        raise ValueError("Labels, predictions and decision scores must have equal nonempty lengths")
    for values, name in ((truth, "True labels"), (predictions, "Predicted labels")):
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0=HC and 1=ADHD")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("Each test fold must contain both HC and ADHD samples")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("AUC requires finite real SVM decision scores")
    truth = truth.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    )
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "BAcc": float((sensitivity + specificity) / 2.0),
        "AUC": float(roc_auc_score(truth, scores)),
        "Sens": float(sensitivity),
        "Spec": float(specificity),
        "F1": float(f1_score(truth, predictions, pos_label=1, zero_division=0)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "n_samples": int(truth.size),
    }


def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
    ddof: int = 1,
) -> pd.DataFrame:
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)) or ddof not in (0, 1):
        raise ValueError("Standard-deviation ddof must be 0 or 1")
    if not isinstance(fold_metrics, pd.DataFrame) or len(fold_metrics) != 10:
        raise ValueError("Metric summaries require exactly ten fold rows")
    if fold_metrics.columns.duplicated().any():
        raise ValueError("Fold metric columns must be unique")
    missing = {"fold", *METRIC_NAMES}.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"Missing fold metric columns: {sorted(missing)}")
    fold_column = pd.to_numeric(fold_metrics["fold"], errors="raise")
    if np.iscomplexobj(fold_column.to_numpy()):
        raise ValueError("Fold identifiers must be real integers")
    folds = fold_column.to_numpy(dtype=np.float64)
    if not np.isfinite(folds).all() or not np.array_equal(np.sort(folds), np.arange(1, 11, dtype=np.float64)):
        raise ValueError("Fold identifiers must contain each integer from 1 through 10 exactly once")
    rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        column = pd.to_numeric(fold_metrics[metric], errors="raise")
        if np.iscomplexobj(column.to_numpy()):
            raise ValueError(f"{metric} fold values must be real")
        values = column.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{metric} fold values must be finite numbers between zero and one")
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=int(ddof)))
        standard_deviation = float(np.sqrt(variance))
        rows.append({
            "metric": metric,
            "n_folds": 10,
            "ddof": int(ddof),
            "mean": mean,
            "std": standard_deviation,
            "variance": variance,
            "mean_percent": mean * 100.0,
            "std_percentage_points": standard_deviation * 100.0,
            "variance_percentage_points_squared": variance * 10000.0,
            "formatted_mean_std_percent": f"{mean * 100.0:.2f} ± {standard_deviation * 100.0:.2f}",
        })
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify saved epoch-level spectral entropy features from extract_fs_sfc_se_feature.py with a linear SVM using the existing subject-disjoint ten-fold partitions.")
    parser.add_argument("feature_directory", type=Path, help="One SE configuration directory containing features.npz and folds/fold_01 through fold_10.")
    parser.add_argument("--output-dir", type=Path, help="Parent for configuration-specific result folders; default: <feature_directory>/classification.")
    parser.add_argument("--C", dest="c", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True, help="Fit StandardScaler on each training partition only; enabled by default.")
    parser.add_argument("--std-ddof", type=int, choices=(0, 1), default=1, help="Degrees of freedom for the standard deviation and variance across ten test folds.")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    arguments.feature_directory = arguments.feature_directory.expanduser().resolve()
    if not arguments.feature_directory.is_dir():
        parser.error("feature_directory must be an existing SE configuration directory.")
    if arguments.output_dir is None:
        arguments.output_dir = arguments.feature_directory / "classification"
    arguments.output_dir = arguments.output_dir.expanduser().resolve()
    for option, value in (("--C", arguments.c), ("--tol", arguments.tol)):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{option} must be finite and positive.")
    if arguments.max_iter != -1 and arguments.max_iter <= 0:
        parser.error("--max-iter must be -1 for no limit or a positive integer.")
    for protected in (arguments.feature_directory / "folds", arguments.feature_directory / "subjects"):
        if arguments.output_dir == protected or arguments.output_dir.is_relative_to(protected):
            parser.error("Classification outputs must be outside the input folds and subjects directories.")
    return arguments


def classification_configuration(arguments: argparse.Namespace, dataset: FeatureDataset) -> dict[str, Any]:
    directory = arguments.feature_directory
    metadata_name = "epochs"
    paths = [directory / name for name in ("features.npz", f"{metadata_name}.csv", "subject_folds.csv", "configuration.json", "summary.json", "feature_schema.json")]
    for fold in range(1, 11):
        fold_directory = directory / "folds" / f"fold_{fold:02d}"
        paths.extend(fold_directory / name for name in ("train.npz", "test.npz", "split.json", f"train_{metadata_name}.csv", f"test_{metadata_name}.csv", "train_subjects.csv", "test_subjects.csv"))
    seed = dataset.configuration.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("The feature configuration must contain its valid integer cross-validation seed")
    return {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "classifier": "sklearn.svm.SVC",
        "kernel": "linear",
        "C": arguments.c,
        "class_weight": None if arguments.class_weight == "none" else arguments.class_weight,
        "standardize": arguments.standardize,
        "standardization_scope": "fit on each training fold only, then transform its training and test samples" if arguments.standardize else "none",
        "probability_estimation": False,
        "class_weight_scope": "training epoch counts only" if arguments.class_weight == "balanced" else "none",
        "reproducibility": "reuse saved subject folds and sample order; record input hashes and software versions; SVC ignores random_state when probability=False",
        "decision_scores": "SVC.decision_function; positive values favor ADHD (class 1)",
        "label_predictions": "SVC.predict using its default decision boundary",
        "tol": arguments.tol,
        "max_iter": arguments.max_iter,
        "save_models": arguments.save_models,
        "cross_validation": "reuse the ten saved subject-disjoint folds without repartitioning",
        "n_splits": 10,
        "split_seed": seed,
        "sample_unit": "epoch",
        "feature_family": "spectral_entropy",
        "label_mapping": {"HC": 0, "ADHD": 1},
        "metrics": list(METRIC_NAMES),
        "fold_aggregation": "unweighted arithmetic mean of the ten held-out fold metrics",
        "std_ddof": arguments.std_ddof,
        "std_definition": "sample standard deviation" if arguments.std_ddof else "population standard deviation",
        "paper_table_format": "mean plus/minus standard deviation, in percent",
        "hyperparameter_search": "none; the supplied settings are fixed across all folds",
        "feature_directory": str(directory),
        "feature_names": dataset.feature_names,
        "feature_shape": list(dataset.features.shape),
        "feature_configuration": dataset.configuration,
        "input_files": [{"path": path.relative_to(directory).as_posix(), "sha256": file_sha256(path)} for path in paths],
        "source_script_sha256": file_sha256(Path(__file__)),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }


def prepare_output_directory(arguments: argparse.Namespace, configuration: dict[str, Any]) -> Path:
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    c_name = format(arguments.c, ".8g").replace(".", "p")
    name = f"linear_svm_C-{c_name}_scale-{int(arguments.standardize)}_weight-{arguments.class_weight}_sd-{arguments.std_ddof}_{signature}"
    root = arguments.output_dir.resolve()
    directory = root / name
    resolved = directory.resolve()
    if directory.is_symlink() or not resolved.is_relative_to(root) or resolved.parent != root or arguments.feature_directory.is_relative_to(resolved):
        raise ValueError("The result directory must be a dedicated child of the classification output directory")
    if directory.exists() and any(directory.iterdir()):
        if not arguments.overwrite:
            raise RuntimeError(f"Classification output already exists: {directory}. Use --overwrite to replace this result configuration.")
        shutil.rmtree(resolved)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "configuration.json", configuration)
    return directory


def build_classifier(arguments: argparse.Namespace, seed: int) -> Pipeline:
    steps = []
    if arguments.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svm", SVC(
        C=arguments.c,
        kernel="linear",
        class_weight=None if arguments.class_weight == "none" else "balanced",
        probability=False,
        tol=arguments.tol,
        max_iter=arguments.max_iter,
        random_state=seed,
    )))
    return Pipeline(steps)


def save_model_parameters(model: Pipeline, feature_names: Sequence[str], directory: Path) -> None:
    svm = model.named_steps["svm"]
    coefficients = np.asarray(svm.coef_, dtype=np.float64)[0]
    intercept = float(svm.intercept_[0])
    if "scaler" in model.named_steps:
        scaler = model.named_steps["scaler"]
        means = np.asarray(scaler.mean_, dtype=np.float64)
        scales = np.asarray(scaler.scale_, dtype=np.float64)
    else:
        means = np.zeros(len(feature_names), dtype=np.float64)
        scales = np.ones(len(feature_names), dtype=np.float64)
    original_coefficients = coefficients / scales
    original_intercept = intercept - float(np.dot(original_coefficients, means))
    pd.DataFrame({
        "feature": list(feature_names),
        "model_coefficient": coefficients,
        "original_scale_coefficient": original_coefficients,
        "scaler_mean": means,
        "scaler_scale": scales,
    }).to_csv(directory / "model_coefficients.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "model_parameters.json", {
        "classes": svm.classes_.astype(int).tolist(),
        "intercept_model_scale": intercept,
        "intercept_original_feature_scale": original_intercept,
        "support_vector_counts": svm.n_support_.astype(int).tolist(),
        "fit_status": int(svm.fit_status_),
        "training_only_standardization": "scaler" in model.named_steps,
        "decision_formula_original_scale": "X @ original_scale_coefficient + intercept_original_feature_scale",
    })


def classify_fold(
    fold: int,
    dataset: FeatureDataset,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_indices = np.flatnonzero(dataset.folds != fold)
    test_indices = np.flatnonzero(dataset.folds == fold)
    train_subjects = set(dataset.subjects[train_indices].tolist())
    test_subjects = set(dataset.subjects[test_indices].tolist())
    if not len(train_indices) or not len(test_indices) or train_subjects.intersection(test_subjects):
        raise ValueError(f"Fold {fold} has an empty partition or subject overlap")
    model = build_classifier(arguments, seed=int(dataset.configuration["seed"]))
    model.fit(dataset.features[train_indices], dataset.labels[train_indices])
    svm = model.named_steps["svm"]
    if svm.fit_status_ != 0:
        raise RuntimeError(f"The linear SVM did not converge in fold {fold}; increase --max-iter or use -1")
    if not np.array_equal(svm.classes_, np.array([0, 1])):
        raise RuntimeError("SVM class order must be HC=0 followed by ADHD=1")
    predictions = np.asarray(model.predict(dataset.features[test_indices]), dtype=np.int64)
    decision_scores = np.asarray(model.decision_function(dataset.features[test_indices]), dtype=np.float64)
    metrics = compute_binary_metrics(dataset.labels[test_indices], predictions, decision_scores)
    record = {
        "fold": fold,
        "train_subject_count": len(train_subjects),
        "test_subject_count": len(test_subjects),
        "train_epoch_count": len(train_indices),
        "test_epoch_count": len(test_indices),
        **metrics,
    }
    directory = output_directory / "folds" / f"fold_{fold:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    table = dataset.epochs.iloc[test_indices].copy().reset_index(drop=True)
    table["y_true"] = dataset.labels[test_indices]
    table["y_pred"] = predictions
    table["decision_score"] = decision_scores
    table["correct"] = table["y_true"].eq(table["y_pred"])
    table.to_csv(directory / "test_predictions.csv", index=False, encoding="utf-8-sig")
    matrix = confusion_matrix(dataset.labels[test_indices], predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=pd.Index(["HC", "ADHD"], name="true_label"), columns=["HC", "ADHD"]).to_csv(directory / "confusion_matrix.csv", encoding="utf-8-sig")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(dataset.labels[test_indices], decision_scores, pos_label=1)
    pd.DataFrame({"false_positive_rate": false_positive_rate, "true_positive_rate": true_positive_rate, "decision_threshold": thresholds}).to_csv(directory / "roc_curve.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "metrics.json", record)
    write_json(directory / "split.json", {
        "fold": fold,
        "source_fold": str(arguments.feature_directory / "folds" / f"fold_{fold:02d}"),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "train_sample_indices": train_indices.tolist(),
        "test_sample_indices": test_indices.tolist(),
        "subject_overlap": False,
        "sample_unit": "epoch",
        "feature_family": "spectral_entropy",
    })
    save_model_parameters(model, dataset.feature_names, directory)
    if arguments.save_models:
        import joblib

        joblib.dump(model, directory / "model.joblib")
    return record, table


def save_final_results(
    fold_records: Sequence[dict[str, Any]],
    predictions: Sequence[pd.DataFrame],
    dataset: FeatureDataset,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> pd.DataFrame:
    fold_metrics = pd.DataFrame(fold_records).sort_values("fold").reset_index(drop=True)
    statistics = summarize_fold_metrics(fold_metrics, ddof=arguments.std_ddof)
    held_out = pd.concat(predictions, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
    identity_column = "epoch_id"
    if not np.array_equal(held_out["sample_index"].to_numpy(), np.arange(len(dataset.labels))) or held_out[identity_column].duplicated().any():
        raise RuntimeError("Every epoch must have exactly one held-out prediction")
    if not np.array_equal(held_out["subject"].to_numpy(), dataset.subjects) or not np.array_equal(held_out["state"].to_numpy(), dataset.states):
        raise RuntimeError("Held-out predictions do not preserve the original subjects and states")
    if not np.array_equal(held_out["epoch_id"].to_numpy(), dataset.epoch_ids):
        raise RuntimeError("Held-out predictions do not preserve the original epoch identifiers")
    if not np.array_equal(held_out["y_true"].to_numpy(), dataset.labels) or not np.array_equal(held_out["fold"].to_numpy(), dataset.folds):
        raise RuntimeError("Held-out predictions do not match the original labels and folds")
    fold_metrics.to_csv(output_directory / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(output_directory / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_directory / "metrics_summary.json", statistics.to_dict(orient="records"))
    held_out.to_csv(output_directory / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    prediction_arrays = {
        "y_true": held_out["y_true"].to_numpy(dtype=np.int64),
        "y_pred": held_out["y_pred"].to_numpy(dtype=np.int64),
        "decision_scores": held_out["decision_score"].to_numpy(dtype=np.float64),
        "subjects": dataset.subjects.astype(str),
        "folds": dataset.folds.astype(np.int64),
        "sample_indices": np.arange(len(dataset.labels), dtype=np.int64),
        "feature_names": np.asarray(dataset.feature_names, dtype=str),
    }
    prediction_arrays["epoch_ids"] = dataset.epoch_ids.astype(str)
    np.savez_compressed(output_directory / "out_of_fold_predictions.npz", **prediction_arrays)
    lookup = statistics.set_index("metric")
    bands = dataset.configuration.get("bands", dataset.feature_names)
    paper_row = {
        "feature_family": "spectral_entropy",
        "sample_unit": "epoch",
        "graph": dataset.configuration.get("graph"),
        "condition": dataset.configuration.get("condition"),
        "bands": bands if isinstance(bands, str) else "+".join(bands),
        "fc_method": dataset.configuration.get("fc_method"),
        "classifier": "linear SVM",
        **{f"{metric} (%)": lookup.loc[metric, "formatted_mean_std_percent"] for metric in METRIC_NAMES},
    }
    pd.DataFrame([paper_row]).to_csv(output_directory / "paper_results.csv", index=False, encoding="utf-8-sig")
    write_json(output_directory / "summary.json", {
        "status": "complete",
        "classifier": "linear SVM",
        "n_splits": 10,
        "subject_count": len(dataset.subject_folds),
        "epoch_count": len(dataset.labels),
        "feature_family": "spectral_entropy",
        "feature_names": dataset.feature_names,
        "split_seed": int(dataset.configuration["seed"]),
        "unit_of_prediction": "epoch",
        "sample_unit": "epoch",
        "subject_disjoint_folds": True,
        "each_epoch_tested_once": True,
        "metric_summary": statistics.to_dict(orient="records"),
        "aggregation": "each test fold contributes equally to the reported mean, standard deviation and variance",
        "std_ddof": arguments.std_ddof,
        "formatted_spread": "standard deviation, not variance",
        "outputs": {
            "per_fold_metrics": "fold_metrics.csv",
            "statistics": "metrics_summary.csv",
            "paper_table": "paper_results.csv",
            "held_out_predictions": "out_of_fold_predictions.csv",
            "per_fold_details": "folds/fold_01 through fold_10",
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return statistics


def main() -> int:
    arguments = parse_arguments()
    output_directory = None
    active_fold = None
    try:
        print(f"Validating saved features and ten-fold partitions: {arguments.feature_directory}", flush=True)
        dataset = load_feature_dataset(arguments.feature_directory)
        configuration = classification_configuration(arguments, dataset)
        output_directory = prepare_output_directory(arguments, configuration)
        dataset.subject_folds.to_csv(output_directory / "subject_folds.csv", index=False, encoding="utf-8-sig")
        write_json(output_directory / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
        fold_records = []
        predictions = []
        for fold in range(1, 11):
            active_fold = fold
            record, held_out = classify_fold(fold, dataset, arguments, output_directory)
            fold_records.append(record)
            predictions.append(held_out)
            pd.DataFrame(fold_records).to_csv(output_directory / "fold_metrics.csv", index=False, encoding="utf-8-sig")
            print(f"Fold {fold:02d}/10: BAcc={100 * record['BAcc']:.2f}%, AUC={100 * record['AUC']:.2f}%", flush=True)
        statistics = save_final_results(fold_records, predictions, dataset, arguments, output_directory)
        write_json(output_directory / "run_status.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
        for row in statistics.itertuples(index=False):
            print(f"{row.metric}: {row.formatted_mean_std_percent}%", flush=True)
        print(f"Classification results saved to {output_directory}", flush=True)
        return 0
    except Exception as error:
        if output_directory is not None:
            write_json(output_directory / "run_status.json", {
                "status": "failed", "fold": active_fold,
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
