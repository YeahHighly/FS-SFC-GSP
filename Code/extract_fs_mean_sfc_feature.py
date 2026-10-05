from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DEFAULT_SFC_DIRECTORY = PROJECT_DIRECTORY / "SFC"
DEFAULT_FEATURE_DIRECTORY = PROJECT_DIRECTORY / "SE"
DEFAULT_LABELS_FILE = PROJECT_DIRECTORY / "Data" / "xxxxx.csv"
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FEATURE_SCHEMA_VERSION = 1
DEFAULT_RANDOM_SEED = 1024


def _mean_read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _mean_integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    number = float(value)
    if not math.isfinite(number) or number < minimum or abs(number - round(number)) > 1e-8:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(round(number))


def load_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path, sep="\t", encoding="utf-8-sig", keep_default_na=False)
    if not {"index", "name"}.issubset(nodes.columns) or len(nodes) != 90:
        raise ValueError(f"Node table must contain 90 rows with index and name columns: {path}")
    indices = [_mean_integer(value, "Node index", 1) for value in nodes["index"]]
    if indices != list(range(1, 91)):
        raise ValueError(f"Node table rows must be ordered by consecutive indices 1-90: {path}")
    names = nodes["name"].astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError(f"Node names must be nonempty and unique: {path}")
    nodes = nodes.copy()
    nodes["index"] = indices
    nodes["name"] = names
    if "original_value" in nodes.columns:
        values = [_mean_integer(value, "Original atlas label", 1) for value in nodes["original_value"]]
        if len(set(values)) != 90:
            raise ValueError(f"Original atlas labels must be unique: {path}")
        nodes["original_value"] = values
    return nodes.reset_index(drop=True)


def _mean_validate_node_alignment(first: pd.DataFrame, second: pd.DataFrame) -> None:
    columns = ["index", "name"]
    if "original_value" in first.columns and "original_value" in second.columns:
        columns.append("original_value")
    if len(first) != 90 or len(second) != 90:
        raise ValueError("SC and FC require exactly 90 ROI nodes")
    for column in columns:
        if column not in first.columns or column not in second.columns or first[column].tolist() != second[column].tolist():
            raise ValueError(f"SC/FC node ordering differs in column {column}")


def load_subject_metadata(fc_directory: Path, subject: str) -> dict:
    summary = _mean_read_json(fc_directory / "summary.json")
    if summary.get("processing_status") != "complete" or summary.get("subject") != subject:
        raise ValueError("FC summary must report complete processing for the requested subject")
    if _mean_integer(summary.get("roi_count", -1), "FC ROI count", 1) != 90:
        raise ValueError("FC summary must describe 90 ROIs")
    origin = str(summary.get("time_series_space", "")).casefold()
    if "source" not in origin or "mri" not in origin or summary.get("sensor_space_connectivity") is not False:
        raise ValueError("FC inputs must be individual-MRI source-space ROI signals, not sensor-space connectivity")
    if summary.get("fc_schema_version") != 2:
        raise ValueError(f"FC outputs must include completed full matrices and their normalisation metadata. Rebuild functional connectivity for {subject}.")
    epoch_seconds = float(summary.get("epoch_seconds", float("nan")))
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0.0:
        raise ValueError(f"FC summary has an invalid epoch duration for {subject}")
    return summary


def load_structural_matrix(sc_directory: Path, subject: str, nodes: pd.DataFrame) -> np.ndarray:
    summary = _mean_read_json(sc_directory / "summary.json")
    if summary.get("processing_status") != "complete" or summary.get("subject") != subject:
        raise ValueError("SC summary must report complete processing for the requested subject")
    if _mean_integer(summary.get("atlas_nodes", -1), "SC atlas node count", 1) != 90:
        raise ValueError("SC summary must describe 90 atlas nodes")
    _mean_validate_node_alignment(load_nodes(sc_directory / "nodes.tsv"), nodes)
    matrix_path = sc_directory / "connectome_sift2.npy"
    if matrix_path.is_file():
        matrix = np.load(matrix_path, allow_pickle=False)
        if not isinstance(matrix, np.ndarray) or matrix.dtype.kind not in "fiu":
            raise ValueError("SC weights must be a real numeric array")
        matrix = np.asarray(matrix, dtype=np.float64)
    else:
        matrix = np.loadtxt(sc_directory / "connectome_sift2.csv", delimiter=",", dtype=np.float64)
    if matrix.shape != (90, 90) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("SC weights must be finite, nonnegative, and shaped (90, 90)")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
        raise ValueError("SC weight matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-8):
        raise ValueError("SC weight matrix must have a zero diagonal")
    return matrix


def _validated_nonnegative_connectivity(matrix: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(matrix):
        raise ValueError("Connectivity weights must be real")
    values = np.array(matrix, dtype=np.float64, copy=True)
    if values.shape != (90, 90) or not np.isfinite(values).all():
        raise ValueError("Connectivity must be a finite 90 by 90 matrix")
    tolerance = max(1e-12, float(np.max(np.abs(values))) * 1e-10)
    if np.any(values < -tolerance):
        raise ValueError("Connectivity contains negative weights beyond numerical tolerance")
    if not np.allclose(values, values.T, rtol=1e-7, atol=tolerance):
        raise ValueError("Connectivity is not symmetric within numerical tolerance")
    values = values * 0.5 + values.T * 0.5
    np.maximum(values, 0.0, out=values)
    np.fill_diagonal(values, 0.0)
    if not np.any(values > 0.0):
        raise ValueError("Connectivity has no positive off-diagonal edges")
    return values


def normalize_structural_connectivity(matrix: np.ndarray) -> np.ndarray:
    values = _validated_nonnegative_connectivity(matrix)
    values = np.log1p(values)
    maximum = float(values.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Connectivity has no valid positive normalization factor")
    normalized = values / maximum
    np.fill_diagonal(normalized, 0.0)
    return normalized


def normalize_subject_id(value: str) -> str:
    label = str(value).strip()
    if label.startswith("sub-"):
        label = label[4:]
    if not label or not label.isascii() or not label.isalnum():
        raise ValueError("Subject IDs must contain only ASCII letters and digits, optionally prefixed by sub-.")
    return f"sub-{label}"


def _fold_canonical_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing label columns: {sorted(required.difference(frame.columns))}")
    if frame.empty or frame["subject"].isna().any() or frame["label"].isna().any():
        raise ValueError("Labels must contain nonempty subjects and labels.")
    result = frame.copy()
    result["subject"] = result["subject"].map(normalize_subject_id)
    if result["subject"].duplicated().any():
        raise ValueError("Each subject must occur exactly once in the label table.")
    values = result["label"].astype(str).str.strip().str.upper()
    encoded = values.map({"0": 0, "HC": 0, "1": 1, "ADHD": 1})
    if encoded.isna().any():
        raise ValueError("Labels must be HC/ADHD or 0/1, with 0=HC and 1=ADHD.")
    result["label"] = encoded.astype(np.int64)
    expected_names = result["label"].map({0: "HC", 1: "ADHD"})
    if "label_name" in result.columns:
        names = result["label_name"].astype(str).str.strip().str.upper()
        if not names.eq(expected_names).all():
            raise ValueError("label_name disagrees with its numeric label.")
    result["label_name"] = expected_names
    columns = ["subject", "label", "label_name"]
    columns.extend(name for name in ("age", "sex") if name in result.columns)
    return result[columns].sort_values("subject").reset_index(drop=True)


def load_labels(
    path: Path,
    subject_column: str = "subject_id",
    label_column: str = "label",
    subjects: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".tsv"):
        raise ValueError("The label file must be CSV or TSV.")
    if subject_column == label_column:
        raise ValueError("Subject and label columns must be different.")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), None)
    if not header or len(header) != len(set(header)):
        raise ValueError("The label file must have unique column names.")
    frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {subject_column, label_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing label-file columns: {sorted(missing)}")
    selected = pd.DataFrame({"subject": frame[subject_column], "label": frame[label_column]})
    for name in ("age", "sex"):
        if name in frame.columns and name not in (subject_column, label_column):
            selected[name] = frame[name]
    labels = _fold_canonical_labels(selected)
    if subjects is not None:
        requested = [normalize_subject_id(subject) for subject in subjects]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("The requested subject list must be nonempty and unique.")
        missing_subjects = set(requested).difference(labels["subject"])
        if missing_subjects:
            raise ValueError(f"Missing labels for {len(missing_subjects)} requested subjects.")
        labels = labels.loc[labels["subject"].isin(requested)].reset_index(drop=True)
    return labels


def _fold_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fold_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fold_validate_assignments(assignments: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "fold" not in assignments.columns:
        raise ValueError("The subject-fold manifest requires a fold column.")
    canonical = _fold_canonical_labels(assignments)
    expected = _fold_canonical_labels(labels)[["subject", "label", "label_name"]]
    if not canonical[["subject", "label", "label_name"]].equals(expected):
        raise ValueError("The subject-fold manifest does not match the complete labeled cohort.")
    folds = pd.to_numeric(assignments["fold"], errors="raise")
    if not folds.isin(range(1, 11)).all():
        raise ValueError("Fold indices must be integers from 1 through 10.")
    by_subject = dict(zip(assignments["subject"].map(normalize_subject_id), folds.astype(np.int64)))
    canonical = canonical[["subject", "label", "label_name"]].copy()
    canonical["fold"] = canonical["subject"].map(by_subject).astype(np.int64)
    if set(canonical["fold"]) != set(range(1, 11)):
        raise ValueError("The manifest must contain exactly ten nonempty folds.")
    for fold in range(1, 11):
        if set(canonical.loc[canonical["fold"].eq(fold), "label"]) != {0, 1}:
            raise ValueError(f"Test fold {fold} must contain both HC and ADHD subjects.")
    return canonical


def ensure_subject_folds(
    labels: pd.DataFrame,
    split_directory: Path,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("The split seed must be an integer between 0 and 2**32 - 1.")
    seed = int(seed)
    canonical = _fold_canonical_labels(labels)[["subject", "label", "label_name"]]
    counts = canonical["label"].value_counts()
    if set(counts.index) != {0, 1} or any(int(counts.get(label, 0)) < 10 for label in (0, 1)):
        raise ValueError("Ten-fold stratification requires at least ten HC and ten ADHD subjects.")
    cohort = {
        "seed": seed,
        "n_splits": 10,
        "subjects": canonical[["subject", "label"]].to_dict(orient="records"),
    }
    cohort_hash = hashlib.sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    split_directory = Path(split_directory)
    split_directory.mkdir(parents=True, exist_ok=True)
    csv_path = split_directory / f"subject_folds_{cohort_hash}.csv"
    json_path = split_directory / f"subject_folds_{cohort_hash}.json"
    if csv_path.exists() != json_path.exists():
        raise RuntimeError("The persistent subject-fold manifest is incomplete; restore its CSV and JSON together.")
    if csv_path.exists():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("cohort_hash") != cohort_hash or metadata.get("seed") != seed or metadata.get("n_splits") != 10:
            raise ValueError("The persistent fold metadata does not match this cohort and seed.")
        if metadata.get("csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
            raise ValueError("The persistent subject-fold CSV does not match its saved checksum.")
        assignments = pd.read_csv(csv_path, dtype={"subject": str, "label_name": str}, keep_default_na=False)
        assignments = _fold_validate_assignments(assignments, canonical)
    else:
        assignments = canonical.copy()
        fold_indices = np.zeros(len(canonical), dtype=np.int64)
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for fold, (_, test_indices) in enumerate(splitter.split(np.zeros((len(canonical), 1)), canonical["label"].to_numpy()), start=1):
            if np.any(fold_indices[test_indices] != 0):
                raise RuntimeError("A subject was assigned to more than one test fold.")
            fold_indices[test_indices] = fold
        assignments["fold"] = fold_indices
        assignments = _fold_validate_assignments(assignments, canonical)
        _fold_write_csv(assignments, csv_path)
        metadata = {
            "cohort_hash": cohort_hash,
            "seed": seed,
            "n_splits": 10,
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "shuffle": True,
            "stratification_unit": "subject",
            "label_mapping": {"HC": 0, "ADHD": 1},
            "subject_count": len(assignments),
            "class_counts": {"HC": int(counts[0]), "ADHD": int(counts[1])},
            "csv_file": csv_path.name,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "cohort": cohort["subjects"],
        }
        _fold_write_json(metadata, json_path)
    assignments.attrs["fold_manifest"] = {"csv": str(csv_path), "json": str(json_path)}
    assignments.attrs["fold_manifest_csv"] = str(csv_path)
    assignments.attrs["fold_manifest_json"] = str(json_path)
    assignments.attrs["cohort_hash"] = cohort_hash
    assignments.attrs["seed"] = seed
    return assignments


def _fold_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Fold NPZ files cannot contain object arrays.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mean_validate_export(
    features: np.ndarray,
    samples: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    required_samples = {"sample_index", "subject", "label", "label_name", "fold", "state"}
    required_assignments = {"subject", "label", "label_name", "fold"}
    if not isinstance(samples, pd.DataFrame) or samples.empty or samples.columns.duplicated().any() or not required_samples.issubset(samples.columns):
        raise ValueError("Subject samples require unique columns and complete sample, label, state, and fold metadata")
    if not isinstance(assignments, pd.DataFrame) or assignments.empty or assignments.columns.duplicated().any() or not required_assignments.issubset(assignments.columns):
        raise ValueError("A complete existing subject-fold assignment table is required")
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name or name != name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Feature names must be unique nonempty strings")
    if set(names).intersection(samples.columns):
        raise ValueError("Feature names must not collide with sample metadata columns")
    original = np.asarray(features)
    if original.dtype.kind not in "fiu" or original.ndim != 2 or original.shape != (len(samples), len(names)):
        raise ValueError("Features must be a real matrix with one row per subject and one column per feature")
    values = np.asarray(original, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Mean SFC features must be finite")
    metadata = samples.copy().reset_index(drop=True)
    manifest = dict(assignments.attrs)
    canonical = assignments.loc[:, ["subject", "label", "label_name", "fold"]].copy().reset_index(drop=True)
    for table, description in ((metadata, "Samples"), (canonical, "Subject assignments")):
        if table[["subject", "label", "label_name", "fold"]].isna().any().any():
            raise ValueError(f"{description} contain missing required values")
        subjects = table["subject"].to_numpy()
        if any(not isinstance(subject, str) or not subject or subject != subject.strip() for subject in subjects) or table["subject"].duplicated().any():
            raise ValueError(f"{description} must contain exactly one row per nonempty subject identifier")
        for column, allowed in (("label", (0, 1)), ("fold", tuple(range(1, 11)))):
            numeric = pd.to_numeric(table[column], errors="raise")
            if numeric.dtype.kind not in "fiu" or not numeric.isin(allowed).all():
                raise ValueError(f"{description} contain invalid {column} values")
            table[column] = numeric.astype(np.int64)
        expected_names = table["label"].map({0: "HC", 1: "ADHD"})
        if not table["label_name"].eq(expected_names).all():
            raise ValueError(f"{description} must use labels HC=0 and ADHD=1 consistently")
    if set(metadata["subject"]) != set(canonical["subject"]):
        raise ValueError("Samples must contain the entire assigned cohort exactly once without excluding subjects")
    lookup = canonical.set_index("subject")
    for column in ("label", "fold"):
        if not np.array_equal(metadata[column].to_numpy(), lookup.loc[metadata["subject"], column].to_numpy()):
            raise ValueError(f"Sample {column} values disagree with the existing subject assignments")
    indices = pd.to_numeric(metadata["sample_index"], errors="raise").to_numpy()
    if indices.dtype.kind not in "fiu" or not np.array_equal(indices, np.arange(len(metadata))):
        raise ValueError("Sample indices must be consecutive zero-based feature row positions")
    metadata["sample_index"] = np.arange(len(metadata), dtype=np.int64)
    if metadata["state"].isna().any() or not metadata["state"].isin(("EO", "EC", "ALL")).all() or metadata["state"].nunique() != 1:
        raise ValueError("A subject dataset must contain one common EO, EC, or ALL condition")
    canonical = canonical.sort_values("subject").reset_index(drop=True)
    for fold in range(1, 11):
        for mask in (canonical["fold"].eq(fold), canonical["fold"].ne(fold)):
            if set(canonical.loc[mask, "label"]) != {0, 1}:
                raise ValueError(f"Saved fold {fold} must contain both HC and ADHD subjects in train and test partitions")
    seed = manifest.get("seed")
    cohort_hash = manifest.get("cohort_hash")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Subject assignments must retain their saved split seed")
    if not isinstance(cohort_hash, str) or len(cohort_hash) != 64 or any(character not in "0123456789abcdef" for character in cohort_hash):
        raise ValueError("Subject assignments must retain their saved cohort hash")
    if not isinstance(manifest.get("fold_manifest"), dict) or set(manifest["fold_manifest"]) != {"csv", "json"}:
        raise ValueError("Subject assignments must retain their shared fold-manifest metadata")
    canonical.attrs = manifest
    return values, metadata, canonical, names


def save_feature_dataset(
    features: np.ndarray,
    samples: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
    output_directory: Path,
) -> None:
    values, metadata, canonical, names = _mean_validate_export(features, samples, assignments, feature_names)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _fold_write_csv(metadata, output_directory / "samples.csv")
    _fold_write_csv(canonical, output_directory / "subject_folds.csv")
    _fold_write_csv(pd.concat([metadata, pd.DataFrame(values, columns=names)], axis=1), output_directory / "features.csv")
    np.save(output_directory / "features.npy", values, allow_pickle=False)
    arrays = {
        "X": values,
        "y": metadata["label"].to_numpy(dtype=np.int64),
        "subjects": metadata["subject"].to_numpy(dtype=np.str_),
        "states": metadata["state"].to_numpy(dtype=np.str_),
        "folds": metadata["fold"].to_numpy(dtype=np.int64),
        "sample_indices": metadata["sample_index"].to_numpy(dtype=np.int64),
        "feature_names": np.asarray(names, dtype=np.str_),
    }
    _fold_write_npz(output_directory / "features.npz", arrays)
    write_json(output_directory / "feature_schema.json", {
        "feature_family": "mean_sfc",
        "feature_names": names,
        "sample_unit": "subject",
        "sample_count": len(metadata),
        "subject_count": len(canonical),
        "X_shape": list(values.shape),
        "X_dtype": "float64",
        "description": "One row of frequency-specific mean SFC features per subject",
        "feature_scaling": "none; fit any later scaler on classification training folds only",
        "label_mapping": {"HC": 0, "ADHD": 1},
        "npz_keys": list(arrays),
        "loading": "np.load(path, allow_pickle=False)",
        "sample_metadata": "samples.csv",
        "subject_folds": "subject_folds.csv",
    })


def export_fold_features(
    features: np.ndarray,
    samples: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
    output_directory: Path,
) -> None:
    values, metadata, canonical, names = _mean_validate_export(features, samples, assignments, feature_names)
    manifest = dict(canonical.attrs)
    global_indices = np.arange(len(metadata), dtype=np.int64)
    folds = metadata["fold"].to_numpy(dtype=np.int64)
    held_out_counts = np.zeros(len(metadata), dtype=np.int64)
    plans = []
    for fold in range(1, 11):
        train_indices, test_indices = global_indices[folds != fold], global_indices[folds == fold]
        if not len(train_indices) or not len(test_indices) or set(metadata.iloc[train_indices]["subject"]).intersection(metadata.iloc[test_indices]["subject"]):
            raise ValueError(f"Saved fold {fold} has an empty partition or subject leakage")
        held_out_counts[test_indices] += 1
        plans.append((fold, train_indices, test_indices))
    if not np.all(held_out_counts == 1):
        raise ValueError("Every subject sample must appear in exactly one held-out test partition")
    for fold, train_indices, test_indices in plans:
        directory = Path(output_directory) / "folds" / f"fold_{fold:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        partition_counts = {}
        for partition, indices in (("train", train_indices), ("test", test_indices)):
            partition_samples = metadata.iloc[indices].copy()
            partition_subjects = canonical.loc[canonical["subject"].isin(partition_samples["subject"])].copy()
            _fold_write_npz(directory / f"{partition}.npz", {
                "X": values[indices],
                "y": partition_samples["label"].to_numpy(dtype=np.int64),
                "sample_indices": indices,
                "subjects": partition_samples["subject"].to_numpy(dtype=np.str_),
                "feature_names": np.asarray(names, dtype=np.str_),
            })
            _fold_write_csv(partition_samples, directory / f"{partition}_samples.csv")
            _fold_write_csv(partition_subjects, directory / f"{partition}_subjects.csv")
            partition_counts[partition] = {
                "sample_count": len(partition_samples),
                "subject_count": len(partition_subjects),
                "class_subject_counts": {"HC": int(partition_subjects["label"].eq(0).sum()), "ADHD": int(partition_subjects["label"].eq(1).sum())},
                "subjects": partition_subjects["subject"].tolist(),
            }
        write_json(directory / "split.json", {
            "fold": fold,
            "n_splits": 10,
            "split_unit": "subject",
            "sample_unit": "subject",
            "feature_family": "mean_sfc",
            "feature_count": len(names),
            "label_mapping": {"HC": 0, "ADHD": 1},
            "cohort_hash": manifest["cohort_hash"],
            "seed": int(manifest["seed"]),
            "fold_manifest": manifest["fold_manifest"],
            "feature_scaling": "none",
            "subject_overlap": False,
            "held_out_once": True,
            "partitions": partition_counts,
        })


def saved_matrix_path(directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.npy"
    if not path.is_file():
        path = directory / f"{stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved connectivity matrix: {directory / stem}")
    return path


def load_saved_functional(directory: Path, state: str, band: str, method: str, epoch_seconds: float) -> tuple[np.ndarray, dict]:
    metadata = _mean_read_json(directory / "metadata.json")
    if metadata.get("fc_schema_version") != 2 or metadata.get("fc_estimation_unit") != "epoch":
        raise ValueError(f"Functional connectivity must be computed upstream with the current FC schema: {directory}")
    if metadata.get("condition") != state or metadata.get("frequency_band") != band:
        raise ValueError(f"Saved FC condition or frequency band disagrees with the selected input: {directory}")
    frequencies = np.asarray(metadata.get("frequency_range_hz", []), dtype=float)
    if frequencies.shape != (2,) or not np.allclose(frequencies, BANDS[band], rtol=0.0, atol=1e-8):
        raise ValueError(f"Saved FC frequency bounds disagree with the selected band: {directory}")
    if metadata.get("roi_count") != 90 or metadata.get("roi_order") != "nodes.tsv indices 1-90":
        raise ValueError(f"Saved FC must use the shared AAL90 node order: {directory}")
    origin = str(metadata.get("signal_origin", "")).casefold()
    if "mri" not in origin or "source" not in origin:
        raise ValueError(f"Saved FC must describe individual-MRI source-space ROI connectivity: {directory}")
    if "topk" in metadata or metadata.get("primary_connectivity") != "full nonnegative connectivity matrix" or metadata.get("matrix_diagonal") != "zero":
        raise ValueError(f"Mean SFC requires full upstream matrices before sparsification: {directory}")
    duration = float(metadata.get("epoch_seconds", float("nan")))
    if not math.isfinite(duration) or not math.isclose(duration, epoch_seconds, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Band and subject FC estimation durations disagree: {directory}")
    _mean_integer(metadata.get("epoch_count", -1), "Upstream FC estimation epoch count", 1)
    methods = metadata.get("connectivity_methods")
    outputs = metadata.get("connectivity_outputs")
    if not isinstance(methods, list) or method not in methods or not isinstance(outputs, dict) or method not in outputs:
        raise ValueError(f"Saved {method} connectivity is unavailable: {directory}")
    record = outputs[method]
    if not isinstance(record, dict) or record.get("normalisation") != "divide by maximum positive off-diagonal raw weight; all-zero matrices remain zero":
        raise ValueError(f"Unexpected upstream FC normalisation: {directory}")
    matrices = []
    for suffix, npy_key, csv_key in (("", "connectivity_npy", "connectivity_csv"), ("_normalised", "normalised_npy", "normalised_csv")):
        stem = f"connectivity_{method}{suffix}"
        if record.get(npy_key) != f"{stem}.npy" or record.get(csv_key) != f"{stem}.csv":
            raise ValueError(f"Unexpected upstream FC file mapping: {directory}")
        path = saved_matrix_path(directory, stem)
        matrix = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
        if not isinstance(matrix, np.ndarray) or matrix.dtype.kind not in "fiu":
            raise ValueError(f"FC weights must be real numeric values: {path}")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (90, 90) or not np.isfinite(matrix).all() or np.any(matrix < 0.0) or np.any(matrix > 1.0 + 1e-7):
            raise ValueError(f"FC weights must form a finite 90 by 90 matrix in [0, 1]: {path}")
        if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10) or not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"FC weights must be symmetric with a zero diagonal: {path}")
        matrices.append(matrix)
    raw, normalised = matrices
    maximum = float(record.get("normalisation_maximum", float("nan")))
    if not math.isfinite(maximum) or maximum < 0.0 or not np.isclose(maximum, float(raw.max()), rtol=1e-7, atol=1e-12):
        raise ValueError(f"The FC normalisation factor does not match its raw matrix: {directory}")
    if maximum == 0.0:
        if record.get("all_zero") is not True or np.any(raw != 0.0) or np.any(normalised != 0.0):
            raise ValueError(f"Inconsistent all-zero FC matrix or metadata: {directory}")
    elif record.get("all_zero") is not False or not np.isclose(float(normalised.max()), 1.0, rtol=1e-7, atol=1e-10) or not np.allclose(normalised * maximum, raw, rtol=1e-7, atol=1e-12):
        raise ValueError(f"Normalised FC does not match the saved raw matrix: {directory}")
    return normalised, metadata


def resolve_upstream_configuration(arguments: argparse.Namespace, summaries: Sequence[dict[str, Any]]) -> None:
    if not summaries:
        raise ValueError("At least one completed FC summary is required")
    durations = [float(summary["epoch_seconds"]) for summary in summaries]
    if any(not math.isclose(duration, durations[0], rel_tol=0.0, abs_tol=1e-9) for duration in durations):
        raise ValueError("All subjects must use the same upstream FC estimation epoch duration")
    arguments.upstream_epoch_seconds = durations[0]
    primary_methods = set()
    for summary in summaries:
        methods = summary.get("connectivity_methods")
        primary = summary.get("primary_method", summary.get("primary_connectivity"))
        if not isinstance(methods, list) or not methods or any(method not in ("pearson", "wpli") for method in methods) or primary not in methods:
            raise ValueError("FC summaries must identify valid saved methods and their primary method")
        primary_methods.add(primary)
    if arguments.fc_method == "auto":
        if len(primary_methods) != 1:
            raise ValueError("Subjects have different primary FC methods; select one saved method with --fc-method")
        arguments.fc_method = primary_methods.pop()
    if any(arguments.fc_method not in summary["connectivity_methods"] for summary in summaries):
        raise ValueError(f"Every subject must have saved {arguments.fc_method} connectivity")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the subject-level mean SFC baseline from the full frequency-specific matrices before sparsification, as defined in Table 2.")
    parser.add_argument("--bands", nargs="+", type=str.lower, choices=(*BANDS, "all"), default=["all"])
    parser.add_argument("--condition", type=str.upper, choices=("EC", "EO", "ALL"), default="EC", help="EC matches Table 2; ALL reads the FC builder's saved pooled EO+EC matrix.")
    parser.add_argument("--fc-method", "--method", dest="fc_method", choices=("auto", "pearson", "wpli"), default="auto", help="Select a saved upstream matrix; auto inherits the common primary FC method.")
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subject", nargs="+", type=normalize_subject_id)
    parser.add_argument("--sfc-dir", type=Path, default=DEFAULT_SFC_DIRECTORY)
    parser.add_argument("--output-dir", "--se-dir", dest="output_dir", type=Path, default=DEFAULT_FEATURE_DIRECTORY)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FEATURE_DIRECTORY / "_folds", help="Shared subject-fold manifests used by the SE feature extractor.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if "all" in arguments.bands:
        if len(arguments.bands) != 1:
            parser.error("Use ALL alone or specify individual frequency bands.")
        arguments.bands = list(BANDS)
    else:
        arguments.bands = [band for band in BANDS if band in arguments.bands]
    if not 0 <= arguments.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if not arguments.subject_column.strip() or not arguments.label_column.strip():
        parser.error("Subject and label column names must be nonempty.")
    for name in ("labels_file", "sfc_dir", "output_dir", "fold_dir"):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    if arguments.subject:
        arguments.subject = list(dict.fromkeys(arguments.subject))
    for destination in (arguments.output_dir, arguments.fold_dir):
        if destination.is_relative_to(arguments.sfc_dir):
            parser.error("Feature outputs and fold manifests must be outside the input SFC directory.")
    return arguments


def input_fingerprints(arguments: argparse.Namespace, labels: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for subject in labels["subject"]:
        root = arguments.sfc_dir / subject
        structural = root / "structural_connectivity"
        functional = root / "functional_connectivity"
        paths = [structural / "summary.json", structural / "nodes.tsv", saved_matrix_path(structural, "connectome_sift2"), functional / "summary.json", functional / "nodes.tsv"]
        for band in arguments.bands:
            directory = functional / arguments.condition / band
            paths.extend((directory / "metadata.json", saved_matrix_path(directory, f"connectivity_{arguments.fc_method}"), saved_matrix_path(directory, f"connectivity_{arguments.fc_method}_normalised")))
        for path in paths:
            records.append({"path": path.relative_to(arguments.sfc_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def configuration_record(arguments: argparse.Namespace, labels: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_family": "mean_sfc",
        "sample_unit": "subject",
        "graph": "SFC",
        "bands": arguments.bands,
        "frequency_bands_hz": {band: list(BANDS[band]) for band in arguments.bands},
        "condition": arguments.condition,
        "fc_method": arguments.fc_method,
        "fc_source": "saved upstream full normalised matrices",
        "upstream_fc_epoch_seconds": arguments.upstream_epoch_seconds,
        "structural_normalisation": "log1p of raw SIFT2 weights divided by the maximum off-diagonal log1p weight",
        "functional_normalisation": "load upstream weights already divided by their maximum off-diagonal raw FC weight; all-zero FC remains zero",
        "sfc_fusion": "elementwise product of normalised SC and FC, without further rescaling",
        "feature_formula": "mean_sfc = sum(SC_normalised * FC_normalised) / (90 * 90)",
        "averaging_scope": "all 8100 entries, including both triangles, the zero diagonal, and all zero-valued entries",
        "sparsification": "none; extract the matrix mean before any edge selection",
        "all_zero_sfc_policy": "retain the valid subject with feature value zero and record QC",
        "combined_condition_policy": "ALL reads the independently saved FC matrix estimated upstream from pooled complete EO and EC epochs",
        "band_selection_policy": "ALL concatenates five band-specific matrix means into one vector per subject",
        "n_splits": 10,
        "seed": arguments.seed,
        "cross_validation": "ten subject-stratified folds shared with the SE extractor; one sample per subject",
        "feature_scaling": "none; fit any later scaler on classification training folds only",
        "subject_fold_manifest": dict(assignments.attrs),
        "sfc_directory": str(arguments.sfc_dir),
        "input_files": input_fingerprints(arguments, labels),
        "labels_file": str(arguments.labels_file),
        "labels_file_sha256": hashlib.sha256(arguments.labels_file.read_bytes()).hexdigest(),
        "subject_column": arguments.subject_column,
        "label_column": arguments.label_column,
        "label_mapping": {"HC": 0, "ADHD": 1},
        "cohort": labels[["subject", "label"]].to_dict(orient="records"),
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paper_reference": "Equation (6), Section 2.6 mean SFC baseline, and Table 2 (EC condition)",
    }


def prepare_configuration_directory(arguments: argparse.Namespace, configuration: dict[str, Any]) -> Path:
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    bands = "ALL" if arguments.bands == list(BANDS) else "-".join(arguments.bands)
    root = arguments.output_dir.resolve()
    directory = root / f"mean_SFC_fc-{arguments.fc_method}_bands-{bands}_state-{arguments.condition}_seed-{arguments.seed}_{signature}"
    resolved = directory.resolve()
    protected = (arguments.sfc_dir, arguments.labels_file, arguments.fold_dir)
    if directory.is_symlink() or resolved.parent != root or any(path.is_relative_to(resolved) for path in protected):
        raise ValueError("The output must be a dedicated configuration directory that cannot contain input data or shared fold manifests")
    if directory.exists() and any(directory.iterdir()):
        if not arguments.overwrite:
            raise RuntimeError(f"Output already exists: {directory}. Use --overwrite to replace this configuration.")
        shutil.rmtree(resolved)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "configuration.json", configuration)
    return directory


def extract_subject_features(subject: str, arguments: argparse.Namespace, output_directory: Path) -> tuple[np.ndarray, dict[str, Any]]:
    root = arguments.sfc_dir / subject
    fc_directory = root / "functional_connectivity"
    nodes = load_nodes(fc_directory / "nodes.tsv")
    structural = normalize_structural_connectivity(load_structural_matrix(root / "structural_connectivity", subject, nodes))
    directory = output_directory / "subjects" / subject
    directory.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(directory / "nodes.tsv", sep="\t", index=False)
    np.save(directory / "structural_normalised.npy", structural, allow_pickle=False)
    features = np.empty(len(arguments.bands), dtype=np.float64)
    band_reports = {}
    for index, band in enumerate(arguments.bands):
        input_directory = fc_directory / arguments.condition / band
        functional, metadata = load_saved_functional(input_directory, arguments.condition, band, arguments.fc_method, arguments.upstream_epoch_seconds)
        coupling = structural * functional
        features[index] = float(np.mean(coupling, dtype=np.float64))
        band_directory = directory / arguments.condition / band
        band_directory.mkdir(parents=True, exist_ok=True)
        np.save(band_directory / "functional_normalised.npy", functional, allow_pickle=False)
        np.save(band_directory / "sfc_full.npy", coupling, allow_pickle=False)
        np.savetxt(band_directory / "sfc_full.csv", coupling, delimiter=",", fmt="%.17g")
        report = {
            "subject": subject, "condition": arguments.condition, "band": band,
            "fc_method": arguments.fc_method, "mean_sfc": float(features[index]),
            "matrix_shape": [90, 90], "averaged_entry_count": int(coupling.size),
            "zero_entry_count": int(np.count_nonzero(coupling == 0.0)),
            "nonzero_entry_count": int(np.count_nonzero(coupling)),
            "matrix_sum": float(coupling.sum()), "matrix_maximum": float(coupling.max()),
            "all_zero_functional": bool(not np.any(functional)),
            "all_zero_sfc": bool(not np.any(coupling)),
            "sparsification": "none", "diagonal_included_in_mean": True,
            "input_directory": str(input_directory), "upstream_metadata": metadata,
        }
        write_json(band_directory / "qc.json", report)
        band_reports[band] = report
    if not np.isfinite(features).all() or np.any(features < 0.0) or np.any(features > 1.0 + 1e-7):
        raise ValueError(f"Mean SFC features are outside their valid finite range for {subject}")
    names = [f"mean_SFC_{band}" for band in arguments.bands]
    np.save(directory / "features.npy", features[np.newaxis, :], allow_pickle=False)
    pd.DataFrame([dict(zip(names, features))]).to_csv(directory / "features.csv", index=False)
    report = {"subject": subject, "sample_unit": "subject", "sample_count": 1, "feature_names": names, "condition": arguments.condition, "bands": band_reports}
    write_json(directory / "summary.json", report)
    return features, report


def main() -> int:
    arguments = parse_arguments()
    output_directory = None
    active_subject = None
    try:
        labels = load_labels(arguments.labels_file, arguments.subject_column, arguments.label_column, arguments.subject)
        summaries = []
        for subject in labels["subject"]:
            active_subject = subject
            directory = arguments.sfc_dir / subject / "functional_connectivity"
            summaries.append(load_subject_metadata(directory, subject))
            load_structural_matrix(arguments.sfc_dir / subject / "structural_connectivity", subject, load_nodes(directory / "nodes.tsv"))
        resolve_upstream_configuration(arguments, summaries)
        assignments = ensure_subject_folds(labels, arguments.fold_dir, arguments.seed)
        configuration = configuration_record(arguments, labels, assignments)
        output_directory = prepare_configuration_directory(arguments, configuration)
        write_json(output_directory / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
        rows = []
        records = []
        for position, subject in enumerate(labels["subject"], start=1):
            active_subject = subject
            print(f"[{position}/{len(labels)}] {subject}", flush=True)
            features, report = extract_subject_features(subject, arguments, output_directory)
            rows.append(features)
            records.append({"subject": subject, "sample_count": 1, "status": "complete", "all_zero_band_count": sum(int(value["all_zero_sfc"]) for value in report["bands"].values())})
            pd.DataFrame(records).to_csv(output_directory / "subject_processing.csv", index=False)
        features = np.stack(rows).astype(np.float64, copy=False)
        samples = labels.copy().reset_index(drop=True)
        samples.insert(0, "sample_index", np.arange(len(samples), dtype=np.int64))
        samples["fold"] = samples["subject"].map(assignments.set_index("subject")["fold"]).astype(np.int64)
        samples["state"] = arguments.condition
        names = [f"mean_SFC_{band}" for band in arguments.bands]
        save_feature_dataset(features, samples, assignments, names, output_directory)
        export_fold_features(features, samples, assignments, names, output_directory)
        write_json(output_directory / "summary.json", {
            "status": "complete", "feature_family": "mean_sfc", "sample_unit": "subject",
            "subject_count": len(samples), "sample_count": len(samples), "feature_names": names,
            "feature_shape": list(features.shape), "graph": "SFC", "condition": arguments.condition,
            "n_splits": 10, "subject_disjoint_folds": True, "one_sample_per_subject": True,
            "classifier_trained": False, "features_file": "features.npz", "sample_metadata": "samples.csv",
            "subject_folds": "subject_folds.csv", "fold_directory": "folds",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": getattr(sys.modules.get("sklearn"), "__version__", "unknown")},
        })
        write_json(output_directory / "run_status.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
        print(f"Saved subject-level mean SFC features and ten-fold datasets to {output_directory}", flush=True)
        return 0
    except Exception as error:
        if output_directory is not None:
            write_json(output_directory / "run_status.json", {"status": "failed", "subject": active_subject, "error": f"{type(error).__name__}: {error}", "updated_at": datetime.now(timezone.utc).isoformat()})
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
