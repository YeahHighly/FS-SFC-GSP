from __future__ import annotations

import argparse
import csv
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


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DEFAULT_SFC_DIRECTORY = PROJECT_DIRECTORY / "SFC"
DEFAULT_SE_DIRECTORY = PROJECT_DIRECTORY / "SE"
DEFAULT_LABELS_FILE = PROJECT_DIRECTORY / "Data" / "xxxxx.csv"
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
ENTROPY_EPSILON = 1e-12
FEATURE_SCHEMA_VERSION = 2
DEFAULT_RANDOM_SEED = 1024


@dataclass(frozen=True)
class BandInput:
    timeseries: np.ndarray
    sample_times: np.ndarray
    sfreq: float
    segments: pd.DataFrame
    directory: Path
    metadata: dict
    epochs: pd.DataFrame


_SE_INPUT_BANDS = BANDS
_SE_SEGMENT_COLUMNS = [
    "segment", "state", "source_onset_s", "source_offset_s",
    "trimmed_onset_s", "trimmed_offset_s", "duration_s",
    "original_start_sample", "original_stop_sample_exclusive",
    "concatenated_start_sample", "concatenated_stop_sample_exclusive",
]
_SE_SCAN_CHUNK = 32768


def _se_read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _se_integer(value: Any, label: str, minimum: int = 0) -> int:
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
    indices = [_se_integer(value, "Node index", 1) for value in nodes["index"]]
    if indices != list(range(1, 91)):
        raise ValueError(f"Node table rows must be ordered by consecutive indices 1-90: {path}")
    names = nodes["name"].astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError(f"Node names must be nonempty and unique: {path}")
    nodes = nodes.copy()
    nodes["index"] = indices
    nodes["name"] = names
    if "original_value" in nodes.columns:
        values = [_se_integer(value, "Original atlas label", 1) for value in nodes["original_value"]]
        if len(set(values)) != 90:
            raise ValueError(f"Original atlas labels must be unique: {path}")
        nodes["original_value"] = values
    return nodes.reset_index(drop=True)


def _se_validate_node_alignment(first: pd.DataFrame, second: pd.DataFrame) -> None:
    columns = ["index", "name"]
    if "original_value" in first.columns and "original_value" in second.columns:
        columns.append("original_value")
    if len(first) != 90 or len(second) != 90:
        raise ValueError("SC and FC require exactly 90 ROI nodes")
    for column in columns:
        if column not in first.columns or column not in second.columns or first[column].tolist() != second[column].tolist():
            raise ValueError(f"SC/FC node ordering differs in column {column}")


def load_subject_metadata(fc_directory: Path, subject: str) -> dict:
    summary = _se_read_json(fc_directory / "summary.json")
    if summary.get("processing_status") != "complete" or summary.get("subject") != subject:
        raise ValueError("FC summary must report complete processing for the requested subject")
    if _se_integer(summary.get("roi_count", -1), "FC ROI count", 1) != 90:
        raise ValueError("FC summary must describe 90 ROIs")
    origin = str(summary.get("time_series_space", "")).casefold()
    if "source" not in origin or "mri" not in origin or summary.get("sensor_space_connectivity") is not False:
        raise ValueError("FC inputs must be individual-MRI source-space ROI signals, not sensor-space connectivity")
    if summary.get("fc_schema_version") != 2:
        raise ValueError(f"FC outputs must include saved epoch definitions and normalised matrices. Rebuild functional connectivity for {subject}.")
    epoch_seconds = float(summary.get("epoch_seconds", float("nan")))
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0.0:
        raise ValueError(f"FC summary has an invalid epoch duration for {subject}")
    return summary


def _se_validate_segments(
    segments: pd.DataFrame,
    sample_times: np.ndarray,
    sample_count: int,
    sfreq: float,
    state: str,
) -> pd.DataFrame:
    missing = set(_SE_SEGMENT_COLUMNS).difference(segments.columns)
    if missing or segments.empty:
        raise ValueError(f"Missing or empty segment table; required columns missing: {sorted(missing)}")
    segments = segments.loc[:, _SE_SEGMENT_COLUMNS].copy().reset_index(drop=True)
    integer_columns = [
        "segment", "original_start_sample", "original_stop_sample_exclusive",
        "concatenated_start_sample", "concatenated_stop_sample_exclusive",
    ]
    for column in integer_columns:
        segments[column] = [_se_integer(value, column, 1 if column == "segment" else 0) for value in segments[column]]
    if segments["segment"].tolist() != list(range(1, len(segments) + 1)):
        raise ValueError("Segment identifiers must be consecutive and ordered from 1")
    segments["state"] = segments["state"].astype(str).str.strip()
    actual_states = set(segments["state"])
    expected_states = {"EO", "EC"} if state == "ALL" else {state}
    if actual_states != expected_states:
        raise ValueError("Segment state labels do not match the selected condition")
    for column in ("source_onset_s", "source_offset_s", "trimmed_onset_s", "trimmed_offset_s", "duration_s"):
        segments[column] = pd.to_numeric(segments[column], errors="raise").astype(float)
        if not np.isfinite(segments[column].to_numpy()).all():
            raise ValueError(f"Segment timing column contains nonfinite values: {column}")
    previous_concat_stop = 0
    previous_original_stop = 0
    tolerance = max(1e-8, 1e-6 / sfreq)
    for row in segments.itertuples(index=False):
        start, stop = int(row.concatenated_start_sample), int(row.concatenated_stop_sample_exclusive)
        original_start, original_stop = int(row.original_start_sample), int(row.original_stop_sample_exclusive)
        if start != previous_concat_stop or not 0 <= start < stop <= sample_count:
            raise ValueError("Segments must cover the concatenated signal exactly once without gaps or overlap")
        if original_start < previous_original_stop or original_stop <= original_start or original_stop - original_start != stop - start:
            raise ValueError("Original sample intervals must be ordered, nonoverlapping, and match concatenated lengths")
        expected_onset, expected_offset = original_start / sfreq, original_stop / sfreq
        if not math.isclose(row.trimmed_onset_s, expected_onset, rel_tol=0.0, abs_tol=tolerance) or not math.isclose(row.trimmed_offset_s, expected_offset, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("Trimmed segment times disagree with original sample indices")
        if not math.isclose(row.duration_s, (stop - start) / sfreq, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("Segment duration disagrees with its sample count")
        if row.source_onset_s < -tolerance or row.source_onset_s > expected_onset + tolerance or row.source_offset_s < expected_offset - tolerance:
            raise ValueError("Source segment bounds do not contain the retained interval")
        for chunk_start in range(start, stop, _SE_SCAN_CHUNK):
            chunk_stop = min(stop, chunk_start + _SE_SCAN_CHUNK)
            expected_times = (original_start + np.arange(chunk_start - start, chunk_stop - start, dtype=np.float64)) / sfreq
            if not np.allclose(sample_times[chunk_start:chunk_stop], expected_times, rtol=0.0, atol=tolerance):
                raise ValueError("Saved sample times disagree with the original recording clock within a segment")
        previous_concat_stop, previous_original_stop = stop, original_stop
    if previous_concat_stop != sample_count:
        raise ValueError("Segment table does not cover all saved time-series samples")
    return segments


def load_band_input(fc_directory: Path, state: str, band: str, nodes: pd.DataFrame) -> BandInput:
    if state not in ("EO", "EC", "ALL") or band not in _SE_INPUT_BANDS:
        raise ValueError("Unsupported state or frequency band")
    _se_validate_node_alignment(load_nodes(fc_directory / "nodes.tsv"), nodes)
    directory = fc_directory / state / band
    metadata = _se_read_json(directory / "metadata.json")
    if metadata.get("fc_schema_version") != 2 or metadata.get("fc_estimation_unit") != "epoch":
        raise ValueError(f"FC inputs must be computed upstream using saved epochs: {directory}")
    if metadata.get("condition") != state or metadata.get("frequency_band") != band:
        raise ValueError("Band metadata condition or frequency-band label is inconsistent")
    if _se_integer(metadata.get("roi_count", -1), "Band ROI count", 1) != 90:
        raise ValueError("Band metadata must describe 90 ROIs")
    if metadata.get("roi_order") != "nodes.tsv indices 1-90":
        raise ValueError("Band metadata must specify nodes.tsv indices 1-90 as the ROI order")
    origin = str(metadata.get("signal_origin", "")).casefold()
    if "source" not in origin or "mri" not in origin:
        raise ValueError("Band signals must come from individual-MRI source reconstruction")
    frequencies = np.asarray(metadata.get("frequency_range_hz", []), dtype=float)
    expected_range = np.asarray(_SE_INPUT_BANDS[band], dtype=float)
    if frequencies.shape != (2,) or not np.isfinite(frequencies).all() or not np.allclose(frequencies, expected_range, rtol=0.0, atol=1e-8):
        raise ValueError("Band metadata frequency bounds do not match the selected band")
    sfreq = float(metadata.get("sampling_frequency_hz", float("nan")))
    if not math.isfinite(sfreq) or sfreq <= 2.0 * frequencies[1]:
        raise ValueError("Sampling frequency must be finite and exceed twice the band upper bound")
    for field, expected_name in (("time_series_file", "roi_timeseries.npy"), ("sample_times_file", "sample_times_seconds.npy"), ("segments_file", "segments.tsv"), ("epochs_file", "epochs.tsv")):
        if metadata.get(field) != expected_name:
            raise ValueError(f"Unsupported band metadata file mapping for {field}")
    timeseries = np.load(directory / "roi_timeseries.npy", mmap_mode="r", allow_pickle=False)
    sample_times = np.load(directory / "sample_times_seconds.npy", mmap_mode="r", allow_pickle=False)
    if not isinstance(timeseries, np.ndarray) or timeseries.ndim != 2 or timeseries.shape[0] != 90 or timeseries.shape[1] == 0 or timeseries.dtype.kind not in "fiu":
        raise ValueError("Band time series must be a real numeric array with shape (90, N), N > 0")
    sample_count = timeseries.shape[1]
    if not isinstance(sample_times, np.ndarray) or sample_times.shape != (sample_count,) or sample_times.dtype.kind not in "fiu":
        raise ValueError("Sample times must be a real numeric vector matching the signal length")
    if _se_integer(metadata.get("sample_count", -1), "Band sample count", 1) != sample_count:
        raise ValueError("Band metadata sample count does not match the saved array")
    if metadata.get("time_series_shape") != [90, sample_count]:
        raise ValueError("Band metadata time-series shape does not match the saved array")
    for start in range(0, sample_count, _SE_SCAN_CHUNK):
        stop = min(sample_count, start + _SE_SCAN_CHUNK)
        if not np.isfinite(timeseries[:, start:stop]).all() or not np.isfinite(sample_times[start:stop]).all():
            raise ValueError("Band time series or timestamps contain nonfinite values")
    segments = pd.read_csv(directory / "segments.tsv", sep="\t", encoding="utf-8-sig")
    segments = _se_validate_segments(segments, sample_times, sample_count, sfreq, state)
    if _se_integer(metadata.get("segment_count", -1), "Band segment count", 1) != len(segments):
        raise ValueError("Band metadata segment count does not match segments.tsv")
    epochs = pd.read_csv(directory / "epochs.tsv", sep="\t", encoding="utf-8-sig")
    epochs = _se_validate_epochs(epochs, segments, sample_count, sfreq, state, metadata)
    return BandInput(timeseries, sample_times, sfreq, segments, directory, metadata, epochs)


_SE_EPOCH_COLUMNS = [
    "epoch_index", "state", "segment",
    "concatenated_start_sample", "concatenated_stop_sample_exclusive",
    "original_start_sample", "original_stop_sample_exclusive",
    "onset_seconds", "offset_seconds", "duration_seconds",
]


def _se_validate_epochs(
    epochs: pd.DataFrame,
    segments: pd.DataFrame,
    sample_count: int,
    sfreq: float,
    state: str,
    metadata: dict,
) -> pd.DataFrame:
    if state not in ("EO", "EC", "ALL") or not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Saved epoch validation requires a valid condition and sampling frequency")
    if metadata.get("epochs_file") != "epochs.tsv" or metadata.get("fc_estimation_unit") != "epoch":
        raise ValueError("FC must provide epochs.tsv and report fc_estimation_unit=epoch; rebuild FC")
    epoch_seconds = float(metadata.get("epoch_seconds", float("nan")))
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("Saved epoch duration must be finite and positive")
    epoch_samples = _se_integer(metadata.get("epoch_samples", -1), "Saved epoch samples", 4)
    if _se_integer(epoch_seconds * sfreq, "Saved epoch duration in samples", 4) != epoch_samples:
        raise ValueError("Saved epoch duration and sample count disagree")
    expected_total = _se_integer(metadata.get("epoch_count", -1), "Saved epoch count", 1)
    sample_count = _se_integer(sample_count, "Saved signal sample count", 1)
    missing = set(_SE_EPOCH_COLUMNS).difference(epochs.columns)
    if missing or epochs.empty or len(epochs) != expected_total or epochs.columns.duplicated().any():
        raise ValueError(f"Invalid saved epoch table or metadata epoch count; missing columns: {sorted(missing)}")
    if segments.empty or not set(_SE_SEGMENT_COLUMNS).issubset(segments.columns):
        raise ValueError("Saved epochs require a validated nonempty segment table")
    epochs = epochs.loc[:, _SE_EPOCH_COLUMNS].copy().reset_index(drop=True)
    for column in (
        "epoch_index", "segment", "concatenated_start_sample",
        "concatenated_stop_sample_exclusive", "original_start_sample",
        "original_stop_sample_exclusive",
    ):
        epochs[column] = [_se_integer(value, column, 1 if column == "segment" else 0) for value in epochs[column]]
    if epochs["epoch_index"].tolist() != list(range(expected_total)):
        raise ValueError("Saved epoch indices must be consecutive, zero-based, and in file order")
    epochs["state"] = epochs["state"].astype(str).str.strip()
    for column in ("onset_seconds", "offset_seconds", "duration_seconds"):
        epochs[column] = pd.to_numeric(epochs[column], errors="raise").astype(float)
        if not np.isfinite(epochs[column].to_numpy()).all():
            raise ValueError(f"Saved epoch column contains nonfinite values: {column}")
    cursor = 0
    previous_stop = 0
    previous_original_stop = 0
    tolerance = max(1e-8, 1e-6 / sfreq)
    for segment in segments.itertuples(index=False):
        segment_start = _se_integer(segment.concatenated_start_sample, "Segment start")
        segment_stop = _se_integer(segment.concatenated_stop_sample_exclusive, "Segment stop", 1)
        segment_original_start = _se_integer(segment.original_start_sample, "Original segment start")
        segment_original_stop = _se_integer(segment.original_stop_sample_exclusive, "Original segment stop", 1)
        segment_id = _se_integer(segment.segment, "Segment identifier", 1)
        if not 0 <= segment_start < segment_stop <= sample_count or segment_original_stop - segment_original_start != segment_stop - segment_start:
            raise ValueError("Saved epoch segment bounds are invalid")
        if segment.state not in ("EO", "EC") or (state != "ALL" and segment.state != state):
            raise ValueError("Segment state does not match the selected condition")
        count = (segment_stop - segment_start) // epoch_samples
        if cursor + count > expected_total:
            raise ValueError("Saved epochs omit complete windows; only incomplete segment tails may be discarded")
        for offset in range(count):
            row = epochs.iloc[cursor]
            start = segment_start + offset * epoch_samples
            stop = start + epoch_samples
            original_start = segment_original_start + offset * epoch_samples
            original_stop = original_start + epoch_samples
            if row["segment"] != segment_id or row["state"] != segment.state:
                raise ValueError("Saved epoch segment or EO/EC label differs from its source segment")
            if row["concatenated_start_sample"] != start or row["concatenated_stop_sample_exclusive"] != stop:
                raise ValueError("Saved epochs must cover every complete fixed-length window without overlap or rejection")
            if row["original_start_sample"] != original_start or row["original_stop_sample_exclusive"] != original_stop:
                raise ValueError("Saved epoch original sample indices do not match its segment")
            if start < previous_stop or original_start < previous_original_stop:
                raise ValueError("Saved epochs overlap or are out of recording-time order")
            expected_times = (original_start / sfreq, original_stop / sfreq, epoch_samples / sfreq)
            actual_times = (row["onset_seconds"], row["offset_seconds"], row["duration_seconds"])
            if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance) for actual, expected in zip(actual_times, expected_times)):
                raise ValueError("Saved epoch timestamps disagree with its recording sample indices")
            previous_stop, previous_original_stop = stop, original_stop
            cursor += 1
    if cursor != expected_total:
        raise ValueError("Saved epochs contain extra windows or cross a segment boundary")
    return epochs


def aligned_epoch_table(bands: dict[str, BandInput], state: str, epoch_seconds: float) -> pd.DataFrame:
    if not bands or state not in ("EO", "EC", "ALL"):
        raise ValueError("Aligned epoch loading requires bands and a valid condition")
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("Requested epoch duration must be finite and positive")
    first = next(iter(bands.values()))
    expected_samples = _se_integer(epoch_seconds * first.sfreq, "Requested epoch samples", 4)
    for band, current in bands.items():
        metadata = current.metadata
        if metadata.get("condition") != state or metadata.get("frequency_band") != band:
            raise ValueError("Band identity does not match the requested saved epoch configuration")
        saved_seconds = float(metadata.get("epoch_seconds", float("nan")))
        if not math.isfinite(saved_seconds) or not math.isclose(saved_seconds, epoch_seconds, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("Requested epoch duration differs from upstream FC; rebuild FC with the requested duration")
        if _se_integer(metadata.get("epoch_samples", -1), "Saved epoch samples", 4) != expected_samples:
            raise ValueError("Requested epoch sample count differs from upstream FC")
        if metadata.get("epochs_file") != "epochs.tsv" or metadata.get("fc_estimation_unit") != "epoch":
            raise ValueError("Upstream FC must provide its saved estimation epochs")
        if not math.isclose(current.sfreq, first.sfreq, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("All bands must have identical sampling frequencies")
        if current.timeseries.shape != first.timeseries.shape or current.sample_times.shape != first.sample_times.shape:
            raise ValueError("All bands must have identical ROI and time dimensions")
        for key in ("sample_count", "roi_order", "epoch_samples", "epoch_count", "epochs_file", "fc_estimation_unit"):
            if metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"Frequency-band metadata differ in {key}")
        if not current.segments.loc[:, _SE_SEGMENT_COLUMNS].equals(first.segments.loc[:, _SE_SEGMENT_COLUMNS]):
            raise ValueError("All bands must share identical source segments and timing")
        if not current.epochs.loc[:, _SE_EPOCH_COLUMNS].equals(first.epochs.loc[:, _SE_EPOCH_COLUMNS]):
            raise ValueError("All bands must share exactly the same persisted epochs")
        for start in range(0, first.sample_times.size, 32768):
            stop = min(first.sample_times.size, start + 32768)
            if not np.array_equal(current.sample_times[start:stop], first.sample_times[start:stop]):
                raise ValueError("All bands must share exactly the same sample-time axis")
    return first.epochs.loc[:, _SE_EPOCH_COLUMNS].copy().reset_index(drop=True)


def load_structural_matrix(sc_directory: Path, subject: str, nodes: pd.DataFrame) -> np.ndarray:
    summary = _se_read_json(sc_directory / "summary.json")
    if summary.get("processing_status") != "complete" or summary.get("subject") != subject:
        raise ValueError("SC summary must report complete processing for the requested subject")
    if _se_integer(summary.get("atlas_nodes", -1), "SC atlas node count", 1) != 90:
        raise ValueError("SC summary must describe 90 atlas nodes")
    _se_validate_node_alignment(load_nodes(sc_directory / "nodes.tsv"), nodes)
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


def build_graph(
    matrix: np.ndarray,
    topk: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    values = _validated_nonnegative_connectivity(matrix)
    if isinstance(topk, (bool, np.bool_)) or not isinstance(topk, (int, np.integer)):
        raise ValueError("topk must be an integer")
    if not 0 <= topk < 90:
        raise ValueError("topk must be between 0 and 89; zero retains all positive edges")
    if topk == 0:
        adjacency = values.copy()
    else:
        selected = np.zeros((90, 90), dtype=bool)
        for row in range(90):
            candidates = np.flatnonzero(values[row] > 0.0)
            order = np.lexsort((candidates, -values[row, candidates]))
            selected[row, candidates[order[:topk]]] = True
        adjacency = np.where(selected | selected.T, values, 0.0)
    degree = adjacency.sum(axis=1)
    if not np.isfinite(degree).all():
        raise ValueError("Graph weighted degrees exceed the supported numerical range")
    positive_degree = degree > 0.0
    inverse_sqrt = np.zeros(90, dtype=np.float64)
    inverse_sqrt[positive_degree] = 1.0 / np.sqrt(degree[positive_degree])
    laplacian = np.eye(90) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    laplacian = laplacian * 0.5 + laplacian.T * 0.5
    if not np.isfinite(laplacian).all():
        raise ValueError("Normalized graph Laplacian contains invalid values")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError("Graph eigendecomposition contains invalid values")
    tolerance = 1e-8
    if eigenvalues[0] < -tolerance or eigenvalues[-1] > 2.0 + tolerance:
        raise ValueError("Normalized Laplacian eigenvalues fall outside zero to two")
    if np.any(np.diff(eigenvalues) < -tolerance):
        raise ValueError("Graph eigenvalues are not sorted in ascending order")
    orthogonality_error = float(np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(90))))
    residual_error = float(np.max(np.abs(laplacian @ eigenvectors - eigenvectors * eigenvalues)))
    if orthogonality_error > tolerance or residual_error > tolerance:
        raise ValueError("Graph eigendecomposition failed orthogonality or residual validation")
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    neighbors = adjacency > 0.0
    unvisited = set(range(90))
    component_sizes: list[int] = []
    while unvisited:
        pending = [min(unvisited)]
        unvisited.remove(pending[0])
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            for neighbor in np.flatnonzero(neighbors[node]):
                neighbor = int(neighbor)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    pending.append(neighbor)
        component_sizes.append(size)
    qc = {
        "node_count": 90,
        "edge_count": int(np.count_nonzero(np.triu(neighbors, k=1))),
        "topk": int(topk),
        "selection": "all positive off-diagonal edges" if topk == 0 else "node-wise positive top-k followed by symmetric union",
        "tie_break": "ascending node index",
        "retained_weights": "original input weights without rescaling",
        "weighted_degrees": degree.tolist(),
        "neighbor_counts": np.count_nonzero(neighbors, axis=1).tolist(),
        "isolated_node_indices": (np.flatnonzero(~positive_degree) + 1).tolist(),
        "connected_component_count": len(component_sizes),
        "connected_component_sizes": component_sizes,
        "laplacian": "I - D^(-1/2) A D^(-1/2); inverse degree zero and diagonal one for isolated nodes",
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "orthogonality_max_absolute_error": orthogonality_error,
        "eigendecomposition_max_absolute_residual": residual_error,
    }
    return adjacency, eigenvalues, eigenvectors, qc


def graph_spectral_entropy(
    epoch: np.ndarray,
    eigenvectors: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[float, np.ndarray, float]:
    if np.iscomplexobj(epoch) or np.iscomplexobj(eigenvectors):
        raise ValueError("ROI signals and graph eigenvectors must be real")
    values = np.asarray(epoch, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 90 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("An entropy epoch must have 90 finite ROI rows and at least two samples")
    if basis.shape != (90, 90) or not np.isfinite(basis).all():
        raise ValueError("Graph Fourier basis must be a finite 90 by 90 array")
    if not np.allclose(basis.T @ basis, np.eye(90), rtol=1e-7, atol=1e-8):
        raise ValueError("Graph Fourier basis is not orthonormal")
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("Relative energy epsilon must be finite and between zero and one")
    input_scale = float(np.max(np.abs(values)))
    if input_scale == 0.0:
        raise ValueError("Cannot compute graph spectral entropy for an all-zero epoch")
    centered = values / input_scale
    constant = np.ptp(centered, axis=1) == 0.0
    centered -= centered.mean(axis=1, keepdims=True)
    centered[constant] = 0.0
    centered_scale = float(np.max(np.abs(centered)))
    if centered_scale == 0.0:
        raise ValueError("Cannot compute graph spectral entropy for an epoch with zero centered energy")
    centered /= centered_scale
    transformed = basis.T @ centered
    mode_energy = np.mean(transformed * transformed, axis=1)
    total_energy_scaled = float(mode_energy.sum())
    centered_energy_scaled = float(np.sum(centered * centered) / centered.shape[1])
    if not np.isfinite(total_energy_scaled) or total_energy_scaled <= 0.0:
        raise ValueError("Graph spectral energy is zero or invalid")
    if not np.isclose(total_energy_scaled, centered_energy_scaled, rtol=1e-7, atol=0.0):
        raise ValueError("Graph Fourier transform did not preserve epoch energy")
    probabilities = mode_energy / (total_energy_scaled * (1.0 + epsilon))
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive]))) / float(np.log(90.0))
    root_total_energy = float(np.sqrt(centered_energy_scaled) * centered_scale * input_scale)
    total_centered_energy = root_total_energy * root_total_energy
    if not np.isfinite(total_centered_energy) or total_centered_energy <= 0.0:
        raise ValueError("Physical centered epoch energy is outside the positive float64 range")
    if not np.isfinite(entropy) or not -1e-10 <= entropy <= 1.0 + 1e-10:
        raise ValueError("Normalized graph spectral entropy is outside zero to one")
    return float(np.clip(entropy, 0.0, 1.0)), probabilities, total_centered_energy


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


def export_fold_features(
    features: np.ndarray,
    epochs: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
    output_directory: Path,
) -> None:
    required = {"subject", "label", "epoch_id", "state"}
    if not required.issubset(epochs.columns):
        raise ValueError(f"Missing epoch columns: {sorted(required.difference(epochs.columns))}")
    if epochs.empty or epochs[list(required)].isna().any().any():
        raise ValueError("Every epoch must have a subject, label, epoch_id, and state.")
    if np.iscomplexobj(features):
        raise ValueError("Fold features must be real-valued.")
    values = np.asarray(features, dtype=np.float64)
    names = [str(name) for name in feature_names]
    if values.ndim != 2 or values.shape != (len(epochs), len(names)) or not names or len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("Features must have one row per epoch and one column per unique nonempty feature name.")
    if not np.isfinite(values).all():
        raise ValueError("Fold features contain nonfinite values.")
    manifest_attributes = dict(assignments.attrs)
    canonical = _fold_canonical_labels(assignments)
    assignments = _fold_validate_assignments(assignments, canonical)
    metadata = epochs.copy().reset_index(drop=True)
    metadata["subject"] = metadata["subject"].map(normalize_subject_id)
    labels = pd.to_numeric(metadata["label"], errors="raise")
    if not labels.isin((0, 1)).all():
        raise ValueError("Epoch labels must use 0=HC and 1=ADHD.")
    metadata["label"] = labels.astype(np.int64)
    for name in ("epoch_id", "state"):
        metadata[name] = metadata[name].astype(str)
        if metadata[name].str.strip().eq("").any():
            raise ValueError(f"Epoch {name} values must be nonempty.")
    if set(metadata["subject"]) != set(assignments["subject"]):
        raise ValueError("Feature epochs and fold assignments must contain exactly the same subjects; no subjects may be silently excluded.")
    subject_labels = assignments.set_index("subject")["label"]
    if not metadata["subject"].map(subject_labels).eq(metadata["label"]).all():
        raise ValueError("Epoch labels disagree with subject-fold labels.")
    subject_folds = assignments.set_index("subject")["fold"]
    epoch_folds = metadata["subject"].map(subject_folds).to_numpy(dtype=np.int64)
    global_indices = np.arange(len(metadata), dtype=np.int64)
    if "sample_index" in metadata.columns:
        existing_indices = pd.to_numeric(metadata["sample_index"], errors="raise").to_numpy()
        if not np.array_equal(existing_indices, global_indices):
            raise ValueError("Existing sample_index values must match the feature row positions.")
    metadata["sample_index"] = global_indices
    held_out_counts = np.zeros(len(metadata), dtype=np.int64)
    plan = []
    for fold in range(1, 11):
        test_indices = global_indices[epoch_folds == fold]
        train_indices = global_indices[epoch_folds != fold]
        train_subjects = set(metadata.iloc[train_indices]["subject"])
        test_subjects = set(metadata.iloc[test_indices]["subject"])
        if not len(train_indices) or not len(test_indices) or train_subjects.intersection(test_subjects):
            raise RuntimeError(f"Fold {fold} has an empty partition or subject leakage.")
        for indices in (train_indices, test_indices):
            if set(metadata.iloc[indices]["label"]) != {0, 1}:
                raise RuntimeError(f"Fold {fold} must contain both classes in train and test partitions.")
        held_out_counts[test_indices] += 1
        plan.append((fold, train_indices, test_indices))
    if not np.all(held_out_counts == 1):
        raise RuntimeError("Every feature row must occur in exactly one held-out test partition.")
    folds_directory = Path(output_directory) / "folds"
    folds_directory.mkdir(parents=True, exist_ok=True)
    for fold, train_indices, test_indices in plan:
        directory = folds_directory / f"fold_{fold:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        partition_counts = {}
        for partition, indices in (("train", train_indices), ("test", test_indices)):
            partition_epochs = metadata.iloc[indices].copy()
            partition_subjects = assignments.loc[assignments["subject"].isin(partition_epochs["subject"])].copy()
            arrays = {
                "X": values[indices],
                "y": partition_epochs["label"].to_numpy(dtype=np.int64),
                "sample_indices": indices.astype(np.int64),
                "subjects": partition_epochs["subject"].to_numpy(dtype=np.str_),
                "epoch_ids": partition_epochs["epoch_id"].to_numpy(dtype=np.str_),
                "feature_names": np.asarray(names, dtype=np.str_),
            }
            _fold_write_npz(directory / f"{partition}.npz", arrays)
            _fold_write_csv(partition_epochs, directory / f"{partition}_epochs.csv")
            _fold_write_csv(partition_subjects, directory / f"{partition}_subjects.csv")
            partition_counts[partition] = {
                "epoch_count": len(partition_epochs),
                "subject_count": len(partition_subjects),
                "class_subject_counts": {"HC": int(partition_subjects["label"].eq(0).sum()), "ADHD": int(partition_subjects["label"].eq(1).sum())},
                "subjects": partition_subjects["subject"].tolist(),
            }
        _fold_write_json({
            "fold": fold,
            "n_splits": 10,
            "split_unit": "subject",
            "feature_count": len(names),
            "label_mapping": {"HC": 0, "ADHD": 1},
            "cohort_hash": manifest_attributes.get("cohort_hash"),
            "seed": manifest_attributes.get("seed"),
            "fold_manifest": manifest_attributes.get("fold_manifest"),
            "feature_scaling": "none",
            "subject_overlap": False,
            "held_out_once": True,
            "partitions": partition_counts,
        }, directory / "split.json")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract epoch-level graph spectral entropy on SC, FC, or SFC graphs and export subject-disjoint ten-fold datasets.")
    parser.add_argument("--graph", type=str.upper, choices=("SC", "FC", "SFC"), default="SFC")
    parser.add_argument("--bands", nargs="+", type=str.lower, choices=(*BANDS, "all"), default=["all"])
    parser.add_argument("--condition", type=str.upper, choices=("EC", "EO", "ALL"), default="EC")
    parser.add_argument("--fc-method", "--method", dest="fc_method", choices=("auto", "pearson", "wpli"), default="auto", help="Select an existing upstream FC matrix; auto uses the common primary method saved by the FC builder.")
    parser.add_argument("--epoch-seconds", type=float, help="Optional check of the saved FC epoch duration; the default inherits the upstream value.")
    parser.add_argument("--topk", type=int, default=5, help="Select k positive edges per node and symmetrise by union; 0 keeps all positive edges.")
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subject", nargs="+", type=normalize_subject_id)
    parser.add_argument("--sfc-dir", type=Path, default=DEFAULT_SFC_DIRECTORY)
    parser.add_argument("--se-dir", type=Path, default=DEFAULT_SE_DIRECTORY)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if "all" in arguments.bands:
        if len(arguments.bands) != 1:
            parser.error("Use ALL alone or specify individual frequency bands.")
        arguments.bands = list(BANDS)
    else:
        arguments.bands = [band for band in BANDS if band in arguments.bands]
    if arguments.epoch_seconds is not None and (not math.isfinite(arguments.epoch_seconds) or arguments.epoch_seconds <= 0):
        parser.error("--epoch-seconds must be a finite positive number.")
    if not 0 <= arguments.topk <= 89:
        parser.error("--topk must be between 0 and 89.")
    if not 0 <= arguments.seed < 2**32:
        parser.error("--seed must be an integer between 0 and 2**32 - 1.")
    if not arguments.subject_column.strip() or not arguments.label_column.strip():
        parser.error("Subject and label column names must be nonempty.")
    for name in ("labels_file", "sfc_dir", "se_dir"):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    if arguments.subject:
        arguments.subject = list(dict.fromkeys(arguments.subject))
    if arguments.se_dir == arguments.sfc_dir or arguments.se_dir.is_relative_to(arguments.sfc_dir):
        parser.error("--se-dir must be outside the input SFC directory.")
    return arguments


def input_fingerprints(arguments: argparse.Namespace, labels: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    states = ("EO", "EC") if arguments.condition == "ALL" else (arguments.condition,)
    for subject in labels["subject"]:
        subject_root = arguments.sfc_dir / subject
        fc_directory = subject_root / "functional_connectivity"
        paths = [fc_directory / "summary.json", fc_directory / "nodes.tsv"]
        if (fc_directory / "configuration.json").is_file():
            paths.append(fc_directory / "configuration.json")
        if arguments.graph in ("SC", "SFC"):
            sc_directory = subject_root / "structural_connectivity"
            matrix = sc_directory / "connectome_sift2.npy"
            if not matrix.is_file():
                matrix = sc_directory / "connectome_sift2.csv"
            paths.extend((sc_directory / "summary.json", sc_directory / "nodes.tsv", matrix))
        for state in states:
            for band in arguments.bands:
                directory = fc_directory / state / band
                paths.extend(directory / name for name in ("metadata.json", "segments.tsv", "epochs.tsv", "roi_timeseries.npy", "sample_times_seconds.npy"))
                if arguments.graph != "SC":
                    for suffix in ("", "_normalised"):
                        matrix = directory / f"connectivity_{arguments.fc_method}{suffix}.npy"
                        if not matrix.is_file():
                            matrix = directory / f"connectivity_{arguments.fc_method}{suffix}.csv"
                        paths.append(matrix)
        for path in paths:
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            records.append({
                "path": path.relative_to(arguments.sfc_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": checksum.hexdigest(),
            })
    return records


def resolve_upstream_configuration(arguments: argparse.Namespace, summaries: Sequence[dict[str, Any]]) -> None:
    if not summaries:
        raise ValueError("At least one completed FC summary is required")
    durations = [float(summary["epoch_seconds"]) for summary in summaries]
    selected_duration = durations[0] if arguments.epoch_seconds is None else arguments.epoch_seconds
    if any(not math.isclose(duration, selected_duration, rel_tol=0.0, abs_tol=1e-9) for duration in durations):
        raise ValueError("All subjects and the requested epoch duration must match the saved FC epochs. Rebuild FC outputs with a common --epoch-seconds value.")
    arguments.epoch_seconds = selected_duration
    if arguments.graph == "SC":
        return
    primary_methods = set()
    for summary in summaries:
        methods = summary.get("connectivity_methods")
        primary = summary.get("primary_method", summary.get("primary_connectivity"))
        if not isinstance(methods, list) or not methods or any(method not in ("pearson", "wpli") for method in methods) or primary not in methods:
            raise ValueError("FC summaries must identify valid saved methods and their primary method")
        primary_methods.add(primary)
    if arguments.fc_method == "auto":
        if len(primary_methods) != 1:
            raise ValueError("Subjects have different primary FC methods. Select one common saved method with --fc-method pearson or --fc-method wpli.")
        arguments.fc_method = primary_methods.pop()
    for summary in summaries:
        if arguments.fc_method not in summary["connectivity_methods"]:
            raise ValueError(f"The requested {arguments.fc_method} matrix was not computed for {summary['subject']}. Generate it with build_functional_connectivity.py first.")


def configuration_record(arguments: argparse.Namespace, labels: pd.DataFrame) -> dict[str, Any]:
    cohort = labels[["subject", "label"]].to_dict(orient="records")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "graph": arguments.graph,
        "bands": arguments.bands,
        "frequency_bands_hz": {band: list(BANDS[band]) for band in arguments.bands},
        "condition": arguments.condition,
        "fc_method": arguments.fc_method if arguments.graph != "SC" else None,
        "fc_source": "saved upstream matrices" if arguments.graph != "SC" else None,
        "epoch_seconds": arguments.epoch_seconds,
        "epoch_overlap_seconds": 0.0,
        "epoch_remainder_policy": "discard incomplete epochs within each continuous segment",
        "epoch_source": "saved FC epochs.tsv; validated and reused without creating new epochs",
        "epoch_artifact_rejection": "none",
        "topk": arguments.topk,
        "symmetrisation": "union of per-node selections; final degree may exceed k",
        "structural_normalisation": "log1p of raw SIFT2 weights divided by the maximum off-diagonal log1p weight",
        "functional_normalisation": "load upstream weights already divided by the maximum off-diagonal FC weight",
        "sfc_fusion": "elementwise product of normalised SC and FC before graph edge selection",
        "laplacian": "I - D^(-1/2) A D^(-1/2); inverse degree is zero at isolated nodes",
        "node_signal": "individual-MRI AAL90 source-space ROI time series",
        "signal_centring": "temporal mean removal within each ROI and epoch; no ROI-wise variance normalisation",
        "entropy": "Shannon entropy of time-averaged graph-mode energies divided by log(90)",
        "entropy_epsilon": ENTROPY_EPSILON,
        "entropy_epsilon_scale": "relative to total centred epoch energy",
        "combined_condition_policy": "pool separately extracted EO and EC epochs, each using its own eye-state graph",
        "band_selection_policy": "ALL concatenates five band-specific entropy features for the same epoch",
        "cross_validation": "ten subject-stratified folds; all epochs from a subject remain in one fold",
        "n_splits": 10,
        "seed": arguments.seed,
        "feature_scaling": "none; fit any later scaler on classification training folds only",
        "sfc_directory": str(arguments.sfc_dir),
        "input_files": input_fingerprints(arguments, labels),
        "labels_file": str(arguments.labels_file),
        "labels_file_sha256": hashlib.sha256(arguments.labels_file.read_bytes()).hexdigest(),
        "subject_column": arguments.subject_column,
        "label_column": arguments.label_column,
        "label_mapping": {"HC": 0, "ADHD": 1},
        "cohort": cohort,
        "source_script_sha256": script_hash,
        "paper_method_sections": ["2.3.2", "2.4.3", "2.5", "2.6", "2.8.1"],
    }


def prepare_configuration_directory(arguments: argparse.Namespace, configuration: dict[str, Any]) -> Path:
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    band_name = "ALL" if arguments.bands == list(BANDS) else "-".join(arguments.bands)
    method_name = "none" if arguments.graph == "SC" else arguments.fc_method
    epoch_name = format(arguments.epoch_seconds, ".8g").replace(".", "p")
    directory = arguments.se_dir / f"{arguments.graph}_fc-{method_name}_bands-{band_name}_state-{arguments.condition}_epoch-{epoch_name}s_topk-{arguments.topk}_seed-{arguments.seed}_{signature}"
    root = arguments.se_dir.resolve()
    resolved = directory.resolve()
    if directory.is_symlink() or not resolved.is_relative_to(root) or resolved.parent != root:
        raise RuntimeError("The configuration output must be a direct child of the SE directory")
    if directory.exists() and any(directory.iterdir()):
        if not arguments.overwrite:
            raise RuntimeError(f"Output already exists: {directory}. Use --overwrite to replace this configuration.")
        shutil.rmtree(resolved)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "configuration.json", configuration)
    return directory


def load_saved_functional(band: BandInput, method: str) -> np.ndarray:
    primary = str(band.metadata.get("primary_connectivity", "")).casefold()
    if "topk" in band.metadata or "sparse" in primary:
        raise RuntimeError(f"Saved FC must contain full connection weights before graph construction: {band.directory}")
    outputs = band.metadata.get("connectivity_outputs", {})
    if not isinstance(outputs, dict) or method not in outputs or method not in band.metadata.get("connectivity_methods", []):
        raise ValueError(f"Saved {method} connectivity is unavailable in {band.directory}. Generate it with build_functional_connectivity.py first.")
    record = outputs[method]
    if not isinstance(record, dict) or not record.get("normalisation"):
        raise ValueError(f"Missing upstream FC normalisation metadata: {band.directory}")
    matrices = []
    for suffix, npy_key, csv_key in (("", "connectivity_npy", "connectivity_csv"), ("_normalised", "normalised_npy", "normalised_csv")):
        stem = f"connectivity_{method}{suffix}"
        if record.get(npy_key) != f"{stem}.npy" or record.get(csv_key) != f"{stem}.csv":
            raise ValueError(f"Unexpected saved FC file mapping for {method}: {band.directory}")
        path = band.directory / f"{stem}.npy"
        if path.is_file():
            matrix = np.load(path, allow_pickle=False)
        else:
            path = band.directory / f"{stem}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing upstream FC matrix: {path}. Generate FC outputs before extracting SE.")
            matrix = np.loadtxt(path, delimiter=",")
        if not isinstance(matrix, np.ndarray) or matrix.dtype.kind not in "fiu":
            raise ValueError(f"Saved functional connectivity must be a real numeric matrix: {path}")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (90, 90) or not np.isfinite(matrix).all() or (matrix < 0.0).any() or (matrix > 1.0 + 1e-7).any():
            raise ValueError(f"Saved FC weights must be finite 90 by 90 values between zero and one: {path}")
        if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10) or not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"Saved FC must be symmetric with a zero diagonal: {path}")
        matrices.append(matrix)
    raw, normalised = matrices
    maximum = float(record.get("normalisation_maximum", float("nan")))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"Saved {method} connectivity has no positive normalisation factor: {band.directory}")
    if not np.isclose(maximum, float(raw.max()), rtol=1e-7, atol=1e-12):
        raise ValueError(f"Saved FC normalisation factor does not match the raw matrix: {band.directory}")
    if not np.isclose(float(normalised.max()), 1.0, rtol=1e-7, atol=1e-10) or not np.allclose(normalised * maximum, raw, rtol=1e-7, atol=1e-12):
        raise ValueError(f"Saved normalised FC does not match the upstream raw matrix: {band.directory}")
    return normalised


def extract_subject_features(
    subject: str,
    label: int,
    fold: int,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    fc_directory = arguments.sfc_dir / subject / "functional_connectivity"
    input_summary = load_subject_metadata(fc_directory, subject)
    nodes = load_nodes(fc_directory / "nodes.tsv")
    subject_directory = output_directory / "subjects" / subject
    subject_directory.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(subject_directory / "nodes.tsv", sep="\t", index=False)
    structural = None
    structural_graph = None
    if arguments.graph in ("SC", "SFC"):
        raw_structural = load_structural_matrix(arguments.sfc_dir / subject / "structural_connectivity", subject, nodes)
        structural = normalize_structural_connectivity(raw_structural)
        np.save(subject_directory / "structural_normalised.npy", structural, allow_pickle=False)
        if arguments.graph == "SC":
            structural_graph = build_graph(structural, topk=arguments.topk)
    states = ("EO", "EC") if arguments.condition == "ALL" else (arguments.condition,)
    state_features = []
    state_epochs = []
    state_reports = {}
    for state in states:
        band_inputs = {band: load_band_input(fc_directory, state, band, nodes) for band in arguments.bands}
        if arguments.graph != "SC":
            primary = input_summary.get("primary_method", input_summary.get("primary_connectivity"))
            for band in band_inputs.values():
                if band.metadata.get("primary_method") != primary:
                    raise ValueError(f"Band and subject metadata disagree about the primary FC method: {band.directory}")
        epochs = aligned_epoch_table(band_inputs, state, arguments.epoch_seconds)
        epochs = epochs.reset_index(drop=True)
        epochs.insert(0, "epoch_id", [f"{subject}_{state}_epoch-{index:06d}" for index in range(len(epochs))])
        epochs.insert(0, "subject", subject)
        epochs.insert(1, "label", label)
        epochs.insert(2, "label_name", "ADHD" if label else "HC")
        epochs.insert(3, "fold", fold)
        bounds = list(zip(epochs["concatenated_start_sample"].astype(int), epochs["concatenated_stop_sample_exclusive"].astype(int)))
        features = np.empty((len(epochs), len(arguments.bands)), dtype=np.float64)
        state_directory = subject_directory / state
        state_directory.mkdir(exist_ok=True)
        band_reports = {}
        for band_index, band_name in enumerate(arguments.bands):
            band = band_inputs[band_name]
            band_directory = state_directory / band_name
            band_directory.mkdir(exist_ok=True)
            functional = None
            if arguments.graph != "SC":
                functional = load_saved_functional(band, arguments.fc_method)
            if arguments.graph == "SC":
                graph_weights = structural
                adjacency, eigenvalues, eigenvectors, graph_qc = structural_graph
            else:
                graph_weights = functional if arguments.graph == "FC" else structural * functional
                adjacency, eigenvalues, eigenvectors, graph_qc = build_graph(graph_weights, topk=arguments.topk)
            degree = adjacency.sum(axis=1)
            inverse_degree = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0.0)
            laplacian = np.eye(len(degree)) - inverse_degree[:, None] * adjacency * inverse_degree[None, :]
            np.savez_compressed(
                band_directory / "graph.npz",
                connectivity=graph_weights,
                adjacency=adjacency,
                laplacian=laplacian,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                roi_indices=nodes["index"].to_numpy(dtype=np.int64),
                roi_names=nodes["name"].to_numpy(dtype=str),
            )
            probabilities = np.empty((len(epochs), 90), dtype=np.float64)
            total_energies = np.empty(len(epochs), dtype=np.float64)
            for epoch_index, (start, stop) in enumerate(bounds):
                try:
                    entropy, probabilities[epoch_index], total_energies[epoch_index] = graph_spectral_entropy(
                        band.timeseries[:, start:stop], eigenvectors, epsilon=ENTROPY_EPSILON,
                    )
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(f"Invalid epoch {epochs.iloc[epoch_index]['epoch_id']} in {band_name}: {error}") from error
                features[epoch_index, band_index] = entropy
            np.save(band_directory / "spectral_entropy.npy", features[:, band_index], allow_pickle=False)
            np.save(band_directory / "normalised_mode_energy.npy", probabilities, allow_pickle=False)
            np.save(band_directory / "total_centred_energy.npy", total_energies, allow_pickle=False)
            graph_qc = dict(graph_qc)
            graph_qc.update({
                "subject": subject,
                "state": state,
                "band": band_name,
                "graph_type": arguments.graph,
                "fc_method": arguments.fc_method if functional is not None else None,
                "fc_source": "saved upstream normalised matrix" if functional is not None else None,
                "epoch_count": len(epochs),
                "entropy_min": float(features[:, band_index].min()),
                "entropy_max": float(features[:, band_index].max()),
                "sampling_frequency_hz": band.sfreq,
                "roi_input_directory": str(band.directory),
                "input_metadata": band.metadata,
            })
            write_json(band_directory / "qc.json", graph_qc)
            band_reports[band_name] = graph_qc
        epochs.to_csv(state_directory / "epochs.csv", index=False)
        np.save(state_directory / "features.npy", features, allow_pickle=False)
        state_features.append(features)
        state_epochs.append(epochs)
        state_reports[state] = {
            "epoch_count": len(epochs),
            "bands": band_reports,
            "epochs_file": f"{state}/epochs.csv",
            "features_file": f"{state}/features.npy",
        }
    features = np.concatenate(state_features, axis=0)
    epochs = pd.concat(state_epochs, ignore_index=True)
    order = np.argsort(epochs["onset_seconds"].to_numpy(), kind="stable")
    features = features[order]
    epochs = epochs.iloc[order].reset_index(drop=True)
    epochs.to_csv(subject_directory / "epochs.csv", index=False)
    np.save(subject_directory / "features.npy", features, allow_pickle=False)
    report = {
        "subject": subject,
        "label": label,
        "fold": fold,
        "epoch_count": len(epochs),
        "feature_names": [f"SE_{band}" for band in arguments.bands],
        "states": state_reports,
        "upstream_fc_summary": input_summary,
    }
    write_json(subject_directory / "summary.json", report)
    return features, epochs, report


def save_feature_dataset(
    features: np.ndarray,
    epochs: pd.DataFrame,
    labels: pd.DataFrame,
    feature_names: Sequence[str],
    output_directory: Path,
) -> None:
    if features.shape != (len(epochs), len(feature_names)) or not np.isfinite(features).all():
        raise RuntimeError("Feature dimensions or numerical values are invalid")
    if epochs["epoch_id"].duplicated().any():
        raise RuntimeError("Epoch identifiers must be unique")
    if set(epochs["subject"]) != set(labels["subject"]):
        raise RuntimeError("Every selected subject must contribute at least one epoch")
    epochs.to_csv(output_directory / "epochs.csv", index=False)
    pd.concat([epochs.reset_index(drop=True), pd.DataFrame(features, columns=feature_names)], axis=1).to_csv(output_directory / "features.csv", index=False)
    np.save(output_directory / "features.npy", features, allow_pickle=False)
    np.savez_compressed(
        output_directory / "features.npz",
        X=features,
        y=epochs["label"].to_numpy(dtype=np.int64),
        subjects=epochs["subject"].to_numpy(dtype=str),
        epoch_ids=epochs["epoch_id"].to_numpy(dtype=str),
        states=epochs["state"].to_numpy(dtype=str),
        folds=epochs["fold"].to_numpy(dtype=np.int64),
        sample_indices=np.arange(len(epochs), dtype=np.int64),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    table = pd.DataFrame(features, columns=feature_names)
    table.insert(0, "subject", epochs["subject"].to_numpy())
    mean_features = table.groupby("subject", sort=True)[list(feature_names)].mean().reset_index()
    counts = epochs.groupby("subject").size().rename("epoch_count").reset_index()
    mean_features = labels.merge(counts, on="subject", validate="one_to_one").merge(mean_features, on="subject", validate="one_to_one")
    mean_features.to_csv(output_directory / "subject_mean_features.csv", index=False)
    write_json(output_directory / "feature_schema.json", {
        "feature_names": list(feature_names),
        "sample_unit": "one retained epoch",
        "X_shape": list(features.shape),
        "X_dtype": "float64",
        "normalisation": "graph spectral entropy divided by log(90); no cohort or fold feature scaling",
        "label_mapping": {"HC": 0, "ADHD": 1},
        "npz_keys": ["X", "y", "subjects", "epoch_ids", "states", "folds", "sample_indices", "feature_names"],
        "loading": "np.load(path, allow_pickle=False)",
        "epoch_metadata": "epochs.csv",
        "subject_means": "subject_mean_features.csv; separate aggregate for descriptive analysis",
    })


def main() -> int:
    arguments = parse_arguments()
    output_directory = None
    active_subject = None
    try:
        labels = load_labels(arguments.labels_file, arguments.subject_column, arguments.label_column, arguments.subject)
        upstream_summaries = []
        for subject in labels["subject"]:
            active_subject = subject
            fc_directory = arguments.sfc_dir / subject / "functional_connectivity"
            upstream_summaries.append(load_subject_metadata(fc_directory, subject))
            nodes = load_nodes(fc_directory / "nodes.tsv")
            if arguments.graph in ("SC", "SFC"):
                load_structural_matrix(arguments.sfc_dir / subject / "structural_connectivity", subject, nodes)
        resolve_upstream_configuration(arguments, upstream_summaries)
        configuration = configuration_record(arguments, labels)
        assignments = ensure_subject_folds(labels, arguments.se_dir / "_folds", arguments.seed)
        configuration["subject_fold_manifest"] = dict(assignments.attrs)
        output_directory = prepare_configuration_directory(arguments, configuration)
        labels.to_csv(output_directory / "subjects.csv", index=False)
        assignments.to_csv(output_directory / "subject_folds.csv", index=False)
        write_json(output_directory / "run_status.json", {
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "subject_count": len(labels),
        })
        all_features = []
        all_epochs = []
        records = []
        fold_by_subject = assignments.set_index("subject")["fold"].to_dict()
        for position, row in enumerate(labels.itertuples(index=False), start=1):
            active_subject = row.subject
            print(f"[{position}/{len(labels)}] {row.subject}", flush=True)
            features, epochs, report = extract_subject_features(
                row.subject, int(row.label), int(fold_by_subject[row.subject]), arguments, output_directory,
            )
            all_features.append(features)
            all_epochs.append(epochs)
            records.append({"subject": row.subject, "label": int(row.label), "fold": int(fold_by_subject[row.subject]), "epoch_count": len(epochs), "status": "complete"})
            pd.DataFrame(records).to_csv(output_directory / "subject_processing.csv", index=False)
        features = np.concatenate(all_features, axis=0)
        epochs = pd.concat(all_epochs, ignore_index=True)
        epochs.insert(0, "sample_index", np.arange(len(epochs)))
        feature_names = [f"SE_{band}" for band in arguments.bands]
        save_feature_dataset(features, epochs, labels, feature_names, output_directory)
        export_fold_features(features, epochs, assignments, feature_names, output_directory)
        write_json(output_directory / "summary.json", {
            "status": "complete",
            "subject_count": len(labels),
            "epoch_count": len(epochs),
            "feature_names": feature_names,
            "feature_shape": list(features.shape),
            "graph": arguments.graph,
            "condition": arguments.condition,
            "n_splits": 10,
            "subject_disjoint_folds": True,
            "classifier_trained": False,
            "features_file": "features.npz",
            "epoch_metadata": "epochs.csv",
            "subject_folds": "subject_folds.csv",
            "fold_directory": "folds",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": getattr(sys.modules.get("sklearn"), "__version__", "unknown")},
        })
        write_json(output_directory / "run_status.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
        print(f"Saved spectral entropy features and ten-fold datasets to {output_directory}", flush=True)
        return 0
    except Exception as error:
        if output_directory is not None:
            write_json(output_directory / "run_status.json", {
                "status": "failed", "subject": active_subject,
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
