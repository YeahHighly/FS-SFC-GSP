from __future__ import annotations

import argparse
import csv
import os
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

try:
    import mne
    from mne.coreg import Coregistration
    from mne.minimum_norm import (
        apply_inverse_raw,
        make_inverse_operator,
        write_inverse_operator,
    )
except ImportError:
    mne = None
    Coregistration = None
    apply_inverse_raw = None
    make_inverse_operator = None
    write_inverse_operator = None

try:
    import nibabel as nib
except ImportError:
    nib = None


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "Data"
SFC_DIRECTORY = PROJECT_DIRECTORY / "SFC"
DEFAULT_OUTPUT_NAME = "functional_connectivity"

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
CONDITIONS = ("EO", "EC", "ALL")
CONNECTIVITY_METHODS = ("pearson", "wpli")
FC_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EEGInputs:
    eeg_set: Path
    events_tsv: Path
    channels_tsv: Path
    eeg_json: Path


@dataclass(frozen=True)
class StructuralInputs:
    atlas_t1: Path
    t1_brain: Path
    nodes_tsv: Path
    subjects_dir: Path
    fs_subject_dir: Path
    summary_json: Path
    scalp_surface: Path
    scalp_surface_low_res: Path


@dataclass(frozen=True)
class Segment:
    condition: str
    source_onset: float
    source_offset: float
    onset: float
    offset: float
    start_sample: int
    stop_sample: int

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @property
    def sample_count(self) -> int:
        return self.stop_sample - self.start_sample


def load_and_validate_nodes(nodes_tsv: Path) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_tsv, sep="\t")
    required_columns = {"index", "name"}
    missing = required_columns.difference(nodes.columns)
    if missing:
        raise RuntimeError(f"nodes.tsv is missing required columns: {sorted(missing)}")
    nodes = nodes.sort_values("index").reset_index(drop=True)
    indices = nodes["index"].astype(int).tolist()
    if indices != list(range(1, 91)):
        raise RuntimeError("nodes.tsv indices must be consecutive integers from 1 to 90")
    if nodes["name"].astype(str).duplicated().any():
        raise RuntimeError("nodes.tsv contains duplicate brain region names")
    return nodes


def _robust_bad_channels(raw: Any, z_threshold: float) -> pd.DataFrame:
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    data = raw.get_data(picks=eeg_picks)
    sfreq = float(raw.info["sfreq"])
    sos = signal.butter(4, (1.0, 45.0), btype="bandpass", fs=sfreq, output="sos")
    finite_input = np.isfinite(data).all(axis=1)
    if not finite_input.any():
        raise RuntimeError("No EEG channels contain entirely finite samples")
    standard_deviation = np.full(data.shape[0], np.nan, dtype=float)
    filtered = signal.sosfiltfilt(sos, data[finite_input], axis=1)
    standard_deviation[finite_input] = np.std(filtered, axis=1)
    finite = finite_input & np.isfinite(standard_deviation)
    usable = finite & (standard_deviation > 0.0)
    positive = standard_deviation[usable]
    typical = float(np.median(positive)) if positive.size else 0.0
    robust_z = np.full(data.shape[0], np.nan, dtype=float)
    if positive.size:
        log_values = np.log(positive)
        median = float(np.median(log_values))
        mad = float(np.median(np.abs(log_values - median)))
        denominator = max(1.4826 * mad, np.finfo(float).eps)
        robust_z[usable] = (log_values - median) / denominator
    flat = finite & (standard_deviation < max(1e-12, typical * 1e-4))
    outlier = np.abs(robust_z) > z_threshold
    nonfinite = ~finite
    bad = flat | outlier | nonfinite
    names = np.asarray([raw.ch_names[index] for index in eeg_picks])
    reason = np.full(len(names), "", dtype=object)
    reason[flat] = "flat"
    reason[outlier & ~flat] = "variance_outlier"
    reason[nonfinite] = "nonfinite"
    return pd.DataFrame(
        {
            "channel": names,
            "filtered_std_uV": standard_deviation * 1e6,
            "robust_log_std_z": robust_z,
            "is_bad": bad,
            "reason": reason,
        }
    )


def _median_psd(
    raw: Any, maximum_frequency: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    data = raw.get_data(picks="eeg", reject_by_annotation="omit")
    sfreq = float(raw.info["sfreq"])
    segment_samples = min(data.shape[1], max(256, int(round(4.0 * sfreq))))
    frequencies, power = signal.welch(
        data,
        fs=sfreq,
        nperseg=segment_samples,
        noverlap=segment_samples // 2,
        axis=1,
    )
    keep = frequencies <= maximum_frequency
    return frequencies[keep], np.median(power[:, keep], axis=0)


def prepare_atlas_mgz(
    atlas_nifti: Path,
    fs_brain_mgz: Path,
    output_mgz: Path,
) -> tuple[Any, Any]:
    atlas_image = nib.load(str(atlas_nifti))
    atlas_data = np.asarray(atlas_image.dataobj)
    if atlas_data.ndim != 3 or not np.isfinite(atlas_data).all():
        raise RuntimeError("The individual MRI atlas must be a finite three-dimensional label image.")
    rounded = np.rint(atlas_data)
    if not np.allclose(atlas_data, rounded, atol=1e-4):
        raise RuntimeError("AAL-90 NIfTI contains noninteger labels")
    labels = set(np.unique(rounded.astype(np.int32)).tolist())
    if not set(range(1, 91)).issubset(labels):
        missing = sorted(set(range(1, 91)).difference(labels))
        raise RuntimeError(f"AAL-90 NIfTI is missing labels: {missing}")

    if labels.difference(set(range(91))):
        raise RuntimeError("The individual MRI atlas contains labels outside 0 through 90.")
    brain_image = nib.load(str(fs_brain_mgz))
    if not np.allclose(atlas_image.affine, brain_image.affine, atol=1e-3, rtol=0.0):
        raise RuntimeError("The AAL atlas and FreeSurfer brain do not share the same scanner-RAS affine.")
    if tuple(atlas_image.shape[:3]) != tuple(brain_image.shape[:3]):
        raise RuntimeError(
            "AAL-90 T1 atlas dimensions do not match FreeSurfer brain.mgz: "
            f"{atlas_image.shape[:3]} vs {brain_image.shape[:3]}"
        )
    atlas_mgh = nib.MGHImage(rounded.astype(np.uint8), atlas_image.affine)
    nib.save(atlas_mgh, str(output_mgz))
    written = nib.load(str(output_mgz))
    atlas_tkr = written.header.get_vox2ras_tkr()
    brain_tkr = brain_image.header.get_vox2ras_tkr()
    if not np.allclose(atlas_tkr, brain_tkr, atol=1e-3):
        raise RuntimeError(
            "The converted AAL-90 MGZ and FreeSurfer MRI surface-RAS grids do not match"
        )
    return written, brain_image


def _plot_coregistration_points(
    raw: Any,
    sphere: Any,
    output_path: Path,
) -> None:
    eeg_points = np.array(
        [
            point["r"]
            for point in raw.info["dig"]
            if int(point["kind"]) == int(mne.io.constants.FIFF.FIFFV_POINT_EEG)
        ]
    )
    figure = plt.figure(figsize=(7, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        eeg_points[:, 0] * 1000,
        eeg_points[:, 1] * 1000,
        eeg_points[:, 2] * 1000,
        s=9,
        c=eeg_points[:, 2],
        cmap="coolwarm",
    )
    center = np.asarray(sphere["r0"]) * 1000
    radius = float(sphere["layers"][-1]["rad"]) * 1000
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 24)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(x, y, z, color="gray", alpha=0.12, linewidth=0.4)
    axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    axis.set_box_aspect((1, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_source_model_and_roi_timeseries(
    raw: Any,
    structural: StructuralInputs,
    nodes: pd.DataFrame,
    output_directory: Path,
    subject: str,
    source_spacing_mm: float,
    snr: float,
    inverse_method: str,
    n_jobs: int,
    coreg_icp_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_directory = output_directory / "model"
    qc_directory = output_directory / "qc"
    atlas_mgz = model_directory / "aal90_t1.mgz"
    prepare_atlas_mgz(
        structural.atlas_t1,
        structural.fs_subject_dir / "mri" / "brain.mgz",
        atlas_mgz,
    )

    coreg = Coregistration(
        raw.info,
        subject,
        subjects_dir=structural.subjects_dir,
        fiducials="estimated",
        on_defects="warn",
    )
    coreg.fit_fiducials(verbose="error")
    coreg.fit_icp(
        n_iterations=coreg_icp_iterations,
        hsp_weight=0.0,
        eeg_weight=1.0,
        verbose="error",
    )
    trans_path = model_directory / f"{subject}-head_to_mri-trans.fif"
    mne.write_trans(trans_path, coreg.trans, overwrite=True, verbose="error")
    distances = np.asarray(coreg.compute_dig_mri_distances()) * 1000.0

    sphere = mne.make_sphere_model(
        r0="auto", head_radius="auto", info=raw.info, verbose="error"
    )
    _plot_coregistration_points(
        raw, sphere, qc_directory / "eeg_montage_and_fitted_sphere.png"
    )
    coreg_metrics = {
        "coordinate_source": "electrode coordinates from the EEGLAB montage; individual digitisation not independently verified",
        "coregistration_method": "estimated MRI fiducials + EEG-to-scalp ICP",
        "conductor_model": "fitted four-shell sphere",
        "dig_to_scalp_distance_mm": {
            "count": int(distances.size),
            "median": float(np.median(distances)) if distances.size else None,
            "mean": float(np.mean(distances)) if distances.size else None,
            "maximum": float(np.max(distances)) if distances.size else None,
        },
        "manual_coregistration_review_required": True,
    }
    (qc_directory / "coregistration_metrics.json").write_text(
        json.dumps(coreg_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    head_center = np.asarray(sphere["r0"], dtype=float)
    mri_center = mne.transforms.apply_trans(coreg.trans, head_center)
    inner_skull_radius = float(sphere["layers"][0]["rad"])
    source_radius = inner_skull_radius - 0.003
    if source_radius <= 0:
        raise RuntimeError("The automatically fitted spherical head model has an invalid inner-skull radius")
    volume_labels = {str(row["name"]): int(row["index"]) for _, row in nodes.iterrows()}
    volume_src = mne.setup_volume_source_space(
        subject=subject,
        pos=source_spacing_mm,
        mri=atlas_mgz,
        sphere=(*mri_center.tolist(), source_radius),
        bem=None,
        surface=None,
        mindist=0.0,
        exclude=0.0,
        subjects_dir=structural.subjects_dir,
        volume_label=volume_labels,
        add_interpolator=False,
        sphere_units="m",
        single_volume=True,
        n_jobs=n_jobs,
        verbose="error",
    )
    if len(volume_src) != 1:
        raise RuntimeError(f"Expected one joint AAL volume source space, found {len(volume_src)}")


    atlas_image = nib.load(atlas_mgz)
    if not hasattr(atlas_image.header, "get_vox2ras_tkr"):
        raise RuntimeError("The AAL MGZ header is missing the vox2ras_tkr transform")
    volume_vertices = np.asarray(volume_src[0]["vertno"], dtype=int)
    active_rr = np.asarray(volume_src[0]["rr"])[volume_vertices]
    active_ras_mm = active_rr * 1000.0
    ras_to_voxel = np.linalg.inv(atlas_image.header.get_vox2ras_tkr())
    active_voxels = np.rint(
        nib.affines.apply_affine(ras_to_voxel, active_ras_mm)
    ).astype(int)
    atlas_shape = np.asarray(atlas_image.shape[:3], dtype=int)
    if ((active_voxels < 0) | (active_voxels >= atlas_shape)).any():
        raise RuntimeError("AAL volume source points lie outside the parcellation image bounds")
    atlas_data = np.asarray(atlas_image.dataobj)
    active_labels = atlas_data[
        active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]
    ].astype(int)
    roi_source_rows = {
        int(index): np.flatnonzero(active_labels == int(index))
        for index in nodes["index"]
    }
    expected_names = nodes["name"].astype(str).tolist()
    source_counts = pd.DataFrame(
        {
            "index": nodes["index"].astype(int),
            "name": expected_names,
            "source_count": [
                int(roi_source_rows[int(index)].size) for index in nodes["index"]
            ],
        }
    )
    source_counts.to_csv(
        qc_directory / "source_counts_per_roi.tsv", sep="\t", index=False
    )
    if (source_counts["source_count"] <= 0).any():
        empty = source_counts.loc[source_counts["source_count"] <= 0, "name"].tolist()
        raise RuntimeError(
            f"The following AAL regions contain no source points: {empty}. Reduce --source-spacing-mm"
        )


    normals = active_rr - mri_center
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > np.finfo(float).eps
    normals[nonzero] /= lengths[nonzero, np.newaxis]
    normals[~nonzero] = (0.0, 0.0, 1.0)
    src = mne.setup_volume_source_space(
        subject=subject,
        pos={"rr": active_rr, "nn": normals},
        mri=None,
        sphere=(*mri_center.tolist(), source_radius),
        bem=None,
        surface=None,
        mindist=0.0,
        exclude=0.0,
        subjects_dir=structural.subjects_dir,
        sphere_units="m",
        n_jobs=n_jobs,
        verbose="error",
    )
    active_vertices = np.asarray(src[0]["vertno"], dtype=int)
    discrete_rr = np.asarray(src[0]["rr"])[active_vertices]
    if discrete_rr.shape != active_rr.shape or not np.allclose(
        discrete_rr, active_rr, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("The discrete source space changed the ordering of AAL source points")


    np.savez_compressed(
        model_directory / "roi_source_mapping.npz",
        roi_indices=nodes["index"].to_numpy(dtype=np.int32),
        source_vertices=active_vertices,
        source_positions_mri_m=active_rr,
        source_normals_mri=normals,
        source_atlas_labels=active_labels,
        source_atlas_voxels=active_voxels,
    )
    for source_space in src:
        mne.transform_surface_to(source_space, "head", coreg.trans, copy=False)
    src.save(model_directory / f"{subject}-aal90-discrete-src.fif", overwrite=True)

    forward = mne.make_forward_solution(
        raw.info,
        trans=None,
        src=src,
        bem=sphere,
        meg=False,
        eeg=True,
        mindist=0.0,
        n_jobs=n_jobs,
        verbose="error",
    )
    forward = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=True,
        copy=False,
        use_cps=False,
        verbose="error",
    )
    mne.write_forward_solution(
        model_directory / f"{subject}-aal90-eeg-fwd.fif",
        forward,
        overwrite=True,
        verbose="error",
    )
    covariance = mne.make_ad_hoc_cov(raw.info, verbose="error")
    inverse = make_inverse_operator(
        raw.info,
        forward,
        covariance,
        loose=0.0,
        depth=None,
        fixed=True,
        rank="info",
        verbose="error",
    )
    write_inverse_operator(
        model_directory / f"{subject}-aal90-eeg-inv.fif",
        inverse,
        overwrite=True,
        verbose="error",
    )
    source_estimate = apply_inverse_raw(
        raw,
        inverse,
        lambda2=1.0 / (snr * snr),
        method=inverse_method,
        pick_ori=None,
        buffer_size=max(256, round(raw.info["sfreq"] * 8.0)),
        verbose="error",
    )

    if len(source_estimate.vertices) != 1 or not np.array_equal(
        source_estimate.vertices[0], active_vertices
    ):
        vertex_shapes = [np.asarray(item).shape for item in source_estimate.vertices]
        raise RuntimeError(
            "Internal error: source-estimate vertices do not match the AAL source-point ordering; "
            f"estimate vertices={vertex_shapes}, src vertices={active_vertices.shape}"
        )
    roi_time_series = np.vstack(
        [
            np.mean(source_estimate.data[roi_source_rows[int(index)]], axis=0)
            for index in nodes["index"]
        ]
    ).astype(np.float32)
    model_summary = {
        **coreg_metrics,
        "source_space": "individual-MRI joint AAL-90 volume grid",
        "source_spacing_mm": source_spacing_mm,
        "source_count_total": int(source_counts["source_count"].sum()),
        "source_count_min_per_roi": int(source_counts["source_count"].min()),
        "source_count_max_per_roi": int(source_counts["source_count"].max()),
        "source_orientation": "fixed outward radial; signed",
        "roi_aggregation": "mean signed source estimate within each individual-MRI AAL90 parcel",
        "roi_mapping_file": "model/roi_source_mapping.npz",
        "time_series_space": "individual MRI AAL90 source-space ROI",
        "inverse_method": inverse_method,
        "inverse_snr": snr,
    }
    return roi_time_series, model_summary


def _bandpass_segment(
    data: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1).astype(np.float32)


def _write_segments_table(
    path: Path, segments: Sequence[Segment], sfreq: float
) -> None:
    concatenated_start = 0
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, start=1):
        concatenated_stop = concatenated_start + segment.sample_count
        rows.append(
            {
                "segment": segment_index,
                "state": segment.condition,
                "source_onset_s": segment.source_onset,
                "source_offset_s": segment.source_offset,
                "trimmed_onset_s": segment.onset,
                "trimmed_offset_s": segment.offset,
                "duration_s": segment.sample_count / sfreq,
                "original_start_sample": segment.start_sample,
                "original_stop_sample_exclusive": segment.stop_sample,
                "concatenated_start_sample": concatenated_start,
                "concatenated_stop_sample_exclusive": concatenated_stop,
            }
        )
        concatenated_start = concatenated_stop
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _plot_timeseries_preview(
    data: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    condition: str,
    band: str,
    output_path: Path,
) -> None:
    preview_samples = min(data.shape[1], int(round(20.0 * sfreq)))
    preview = np.array(data[:10, :preview_samples], dtype=float, copy=True)
    preview -= preview.mean(axis=1, keepdims=True)
    scale = np.median(np.std(preview, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    preview /= scale
    offsets = np.arange(10)[::-1] * 6.0
    time = np.arange(preview_samples) / sfreq
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for index in range(10):
        axis.plot(time, preview[index] + offsets[index], linewidth=0.55)
    axis.set_yticks(offsets, names[:10])
    axis.set(
        xlabel="Concatenated time (s; first 20 s shown)",
        title=f"{condition} | {band} | first 10 AAL ROI time series",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def preflight_report(
    eeg_inputs: EEGInputs,
    structural: StructuralInputs,
    nodes: pd.DataFrame,
) -> dict[str, Any]:
    eeg_metadata = json.loads(eeg_inputs.eeg_json.read_text(encoding="utf-8-sig"))
    events = load_and_validate_events(
        eeg_inputs.events_tsv, float(eeg_metadata["RecordingDuration"])
    )
    channels = pd.read_csv(eeg_inputs.channels_tsv, sep="\t")
    return {
        "status": "preflight_pass",
        "eeg_set": str(eeg_inputs.eeg_set),
        "events_tsv": str(eeg_inputs.events_tsv),
        "eeg_channels": int(len(channels)),
        "sampling_frequency_hz": float(eeg_metadata["SamplingFrequency"]),
        "recording_duration_seconds": float(eeg_metadata["RecordingDuration"]),
        "eyes_open_events": int(
            (events["value"].astype(str) == "instructed_toOpenEyes").sum()
        ),
        "eyes_closed_events": int(
            (events["value"].astype(str) == "instructed_toCloseEyes").sum()
        ),
        "aal_atlas_t1": str(structural.atlas_t1),
        "aal_nodes": int(len(nodes)),
        "subjects_dir": str(structural.subjects_dir),
        "freesurfer_subject": str(structural.fs_subject_dir),
        "scalp_surface": str(structural.scalp_surface),
        "scalp_surface_present": structural.scalp_surface.is_file(),
        "scalp_surface_low_res": str(structural.scalp_surface_low_res),
        "scalp_surface_low_res_present": structural.scalp_surface_low_res.is_file(),
        "mne_version": mne.__version__,
        "nibabel_version": nib.__version__,
        "coordinate_warning": (
            "EEGLAB montage coordinates; individual digitisation not independently verified"
        ),
    }

def _bad_annotation_intervals(raw: Any) -> list[tuple[float, float]]:
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    intervals: list[tuple[float, float]] = []
    for annotation in raw.annotations:
        if not str(annotation["description"]).lower().startswith("bad"):
            continue
        start = float(annotation["onset"]) - float(raw.first_time)
        stop = start + max(float(annotation["duration"]), 1.0 / sfreq)
        start, stop = max(0.0, start), min(duration, stop)
        if math.isfinite(start) and math.isfinite(stop) and stop > start:
            intervals.append((start, stop))
    return intervals


def _write_processing_status(output_directory: Path, subject: str, status: str, stage: str, error: str = "") -> None:
    record = {
        "subject": subject,
        "status": status,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    path = output_directory / "run_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output_directory(output_directory: Path, overwrite: bool) -> None:
    resolved = output_directory.resolve()
    expected_root = SFC_DIRECTORY.resolve()
    if output_directory.is_symlink() or not resolved.is_relative_to(expected_root) or resolved.name != DEFAULT_OUTPUT_NAME:
        raise RuntimeError("The functional-connectivity output must be a subject directory within SFC.")
    if output_directory.exists() and any(output_directory.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output already exists: {output_directory}. Use --overwrite to replace it.")
        shutil.rmtree(resolved)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "model").mkdir(exist_ok=True)
    (output_directory / "qc").mkdir(exist_ok=True)


def process_subject(arguments: argparse.Namespace, subject: str) -> dict[str, Any]:
    structural_output = SFC_DIRECTORY / subject / "structural_connectivity"
    output_directory = SFC_DIRECTORY / subject / DEFAULT_OUTPUT_NAME
    eeg_inputs = discover_eeg_inputs(
        arguments.data_dir / "EEG" / subject,
        session=arguments.eeg_session,
        eeg_set_name=arguments.eeg_set_name,
    )
    structural = discover_structural_inputs(
        structural_output, subject, arguments.subjects_dir,
    )
    nodes = load_and_validate_nodes(structural.nodes_tsv)
    if arguments.preflight:
        report = preflight_report(eeg_inputs, structural, nodes)
        report.update({
            "fc_schema_version": FC_SCHEMA_VERSION,
            "subject": subject,
            "status": "preflight_pass",
            "output_directory": str(output_directory),
            "time_series_space": "individual MRI AAL90 source-space ROI",
            "connectivity_methods": arguments.connectivity_methods,
            "primary_connectivity": arguments.connectivity_methods[0],
            "conditions": list(CONDITIONS),
            "epoch_seconds": arguments.epoch_seconds,
            "fc_estimation_unit": "epoch",
        })
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    _prepare_output_directory(output_directory, arguments.overwrite)
    logger = logging.getLogger("hbn_functional_connectivity")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output_directory / "processing.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    stage = "initialization"
    try:
        _write_processing_status(output_directory, subject, "running", stage)
        shutil.copy2(structural.nodes_tsv, output_directory / "nodes.tsv")
        configuration = {
            "fc_schema_version": FC_SCHEMA_VERSION,
            "subject": subject,
            "eeg_set": str(eeg_inputs.eeg_set),
            "events_tsv": str(eeg_inputs.events_tsv),
            "channels_tsv": str(eeg_inputs.channels_tsv),
            "structural_output": str(structural_output),
            "atlas_in_individual_t1_space": str(structural.atlas_t1),
            "freesurfer_subject_directory": str(structural.fs_subject_dir),
            "connectivity_methods": arguments.connectivity_methods,
            "primary_connectivity": arguments.connectivity_methods[0],
            "conditions": list(CONDITIONS),
            "epoch_seconds": arguments.epoch_seconds,
            "fc_estimation_unit": "epoch",
            "analysis_sfreq": arguments.analysis_sfreq,
            "source_spacing_mm": arguments.source_spacing_mm,
            "snr": arguments.snr,
            "inverse_method": arguments.inverse_method,
            "transition_trim": arguments.transition_trim,
            "min_segment_duration": arguments.min_segment_duration,
            "bad_channel_z": arguments.bad_channel_z,
            "coreg_icp_iterations": arguments.coreg_icp_iterations,
        }
        (output_directory / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        stage = "eeg_preprocessing"
        _write_processing_status(output_directory, subject, "running", stage)
        logger.info("Preprocessing EEG for individual-MRI source reconstruction: %s", eeg_inputs.eeg_set)
        raw, bad_table = preprocess_eeg(
            eeg_inputs, arguments.analysis_sfreq, arguments.n_jobs,
            arguments.bad_channel_z, output_directory / "qc",
        )
        sfreq = float(raw.info["sfreq"])
        events = load_and_validate_events(eeg_inputs.events_tsv, raw.n_times / sfreq)
        bad_intervals = _bad_annotation_intervals(raw)
        segments_by_condition = create_condition_segments(
            events, sfreq, raw.n_times, arguments.transition_trim,
            arguments.min_segment_duration, bad_intervals=bad_intervals,
        )
        pd.DataFrame(bad_intervals, columns=["onset_seconds", "offset_seconds"]).to_csv(
            output_directory / "qc" / "excluded_annotation_intervals.tsv", sep="\t", index=False,
        )
        all_segments = [
            {
                "condition": segment.condition,
                "source_onset_s": segment.source_onset,
                "source_offset_s": segment.source_offset,
                "trimmed_onset_s": segment.onset,
                "trimmed_offset_s": segment.offset,
                "duration_s": segment.duration,
            }
            for segment in segments_by_condition["ALL"]
        ]
        pd.DataFrame(all_segments).to_csv(output_directory / "qc" / "condition_segments.tsv", sep="\t", index=False)
        stage = "mri_source_reconstruction"
        _write_processing_status(output_directory, subject, "running", stage)
        logger.info("Reconstructing signed source estimates in the individual-MRI AAL90 space")
        roi_time_series, model_summary = build_source_model_and_roi_timeseries(
            raw, structural, nodes, output_directory, subject,
            arguments.source_spacing_mm, arguments.snr, arguments.inverse_method,
            arguments.n_jobs, arguments.coreg_icp_iterations,
        )
        if roi_time_series.shape != (90, raw.n_times) or not np.isfinite(roi_time_series).all():
            raise RuntimeError(f"Invalid individual-MRI ROI time series: {roi_time_series.shape}")
        np.save(output_directory / "model" / "roi_timeseries_broadband.npy", roi_time_series, allow_pickle=False)
        np.save(output_directory / "model" / "sample_times_seconds.npy", np.arange(raw.n_times) / sfreq, allow_pickle=False)
        stage = "roi_functional_connectivity"
        _write_processing_status(output_directory, subject, "running", stage)
        logger.info("Computing %s on %.6g-second epochs within EO, EC and ALL ROI signals", arguments.connectivity_methods, arguments.epoch_seconds)
        derivatives = export_condition_band_results(
            roi_time_series, segments_by_condition, sfreq, nodes, output_directory,
            arguments.connectivity_methods,
            epoch_seconds=arguments.epoch_seconds,
        )
        eeg_metadata = json.loads(eeg_inputs.eeg_json.read_text(encoding="utf-8-sig"))
        summary = {
            "fc_schema_version": FC_SCHEMA_VERSION,
            "subject": subject,
            "status": "complete",
            "processing_status": "complete",
            "output_directory": str(output_directory),
            "eeg_input": str(eeg_inputs.eeg_set),
            "events_input": str(eeg_inputs.events_tsv),
            "structural_output": str(structural_output),
            "aal_parcellation": str(structural.atlas_t1),
            "roi_count": 90,
            "time_series_space": "individual MRI AAL90 source-space ROI",
            "connectivity_input": "signed source estimates averaged within individual-MRI anatomical parcels",
            "sensor_space_connectivity": False,
            "roi_timeseries_broadband": "model/roi_timeseries_broadband.npy",
            "input_sampling_frequency_hz": eeg_metadata.get("SamplingFrequency"),
            "analysis_sampling_frequency_hz": sfreq,
            "input_channel_count": len(raw.ch_names),
            "bad_channels": bad_table.loc[bad_table["is_bad"], "channel"].astype(str).tolist(),
            "reference": "average-reference projection",
            "frequency_bands_hz": {name: list(values) for name, values in BANDS.items()},
            "conditions": list(CONDITIONS),
            "epoch_seconds": arguments.epoch_seconds,
            "epoch_samples": int(round(arguments.epoch_seconds * sfreq)),
            "fc_estimation_unit": "epoch",
            "epoch_overlap_seconds": 0.0,
            "epoch_artifact_rejection": "none",
            "epoch_remainder_policy": "discard incomplete epoch tails separately within each retained continuous segment",
            "condition_definition": "EO/EC state intervals split at boundaries and BAD annotations, trimmed at each edge; ALL is their chronological union",
            "combined_condition_connectivity": "estimate ALL connectivity from the chronological union of the complete EO and EC epochs, keeping their actual state labels in epochs.tsv",
            "transition_trim_seconds_each_edge": arguments.transition_trim,
            "connectivity_methods": arguments.connectivity_methods,
            "primary_connectivity": arguments.connectivity_methods[0],
            "pearson_negative_policy": "equal-weight Fisher-z mean of signed correlations across fixed-length epochs, inverse transform, then set nonpositive values to zero",
            "wpli_aggregation": "compute Hilbert analytic signals independently per fixed-length epoch; pool signed and absolute imaginary cross-spectrum sums across epochs before taking the ratio",
            "connectivity_matrices": "full nonnegative symmetric raw matrices and matrices normalised by their maximum off-diagonal raw weight; zero diagonal",
            "model": model_summary,
            "derivatives": derivatives,
            "manual_qc_required": True,
            "scientific_limitations": [
                "Electrode coordinates are read from the EEGLAB montage; individual digitisation has not been independently verified.",
                "The fitted spherical conductor is an approximation to the individual head anatomy.",
                "Source-space ROI signals are inverse-model estimates and remain susceptible to source leakage.",
            ],
            "software": {
                "python": sys.version.split()[0], "mne": mne.__version__,
                "numpy": np.__version__, "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
                "nibabel": nib.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
            },
        }
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_processing_status(output_directory, subject, "complete", "complete")
        logger.info("Functional-connectivity results saved to %s", output_directory)
        return {
            "fc_schema_version": FC_SCHEMA_VERSION,
            "subject": subject,
            "status": "complete",
            "output_directory": str(output_directory),
            "summary_json": str(output_directory / "summary.json"),
            "connectivity_methods": arguments.connectivity_methods,
            "primary_connectivity": arguments.connectivity_methods[0],
            "conditions": list(CONDITIONS),
            "epoch_seconds": arguments.epoch_seconds,
            "fc_estimation_unit": "epoch",
        }
    except Exception as error:
        _write_processing_status(output_directory, subject, "failed", stage, str(error))
        logger.exception("Functional-connectivity processing failed")
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def _require_dependencies() -> None:
    missing = [name for name, module in (("mne", mne), ("nibabel", nib)) if module is None]
    if missing:
        raise RuntimeError("Missing Python dependencies: " + ", ".join(missing))


def _eeg_bids_entities(filename: str, suffix: str) -> dict[str, str]:
    if not filename.endswith("_" + suffix):
        return {}
    parts = filename[:-(len(suffix) + 1)].split("_")
    entities = {}
    for part in parts:
        if "-" not in part:
            return {}
        key, value = part.split("-", 1)
        if not key or not value or key in entities:
            return {}
        entities[key] = value
    return entities


def _eeg_sidecar(eeg_set: Path, suffix: str) -> Path:
    prefix = eeg_set.name.removesuffix("_eeg.set")
    exact = eeg_set.with_name(f"{prefix}_{suffix}")
    if exact.is_file():
        if exact.stat().st_size == 0:
            raise RuntimeError(f"EEG sidecar is empty: {exact}")
        return exact
    recording_entities = _eeg_bids_entities(eeg_set.name, "eeg.set")
    candidates = []
    for path in sorted(eeg_set.parent.glob(f"*_{suffix}")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        entities = _eeg_bids_entities(path.name, suffix)
        if entities and all(recording_entities.get(key) == value for key, value in entities.items()):
            candidates.append((len(entities), path))
    if candidates:
        specificity = max(score for score, _ in candidates)
        best = [path for score, path in candidates if score == specificity]
        if len(best) == 1:
            return best[0]
    rendered = "\n".join(str(path) for _, path in candidates) or "(none)"
    raise RuntimeError(f"No unique {suffix} sidecar associated with {eeg_set.name}:\n{rendered}")


def discover_eeg_inputs(
    eeg_directory: Path,
    session: str | None = None,
    eeg_set_name: str | None = None,
) -> EEGInputs:
    if not eeg_directory.is_dir():
        raise RuntimeError(f"EEG subject directory does not exist: {eeg_directory}")
    if eeg_directory.name == "eeg":
        directories = [eeg_directory]
    else:
        directories = [eeg_directory / "eeg"]
        directories.extend(path / "eeg" for path in sorted(eeg_directory.glob("ses-*")) if path.is_dir())
    if session is not None:
        label = session.removeprefix("ses-")
        if not label or not label.isascii() or not label.isalnum():
            raise RuntimeError(f"Invalid EEG session label: {session!r}")
        directories = [path for path in directories if path.parent.name == f"ses-{label}"]
    if eeg_set_name is not None:
        if not eeg_set_name or Path(eeg_set_name).name != eeg_set_name or "/" in eeg_set_name or "\\" in eeg_set_name:
            raise RuntimeError("EEG SET selection must be an exact filename")
    matches = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_eeg.set")):
            entities = _eeg_bids_entities(path.name, "eeg.set")
            if entities.get("task", "").casefold() != "restingstate":
                continue
            if eeg_set_name is not None and path.name != eeg_set_name:
                continue
            if not path.is_file() or path.stat().st_size == 0:
                continue
            subject_directory = next((parent for parent in directory.parents if parent.name.startswith("sub-")), None)
            if subject_directory is not None and entities.get("sub") != subject_directory.name[4:]:
                raise RuntimeError(f"EEG filename subject differs from its subject directory: {path}")
            if directory.parent.name.startswith("ses-") and entities.get("ses") != directory.parent.name[4:]:
                raise RuntimeError(f"EEG filename session differs from its session directory: {path}")
            matches.append(path)
    if len(matches) != 1:
        rendered = "\n".join(f"  - {path}" for path in matches) or "  (none)"
        raise RuntimeError(f"Expected one RestingState EEG recording; found {len(matches)}. Select its session and exact SET filename:\n{rendered}")
    eeg_set = matches[0]
    return EEGInputs(
        eeg_set=eeg_set,
        events_tsv=_eeg_sidecar(eeg_set, "events.tsv"),
        channels_tsv=_eeg_sidecar(eeg_set, "channels.tsv"),
        eeg_json=_eeg_sidecar(eeg_set, "eeg.json"),
    )


def discover_structural_inputs(
    structural_output: Path,
    subject: str,
    subjects_dir_override: Path | None,
) -> StructuralInputs:
    summary_json = structural_output / "summary.json"
    atlas_t1 = structural_output / "qc" / "nodes_in_t1_space.nii.gz"
    t1_brain = structural_output / "qc" / "freesurfer_brain_t1.nii.gz"
    nodes_tsv = structural_output / "nodes.tsv"
    for path in (summary_json, atlas_t1, t1_brain, nodes_tsv):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required structural-connectivity output is missing: {path}")
    summary = json.loads(summary_json.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, dict) or summary.get("processing_status") != "complete":
        raise RuntimeError("Structural-connectivity summary does not report processing_status=complete")
    if summary.get("subject") != subject or summary.get("freesurfer_subject_id", subject) != subject:
        raise RuntimeError("Structural-connectivity subject identifiers do not match the requested EEG subject")
    if not subject or Path(subject).name != subject or "/" in subject or "\\" in subject or subject in (".", ".."):
        raise RuntimeError("Invalid subject identifier")
    subjects_dir = subjects_dir_override.resolve() if subjects_dir_override is not None else structural_output / "intermediates" / "freesurfer"
    fs_subject_dir = subjects_dir / subject
    scalp_surface = fs_subject_dir / "surf" / "lh.seghead"
    scalp_surface_low_res = fs_subject_dir / "bem" / "outer_skin.surf"
    required = (
        fs_subject_dir / "mri" / "orig.mgz",
        fs_subject_dir / "mri" / "brain.mgz",
        fs_subject_dir / "mri" / "transforms" / "talairach.xfm",
        scalp_surface,
        scalp_surface_low_res,
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Required local FreeSurfer outputs are missing; complete structural connectivity first:\n{rendered}")
    return StructuralInputs(
        atlas_t1=atlas_t1,
        t1_brain=t1_brain,
        nodes_tsv=nodes_tsv,
        subjects_dir=subjects_dir,
        fs_subject_dir=fs_subject_dir,
        summary_json=summary_json,
        scalp_surface=scalp_surface,
        scalp_surface_low_res=scalp_surface_low_res,
    )


def load_and_validate_events(events_tsv: Path, recording_duration: float) -> pd.DataFrame:
    if not math.isfinite(recording_duration) or recording_duration <= 0:
        raise RuntimeError("EEG recording duration must be finite and positive")
    events = pd.read_csv(events_tsv, sep="\t")
    events.columns = events.columns.str.strip()
    if events.columns.duplicated().any():
        raise RuntimeError("events.tsv contains duplicate column names")
    missing = {"onset", "value"}.difference(events.columns)
    if missing:
        raise RuntimeError(f"events.tsv is missing columns: {sorted(missing)}")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    onset = events["onset"].to_numpy(dtype=float)
    if not np.isfinite(onset).all() or (onset < 0).any() or (onset > recording_duration + 1.0).any():
        raise RuntimeError("events.tsv contains nonfinite or out-of-range onset times")
    events["value"] = events["value"].fillna("").astype(str).str.strip()
    required_values = {"instructed_toOpenEyes", "instructed_toCloseEyes"}
    if not required_values.issubset(set(events["value"])):
        raise RuntimeError("events.tsv must explicitly identify both instructed_toOpenEyes and instructed_toCloseEyes")
    states = events[events["value"].isin(required_values)]
    if (states.groupby("onset")["value"].nunique() > 1).any():
        raise RuntimeError("Conflicting EO and EC state events share the same onset")
    return events.sort_values("onset", kind="stable").reset_index(drop=True)


def _subtract_bad_intervals(
    start: float,
    stop: float,
    bad_intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    pieces = [(start, stop)]
    for bad_start, bad_stop in bad_intervals:
        remaining = []
        for piece_start, piece_stop in pieces:
            if bad_stop <= piece_start or bad_start >= piece_stop:
                remaining.append((piece_start, piece_stop))
                continue
            if piece_start < bad_start:
                remaining.append((piece_start, min(bad_start, piece_stop)))
            if bad_stop < piece_stop:
                remaining.append((max(bad_stop, piece_start), piece_stop))
        pieces = remaining
    return pieces


def create_condition_segments(
    events: pd.DataFrame,
    sfreq: float,
    n_times: int,
    transition_trim: float,
    min_segment_duration: float,
    bad_intervals: Sequence[tuple[float, float]] | None = None,
) -> dict[str, list[Segment]]:
    if not math.isfinite(sfreq) or sfreq <= 0 or n_times <= 0:
        raise RuntimeError("EEG sampling frequency and sample count must be positive")
    if not math.isfinite(transition_trim) or transition_trim < 0:
        raise RuntimeError("Transition trim must be finite and nonnegative")
    if not math.isfinite(min_segment_duration) or min_segment_duration <= 0:
        raise RuntimeError("Minimum segment duration must be finite and positive")
    terminal_time = n_times / sfreq
    excluded = []
    for start, stop in bad_intervals if bad_intervals is not None else ():
        start, stop = float(start), float(stop)
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise RuntimeError("Bad intervals must have finite ordered start/stop times")
        if stop >= 0 and start <= terminal_time:
            excluded.append((max(0.0, start), min(terminal_time, stop)))
    excluded.sort()
    state_events = events[events["value"].isin(("instructed_toOpenEyes", "instructed_toCloseEyes"))]
    state_events = state_events.drop_duplicates(subset=["onset", "value"]).sort_values("onset", kind="stable").reset_index(drop=True)
    boundaries = sorted(float(value) for value in events.loc[events["value"].astype(str).str.casefold() == "boundary", "onset"])
    segments = {"EO": [], "EC": []}
    for index, row in state_events.iterrows():
        condition = "EO" if row["value"] == "instructed_toOpenEyes" else "EC"
        source_onset = float(row["onset"])
        source_offset = float(state_events.iloc[index + 1]["onset"]) if index + 1 < len(state_events) else terminal_time
        boundary = next((value for value in boundaries if value > source_onset), terminal_time)
        source_offset = min(source_offset, boundary, terminal_time)
        if source_offset <= source_onset:
            continue
        for piece_start, piece_stop in _subtract_bad_intervals(source_onset, source_offset, excluded):
            onset = piece_start + transition_trim
            offset = piece_stop - transition_trim
            if offset - onset < min_segment_duration:
                continue
            start_sample = max(0, int(math.ceil(onset * sfreq)))
            stop_sample = min(n_times, int(math.floor(offset * sfreq)))
            if (stop_sample - start_sample) / sfreq < min_segment_duration:
                continue
            segments[condition].append(Segment(
                condition=condition,
                source_onset=piece_start,
                source_offset=piece_stop,
                onset=start_sample / sfreq,
                offset=stop_sample / sfreq,
                start_sample=start_sample,
                stop_sample=stop_sample,
            ))
    if not segments["EO"] or not segments["EC"]:
        raise RuntimeError("EO or EC has no valid segment after boundaries, bad intervals, and transition trimming")
    segments["ALL"] = sorted(segments["EO"] + segments["EC"], key=lambda segment: segment.onset)
    return segments


def _validate_eeg_coordinates(raw: Any, eeg_names: Sequence[str]) -> None:
    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError("EEGLAB input has no electrode montage")
    positions = montage.get_positions()
    if positions.get("coord_frame") != "head":
        raise RuntimeError("EEG montage must provide electrode coordinates in the head coordinate frame")
    for name in eeg_names:
        coordinates = np.asarray(positions["ch_pos"].get(name, ()), dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all() or not 1e-5 < np.linalg.norm(coordinates) < 0.5:
            raise RuntimeError(f"EEG channel has invalid or implausible head coordinates in metres: {name}")
    fiducials = []
    for label in ("nasion", "lpa", "rpa"):
        coordinates = np.asarray(positions.get(label), dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise RuntimeError(f"EEG montage is missing a valid {label} fiducial")
        fiducials.append(coordinates)
    nasion, lpa, rpa = fiducials
    if not 0.05 < np.linalg.norm(lpa - rpa) < 0.4 or np.linalg.norm(np.cross(lpa - nasion, rpa - nasion)) < 1e-8:
        raise RuntimeError("EEG montage fiducials are degenerate or have implausible units")


def preprocess_eeg(
    inputs: EEGInputs,
    analysis_sfreq: float,
    n_jobs: int,
    bad_channel_z: float,
    qc_directory: Path,
) -> tuple[Any, pd.DataFrame]:
    raw = mne.io.read_raw_eeglab(inputs.eeg_set, preload=True, verbose="error")
    input_events = load_and_validate_events(inputs.events_tsv, raw.n_times / float(raw.info["sfreq"]))
    boundary_onsets = input_events.loc[input_events["value"].str.casefold() == "boundary", "onset"].to_numpy(dtype=float)
    if boundary_onsets.size:
        raw.annotations.append(
            onset=boundary_onsets + float(raw.first_time),
            duration=np.zeros(boundary_onsets.size),
            description=["BAD boundary"] * boundary_onsets.size,
        )
    channels = pd.read_csv(inputs.channels_tsv, sep="\t")
    if "name" not in channels.columns or channels["name"].duplicated().any():
        raise RuntimeError("channels.tsv must contain unique channel names")
    if raw.ch_names != channels["name"].astype(str).tolist():
        raise RuntimeError("channels.tsv and EEGLAB channel names/order do not match")
    type_map = {"EEG": "eeg", "EOG": "eog", "HEOG": "eog", "VEOG": "eog", "ECG": "ecg", "EKG": "ecg", "EMG": "emg", "MISC": "misc", "OTHER": "misc", "TRIG": "stim", "STIM": "stim", "RESP": "resp"}
    if "type" in channels.columns:
        changes = {}
        for _, row in channels.iterrows():
            channel_type = str(row["type"]).strip().upper()
            if channel_type in type_map:
                changes[str(row["name"])] = type_map[channel_type]
            elif channel_type not in ("", "N/A", "NAN"):
                raise RuntimeError(f"Unsupported channels.tsv channel type: {channel_type}")
        if changes:
            raw.set_channel_types(changes, verbose="error")
    eeg_names = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
    if len(eeg_names) < 4:
        raise RuntimeError("At least four EEG channels are required for source modelling")
    if not math.isfinite(analysis_sfreq) or analysis_sfreq <= 90:
        raise RuntimeError("Analysis sampling frequency must be finite and greater than 90 Hz")
    if not math.isfinite(bad_channel_z) or bad_channel_z <= 0:
        raise RuntimeError("Bad-channel threshold must be finite and positive")
    input_sfreq = float(raw.info["sfreq"])
    high_cutoff = min(48.0, analysis_sfreq / 2.0 - 2.0, input_sfreq / 2.0 - 2.0)
    if not math.isfinite(input_sfreq) or high_cutoff < BANDS["gamma"][1]:
        raise RuntimeError("Input and analysis sampling frequencies must support the full 30-45 Hz gamma band")
    prior_bad = set(raw.info["bads"])
    if "status" in channels.columns:
        prior_bad.update(channels.loc[channels["status"].astype(str).str.strip().str.casefold() == "bad", "name"].astype(str))
    raw.info["bads"] = []
    bad_table = _robust_bad_channels(raw, bad_channel_z)
    for index, row in bad_table.iterrows():
        if row["channel"] in prior_bad:
            bad_table.loc[index, "is_bad"] = True
            existing_reason = str(row["reason"]).strip()
            bad_table.loc[index, "reason"] = ";".join(value for value in (existing_reason, "input_bad") if value)
    all_bad = prior_bad | set(bad_table.loc[bad_table["is_bad"], "channel"])
    raw.info["bads"] = [name for name in raw.ch_names if name in all_bad]
    good_eeg = [name for name in eeg_names if name not in all_bad]
    if len(good_eeg) < 4 or len(eeg_names) - len(good_eeg) > max(20, round(len(eeg_names) * 0.2)):
        raise RuntimeError("Too few good EEG channels remain for source modelling")
    _validate_eeg_coordinates(raw, eeg_names)
    before_frequency, before_power = _median_psd(raw.copy().pick(good_eeg))
    raw.filter(l_freq=0.5, h_freq=high_cutoff, picks=good_eeg, skip_by_annotation=("edge", "bad", "boundary"), n_jobs=n_jobs, verbose="error")
    raw.set_eeg_reference("average", projection=True, verbose="error")
    raw.resample(analysis_sfreq, n_jobs=n_jobs, verbose="error")
    after_frequency, after_power = _median_psd(raw.copy().pick(good_eeg))
    bad_table.to_csv(qc_directory / "bad_channels.tsv", sep="\t", index=False)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.semilogy(before_frequency, before_power, label="input median PSD")
    axis.semilogy(after_frequency, after_power, label="processed median PSD")
    for low, high in BANDS.values():
        axis.axvspan(low, high, alpha=0.04, color="tab:blue")
    axis.set(xlabel="Frequency (Hz)", ylabel="Power spectral density (V²/Hz)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(qc_directory / "eeg_psd_before_after.png", dpi=180)
    plt.close(figure)
    figure = raw.plot_sensors(kind="topomap", show_names=False, show=False)
    figure.savefig(qc_directory / "sensor_layout.png", dpi=180)
    plt.close(figure)
    return raw, bad_table


def estimate_epoch_connectivity(
    timeseries: np.ndarray,
    epoch_bounds: Sequence[tuple[int, int]],
    method: str,
) -> np.ndarray:
    if np.iscomplexobj(timeseries):
        raise ValueError("ROI time series must be real")
    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 90 or not np.isfinite(values).all():
        raise ValueError("ROI time series must be a finite array with 90 rows")
    if method not in {"pearson", "wpli"}:
        raise ValueError(f"Unsupported functional connectivity method: {method}")
    bounds = list(epoch_bounds)
    if not bounds:
        raise ValueError("At least one complete epoch is required")
    epoch_length = None
    previous_stop = 0
    amplitude_scale = 0.0
    for start, stop in bounds:
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (start, stop)):
            raise ValueError("Epoch bounds must be integer sample indices")
        if not 0 <= start < stop <= values.shape[1] or start < previous_stop:
            raise ValueError("Epoch bounds must be in range, ordered and nonoverlapping")
        if stop - start < 4 or (epoch_length is not None and stop - start != epoch_length):
            raise ValueError("Functional connectivity requires equal-length epochs with at least four samples")
        epoch_length = stop - start
        previous_stop = stop
        amplitude_scale = max(amplitude_scale, float(np.max(np.abs(values[:, start:stop]))))
    if amplitude_scale == 0.0:
        return np.zeros((90, 90), dtype=np.float64)
    if method == "pearson":
        fisher_total = np.zeros((90, 90), dtype=np.float64)
        limit = np.nextafter(1.0, 0.0)
        for start, stop in bounds:
            epoch = values[:, start:stop] / amplitude_scale
            centered = epoch - epoch.mean(axis=1, keepdims=True)
            centered[np.ptp(epoch, axis=1) == 0.0] = 0.0
            cross_products = centered @ centered.T
            norms = np.sqrt(np.maximum(np.diag(cross_products), 0.0))
            denominator = np.outer(norms, norms)
            correlation = np.divide(cross_products, denominator, out=np.zeros_like(cross_products), where=denominator > 0.0)
            np.clip(correlation, -limit, limit, out=correlation)
            np.fill_diagonal(correlation, 0.0)
            fisher_total += np.arctanh(correlation)
        connectivity = np.tanh(fisher_total / len(bounds))
    else:
        signed_total = np.zeros((90, 90), dtype=np.float64)
        absolute_total = np.zeros_like(signed_total)
        for start, stop in bounds:
            epoch = values[:, start:stop] / amplitude_scale
            analytic = signal.hilbert(epoch, axis=1)
            analytic[np.ptp(epoch, axis=1) == 0.0] = 0.0
            for chunk_start in range(0, epoch.shape[1], 256):
                chunk = analytic[:, chunk_start : chunk_start + 256]
                cross_imaginary = chunk.imag[:, None, :] * chunk.real[None, :, :] - chunk.real[:, None, :] * chunk.imag[None, :, :]
                signed_total += cross_imaginary.sum(axis=2)
                absolute_total += np.abs(cross_imaginary).sum(axis=2)
        signed_total = (signed_total - signed_total.T) / 2.0
        absolute_total = (absolute_total + absolute_total.T) / 2.0
        connectivity = np.divide(np.abs(signed_total), absolute_total, out=np.zeros_like(absolute_total), where=absolute_total > 0.0)
    if not np.isfinite(connectivity).all():
        raise ValueError("Estimated functional connectivity contains invalid values")
    connectivity = (connectivity + connectivity.T) / 2.0
    np.clip(connectivity, 0.0, 1.0, out=connectivity)
    np.fill_diagonal(connectivity, 0.0)
    return connectivity


def _plot_connectivity(
    matrix: np.ndarray,
    names: Sequence[str],
    condition: str,
    band: str,
    method: str,
    output_path: Path,
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(
        matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ticks = np.arange(0, len(names), 10)
    tick_labels = [f"{index + 1}: {names[index]}" for index in ticks]
    axis.set_xticks(ticks, labels=tick_labels, rotation=60, ha="right")
    axis.set_yticks(ticks, labels=tick_labels)
    method_label = "Positive Pearson" if method == "pearson" else "wPLI"
    axis.set(
        title=f"AAL-90 {method_label} | {condition} | {band}",
        xlabel="Target ROI",
        ylabel="Source ROI",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Connectivity weight")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {"minimum": 0.0, "maximum": 1.0, "transform": "none"}


def build_condition_epoch_table(
    segments: Sequence[Segment],
    sfreq: float,
    epoch_seconds: float = 4.0,
) -> tuple[pd.DataFrame, int]:
    if not math.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("Sampling frequency must be finite and positive")
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0.0:
        raise ValueError("Epoch duration must be finite and positive")
    exact_samples = epoch_seconds * sfreq
    if not math.isfinite(exact_samples):
        raise ValueError("Epoch sample count exceeds the supported numerical range")
    epoch_samples = int(round(exact_samples))
    if epoch_samples < 4 or not math.isclose(exact_samples, epoch_samples, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Epoch duration must correspond to an integer sample count of at least four")
    rows: list[dict[str, Any]] = []
    concatenated_start = 0
    previous_stop = 0
    for segment_index, segment in enumerate(segments, start=1):
        if segment.condition not in {"EO", "EC"}:
            raise ValueError("Epoch segments must identify their original EO or EC state")
        if segment.start_sample < previous_stop or segment.stop_sample <= segment.start_sample:
            raise ValueError("Retained segments must be ordered and nonoverlapping")
        for offset in range(0, segment.sample_count - epoch_samples + 1, epoch_samples):
            start = concatenated_start + offset
            stop = start + epoch_samples
            original_start = segment.start_sample + offset
            original_stop = original_start + epoch_samples
            rows.append({
                "epoch_index": len(rows),
                "state": segment.condition,
                "segment": segment_index,
                "concatenated_start_sample": start,
                "concatenated_stop_sample_exclusive": stop,
                "original_start_sample": original_start,
                "original_stop_sample_exclusive": original_stop,
                "onset_seconds": original_start / sfreq,
                "offset_seconds": original_stop / sfreq,
                "duration_seconds": epoch_samples / sfreq,
            })
        concatenated_start += segment.sample_count
        previous_stop = segment.stop_sample
    if not rows:
        raise RuntimeError("No complete fixed-length epochs fit inside the retained segments")
    return pd.DataFrame(rows), epoch_samples


def export_condition_band_results(
    roi_time_series: np.ndarray,
    segments_by_condition: dict[str, list[Segment]],
    sfreq: float,
    nodes: pd.DataFrame,
    output_directory: Path,
    connectivity_methods: Sequence[str],
    epoch_seconds: float = 4.0,
) -> dict[str, Any]:
    source_data = np.asarray(roi_time_series)
    if source_data.ndim != 2 or source_data.shape[0] != 90 or not np.isfinite(source_data).all():
        raise RuntimeError(f"Expected finite source-space ROI time series with 90 rows: {source_data.shape}")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be finite and positive")
    methods = list(dict.fromkeys(connectivity_methods))
    if not methods or any(method not in {"pearson", "wpli"} for method in methods):
        raise ValueError("Choose at least one of pearson or wpli")
    indices = nodes["index"].astype(int).tolist()
    names = nodes["name"].astype(str).tolist()
    if indices != list(range(1, 91)) or len(names) != 90:
        raise RuntimeError("AAL node rows must be ordered consecutively from 1 to 90")
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_segments = segments_by_condition.get(condition, [])
        if not condition_segments:
            raise RuntimeError(f"No retained segments for {condition}")
        for segment in condition_segments:
            if condition != "ALL" and segment.condition != condition:
                raise RuntimeError(f"Retained segment state does not match {condition}")
            if not 0 <= segment.start_sample < segment.stop_sample <= source_data.shape[1]:
                raise RuntimeError(f"Invalid sample interval for {condition}: {segment.start_sample}:{segment.stop_sample}")
            if segment.stop_sample - segment.start_sample < 4:
                raise RuntimeError(f"Insufficient samples in a retained {condition} segment")
        epochs, epoch_samples = build_condition_epoch_table(condition_segments, sfreq, epoch_seconds)
        bounds = list(zip(
            epochs["concatenated_start_sample"].astype(int),
            epochs["concatenated_stop_sample_exclusive"].astype(int),
        ))
        condition_summary: dict[str, Any] = {}
        for band_name, (low, high) in BANDS.items():
            if not 0 < low < high < sfreq / 2.0:
                raise ValueError(f"Invalid {band_name} band for sampling frequency {sfreq}: {low}-{high}")
            band_directory = output_directory / condition / band_name
            qc_directory = band_directory / "qc"
            qc_directory.mkdir(parents=True, exist_ok=True)
            filtered_segments = [
                _bandpass_segment(
                    source_data[:, segment.start_sample : segment.stop_sample],
                    sfreq,
                    low,
                    high,
                )
                for segment in condition_segments
            ]
            concatenated = np.concatenate(filtered_segments, axis=1).astype(np.float32)
            concatenated_times = np.concatenate(
                [
                    np.arange(segment.start_sample, segment.stop_sample) / sfreq
                    for segment in condition_segments
                ]
            ).astype(np.float64)
            np.save(band_directory / "roi_timeseries.npy", concatenated, allow_pickle=False)
            np.save(band_directory / "sample_times_seconds.npy", concatenated_times, allow_pickle=False)
            _write_segments_table(band_directory / "segments.tsv", condition_segments, sfreq)
            epochs.to_csv(band_directory / "epochs.tsv", sep="\t", index=False)
            _plot_timeseries_preview(
                concatenated,
                sfreq,
                names,
                condition,
                band_name,
                band_directory / "roi_timeseries_preview.png",
            )
            pd.DataFrame(
                {
                    "index": indices,
                    "name": names,
                    "zero_variance_epoch_count": np.sum(
                        [np.ptp(concatenated[:, start:stop], axis=1) == 0.0 for start, stop in bounds],
                        axis=0,
                    ),
                }
            ).to_csv(qc_directory / "roi_signal_statistics.tsv", sep="\t", index=False)
            matrix_files: dict[str, str] = {}
            method_outputs: dict[str, dict[str, Any]] = {}
            plot_scaling: dict[str, dict[str, Any]] = {}
            for method in methods:
                matrix = estimate_epoch_connectivity(concatenated, bounds, method)
                maximum_weight = float(matrix.max())
                all_zero = maximum_weight == 0.0
                normalised = matrix / maximum_weight if not all_zero else np.zeros_like(matrix)
                matrix_stem = f"connectivity_{method}"
                normalised_stem = f"{matrix_stem}_normalised"
                np.save(band_directory / f"{matrix_stem}.npy", matrix, allow_pickle=False)
                np.savetxt(band_directory / f"{matrix_stem}.csv", matrix, delimiter=",", fmt="%.17g")
                np.save(band_directory / f"{normalised_stem}.npy", normalised, allow_pickle=False)
                np.savetxt(band_directory / f"{normalised_stem}.csv", normalised, delimiter=",", fmt="%.17g")
                plot_scaling[method] = _plot_connectivity(
                    matrix,
                    names,
                    condition,
                    band_name,
                    method,
                    band_directory / f"{matrix_stem}.png",
                )
                matrix_files[method] = f"{matrix_stem}.npy"
                method_outputs[method] = {
                    "connectivity_npy": f"{matrix_stem}.npy",
                    "connectivity_csv": f"{matrix_stem}.csv",
                    "connectivity_png": f"{matrix_stem}.png",
                    "normalised_npy": f"{normalised_stem}.npy",
                    "normalised_csv": f"{normalised_stem}.csv",
                    "normalisation": "divide by maximum positive off-diagonal raw weight; all-zero matrices remain zero",
                    "normalisation_maximum": maximum_weight,
                    "all_zero": all_zero,
                    "qc_status": "all_zero_connectivity" if all_zero else "valid",
                    "matrix_is_symmetric": bool(np.array_equal(matrix, matrix.T)),
                    "aggregation": (
                        "signed correlations in equal-length nonoverlapping epochs; equal-weight Fisher-z mean across epochs, inverse transform, then nonpositive values set to zero"
                        if method == "pearson"
                        else "compute Hilbert analytic signals independently within each fixed-length epoch; pool signed and absolute imaginary cross-spectrum sums across epochs before taking the absolute-sum ratio"
                    ),
                    "zero_variance_policy": "zero connectivity contribution for a constant ROI signal within an epoch",
                }
            metadata = {
                "fc_schema_version": FC_SCHEMA_VERSION,
                "condition": condition,
                "frequency_band": band_name,
                "frequency_range_hz": [low, high],
                "sampling_frequency_hz": sfreq,
                "roi_count": 90,
                "roi_order": "nodes.tsv indices 1-90",
                "signal_origin": "individual-MRI source reconstruction aggregated to AAL-90 ROI time series",
                "sample_count": int(concatenated.shape[1]),
                "duration_seconds": concatenated.shape[1] / sfreq,
                "segment_count": len(condition_segments),
                "time_series_shape": list(concatenated.shape),
                "time_series_file": "roi_timeseries.npy",
                "sample_times_file": "sample_times_seconds.npy",
                "segments_file": "segments.tsv",
                "epoch_seconds": epoch_seconds,
                "epoch_samples": epoch_samples,
                "epoch_count": len(epochs),
                "epochs_file": "epochs.tsv",
                "fc_estimation_unit": "epoch",
                "epoch_overlap_seconds": 0.0,
                "epoch_artifact_rejection": "none",
                "epoch_remainder_policy": "discard incomplete epoch tails within each retained continuous segment for connectivity estimation; retain the complete segment in saved ROI time series",
                "discarded_epoch_tail_samples": int(concatenated.shape[1] - len(epochs) * epoch_samples),
                "roi_signal_qc_tsv": "qc/roi_signal_statistics.tsv",
                "connectivity_methods": methods,
                "primary_method": methods[0],
                "connectivity_files": matrix_files,
                "connectivity_outputs": method_outputs,
                "connectivity_plot_scaling": plot_scaling,
                "primary_connectivity": "full nonnegative connectivity matrix",
                "connectivity_weight_range": [0.0, 1.0],
                "matrix_diagonal": "zero",
            }
            (band_directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            condition_summary[band_name] = metadata
        summary[condition] = condition_summary
    return summary


def normalize_subject_id(value: str) -> str:
    label = value.strip()
    if label.startswith("sub-"):
        label = label[4:]
    if not label or not label.isascii() or not label.isalnum():
        raise ValueError("Subject IDs must contain only ASCII letters and digits, optionally prefixed by sub-.")
    return f"sub-{label}"


def select_subjects(arguments: argparse.Namespace) -> list[str]:
    requested = [subject for group in arguments.subject for subject in group]
    if requested:
        return list(dict.fromkeys(normalize_subject_id(subject) for subject in requested))
    eeg_directory = arguments.data_dir / "EEG"
    subjects = []
    for summary_path in sorted(SFC_DIRECTORY.glob("sub-*/structural_connectivity/summary.json")):
        subject = summary_path.parent.parent.name
        try:
            subject = normalize_subject_id(subject)
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(summary, dict) or summary.get("processing_status") != "complete":
            continue
        if (eeg_directory / subject).is_dir():
            subjects.append(subject)
    if not subjects:
        raise RuntimeError(
            "No subjects have both a completed structural_connectivity/summary.json in SFC "
            f"and a downloaded EEG directory in {eeg_directory}."
        )
    return list(dict.fromkeys(subjects))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level EEG functional connectivity from Data/EEG and SFC structural results.")
    parser.add_argument("--subject", nargs="+", action="append", default=[], type=normalize_subject_id)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--subjects-dir", type=Path)
    parser.add_argument("--eeg-session")
    parser.add_argument("--eeg-set-name")
    parser.add_argument("--method", "--connectivity-methods", dest="connectivity_methods", nargs="+", choices=CONNECTIVITY_METHODS, default=["wpli"])
    parser.add_argument("--epoch-seconds", type=float, default=4.0)
    parser.add_argument("--analysis-sfreq", type=float, default=125.0)
    parser.add_argument("--source-spacing-mm", type=float, default=8.0)
    parser.add_argument("--snr", type=float, default=3.0)
    parser.add_argument("--inverse-method", choices=("MNE", "dSPM", "sLORETA", "eLORETA"), default="eLORETA")
    parser.add_argument("--transition-trim", type=float, default=2.0)
    parser.add_argument("--min-segment-duration", type=float, default=4.0)
    parser.add_argument("--bad-channel-z", type=float, default=6.0)
    parser.add_argument("--coreg-icp-iterations", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    arguments.data_dir = arguments.data_dir.expanduser().resolve()
    if arguments.subjects_dir is not None:
        arguments.subjects_dir = arguments.subjects_dir.expanduser().resolve()
    arguments.connectivity_methods = list(dict.fromkeys(arguments.connectivity_methods))
    for option, value in (
        ("--epoch-seconds", arguments.epoch_seconds),
        ("--analysis-sfreq", arguments.analysis_sfreq),
        ("--source-spacing-mm", arguments.source_spacing_mm),
        ("--snr", arguments.snr),
        ("--min-segment-duration", arguments.min_segment_duration),
        ("--bad-channel-z", arguments.bad_channel_z),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{option} must be a finite positive number.")
    if arguments.analysis_sfreq <= 90:
        parser.error("--analysis-sfreq must exceed 90 Hz.")
    epoch_samples = arguments.epoch_seconds * arguments.analysis_sfreq
    if not math.isfinite(epoch_samples) or epoch_samples < 4 or not math.isclose(epoch_samples, round(epoch_samples), rel_tol=0.0, abs_tol=1e-8):
        parser.error("--epoch-seconds must correspond to an integer sample count of at least four at --analysis-sfreq.")
    if not math.isfinite(arguments.transition_trim) or arguments.transition_trim < 0:
        parser.error("--transition-trim must be a finite nonnegative number.")
    if arguments.coreg_icp_iterations <= 0:
        parser.error("--coreg-icp-iterations must be a positive integer.")
    if arguments.n_jobs == 0:
        parser.error("--n-jobs must be nonzero.")
    if arguments.eeg_set_name is not None:
        filename = arguments.eeg_set_name
        if not filename or any(character in filename for character in ("/", "\\", "*", "?", "[", "]")) or not filename.lower().endswith(".set"):
            parser.error("--eeg-set-name must be an exact .set filename without a directory or wildcard.")
    if arguments.eeg_session is not None:
        session = arguments.eeg_session
        if session.startswith("ses-"):
            session = session[4:]
        if not session or not session.isascii() or not session.isalnum():
            parser.error("--eeg-session must be an ASCII alphanumeric session label, optionally prefixed by ses-.")
        arguments.eeg_session = f"ses-{session}"
    return arguments


def save_batch_records(records: list[dict[str, Any]]) -> None:
    manifest_path = SFC_DIRECTORY / "functional_connectivity_manifest.csv"
    merged: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                subject = record.get("subject", "")
                if subject:
                    merged[subject] = dict(record)
    for record in records:
        subject = str(record.get("subject", ""))
        if not subject:
            raise ValueError("Every manifest record requires a subject.")
        merged[subject] = dict(record)
    fieldnames = ["subject", "status", "updated_at", "error"]
    for record in merged.values():
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for subject in sorted(merged):
                serialized = {}
                for name, value in merged[subject].items():
                    if isinstance(value, (dict, list, tuple)):
                        serialized[name] = json.dumps(value, ensure_ascii=False, default=str)
                    elif value is None:
                        serialized[name] = ""
                    else:
                        serialized[name] = str(value)
                writer.writerow(serialized)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    arguments = parse_arguments()
    try:
        _require_dependencies()
        subjects = select_subjects(arguments)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    failures = 0
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {subject}", flush=True)
        try:
            record = dict(process_subject(arguments, subject))
            record["subject"] = subject
            if record.get("status") not in ("complete", "preflight_pass"):
                failures += 1
        except Exception as error:
            failures += 1
            record = {
                "subject": subject,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"ERROR: {subject}: {record['error']}", file=sys.stderr, flush=True)
        record.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        try:
            save_batch_records([record])
        except Exception as error:
            print(f"ERROR: Cannot save the batch manifest: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return 1
        print(f"{subject}: {record.get('status', 'unknown')}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
