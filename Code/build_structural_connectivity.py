from __future__ import annotations

import argparse
import csv
import gzip
import os
import struct
import json
import math
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "Data"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "SFC"
DOWNLOAD_MANIFEST_FILENAME = "download_hbn_mri_eeg_data_manifest.csv"
DEFAULT_AAL_ATLAS = SCRIPT_DIRECTORY / "atlases" / "aal90" / "aal90_mni152_2mm.nii.gz"
DEFAULT_AAL_MAPPING = SCRIPT_DIRECTORY / "atlases" / "aal90" / "aal90_mapping.tsv"
PIPELINE_STAGES = (
    "preprocessing",
    "anatomy",
    "registration",
    "fod",
    "tractography",
    "connectome",
    "qc",
)
PIPELINE_PAUSED_EXIT_CODE = 75


@dataclass(frozen=True)
class InputFiles:
    t1w: Path
    dwi: Path
    bval: Path
    bvec: Path
    dwi_json: Path
    fmap_ap: Path | None
    fmap_ap_json: Path | None
    fmap_pa: Path | None
    fmap_pa_json: Path | None
    reverse_phase_encoding_applicable: bool
    reverse_phase_encoding_reason: str


def _nifti_volume_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"Image is not a downloaded gzip NIfTI file: {path}")
        with gzip.open(path, "rb") as handle:
            prefix = handle.read(4)
            if len(prefix) != 4:
                raise RuntimeError(f"Incomplete NIfTI header: {path}")
            little_size = struct.unpack("<i", prefix)[0]
            big_size = struct.unpack(">i", prefix)[0]
            if little_size in (348, 540):
                byte_order, header_size = "<", little_size
            elif big_size in (348, 540):
                byte_order, header_size = ">", big_size
            else:
                raise RuntimeError(f"Unsupported NIfTI header: {path}")
            header = prefix + handle.read(header_size - 4)
        if len(header) != header_size:
            raise RuntimeError(f"Incomplete NIfTI header: {path}")
        if header_size == 348:
            if header[344:348] != b"n+1\x00":
                raise RuntimeError(f"Expected a single-file NIfTI-1 image: {path}")
            dimensions = struct.unpack(byte_order + "8h", header[40:56])
        else:
            if header[4:12] != b"n+2\x00\r\n\x1a\n":
                raise RuntimeError(f"Expected a single-file NIfTI-2 image: {path}")
            dimensions = struct.unpack(byte_order + "8q", header[16:80])
        dimension_count = dimensions[0]
        if dimension_count not in (3, 4):
            raise RuntimeError(f"Only 3D or 4D NIfTI images are supported: {path}")
        if any(size <= 0 for size in dimensions[1:dimension_count + 1]):
            raise RuntimeError(f"NIfTI image has invalid dimensions: {path}")
        return int(dimensions[4]) if dimension_count == 4 else 1
    except (OSError, EOFError, struct.error) as exc:
        raise RuntimeError(f"Cannot read a downloaded NIfTI image: {path}") from exc


def _is_downloaded_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        _nifti_volume_count(path)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _read_json_object(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return metadata


def _find_one(
    directory: Path,
    pattern: str,
    description: str,
    exact_name: str | None = None,
) -> Path:
    if exact_name is not None:
        if not exact_name or Path(exact_name).name != exact_name or "/" in exact_name or "\\" in exact_name:
            raise RuntimeError(f"{description} selection must be an exact filename within {directory}")
        candidate = directory / exact_name
        if not candidate.match(pattern) or not _is_downloaded_image(candidate):
            raise RuntimeError(f"Selected {description} is not a downloaded valid image: {candidate}")
        return candidate
    matches = [path for path in sorted(directory.glob(pattern)) if _is_downloaded_image(path)]
    if len(matches) != 1:
        rendered = "\n".join(f"  - {path}" for path in matches) or "  (none)"
        raise RuntimeError(f"Expected exactly one downloaded {description}; found {len(matches)}:\n{rendered}")
    return matches[0]


def discover_inputs(
    session_directory: Path,
    t1w_name: str | None = None,
    dwi_name: str | None = None,
) -> InputFiles:
    if not session_directory.is_dir():
        raise RuntimeError(f"MRI session directory does not exist: {session_directory}")
    anat = session_directory / "anat"
    dwi_directory = session_directory / "dwi"
    for directory in (anat, dwi_directory):
        if not directory.is_dir():
            raise RuntimeError(f"Required BIDS directory is missing: {directory}")
    t1w = _find_one(anat, "*_T1w.nii.gz", "T1w", t1w_name)
    dwi = _find_one(dwi_directory, "*_dwi.nii.gz", "DWI", dwi_name)
    stem = dwi.name.removesuffix(".nii.gz")
    related = {
        "bval": dwi_directory / f"{stem}.bval",
        "bvec": dwi_directory / f"{stem}.bvec",
        "dwi_json": dwi_directory / f"{stem}.json",
    }
    for label, path in related.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"DWI {label} file is missing or empty: {path}")
    _validate_gradients(related["bval"], related["bvec"], related["dwi_json"], dwi)
    _validate_dwi_phase_encoding_metadata(related["dwi_json"])
    fmap_ap, fmap_ap_json, fmap_pa, fmap_pa_json, applicable, reason = _discover_reverse_phase_encoding_inputs(
        session_directory / "fmap", related["dwi_json"],
    )
    return InputFiles(
        t1w=t1w,
        dwi=dwi,
        bval=related["bval"],
        bvec=related["bvec"],
        dwi_json=related["dwi_json"],
        fmap_ap=fmap_ap,
        fmap_ap_json=fmap_ap_json,
        fmap_pa=fmap_pa,
        fmap_pa_json=fmap_pa_json,
        reverse_phase_encoding_applicable=applicable,
        reverse_phase_encoding_reason=reason,
    )


def _validate_gradients(
    bval: Path,
    bvec: Path,
    dwi_json: Path,
    dwi_image: Path | None = None,
) -> None:
    bvalues = [float(value) for value in bval.read_text(encoding="utf-8-sig").split()]
    vectors = [
        [float(value) for value in row.split()]
        for row in bvec.read_text(encoding="utf-8-sig").splitlines()
        if row.strip()
    ]
    volume_count = _nifti_volume_count(dwi_image if dwi_image is not None else dwi_json.with_suffix(".nii.gz"))
    if len(bvalues) != volume_count:
        raise RuntimeError(f"bval count ({len(bvalues)}) differs from NIfTI volume count ({volume_count})")
    if len(vectors) != 3 or any(len(row) != volume_count for row in vectors):
        raise RuntimeError("bvec must contain three rows matching the NIfTI volume count")
    if any(not math.isfinite(value) or value < 0 for value in bvalues):
        raise RuntimeError("bval values must be finite and nonnegative")
    if any(not math.isfinite(value) for row in vectors for value in row):
        raise RuntimeError("bvec values must be finite")
    metadata = _read_json_object(dwi_json)
    if "NumVolumes" in metadata:
        recorded_count = metadata["NumVolumes"]
        if isinstance(recorded_count, bool) or not isinstance(recorded_count, (int, float)):
            raise RuntimeError("DWI JSON NumVolumes must be a number")
        if not math.isfinite(recorded_count) or recorded_count != volume_count:
            raise RuntimeError("DWI JSON NumVolumes differs from the actual NIfTI volume count")
    abnormal_norms = []
    for index, bvalue in enumerate(bvalues):
        vector_norm = math.sqrt(sum(row[index] * row[index] for row in vectors))
        if bvalue > 50 and not 0.9 <= vector_norm <= 1.1:
            abnormal_norms.append((index + 1, vector_norm))
    if abnormal_norms:
        examples = ", ".join(f"volume {index}: {norm:.4f}" for index, norm in abnormal_norms[:8])
        raise RuntimeError(f"Non-b0 gradient directions are not normalized: {examples}")
    shells = sorted({round(value / 50) * 50 for value in bvalues})
    if len(shells) < 2 or not any(value <= 50 for value in bvalues) or not any(value > 50 for value in bvalues):
        raise RuntimeError(f"DWI requires b0 and diffusion-weighted volumes: {shells}")


def _read_phase_encoding_metadata(path: Path, label: str) -> dict:
    metadata = _read_json_object(path)
    direction = metadata.get("PhaseEncodingDirection")
    readout = metadata.get("TotalReadoutTime")
    if direction not in ("i", "i-", "j", "j-", "k", "k-"):
        raise RuntimeError(f"{label} JSON has no valid PhaseEncodingDirection: {direction!r}")
    if isinstance(readout, bool) or not isinstance(readout, (int, float)) or not math.isfinite(readout) or readout <= 0:
        raise RuntimeError(f"{label} JSON has no finite positive TotalReadoutTime: {readout!r}")
    return metadata


def _validate_dwi_phase_encoding_metadata(dwi_json: Path) -> None:
    _read_phase_encoding_metadata(dwi_json, "DWI")


def _fieldmap_targets_dwi(metadata: dict, dwi_json: Path) -> bool:
    if "IntendedFor" not in metadata:
        return True
    intended = metadata["IntendedFor"]
    references = [intended] if isinstance(intended, str) else intended
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise RuntimeError("Fieldmap IntendedFor must be a string or a list of strings")
    dwi_image = dwi_json.with_suffix(".nii.gz")
    expected = {dwi_image.name, dwi_image.as_posix().lstrip("/")}
    subject_root = next((parent for parent in dwi_image.parents if parent.name.startswith("sub-")), None)
    if subject_root is not None:
        relative = dwi_image.relative_to(subject_root).as_posix()
        expected.update((relative, f"{subject_root.name}/{relative}"))
    for reference in references:
        normalized = reference.replace("\\", "/").strip()
        if normalized.startswith("bids:"):
            normalized = normalized.split(":", 2)[-1]
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.lstrip("/") in expected:
            return True
    return False


def _discover_reverse_phase_encoding_inputs(
    fmap_directory: Path,
    dwi_json: Path,
) -> tuple[Path | None, Path | None, Path | None, Path | None, bool, str]:
    if not fmap_directory.is_dir():
        return None, None, None, None, False, "No fmap directory"
    matches = {"AP": [], "PA": []}
    reasons = []
    for image in sorted(fmap_directory.glob("*_epi.nii.gz")):
        entities = dict(part.split("-", 1) for part in image.name.split("_") if "-" in part)
        direction = entities.get("dir", "").upper()
        if direction not in matches or "dwi" not in image.name.lower() or not _is_downloaded_image(image):
            continue
        sidecar = image.with_name(image.name.removesuffix(".nii.gz") + ".json")
        try:
            metadata = _read_json_object(sidecar)
            if not _fieldmap_targets_dwi(metadata, dwi_json):
                reasons.append(f"{image.name}: IntendedFor does not include the selected DWI")
                continue
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            reasons.append(f"{image.name}: {exc}")
            continue
        matches[direction].append((image, sidecar))
    valid_pairs = []
    for ap, ap_json in matches["AP"]:
        for pa, pa_json in matches["PA"]:
            try:
                _validate_phase_encoding_metadata(dwi_json, ap_json, pa_json)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                reasons.append(f"{ap.name} / {pa.name}: {exc}")
                continue
            valid_pairs.append((ap, ap_json, pa, pa_json))
    if len(valid_pairs) != 1:
        reason = f"No unambiguous applicable AP/PA pair: AP={len(matches['AP'])}, PA={len(matches['PA'])}, valid_pairs={len(valid_pairs)}"
        if reasons:
            reason += "; " + "; ".join(reasons[:4])
        return None, None, None, None, False, reason
    ap, ap_json, pa, pa_json = valid_pairs[0]
    return ap, ap_json, pa, pa_json, True, "AP/PA directions are opposite and DWI/readout metadata match"


def _validate_phase_encoding_metadata(dwi_json: Path, fmap_ap_json: Path, fmap_pa_json: Path) -> None:
    metadata = {
        "DWI": _read_phase_encoding_metadata(dwi_json, "DWI"),
        "AP": _read_phase_encoding_metadata(fmap_ap_json, "AP"),
        "PA": _read_phase_encoding_metadata(fmap_pa_json, "PA"),
    }
    ap_direction = metadata["AP"]["PhaseEncodingDirection"]
    pa_direction = metadata["PA"]["PhaseEncodingDirection"]
    if ap_direction.rstrip("-") != pa_direction.rstrip("-") or ap_direction.endswith("-") == pa_direction.endswith("-"):
        raise RuntimeError(f"AP/PA phase-encoding directions are not opposite: {ap_direction!r}, {pa_direction!r}")
    if metadata["DWI"]["PhaseEncodingDirection"] != ap_direction:
        raise RuntimeError("DWI phase-encoding direction must match the first AP fieldmap")
    readouts = [float(metadata[name]["TotalReadoutTime"]) for name in ("DWI", "AP", "PA")]
    if not math.isclose(max(readouts), min(readouts), rel_tol=0.01):
        raise RuntimeError("DWI and AP/PA TotalReadoutTime values differ by more than 1%")


def _validate_aal90(atlas: Path, mapping: Path) -> None:
    if not atlas.is_file() or atlas.stat().st_size == 0:
        raise RuntimeError(f"缺少 AAL-90 图谱：{atlas}")
    if not mapping.is_file() or mapping.stat().st_size == 0:
        raise RuntimeError(f"缺少 AAL-90 标签表：{mapping}")

    lines = [
        line.split("\t")
        for line in mapping.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or lines[0] != ["index", "original_value", "name"]:
        raise RuntimeError("AAL-90 标签表表头必须为 index/original_value/name")
    if len(lines) != 91:
        raise RuntimeError(f"AAL-90 标签表应有 90 个节点，实际为 {len(lines) - 1}")
    indices = [int(parts[0]) for parts in lines[1:]]
    original_values = [int(parts[1]) for parts in lines[1:]]
    if indices != list(range(1, 91)):
        raise RuntimeError("AAL-90 节点编号必须连续为 1–90")
    if len(set(original_values)) != 90:
        raise RuntimeError("AAL-90 原始图谱值存在重复")
    if any(len(parts) != 3 or not parts[2] for parts in lines[1:]):
        raise RuntimeError("AAL-90 标签表存在无效行")


def _wsl_path(path: Path, distro: str) -> str:
    path = path.absolute()
    if os.name != "nt":
        return str(path)
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        raise RuntimeError("WSL is required when launching from Windows.")
    completed = subprocess.run(
        [wsl, "--distribution", distro, "--exec", "wslpath", "-a", str(path)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Cannot translate path to WSL: {path}: {detail}")
    return completed.stdout.decode("utf-8").strip()


def _quote(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("路径或参数不能包含换行符")
    return shlex.quote(value)


def build_wsl_script(
    *,
    inputs: InputFiles,
    output_directory: Path,
    aal_atlas: Path,
    aal_mapping: Path,
    subject: str,
    distro: str,
    mrtrix_bin: str | None,
    freesurfer_home: str | None,
    freesurfer_license: str | None,
    preprocessing: str,
    threads: int,
    streamlines: int,
    preflight: bool,
    until_stage: str,
) -> str:
    dwi_metadata = _read_json_object(inputs.dwi_json)
    def translate_optional(path: Path | None) -> str:
        return _wsl_path(path, distro) if path is not None else ""

    translated = {
        "T1": _wsl_path(inputs.t1w, distro),
        "DWI": _wsl_path(inputs.dwi, distro),
        "BVAL": _wsl_path(inputs.bval, distro),
        "BVEC": _wsl_path(inputs.bvec, distro),
        "DWI_JSON": _wsl_path(inputs.dwi_json, distro),
        "FMAP_AP": translate_optional(inputs.fmap_ap),
        "FMAP_AP_JSON": translate_optional(inputs.fmap_ap_json),
        "FMAP_PA": translate_optional(inputs.fmap_pa),
        "FMAP_PA_JSON": translate_optional(inputs.fmap_pa_json),
        "OUTPUT": _wsl_path(output_directory, distro),
        "AAL90_MNI": _wsl_path(aal_atlas, distro),
        "AAL90_MAPPING": _wsl_path(aal_mapping, distro),
    }

    substitutions = {
        **translated,
        "SUBJECT": subject,
        "MRTRIX_BIN_REQUEST": mrtrix_bin or "",
        "FS_HOME_REQUEST": freesurfer_home or "",
        "FS_LICENSE_REQUEST": freesurfer_license or "",
        "PREPROCESSING_REQUEST": preprocessing,
        "DWI_PE_DIR": str(dwi_metadata["PhaseEncodingDirection"]),
        "RPE_PAIR_APPLICABLE": (
            "1" if inputs.reverse_phase_encoding_applicable else "0"
        ),
        "RPE_PAIR_REASON": inputs.reverse_phase_encoding_reason,
        "THREADS": str(threads),
        "STREAMLINES": str(streamlines),
        "MODE": "preflight" if preflight else "run",
        "UNTIL_STAGE": until_stage,
    }

    script = _WSL_PIPELINE_TEMPLATE
    for key, value in substitutions.items():
        script = script.replace(f"__{key}__", _quote(value))
    unresolved = [
        token
        for token in script.split()
        if token.startswith("__") and token.endswith("__")
    ]
    if unresolved:
        raise RuntimeError(f"内部错误：未替换脚本变量 {unresolved}")
    return script


_WSL_PIPELINE_TEMPLATE = r"""set -eo pipefail

MODE=__MODE__
SUBJECT=__SUBJECT__
T1=__T1__
DWI=__DWI__
BVAL=__BVAL__
BVEC=__BVEC__
DWI_JSON=__DWI_JSON__
FMAP_AP=__FMAP_AP__
FMAP_AP_JSON=__FMAP_AP_JSON__
FMAP_PA=__FMAP_PA__
FMAP_PA_JSON=__FMAP_PA_JSON__
OUTPUT=__OUTPUT__
AAL90_MNI=__AAL90_MNI__
AAL90_MAPPING=__AAL90_MAPPING__
MRTRIX_BIN_REQUEST=__MRTRIX_BIN_REQUEST__
FS_HOME_REQUEST=__FS_HOME_REQUEST__
FS_LICENSE_REQUEST=__FS_LICENSE_REQUEST__
PREPROCESSING_REQUEST=__PREPROCESSING_REQUEST__
DWI_PE_DIR=__DWI_PE_DIR__
RPE_PAIR_APPLICABLE=__RPE_PAIR_APPLICABLE__
RPE_PAIR_REASON=__RPE_PAIR_REASON__
FS_RECON_MODE=classic_no_v8
THREADS=__THREADS__
STREAMLINES=__STREAMLINES__
UNTIL_STAGE=__UNTIL_STAGE__

WORK="$OUTPUT/intermediates"
SCRATCH="$WORK/scratch"
mkdir -p "$OUTPUT/qc/eddy" "$OUTPUT/checkpoints" "$WORK/done" "$WORK/freesurfer" "$SCRATCH" "$WORK/tmp"
DONE="$WORK/done"
export SUBJECTS_DIR="$WORK/freesurfer"
export OMP_NUM_THREADS="$THREADS"
export TMPDIR="$WORK/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd "$WORK"

find_mrtrix() {
    if [[ -n "$MRTRIX_BIN_REQUEST" ]]; then
        MRTRIX_BIN="${MRTRIX_BIN_REQUEST%/}"
    elif command -v mrinfo >/dev/null 2>&1; then
        MRTRIX_BIN="$(dirname "$(command -v mrinfo)")"
    else
        MRTRIX_BIN=''
        for candidate in \
            "$HOME"/mrtrix3/bin \
            "$HOME"/MRtrix3/bin \
            /usr/local/mrtrix3/bin \
            /opt/mrtrix3/bin
        do
            if [[ -x "$candidate/mrinfo" ]]; then
                MRTRIX_BIN="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$MRTRIX_BIN" || ! -x "$MRTRIX_BIN/mrinfo" ]]; then
        echo "ERROR: 找不到 MRtrix3 bin 目录" >&2
        exit 20
    fi
    export PATH="$MRTRIX_BIN:$PATH"
    MRTRIX_HOME="${MRTRIX_BIN%/bin}"
}

find_freesurfer() {
    local setup=''
    if [[ -n "$FS_HOME_REQUEST" ]]; then
        setup="${FS_HOME_REQUEST%/}/SetUpFreeSurfer.sh"
    elif [[ -n "${FREESURFER_HOME:-}" \
            && -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]]; then
        setup="$FREESURFER_HOME/SetUpFreeSurfer.sh"
    else
        local candidates=()
        local candidate
        for candidate in \
            "$HOME"/freesurfer/SetUpFreeSurfer.sh \
            "$HOME"/freesurfer/*/SetUpFreeSurfer.sh \
            /usr/local/freesurfer/SetUpFreeSurfer.sh \
            /usr/local/freesurfer/*/SetUpFreeSurfer.sh \
            /opt/freesurfer/SetUpFreeSurfer.sh \
            /opt/freesurfer/*/SetUpFreeSurfer.sh
        do
            [[ -f "$candidate" ]] && candidates+=("$candidate")
        done
        if (( ${#candidates[@]} > 0 )); then
            setup="${candidates[-1]}"
        fi
    fi
    if [[ -z "$setup" || ! -f "$setup" ]]; then
        echo "ERROR: 找不到 FreeSurfer SetUpFreeSurfer.sh" >&2
        exit 21
    fi

    export FREESURFER_HOME="${setup%/SetUpFreeSurfer.sh}"
    set +e
    source "$setup" >/dev/null
    set -e

    if [[ -n "$FS_LICENSE_REQUEST" ]]; then
        export FS_LICENSE="$FS_LICENSE_REQUEST"
    elif [[ -z "${FS_LICENSE:-}" || ! -f "$FS_LICENSE" ]]; then
        local license_candidate
        for license_candidate in \
            "$FREESURFER_HOME"/license.txt \
            "$FREESURFER_HOME"/.license \
            "$HOME"/.licenses/freesurfer/license.txt \
            "$HOME"/.freesurfer/license.txt
        do
            if [[ -f "$license_candidate" ]]; then
                export FS_LICENSE="$license_candidate"
                break
            fi
        done
    fi
    if [[ -z "${FS_LICENSE:-}" || ! -f "$FS_LICENSE" ]]; then
        echo "ERROR: 找不到 FreeSurfer license" >&2
        exit 22
    fi
}

find_fsl_and_ants() {
    if ! command -v topup >/dev/null 2>&1; then
        local fsl_setup
        for fsl_setup in \
            "$HOME"/fsl/etc/fslconf/fsl.sh \
            /usr/local/fsl/etc/fslconf/fsl.sh \
            /opt/fsl/etc/fslconf/fsl.sh
        do
            if [[ -f "$fsl_setup" ]]; then
                set +e
                source "$fsl_setup"
                set -e
                break
            fi
        done
    fi

    FSL_AVAILABLE=0
    if command -v topup >/dev/null 2>&1 \
        && { command -v eddy >/dev/null 2>&1 \
             || command -v eddy_openmp >/dev/null 2>&1 \
             || command -v eddy_cuda >/dev/null 2>&1; }
    then
        FSL_AVAILABLE=1
    fi
    if [[ -z "${FSLDIR:-}" ]]; then
        local fsl_root
        for fsl_root in "$HOME"/fsl /usr/local/fsl /opt/fsl
        do
            if [[ -f "$fsl_root/data/standard/MNI152_T1_2mm_brain.nii.gz" ]]; then
                export FSLDIR="$fsl_root"
                break
            fi
        done
    fi

    if ! command -v N4BiasFieldCorrection >/dev/null 2>&1; then
        local ants_root
        local ants_executable=''
        for ants_root in \
            "$HOME"/opt \
            "$HOME"/ants \
            /opt/ants \
            /usr/local/ants
        do
            if [[ -d "$ants_root" ]]; then
                ants_executable="$(
                    find "$ants_root" \
                        -maxdepth 6 \
                        -type f \
                        -name N4BiasFieldCorrection \
                        -perm -u+x \
                        -print \
                        -quit 2>/dev/null
                )"
                if [[ -n "$ants_executable" ]]; then
                    export PATH="$(dirname "$ants_executable"):$PATH"
                    break
                fi
            fi
        done
    fi
    ANTS_AVAILABLE=0
    command -v N4BiasFieldCorrection >/dev/null 2>&1 && ANTS_AVAILABLE=1

    case "$PREPROCESSING_REQUEST" in
        auto)
            if (( FSL_AVAILABLE )); then
                if (( RPE_PAIR_APPLICABLE )); then
                    PREPROCESSING_MODE=fsl
                else
                    PREPROCESSING_MODE=fsl_rpe_none
                fi
            else
                PREPROCESSING_MODE=basic
            fi
            ;;
        fsl)
            if (( ! FSL_AVAILABLE )); then
                echo "ERROR: --preprocessing fsl 已指定，但 WSL 中没有 topup/eddy" >&2
                exit 23
            fi
            if (( RPE_PAIR_APPLICABLE )); then
                PREPROCESSING_MODE=fsl
            else
                PREPROCESSING_MODE=fsl_rpe_none
            fi
            ;;
        basic)
            PREPROCESSING_MODE=basic
            ;;
        *)
            echo "ERROR: 未知预处理模式 $PREPROCESSING_REQUEST" >&2
            exit 24
            ;;
    esac

    if (( ANTS_AVAILABLE )); then
        BIAS_MODE=ants
    elif command -v fast >/dev/null 2>&1; then
        BIAS_MODE=fsl
    else
        BIAS_MODE=none
    fi
}

check_commands() {
    local command_name
    for command_name in \
        mrinfo mrconvert mrcat mrmath dwiextract dwidenoise mrdegibbs \
        dwifslpreproc dwibiascorrect dwi2mask dwi2response dwi2fod mtnormalise 5ttgen 5tt2gmwmi \
        transformconvert mrtransform tckgen tcksift2 tck2connectome \
        mrstats mrcalc tckstats recon-all mkheadsurf mris_remesh \
        bbregister mri_segstats \
        antsRegistrationSyNQuick.sh antsRegistration antsApplyTransforms \
        topup eddy_quad slices fast base64 awk
    do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "ERROR: 找不到必需命令 $command_name" >&2
            exit 25
        fi
    done
    if (( ! FSL_AVAILABLE )); then
        echo "ERROR: AAL-90 流水线需要 FSL topup/eddy 和 MNI152 模板" >&2
        exit 26
    fi
    if (( ! ANTS_AVAILABLE )); then
        echo "ERROR: AAL-90 非线性配准需要 ANTs" >&2
        exit 27
    fi
    MNI_TEMPLATE="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz"
    [[ -f "$MNI_TEMPLATE" ]] || {
        echo "ERROR: 缺少 FSL MNI 模板 $MNI_TEMPLATE" >&2
        exit 28
    }
    [[ -f "$AAL90_MNI" ]] || { echo "ERROR: 缺少 $AAL90_MNI" >&2; exit 29; }
    [[ -f "$AAL90_MAPPING" ]] || {
        echo "ERROR: 缺少 $AAL90_MAPPING" >&2
        exit 30
    }
}

find_mrtrix
find_freesurfer
find_fsl_and_ants
export SUBJECTS_DIR="$WORK/freesurfer"
export OMP_NUM_THREADS="$THREADS"
export TMPDIR="$WORK/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd "$WORK"
check_commands

if [[ "$MODE" == preflight ]]; then
    echo "STATUS=OK"
    echo "SUBJECT=$SUBJECT"
    echo "MRTRIX_BIN=$MRTRIX_BIN"
    echo "MRTRIX_VERSION=$(mrinfo -version 2>&1 | head -n 1)"
    echo "FREESURFER_HOME=$FREESURFER_HOME"
    echo "FREESURFER_VERSION=$(recon-all -version 2>&1 | head -n 1)"
    echo "FREESURFER_RECON_MODE=$FS_RECON_MODE"
    echo "MKHEADSURF_PATH=$(command -v mkheadsurf)"
    echo "MRIS_REMESH_PATH=$(command -v mris_remesh)"
    echo "FS_LICENSE=$FS_LICENSE"
    echo "FSL_AVAILABLE=$FSL_AVAILABLE"
    echo "FSL_VERSION=$(cat "$FSLDIR/etc/fslversion" 2>/dev/null || echo unknown)"
    echo "TOPUP_PATH=$(command -v topup)"
    echo "EDDY_PATH=$(command -v eddy || command -v eddy_openmp || command -v eddy_cuda)"
    echo "EDDY_QUAD_PATH=$(command -v eddy_quad)"
    echo "SLICES_PATH=$(command -v slices)"
    echo "ANTS_AVAILABLE=$ANTS_AVAILABLE"
    echo "ANTS_VERSION=$(antsRegistration --version 2>&1 | head -n 1)"
    echo "N4_PATH=$(command -v N4BiasFieldCorrection)"
    echo "PREPROCESSING_MODE=$PREPROCESSING_MODE"
    echo "BIAS_MODE=$BIAS_MODE"
    echo "DWI_PE_DIR=$DWI_PE_DIR"
    echo "UNTIL_STAGE=$UNTIL_STAGE"
    echo "ATLAS=AAL-90"
    echo "ATLAS_IMAGE=$AAL90_MNI"
    echo "ATLAS_NODE_COUNT=$(awk 'NR > 1 {count++} END {print count+0}' "$AAL90_MAPPING")"
    echo "ATLAS_MAX_LABEL=$(mrstats "$AAL90_MNI" -output max -quiet)"
    echo "MNI_TEMPLATE=$MNI_TEMPLATE"
    echo "OUTPUT=$OUTPUT"
    echo "DWI_HEADER_BEGIN"
    mrconvert "$DWI" - \
        -fslgrad "$BVEC" "$BVAL" \
        -json_import "$DWI_JSON" \
        -quiet |
        mrinfo - \
            -size \
            -spacing \
            -shell_bvalues \
            -property PhaseEncodingDirection \
            -property TotalReadoutTime \
            -quiet
    echo "DWI_HEADER_END"
    echo "RPE_PAIR_APPLICABLE=$RPE_PAIR_APPLICABLE"
    echo "RPE_PAIR_REASON=$RPE_PAIR_REASON"
    if [[ "$PREPROCESSING_MODE" == basic ]]; then
        echo "WARNING=FSL不可用，将跳过topup/eddy运动与畸变校正"
    fi
    if [[ "$BIAS_MODE" == none ]]; then
        echo "WARNING=ANTs和FSL FAST不可用，将跳过DWI偏置场校正"
    fi
    df -h "$OUTPUT" | tail -n 1 | sed 's/^/WSL_DISK=/'
    exit 0
fi

QC_METRICS="$OUTPUT/qc/qc_metrics.tsv"

record_qc() {
    local component="$1"
    local metric="$2"
    local value="$3"
    local status="$4"
    local criterion="$5"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$component" "$metric" "$value" "$status" "$criterion" \
        >> "$QC_METRICS"
}

LOG="$OUTPUT/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

STATE_FILE="$OUTPUT/pipeline_state.json"
EVENT_LOG="$OUTPUT/pipeline_events.tsv"
CHECKPOINTS="$OUTPUT/checkpoints"
CURRENT_STAGE=initialization
CURRENT_STEP=''
CONFIG_HASH=pending
PIPELINE_EXIT_RECORDED=0

if [[ ! -f "$EVENT_LOG" ]]; then
    printf 'timestamp\tevent\tstage\tstep\tdetail\n' > "$EVENT_LOG"
fi

record_event() {
    local event="$1"
    local stage="$2"
    local step="$3"
    local detail="$4"
    detail="${detail//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" \
        "$event" "$stage" "$step" "$detail" \
        >> "$EVENT_LOG"
}

write_state() {
    local status="$1"
    local stage="$2"
    local step="$3"
    local completed_count
    completed_count="$(
        find "$DONE" -maxdepth 1 -type f -name '*.done' -print |
            wc -l |
            tr -d '[:space:]'
    )"
    local temporary_state="$STATE_FILE.tmp.$$"
    cat > "$temporary_state" <<STATE
{
  "subject": "$SUBJECT",
  "status": "$status",
  "current_stage": "$stage",
  "current_step": "$step",
  "until_stage": "$UNTIL_STAGE",
  "completed_checkpoints": $completed_count,
  "configuration_hash": "$CONFIG_HASH",
  "work_directory_wsl": "$WORK",
  "output_directory_wsl": "$OUTPUT",
  "updated_at": "$(date --iso-8601=seconds)",
  "event_log": "pipeline_events.tsv",
  "checkpoint_directory": "checkpoints"
}
STATE
    mv -f "$temporary_state" "$STATE_FILE"
}

handle_pipeline_exit() {
    local exit_code=$?
    if (( PIPELINE_EXIT_RECORDED )); then
        return
    fi
    if (( exit_code != 0 && exit_code != 75 )); then
        trap - EXIT
        record_event FAILED "$CURRENT_STAGE" "$CURRENT_STEP" \
            "exit_code=$exit_code; inspect pipeline.log"
        write_state failed "$CURRENT_STAGE" "$CURRENT_STEP"
        echo "ERROR: 流水线在 $CURRENT_STAGE/$CURRENT_STEP 失败，退出码 $exit_code" >&2
        echo "可修复问题后使用相同命令继续，已完成检查点会自动复用。" >&2
        exit "$exit_code"
    fi
}
trap handle_pipeline_exit EXIT

echo
echo "============================================================"
echo "Structural connectivity pipeline started: $(date --iso-8601=seconds)"
echo "Subject: $SUBJECT"
echo "Output: $OUTPUT"
echo "Work: $WORK"
echo "Preprocessing: $PREPROCESSING_MODE"
if (( ! RPE_PAIR_APPLICABLE )); then
    echo "AP/PA skipped: $RPE_PAIR_REASON"
fi
echo "Bias correction: $BIAS_MODE"
echo "Streamlines: $STREAMLINES"
echo "Run until stage: $UNTIL_STAGE"
echo "============================================================"

CONFIG_CANDIDATE="$WORK/run_configuration.candidate.txt"
CONFIG_SNAPSHOT="$WORK/run_configuration.txt"
input_identity() {
    if [[ -n "$1" && -f "$1" ]]; then
        stat -c '%s:%Y' "$1"
    else
        echo not_used
    fi
}
cat > "$CONFIG_CANDIDATE" <<CONFIG
subject=$SUBJECT
t1w=$T1
dwi=$DWI
bval=$BVAL
bvec=$BVEC
dwi_json=$DWI_JSON
fmap_ap=$FMAP_AP
fmap_ap_json=$FMAP_AP_JSON
fmap_pa=$FMAP_PA
fmap_pa_json=$FMAP_PA_JSON
t1w_identity=$(stat -c '%s:%Y' "$T1")
dwi_identity=$(stat -c '%s:%Y' "$DWI")
bval_identity=$(stat -c '%s:%Y' "$BVAL")
bvec_identity=$(stat -c '%s:%Y' "$BVEC")
dwi_json_identity=$(stat -c '%s:%Y' "$DWI_JSON")
fmap_ap_identity=$(input_identity "$FMAP_AP")
fmap_ap_json_identity=$(input_identity "$FMAP_AP_JSON")
fmap_pa_identity=$(input_identity "$FMAP_PA")
fmap_pa_json_identity=$(input_identity "$FMAP_PA_JSON")
mrtrix_bin=$MRTRIX_BIN
mrtrix_version=$(mrinfo -version 2>&1 | head -n 1)
freesurfer_home=$FREESURFER_HOME
freesurfer_version=$(recon-all -version 2>&1 | head -n 1)
fsl_version=$(cat "$FSLDIR/etc/fslversion")
ants_version=$(antsRegistration --version 2>&1 | head -n 1)
work_directory=$WORK
preprocessing=$PREPROCESSING_MODE
bias_correction=$BIAS_MODE
streamlines=$STREAMLINES
atlas=AAL-90
atlas_image=$AAL90_MNI
atlas_mapping=$AAL90_MAPPING
atlas_image_identity=$(stat -c '%s:%Y' "$AAL90_MNI")
atlas_mapping_identity=$(stat -c '%s:%Y' "$AAL90_MAPPING")
CONFIG

if [[ -f "$CONFIG_SNAPSHOT" ]] \
        && ! cmp -s "$CONFIG_CANDIDATE" "$CONFIG_SNAPSHOT"; then
    diff -u "$CONFIG_SNAPSHOT" "$CONFIG_CANDIDATE" \
        > "$OUTPUT/configuration_mismatch.diff" || true
    CONFIG_HASH="$(sha256sum "$CONFIG_SNAPSHOT" | awk '{print $1}')"
    record_event BLOCKED initialization configuration \
        "configuration changed; inspect configuration_mismatch.diff"
    write_state blocked initialization configuration
    PIPELINE_EXIT_RECORDED=1
    echo "ERROR: 当前参数或输入与已有检查点不一致。" >&2
    echo "Inspect $OUTPUT/configuration_mismatch.diff and archive the existing structural_connectivity directory before changing inputs or parameters." >&2
    exit 34
fi
if [[ ! -f "$CONFIG_SNAPSHOT" ]]; then
    mv -f "$CONFIG_CANDIDATE" "$CONFIG_SNAPSHOT"
else
    rm -f "$CONFIG_CANDIDATE"
fi
CONFIG_HASH="$(sha256sum "$CONFIG_SNAPSHOT" | awk '{print $1}')"
{
    cat "$CONFIG_SNAPSHOT"
    printf 'freesurfer_license=%s\n' "$FS_LICENSE"
    printf 'freesurfer_recon_mode=%s\n' "$FS_RECON_MODE"
    printf 'threads=%s\n' "$THREADS"
    printf 'until_stage=%s\n' "$UNTIL_STAGE"
    printf 'configuration_hash=%s\n' "$CONFIG_HASH"
} > "$OUTPUT/run_configuration.txt"

printf 'component\tmetric\tvalue\tstatus\tcriterion\n' > "$QC_METRICS"

record_event RUN_STARTED initialization '' \
    "until_stage=$UNTIL_STAGE configuration_hash=$CONFIG_HASH"
write_state running initialization ''

is_done() {
    [[ -f "$DONE/$1.done" ]]
}

mark_done() {
    local step_name="$1"
    local completion_time
    completion_time="$(date --iso-8601=seconds)"
    printf '%s\n' "$completion_time" > "$DONE/$step_name.done"
    printf '%s\n' "$completion_time" > "$CHECKPOINTS/$step_name.done"
}

stage_number() {
    case "$1" in
        preprocessing) echo 1 ;;
        anatomy) echo 2 ;;
        registration) echo 3 ;;
        fod) echo 4 ;;
        tractography) echo 5 ;;
        connectome) echo 6 ;;
        qc) echo 7 ;;
        *)
            echo "ERROR: 未知阶段 $1" >&2
            return 1
            ;;
    esac
}

enter_stage() {
    local requested_stage="$1"
    local requested_number
    local limit_number
    if [[ "$CURRENT_STAGE" != initialization \
            && "$CURRENT_STAGE" != "$requested_stage" ]]; then
        record_event STAGE_COMPLETED "$CURRENT_STAGE" '' ''
    fi
    requested_number="$(stage_number "$requested_stage")"
    limit_number="$(stage_number "$UNTIL_STAGE")"
    if (( requested_number > limit_number )); then
        record_event PAUSED "$CURRENT_STAGE" '' \
            "requested stop after $UNTIL_STAGE"
        write_state paused "$CURRENT_STAGE" ''
        echo
        echo "[暂停] 已按要求完成到阶段：$UNTIL_STAGE"
        echo "[续跑] 重新运行并指定更后的 --until-stage，或不指定以跑到 qc。"
        exit 75
    fi
    CURRENT_STAGE="$requested_stage"
    CURRENT_STEP=''
    record_event STAGE_ENTERED "$CURRENT_STAGE" '' ''
    write_state running "$CURRENT_STAGE" ''
    echo
    echo "---------------- Stage: $CURRENT_STAGE ----------------"
}

expected_step_outputs() {
    case "$1" in
        01_import_dwi) printf '%s\n' "$RAW" ;;
        02_denoise) printf '%s\n' "$DENOISED" "$NOISE" ;;
        03_degibbs) printf '%s\n' "$DEGIBBS" ;;
        04a_import_fmap_ap) printf '%s\n' "$AP_MIF" ;;
        04b_import_fmap_pa) printf '%s\n' "$PA_MIF" ;;
        04c_concat_fmaps) printf '%s\n' "$SE_EPI" ;;
        04d_topup_eddy) printf '%s\n' "$PREPROC" "$OUTPUT/qc/eddy" ;;
        04d_eddy_no_reverse_pe) printf '%s\n' "$PREPROC" "$OUTPUT/qc/eddy" ;;
        04_basic_preprocessing) printf '%s\n' "$PREPROC" ;;
        05_bias_ants|05_bias_fsl)
            printf '%s\n' "$BIASCORR" "$WORK/dwi_bias.mif"
            ;;
        05_no_bias_correction) printf '%s\n' "$BIASCORR" ;;
        06_brain_mask) printf '%s\n' "$MASK" ;;
        07_mean_b0) printf '%s\n' "$MEAN_B0" ;;
        08_freesurfer_recon_all)
            printf '%s\n' "$SUBJECTS_DIR/$FS_SUBJECT/mri/brain.mgz"
            ;;
        08b_freesurfer_head_surface)
            printf '%s\n' "$SUBJECTS_DIR/$FS_SUBJECT/surf/lh.seghead"
            ;;
        08c_freesurfer_head_surface_lowres)
            printf '%s\n' "$SUBJECTS_DIR/$FS_SUBJECT/bem/outer_skin.surf"
            ;;
        09_generate_5tt) printf '%s\n' "$FIVE_TT_T1" ;;
        10a_export_freesurfer_brain) printf '%s\n' "$FS_BRAIN_NII" ;;
        10b_register_mni_to_t1)
            printf '%s\n' \
                "${MNI_TO_T1_PREFIX}0GenericAffine.mat" \
                "${MNI_TO_T1_PREFIX}1Warp.nii.gz" \
                "${MNI_TO_T1_PREFIX}Warped.nii.gz"
            ;;
        10c_warp_aal90_to_t1) printf '%s\n' "$NODES_T1_NII" ;;
        10d_import_aal90_t1) printf '%s\n' "$NODES_T1" ;;
        10e_node_table) printf '%s\n' "$OUTPUT/nodes.tsv" ;;
        11a_export_mean_b0) printf '%s\n' "$MEAN_B0_NII" ;;
        11b_bbregister) printf '%s\n' "$REG_DAT" "$REG_FSL" ;;
        11c_convert_transform) printf '%s\n' "$DWI_TO_T1" ;;
        12a_transform_5tt) printf '%s\n' "$FIVE_TT_DWI" ;;
        12b_transform_nodes) printf '%s\n' "$NODES_DWI" ;;
        12c_transform_t1_brain) printf '%s\n' "$FS_BRAIN_DWI" ;;
        12d_export_nodes_dwi) printf '%s\n' "$NODES_DWI_NII" ;;
        13_gmwmi_seed) printf '%s\n' "$GMWMI" ;;
        14_response_dhollander)
            printf '%s\n' \
                "$WM_RESPONSE" "$GM_RESPONSE" "$CSF_RESPONSE" \
                "$WORK/response_voxels.mif"
            ;;
        15_msmt_csd) printf '%s\n' "$WM_FOD" "$GM_FOD" "$CSF_FOD" ;;
        16_mtnormalise)
            printf '%s\n' "$WM_FOD_NORM" "$GM_FOD_NORM" "$CSF_FOD_NORM"
            ;;
        17_act_tractography) printf '%s\n' "$TRACKS" ;;
        18_sift2) printf '%s\n' "$SIFT2_WEIGHTS" "$OUTPUT/sift2_mu.txt" ;;
        19a_connectome_sift2)
            printf '%s\n' \
                "$OUTPUT/connectome_sift2.csv" "$STREAMLINE_ASSIGNMENTS"
            ;;
        19b_connectome_count)
            printf '%s\n' "$OUTPUT/connectome_streamline_count.csv"
            ;;
        19c_connectome_length)
            printf '%s\n' "$OUTPUT/connectome_mean_length_mm.csv"
            ;;
        20a_qc_raw_mean_b0)
            printf '%s\n' "$RAW_MEAN_B0" "$OUTPUT/qc/raw_mean_b0.nii.gz"
            ;;
        20b_qc_processed_mean_b0)
            printf '%s\n' "$OUTPUT/qc/preprocessed_mean_b0.nii.gz"
            ;;
        20c_qc_mask) printf '%s\n' "$OUTPUT/qc/dwi_mask.nii.gz" ;;
        20d_qc_noise) printf '%s\n' "$OUTPUT/qc/denoise_noise_map.nii.gz" ;;
        20e_qc_bias_field) printf '%s\n' "$OUTPUT/qc/dwi_bias_field.nii.gz" ;;
        20f1_qc_freesurfer_t1)
            printf '%s\n' "$OUTPUT/qc/freesurfer_T1.nii.gz"
            ;;
        20f2_qc_freesurfer_brain_t1)
            printf '%s\n' "$OUTPUT/qc/freesurfer_brain_t1.nii.gz"
            ;;
        20g_qc_freesurfer_aseg)
            printf '%s\n' "$OUTPUT/qc/freesurfer_aseg.nii.gz"
            ;;
        20h_qc_freesurfer_wmparc)
            printf '%s\n' "$OUTPUT/qc/freesurfer_wmparc.nii.gz"
            ;;
        20i1_qc_t1_in_dwi_fsl_grid)
            printf '%s\n' "$OUTPUT/qc/freesurfer_brain_in_dwi_space.nii.gz"
            ;;
        20j_qc_nodes_t1)
            printf '%s\n' "$OUTPUT/qc/nodes_in_t1_space.nii.gz"
            ;;
        20k_qc_mni_registration)
            printf '%s\n' "$OUTPUT/qc/mni152_to_t1_warped.nii.gz"
            ;;
        20l_qc_5tt_dwi)
            printf '%s\n' "$OUTPUT/qc/5tt_in_dwi_space.nii.gz"
            ;;
        20m_qc_gmwmi_seed) printf '%s\n' "$OUTPUT/qc/gmwmi_seed.nii.gz" ;;
        20n_qc_wm_fod_l0) printf '%s\n' "$OUTPUT/qc/wm_fod_l0.nii.gz" ;;
        20o_export_gradients)
            printf '%s\n' \
                "$OUTPUT/qc/gradients_after_preprocessing.bvec" \
                "$OUTPUT/qc/gradients_after_preprocessing.bval"
            ;;
        20p_qc_freesurfer_logs)
            printf '%s\n' "$OUTPUT/qc/freesurfer/recon-all.log"
            ;;
        20q_qc_mni_to_t1_montage)
            printf '%s\n' "$OUTPUT/qc/mni_to_t1_registration.gif"
            ;;
        20r_qc_t1_to_dwi_montage)
            printf '%s\n' "$OUTPUT/qc/t1_to_dwi_registration.gif"
            ;;
        20r1_qc_raw_to_preprocessed_b0_montage)
            printf '%s\n' "$OUTPUT/qc/raw_to_preprocessed_b0.gif"
            ;;
        20r2_qc_freesurfer_brainmask_montage)
            printf '%s\n' "$OUTPUT/qc/freesurfer_brainmask.gif"
            ;;
        20s_qc_tractogram_stats)
            printf '%s\n' \
                "$WORK/tckstats_values.txt" "$OUTPUT/qc/tractogram_stats.tsv"
            ;;
    esac
}

checkpoint_outputs_valid() {
    local step_name="$1"
    local output_path
    local output_count=0
    while IFS= read -r output_path; do
        [[ -n "$output_path" ]] || continue
        output_count=$((output_count + 1))
        if [[ -d "$output_path" ]]; then
            if [[ -z "$(find "$output_path" -mindepth 1 -print -quit)" ]]; then
                return 1
            fi
        elif [[ ! -s "$output_path" ]]; then
            return 1
        fi
    done < <(expected_step_outputs "$step_name")
    (( output_count > 0 ))
}

prepare_step() {
    local step_name="$1"
    if is_done "$step_name"; then
        if checkpoint_outputs_valid "$step_name"; then
            echo "[SKIP] $step_name（检查点和输出有效）"
            record_event STEP_SKIPPED "$CURRENT_STAGE" "$step_name" \
                "valid checkpoint"
            return 1
        fi
        echo "[REDO] $step_name（检查点存在，但输出缺失或为空）"
        record_event CHECKPOINT_INVALIDATED "$CURRENT_STAGE" "$step_name" \
            "missing or empty expected output"
        rm -f "$DONE/$step_name.done" "$CHECKPOINTS/$step_name.done"
    fi
    CURRENT_STEP="$step_name"
    echo "[RUN ] $step_name"
    record_event STEP_STARTED "$CURRENT_STAGE" "$CURRENT_STEP" ''
    write_state running "$CURRENT_STAGE" "$CURRENT_STEP"
    return 0
}

complete_step() {
    local step_name="$1"
    if ! checkpoint_outputs_valid "$step_name"; then
        echo "ERROR: $step_name 命令结束，但预期输出缺失或为空" >&2
        return 35
    fi
    mark_done "$step_name"
    record_event STEP_COMPLETED "$CURRENT_STAGE" "$step_name" ''
    CURRENT_STEP=''
    write_state running "$CURRENT_STAGE" ''
}

start_transient_step() {
    CURRENT_STEP="$1"
    echo "[CHECK] $CURRENT_STEP"
    record_event STEP_STARTED "$CURRENT_STAGE" "$CURRENT_STEP" \
        "non-checkpointed validation/report step"
    write_state running "$CURRENT_STAGE" "$CURRENT_STEP"
}

complete_transient_step() {
    local step_name="$1"
    record_event STEP_COMPLETED "$CURRENT_STAGE" "$step_name" \
        "non-checkpointed validation/report step"
    CURRENT_STEP=''
    write_state running "$CURRENT_STAGE" ''
}

run_step() {
    local step_name="$1"
    shift
    if ! prepare_step "$step_name"; then
        return
    fi
    "$@"
    complete_step "$step_name"
}

export_with_reference_strides() {
    local reference_image="$1"
    local input_image="$2"
    local output_image="$3"
    local reference_strides

    reference_strides="$(
        mrinfo "$reference_image" -strides |
            awk '{$1=$1; gsub(/[[:space:]]+/, ","); print}'
    )"
    if [[ -z "$reference_strides" ]]; then
        echo "ERROR: 无法读取参考影像 strides: $reference_image" >&2
        return 36
    fi
    mrconvert "$input_image" "$output_image" \
        -strides "$reference_strides" \
        -force
}

validate_aal90_segmentation() {
    local segmentation="$1"
    local statistics_file="$2"
    mri_segstats \
        --seg "$segmentation" \
        --excludeid 0 \
        --sum "$statistics_file" \
        >/dev/null
    local segment_count
    local minimum_id
    local maximum_id
    segment_count="$(awk '!/^#/ {count++} END {print count+0}' "$statistics_file")"
    minimum_id="$(awk '!/^#/ {if (min == "" || $2 < min) min=$2} END {print min}' "$statistics_file")"
    maximum_id="$(awk '!/^#/ {if ($2 > max) max=$2} END {print max+0}' "$statistics_file")"
    local small_segment_count
    small_segment_count="$(awk '!/^#/ && $3 < 5 {count++} END {print count+0}' "$statistics_file")"

    record_qc atlas label_count "$segment_count" \
        "$([[ "$segment_count" == 90 ]] && echo PASS || echo FAIL)" "90"
    record_qc atlas minimum_label "$minimum_id" \
        "$([[ "$minimum_id" == 1 ]] && echo PASS || echo FAIL)" "1"
    record_qc atlas maximum_label "$maximum_id" \
        "$([[ "$maximum_id" == 90 ]] && echo PASS || echo FAIL)" "90"
    record_qc atlas labels_smaller_than_5_voxels "$small_segment_count" \
        "$([[ "$small_segment_count" == 0 ]] && echo PASS || echo WARN)" \
        "0 preferred; inspect small parcels"

    if [[ "$segment_count" != 90 || "$minimum_id" != 1 || "$maximum_id" != 90 ]]; then
        echo "ERROR: AAL-90 标签验证失败: count=$segment_count, min=$minimum_id, max=$maximum_id" >&2
        exit 31
    fi
}

validate_connectome_matrix() {
    local matrix="$1"
    local matrix_name="$2"
    local results
    results="$(
        awk -F',' '
            function absolute(value) {
                return value < 0 ? -value : value
            }
            NF > 0 {
                rows++
                if (NF != 90) bad_columns=1
                for (column = 1; column <= NF; column++) {
                    raw=$column
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw)
                    lower=tolower(raw)
                    if (lower ~ /nan|inf/ ||
                        raw !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) {
                        bad_numeric=1
                        value=0
                    } else {
                        value=raw+0
                    }
                    matrix[rows,column]=value
                    if (value < -1e-8) negative=1
                    if (rows == column && absolute(value) > 1e-8) bad_diagonal=1
                }
            }
            END {
                dimensions = bad_columns ? rows "xINVALID" : rows "x90"
                for (row = 1; row <= 90; row++) {
                    strength=0
                    for (column = 1; column <= 90; column++) {
                        left=matrix[row,column]
                        right=matrix[column,row]
                        tolerance=1e-6 * (1 + absolute(left) + absolute(right))
                        if (absolute(left-right) > tolerance) bad_symmetry=1
                        strength += absolute(left)
                        if (row < column && absolute(left) > 1e-12) nonzero_edges++
                    }
                    if (strength <= 1e-12) isolated_nodes++
                }
                density=nonzero_edges/(90*89/2)
                printf "%s %d %d %d %d %d %d %.8f\n", \
                    dimensions, bad_numeric+0, negative+0, bad_diagonal+0, \
                    bad_symmetry+0, isolated_nodes+0, nonzero_edges+0, density
            }
        ' "$matrix"
    )"
    local dimensions bad_numeric negative bad_diagonal bad_symmetry
    local isolated_nodes nonzero_edges density
    read -r dimensions bad_numeric negative bad_diagonal bad_symmetry \
        isolated_nodes nonzero_edges density <<< "$results"

    record_qc connectome "${matrix_name}.dimensions" "$dimensions" \
        "$([[ "$dimensions" == 90x90 ]] && echo PASS || echo FAIL)" "90x90"
    record_qc connectome "${matrix_name}.numeric_values" "$bad_numeric" \
        "$([[ "$bad_numeric" == 0 ]] && echo PASS || echo FAIL)" "0 invalid values"
    record_qc connectome "${matrix_name}.negative_values" "$negative" \
        "$([[ "$negative" == 0 ]] && echo PASS || echo FAIL)" "0"
    record_qc connectome "${matrix_name}.nonzero_diagonal" "$bad_diagonal" \
        "$([[ "$bad_diagonal" == 0 ]] && echo PASS || echo FAIL)" "0"
    record_qc connectome "${matrix_name}.asymmetry" "$bad_symmetry" \
        "$([[ "$bad_symmetry" == 0 ]] && echo PASS || echo FAIL)" "0"
    record_qc connectome "${matrix_name}.isolated_nodes" "$isolated_nodes" \
        "$([[ "$isolated_nodes" == 0 ]] && echo PASS || echo WARN)" "0 preferred"
    record_qc connectome "${matrix_name}.nonzero_edges" "$nonzero_edges" \
        REVIEW "interpret with density and cohort distribution"
    record_qc connectome "${matrix_name}.density" "$density" \
        REVIEW "no universal single-subject threshold"

    if [[ "$dimensions" != 90x90 || "$bad_numeric" != 0 || "$negative" != 0 \
            || "$bad_diagonal" != 0 || "$bad_symmetry" != 0 ]]; then
        echo "ERROR: 连接矩阵自动 QC 失败: $matrix ($results)" >&2
        exit 32
    fi
}

enter_stage preprocessing

RAW="$WORK/dwi_raw.mif"
DENOISED="$WORK/dwi_denoised.mif"
NOISE="$WORK/noise.mif"
DEGIBBS="$WORK/dwi_degibbs.mif"
PREPROC="$WORK/dwi_preprocessed.mif"
BIASCORR="$WORK/dwi_biascorrected.mif"
MASK="$WORK/dwi_mask.mif"
MEAN_B0="$WORK/mean_b0.mif"

run_step 01_import_dwi \
    mrconvert "$DWI" "$RAW" \
        -fslgrad "$BVEC" "$BVAL" \
        -json_import "$DWI_JSON" \
        -force
run_step 02_denoise \
    dwidenoise "$RAW" "$DENOISED" \
        -noise "$NOISE" \
        -nthreads "$THREADS" \
        -force
run_step 03_degibbs \
    mrdegibbs "$DENOISED" "$DEGIBBS" \
        -nthreads "$THREADS" \
        -force

if [[ "$PREPROCESSING_MODE" == fsl ]]; then
    AP_MIF="$WORK/fmap_ap.mif"
    PA_MIF="$WORK/fmap_pa.mif"
    SE_EPI="$WORK/fmap_ap_pa.mif"
    run_step 04a_import_fmap_ap \
        mrconvert "$FMAP_AP" "$AP_MIF" \
            -json_import "$FMAP_AP_JSON" \
            -force
    run_step 04b_import_fmap_pa \
        mrconvert "$FMAP_PA" "$PA_MIF" \
            -json_import "$FMAP_PA_JSON" \
            -force
    run_step 04c_concat_fmaps \
        mrcat "$AP_MIF" "$PA_MIF" "$SE_EPI" \
            -axis 3 \
            -force
    run_step 04d_topup_eddy \
        dwifslpreproc "$DEGIBBS" "$PREPROC" \
            -rpe_pair \
            -se_epi "$SE_EPI" \
            -pe_dir "$DWI_PE_DIR" \
            -align_seepi \
            -eddy_options " --slm=linear --repol" \
            -eddyqc_all "$OUTPUT/qc/eddy" \
            -scratch "$SCRATCH" \
            -nocleanup \
            -nthreads "$THREADS" \
            -force
elif [[ "$PREPROCESSING_MODE" == fsl_rpe_none ]]; then
    echo "WARNING: AP/PA 场图对不适用；跳过 topup，使用单相位编码 eddy：$RPE_PAIR_REASON"
    run_step 04d_eddy_no_reverse_pe \
        dwifslpreproc "$DEGIBBS" "$PREPROC" \
            -rpe_none \
            -pe_dir "$DWI_PE_DIR" \
            -eddy_options " --slm=linear --repol" \
            -eddyqc_all "$OUTPUT/qc/eddy" \
            -scratch "$SCRATCH" \
            -nocleanup \
            -nthreads "$THREADS" \
            -force
else
    echo "WARNING: FSL 不可用；跳过 topup/eddy。"
    run_step 04_basic_preprocessing \
        mrconvert "$DEGIBBS" "$PREPROC" -force
fi

case "$BIAS_MODE" in
    ants)
        run_step 05_bias_ants \
            dwibiascorrect ants "$PREPROC" "$BIASCORR" \
                -bias "$WORK/dwi_bias.mif" \
                -scratch "$SCRATCH" \
                -nocleanup \
                -nthreads "$THREADS" \
                -force
        ;;
    fsl)
        run_step 05_bias_fsl \
            dwibiascorrect fsl "$PREPROC" "$BIASCORR" \
                -bias "$WORK/dwi_bias.mif" \
                -scratch "$SCRATCH" \
                -nocleanup \
                -nthreads "$THREADS" \
                -force
        ;;
    none)
        echo "WARNING: ANTs/FSL FAST 不可用；跳过 DWI bias correction。"
        run_step 05_no_bias_correction \
            mrconvert "$PREPROC" "$BIASCORR" -force
        ;;
esac

run_step 06_brain_mask \
    dwi2mask "$BIASCORR" "$MASK" \
        -nthreads "$THREADS" \
        -force

if prepare_step 07_mean_b0; then
    dwiextract "$BIASCORR" - -bzero |
        mrmath - mean "$MEAN_B0" -axis 3 -force
    complete_step 07_mean_b0
fi

enter_stage anatomy

FS_SUBJECT="$SUBJECT"
FS_SUBJECT_DIR="$SUBJECTS_DIR/$FS_SUBJECT"
FS_MODE_DIRECTORY="$WORK/freesurfer_modes"
FS_MODE_FILE="$FS_MODE_DIRECTORY/$FS_SUBJECT.mode"
if prepare_step 08_freesurfer_recon_all; then
    echo "       FreeSurfer recon-all 可能需要数小时"
    mkdir -p "$FS_MODE_DIRECTORY"

    if [[ -d "$FS_SUBJECT_DIR" && ! -f "$FS_MODE_FILE" ]]; then
        failed_run_directory="$WORK/failed_freesurfer_runs"
        failed_run_timestamp="$(date +%Y%m%dT%H%M%S)"
        failed_run_backup="$failed_run_directory/${FS_SUBJECT}.pre-classic-${failed_run_timestamp}"
        mkdir -p "$failed_run_directory"
        echo "[RECOVER] 检测到旧模式的未完成 FreeSurfer subject。"
        echo "          移动到可恢复备份：$failed_run_backup"
        mv "$FS_SUBJECT_DIR" "$failed_run_backup"
        record_event PARTIAL_FREESURFER_ARCHIVED anatomy \
            08_freesurfer_recon_all "$failed_run_backup"
    fi

    if [[ -d "$FS_SUBJECT_DIR" && -f "$FS_MODE_FILE" \
            && ! -f "$FS_SUBJECT_DIR/mri/orig.mgz" ]]; then
        failed_run_directory="$WORK/failed_freesurfer_runs"
        failed_run_timestamp="$(date +%Y%m%dT%H%M%S)"
        failed_run_backup="$failed_run_directory/${FS_SUBJECT}.incomplete-${failed_run_timestamp}"
        mkdir -p "$failed_run_directory"
        echo "[RECOVER] FreeSurfer subject 未完成输入初始化。"
        echo "          移动到可恢复备份：$failed_run_backup"
        mv "$FS_SUBJECT_DIR" "$failed_run_backup"
        rm -f "$FS_MODE_FILE"
        record_event PARTIAL_FREESURFER_ARCHIVED anatomy \
            08_freesurfer_recon_all "$failed_run_backup"
    fi

    if [[ -f "$FS_MODE_FILE" ]] \
            && [[ "$(cat "$FS_MODE_FILE")" != "$FS_RECON_MODE" ]]; then
        echo "ERROR: FreeSurfer 中间结果模式与当前模式不一致：" >&2
        echo "  existing=$(cat "$FS_MODE_FILE")" >&2
        echo "  current=$FS_RECON_MODE" >&2
        exit 36
    fi

    if [[ -f "$FS_SUBJECT_DIR/mri/orig.mgz" ]]; then
        echo "[RESUME] 继续已有 FreeSurfer subject，不再传入 -i。"
        recon-all \
            -no-v8 \
            -subject "$FS_SUBJECT" \
            -all \
            -parallel \
            -openmp "$THREADS"
    else
        printf '%s\n' "$FS_RECON_MODE" > "$FS_MODE_FILE"
        echo "[NEW] 使用经典 FreeSurfer 模式初始化 subject（禁用 SynthSeg）。"
        recon-all \
            -no-v8 \
            -subject "$FS_SUBJECT" \
            -i "$T1" \
            -all \
            -parallel \
            -openmp "$THREADS"
    fi
    complete_step 08_freesurfer_recon_all
fi

run_step 08b_freesurfer_head_surface \
    mkheadsurf \
        -s "$FS_SUBJECT" \
        -sd "$SUBJECTS_DIR"

if prepare_step 08c_freesurfer_head_surface_lowres; then
    mkdir -p "$FS_SUBJECT_DIR/bem"
    mris_remesh \
        --input "$FS_SUBJECT_DIR/surf/lh.seghead" \
        --output "$FS_SUBJECT_DIR/bem/outer_skin.surf" \
        --nvert 2562
    complete_step 08c_freesurfer_head_surface_lowres
fi

FIVE_TT_T1="$WORK/5tt_t1.mif"
FS_PARCELLATION="$FS_SUBJECT_DIR/mri/aparc+aseg.mgz"
run_step 09_generate_5tt \
    5ttgen freesurfer "$FS_PARCELLATION" "$FIVE_TT_T1" \
        -nocrop \
        -scratch "$SCRATCH" \
        -nocleanup \
        -nthreads "$THREADS" \
        -force

enter_stage registration

FS_BRAIN_NII="$WORK/freesurfer_brain.nii.gz"
run_step 10a_export_freesurfer_brain \
    mrconvert "$FS_SUBJECT_DIR/mri/brain.mgz" "$FS_BRAIN_NII" \
        -force

MNI_TO_T1_PREFIX="$WORK/mni_to_t1_"
run_step 10b_register_mni_to_t1 \
    antsRegistrationSyNQuick.sh \
        -d 3 \
        -f "$FS_BRAIN_NII" \
        -m "$MNI_TEMPLATE" \
        -o "$MNI_TO_T1_PREFIX" \
        -t s \
        -n "$THREADS" \
        -p f \
        -y 1

NODES_T1_NII="$WORK/aal90_t1.nii.gz"
run_step 10c_warp_aal90_to_t1 \
    antsApplyTransforms \
        -d 3 \
        -i "$AAL90_MNI" \
        -r "$FS_BRAIN_NII" \
        -o "$NODES_T1_NII" \
        -n GenericLabel \
        -t "${MNI_TO_T1_PREFIX}1Warp.nii.gz" \
        -t "${MNI_TO_T1_PREFIX}0GenericAffine.mat"

NODES_T1="$WORK/aal90_t1.mif"
run_step 10d_import_aal90_t1 \
    mrconvert "$NODES_T1_NII" "$NODES_T1" \
        -datatype uint32 \
        -force

if prepare_step 10e_node_table; then
    cp -f "$AAL90_MAPPING" "$OUTPUT/nodes.tsv"
    complete_step 10e_node_table
fi

MEAN_B0_NII="$WORK/mean_b0.nii.gz"
run_step 11a_export_mean_b0 \
    mrconvert "$MEAN_B0" "$MEAN_B0_NII" -force

REG_DAT="$WORK/dwi_to_t1.dat"
REG_FSL="$WORK/dwi_to_t1_fsl.mat"
if prepare_step 11b_bbregister; then
    bbregister \
        --s "$FS_SUBJECT" \
        --mov "$MEAN_B0_NII" \
        --reg "$REG_DAT" \
        --fslmat "$REG_FSL" \
        --dti \
        --init-coreg
    complete_step 11b_bbregister
fi

DWI_TO_T1="$WORK/dwi_to_t1_mrtrix.txt"
run_step 11c_convert_transform \
    transformconvert \
        "$REG_FSL" \
        "$MEAN_B0_NII" \
        "$FS_SUBJECT_DIR/mri/orig.mgz" \
        flirt_import \
        "$DWI_TO_T1" \
        -force

FIVE_TT_DWI="$WORK/5tt_dwi.mif"
NODES_DWI="$WORK/nodes_dwi.mif"
FS_BRAIN_DWI="$WORK/freesurfer_brain_in_dwi.mif"
run_step 12a_transform_5tt \
    mrtransform "$FIVE_TT_T1" "$FIVE_TT_DWI" \
        -linear "$DWI_TO_T1" \
        -inverse \
        -template "$MEAN_B0" \
        -interp linear \
        -force
run_step 12b_transform_nodes \
    mrtransform "$NODES_T1" "$NODES_DWI" \
        -linear "$DWI_TO_T1" \
        -inverse \
        -template "$MEAN_B0" \
        -interp nearest \
        -datatype uint32 \
        -force
run_step 12c_transform_t1_brain \
    mrtransform "$FS_BRAIN_NII" "$FS_BRAIN_DWI" \
        -linear "$DWI_TO_T1" \
        -inverse \
        -template "$MEAN_B0" \
        -interp cubic \
        -force

NODES_DWI_NII="$OUTPUT/qc/nodes_in_dwi_space.nii.gz"
run_step 12d_export_nodes_dwi \
    mrconvert "$NODES_DWI" "$NODES_DWI_NII" \
        -datatype uint32 \
        -force
start_transient_step 12e_validate_aal90
validate_aal90_segmentation \
    "$NODES_DWI_NII" \
    "$OUTPUT/qc/aal90_segmentation_stats.txt"
complete_transient_step 12e_validate_aal90

enter_stage fod

GMWMI="$WORK/gmwmi_seed.mif"
run_step 13_gmwmi_seed \
    5tt2gmwmi "$FIVE_TT_DWI" "$GMWMI" -force

WM_RESPONSE="$WORK/response_wm.txt"
GM_RESPONSE="$WORK/response_gm.txt"
CSF_RESPONSE="$WORK/response_csf.txt"
if prepare_step 14_response_dhollander; then
    dwi2response dhollander \
        "$BIASCORR" \
        "$WM_RESPONSE" "$GM_RESPONSE" "$CSF_RESPONSE" \
        -mask "$MASK" \
        -voxels "$WORK/response_voxels.mif" \
        -scratch "$SCRATCH" \
        -nocleanup \
        -nthreads "$THREADS" \
        -force
    complete_step 14_response_dhollander
fi

WM_FOD="$WORK/wm_fod.mif"
GM_FOD="$WORK/gm_fod.mif"
CSF_FOD="$WORK/csf_fod.mif"
if prepare_step 15_msmt_csd; then
    dwi2fod msmt_csd \
        "$BIASCORR" \
        "$WM_RESPONSE" "$WM_FOD" \
        "$GM_RESPONSE" "$GM_FOD" \
        "$CSF_RESPONSE" "$CSF_FOD" \
        -mask "$MASK" \
        -nthreads "$THREADS" \
        -force
    complete_step 15_msmt_csd
fi

WM_FOD_NORM="$WORK/wm_fod_normalised.mif"
GM_FOD_NORM="$WORK/gm_fod_normalised.mif"
CSF_FOD_NORM="$WORK/csf_fod_normalised.mif"
run_step 16_mtnormalise \
    mtnormalise \
        "$WM_FOD" "$WM_FOD_NORM" \
        "$GM_FOD" "$GM_FOD_NORM" \
        "$CSF_FOD" "$CSF_FOD_NORM" \
        -mask "$MASK" \
        -nthreads "$THREADS" \
        -force

enter_stage tractography

TRACKS="$WORK/tracks_${STREAMLINES}.tck"
run_step 17_act_tractography \
    tckgen \
        "$WM_FOD_NORM" \
        "$TRACKS" \
        -algorithm iFOD2 \
        -act "$FIVE_TT_DWI" \
        -backtrack \
        -crop_at_gmwmi \
        -seed_gmwmi "$GMWMI" \
        -select "$STREAMLINES" \
        -step 0.9 \
        -minlength 10 \
        -maxlength 250 \
        -cutoff 0.06 \
        -nthreads "$THREADS" \
        -force

SIFT2_WEIGHTS="$WORK/sift2_weights.txt"
run_step 18_sift2 \
    tcksift2 \
        "$TRACKS" \
        "$WM_FOD_NORM" \
        "$SIFT2_WEIGHTS" \
        -act "$FIVE_TT_DWI" \
        -out_mu "$OUTPUT/sift2_mu.txt" \
        -nthreads "$THREADS" \
        -force

enter_stage connectome

STREAMLINE_ASSIGNMENTS="$WORK/streamline_assignments.txt"
run_step 19a_connectome_sift2 \
    tck2connectome \
        "$TRACKS" "$NODES_DWI" \
        "$OUTPUT/connectome_sift2.csv" \
        -tck_weights_in "$SIFT2_WEIGHTS" \
        -out_assignments "$STREAMLINE_ASSIGNMENTS" \
        -symmetric \
        -zero_diagonal \
        -assignment_radial_search 2 \
        -force
run_step 19b_connectome_count \
    tck2connectome \
        "$TRACKS" "$NODES_DWI" \
        "$OUTPUT/connectome_streamline_count.csv" \
        -stat_edge sum \
        -symmetric \
        -zero_diagonal \
        -assignment_radial_search 2 \
        -force
run_step 19c_connectome_length \
    tck2connectome \
        "$TRACKS" "$NODES_DWI" \
        "$OUTPUT/connectome_mean_length_mm.csv" \
        -scale_length \
        -stat_edge mean \
        -symmetric \
        -zero_diagonal \
        -assignment_radial_search 2 \
        -force

start_transient_step 19d_validate_connectomes
validate_connectome_matrix \
    "$OUTPUT/connectome_sift2.csv" sift2
validate_connectome_matrix \
    "$OUTPUT/connectome_streamline_count.csv" streamline_count
validate_connectome_matrix \
    "$OUTPUT/connectome_mean_length_mm.csv" mean_length_mm
complete_transient_step 19d_validate_connectomes

enter_stage qc

RAW_MEAN_B0="$WORK/raw_mean_b0.mif"
if prepare_step 20a_qc_raw_mean_b0; then
    dwiextract "$RAW" - -bzero |
        mrmath - mean "$RAW_MEAN_B0" -axis 3 -force
    mrconvert "$RAW_MEAN_B0" "$OUTPUT/qc/raw_mean_b0.nii.gz" -force
    complete_step 20a_qc_raw_mean_b0
fi
run_step 20b_qc_processed_mean_b0 \
    mrconvert "$MEAN_B0" "$OUTPUT/qc/preprocessed_mean_b0.nii.gz" -force
run_step 20c_qc_mask \
    mrconvert "$MASK" "$OUTPUT/qc/dwi_mask.nii.gz" -force
run_step 20d_qc_noise \
    mrconvert "$NOISE" "$OUTPUT/qc/denoise_noise_map.nii.gz" -force
if [[ -f "$WORK/dwi_bias.mif" ]]; then
    run_step 20e_qc_bias_field \
        mrconvert "$WORK/dwi_bias.mif" "$OUTPUT/qc/dwi_bias_field.nii.gz" -force
fi

run_step 20f1_qc_freesurfer_t1 \
    mrconvert "$FS_SUBJECT_DIR/mri/T1.mgz" \
        "$OUTPUT/qc/freesurfer_T1.nii.gz" \
        -force
run_step 20f2_qc_freesurfer_brain_t1 \
    mrconvert "$FS_BRAIN_NII" "$OUTPUT/qc/freesurfer_brain_t1.nii.gz" -force
run_step 20g_qc_freesurfer_aseg \
    mrconvert "$FS_SUBJECT_DIR/mri/aseg.mgz" \
        "$OUTPUT/qc/freesurfer_aseg.nii.gz" \
        -datatype uint32 \
        -force
run_step 20h_qc_freesurfer_wmparc \
    mrconvert "$FS_SUBJECT_DIR/mri/wmparc.mgz" \
        "$OUTPUT/qc/freesurfer_wmparc.nii.gz" \
        -datatype uint32 \
        -force
run_step 20i1_qc_t1_in_dwi_fsl_grid \
    export_with_reference_strides \
        "$MEAN_B0_NII" \
        "$FS_BRAIN_DWI" \
        "$OUTPUT/qc/freesurfer_brain_in_dwi_space.nii.gz"
run_step 20j_qc_nodes_t1 \
    cp -f "$NODES_T1_NII" "$OUTPUT/qc/nodes_in_t1_space.nii.gz"
run_step 20k_qc_mni_registration \
    cp -f "${MNI_TO_T1_PREFIX}Warped.nii.gz" \
        "$OUTPUT/qc/mni152_to_t1_warped.nii.gz"
run_step 20l_qc_5tt_dwi \
    mrconvert "$FIVE_TT_DWI" "$OUTPUT/qc/5tt_in_dwi_space.nii.gz" -force
run_step 20m_qc_gmwmi_seed \
    mrconvert "$GMWMI" "$OUTPUT/qc/gmwmi_seed.nii.gz" -force
run_step 20n_qc_wm_fod_l0 \
    mrconvert "$WM_FOD_NORM" "$OUTPUT/qc/wm_fod_l0.nii.gz" \
        -coord 3 0 \
        -force
run_step 20o_export_gradients \
    mrinfo "$BIASCORR" \
        -export_grad_fsl \
        "$OUTPUT/qc/gradients_after_preprocessing.bvec" \
        "$OUTPUT/qc/gradients_after_preprocessing.bval"

if prepare_step 20p_qc_freesurfer_logs; then
    mkdir -p "$OUTPUT/qc/freesurfer"
    cp -f "$FS_SUBJECT_DIR/scripts/recon-all.log" \
        "$OUTPUT/qc/freesurfer/recon-all.log"
    cp -f "$FS_SUBJECT_DIR/scripts/recon-all-status.log" \
        "$OUTPUT/qc/freesurfer/recon-all-status.log"
    cp -f "$FS_SUBJECT_DIR/stats/aseg.stats" \
        "$OUTPUT/qc/freesurfer/aseg.stats"
    cp -f "$FS_SUBJECT_DIR/stats/lh.aparc.stats" \
        "$OUTPUT/qc/freesurfer/lh.aparc.stats"
    cp -f "$FS_SUBJECT_DIR/stats/rh.aparc.stats" \
        "$OUTPUT/qc/freesurfer/rh.aparc.stats"
    complete_step 20p_qc_freesurfer_logs
fi

run_step 20q_qc_mni_to_t1_montage \
    slices "$FS_BRAIN_NII" "${MNI_TO_T1_PREFIX}Warped.nii.gz" \
        -o "$OUTPUT/qc/mni_to_t1_registration.gif"
run_step 20r_qc_t1_to_dwi_montage \
    slices "$MEAN_B0_NII" "$OUTPUT/qc/freesurfer_brain_in_dwi_space.nii.gz" \
        -o "$OUTPUT/qc/t1_to_dwi_registration.gif"
run_step 20r1_qc_raw_to_preprocessed_b0_montage \
    slices "$OUTPUT/qc/raw_mean_b0.nii.gz" \
        "$OUTPUT/qc/preprocessed_mean_b0.nii.gz" \
        -o "$OUTPUT/qc/raw_to_preprocessed_b0.gif"
run_step 20r2_qc_freesurfer_brainmask_montage \
    slices "$OUTPUT/qc/freesurfer_T1.nii.gz" \
        "$OUTPUT/qc/freesurfer_brain_t1.nii.gz" \
        -o "$OUTPUT/qc/freesurfer_brainmask.gif"

if prepare_step 20s_qc_tractogram_stats; then
    tckstats "$TRACKS" \
        -histogram "$OUTPUT/qc/streamline_length_histogram.tsv" \
        -output mean \
        -output median \
        -output std \
        -output min \
        -output max \
        -output count \
        -nthreads "$THREADS" \
        -quiet \
        > "$WORK/tckstats_values.txt"
    read -r track_mean track_median track_std track_minimum track_maximum \
        track_count < "$WORK/tckstats_values.txt"
    {
        printf 'metric\tvalue\n'
        printf 'mean_length_mm\t%s\n' "$track_mean"
        printf 'median_length_mm\t%s\n' "$track_median"
        printf 'std_length_mm\t%s\n' "$track_std"
        printf 'minimum_length_mm\t%s\n' "$track_minimum"
        printf 'maximum_length_mm\t%s\n' "$track_maximum"
        printf 'streamline_count\t%s\n' "$track_count"
    } > "$OUTPUT/qc/tractogram_stats.tsv"
    complete_step 20s_qc_tractogram_stats
fi

start_transient_step 20t_compute_qc_metrics
FS_T1_MASK="$WORK/qc_fs_t1_mask.mif"
MNI_T1_MASK="$WORK/qc_mni_t1_mask.mif"
FS_DWI_MASK="$WORK/qc_fs_dwi_mask.mif"
NODES_MASK="$WORK/qc_nodes_mask.mif"
NODES_IN_BRAIN="$WORK/qc_nodes_in_brain.mif"
MNI_FS_INTERSECTION="$WORK/qc_mni_fs_intersection.mif"
DWI_FS_INTERSECTION="$WORK/qc_dwi_fs_intersection.mif"
mrcalc "$FS_BRAIN_NII" 0 -gt "$FS_T1_MASK" -force
mrcalc "${MNI_TO_T1_PREFIX}Warped.nii.gz" 0 -gt "$MNI_T1_MASK" -force
mrcalc "$FS_T1_MASK" "$MNI_T1_MASK" -mult "$MNI_FS_INTERSECTION" -force
mrcalc "$FS_BRAIN_DWI" 0 -gt "$FS_DWI_MASK" -force
mrcalc "$MASK" "$FS_DWI_MASK" -mult "$DWI_FS_INTERSECTION" -force
mrcalc "$NODES_DWI" 0 -gt "$NODES_MASK" -force
mrcalc "$NODES_MASK" "$MASK" -mult "$NODES_IN_BRAIN" -force

count_mask_voxels() {
    mrstats "$1" -mask "$1" -output count -quiet
}

fs_t1_count="$(count_mask_voxels "$FS_T1_MASK")"
mni_t1_count="$(count_mask_voxels "$MNI_T1_MASK")"
mni_fs_intersection_count="$(count_mask_voxels "$MNI_FS_INTERSECTION")"
mni_t1_dice="$(
    awk -v intersection="$mni_fs_intersection_count" \
        -v first="$fs_t1_count" \
        -v second="$mni_t1_count" \
        'BEGIN {
            if (first+second == 0) print "0.000000"
            else printf "%.6f", 2*intersection/(first+second)
        }'
)"
mni_t1_status="$(
    awk -v value="$mni_t1_dice" \
        'BEGIN {print value >= 0.80 ? "PASS" : "WARN"}'
)"
record_qc registration mni_to_t1_brainmask_dice \
    "$mni_t1_dice" "$mni_t1_status" \
    ">=0.80 heuristic; visual review required"

dwi_mask_count="$(count_mask_voxels "$MASK")"
fs_dwi_count="$(count_mask_voxels "$FS_DWI_MASK")"
dwi_fs_intersection_count="$(count_mask_voxels "$DWI_FS_INTERSECTION")"
t1_dwi_dice="$(
    awk -v intersection="$dwi_fs_intersection_count" \
        -v first="$dwi_mask_count" \
        -v second="$fs_dwi_count" \
        'BEGIN {
            if (first+second == 0) print "0.000000"
            else printf "%.6f", 2*intersection/(first+second)
        }'
)"
t1_dwi_status="$(
    awk -v value="$t1_dwi_dice" \
        'BEGIN {print value >= 0.80 ? "PASS" : "WARN"}'
)"
record_qc registration t1_to_dwi_brainmask_dice \
    "$t1_dwi_dice" "$t1_dwi_status" \
    ">=0.80 heuristic; visual review required"
record_qc dwi brain_mask_voxels "$dwi_mask_count" REVIEW \
    "compare with cohort distribution"

node_voxel_count="$(count_mask_voxels "$NODES_MASK")"
node_in_brain_count="$(count_mask_voxels "$NODES_IN_BRAIN")"
node_brain_coverage="$(
    awk -v inside="$node_in_brain_count" -v total="$node_voxel_count" \
        'BEGIN {
            if (total == 0) print "0.000000"
            else printf "%.6f", inside/total
        }'
)"
node_coverage_status="$(
    awk -v value="$node_brain_coverage" \
        'BEGIN {print value >= 0.90 ? "PASS" : "WARN"}'
)"
record_qc atlas parcel_voxels_inside_dwi_mask_fraction \
    "$node_brain_coverage" "$node_coverage_status" \
    ">=0.90 heuristic; visual review required"

raw_b0_mean="$(mrstats "$RAW_MEAN_B0" -mask "$MASK" -output mean -quiet)"
processed_b0_mean="$(mrstats "$MEAN_B0" -mask "$MASK" -output mean -quiet)"
record_qc dwi raw_mean_b0_signal "$raw_b0_mean" REVIEW \
    "compare with processed image and cohort"
record_qc dwi preprocessed_mean_b0_signal "$processed_b0_mean" REVIEW \
    "compare with raw image and cohort"

if [[ -f "$OUTPUT/qc/eddy/eddy_movement_rms" ]]; then
    read -r mean_absolute_rms mean_relative_rms maximum_absolute_rms \
        maximum_relative_rms <<< "$(
            awk '
                {
                    absolute_sum += $1
                    relative_sum += $2
                    if ($1 > absolute_max) absolute_max=$1
                    if ($2 > relative_max) relative_max=$2
                    count++
                }
                END {
                    if (count == 0) print "0 0 0 0"
                    else printf "%.6f %.6f %.6f %.6f", \
                        absolute_sum/count, relative_sum/count, \
                        absolute_max, relative_max
                }
            ' "$OUTPUT/qc/eddy/eddy_movement_rms"
        )"
    record_qc eddy mean_absolute_rms_mm "$mean_absolute_rms" REVIEW \
        "inspect eddy_quad and compare with cohort"
    record_qc eddy mean_relative_rms_mm "$mean_relative_rms" REVIEW \
        "inspect eddy_quad and compare with cohort"
    record_qc eddy maximum_absolute_rms_mm "$maximum_absolute_rms" REVIEW \
        "inspect eddy_quad"
    record_qc eddy maximum_relative_rms_mm "$maximum_relative_rms" REVIEW \
        "inspect eddy_quad"
fi
if [[ -f "$OUTPUT/qc/eddy/eddy_outlier_map" ]]; then
    eddy_outlier_percentage="$(
        awk '
            {
                for (column=1; column<=NF; column++) {
                    outliers += $column
                    total++
                }
            }
            END {
                if (total == 0) print "0.000000"
                else printf "%.6f", 100*outliers/total
            }
        ' "$OUTPUT/qc/eddy/eddy_outlier_map"
    )"
    eddy_outlier_status="$(
        awk -v value="$eddy_outlier_percentage" \
            'BEGIN {print value <= 10 ? "PASS" : "WARN"}'
    )"
    record_qc eddy outlier_slice_percentage "$eddy_outlier_percentage" \
        "$eddy_outlier_status" \
        "<=10% heuristic; inspect eddy_quad"
fi
if [[ -f "$OUTPUT/qc/eddy/quad/index.html" ]]; then
    record_qc eddy eddy_quad_report present PASS \
        "open qc/eddy/quad/index.html"
else
    record_qc eddy eddy_quad_report missing WARN \
        "inspect text outputs in qc/eddy"
fi

if grep -qi 'finished without error' \
        "$FS_SUBJECT_DIR/scripts/recon-all-status.log"; then
    record_qc freesurfer recon_all_completion complete PASS \
        "recon-all status reports completion without error"
else
    record_qc freesurfer recon_all_completion uncertain WARN \
        "inspect qc/freesurfer/recon-all-status.log"
fi
if [[ -s "$FS_SUBJECT_DIR/surf/lh.seghead" \
        && -s "$FS_SUBJECT_DIR/bem/outer_skin.surf" ]]; then
    record_qc freesurfer eeg_scalp_surface present PASS \
        "dense and low-resolution scalp surfaces are available for EEG-to-MRI coregistration"
else
    record_qc freesurfer eeg_scalp_surface missing FAIL \
        "rerun anatomy stage; dense and low-resolution scalp surfaces are required"
fi

read -r track_mean track_median track_std track_minimum track_maximum \
    track_count < "$WORK/tckstats_values.txt"
record_qc tractography streamline_count "$track_count" \
    "$([[ "$track_count" == "$STREAMLINES" ]] && echo PASS || echo WARN)" \
    "requested $STREAMLINES"
record_qc tractography mean_length_mm "$track_mean" REVIEW \
    "10-250 mm generation bounds; compare with cohort"
record_qc tractography median_length_mm "$track_median" REVIEW \
    "compare with cohort"
record_qc tractography minimum_length_mm "$track_minimum" REVIEW \
    "generation minimum approximately 10 mm"
record_qc tractography maximum_length_mm "$track_maximum" REVIEW \
    "generation maximum approximately 250 mm"

read -r assignment_total assignment_valid assignment_percentage <<< "$(
    awk '
        !/^#/ && NF >= 2 {
            total++
            if ($1 > 0 && $2 > 0) assigned++
        }
        END {
            if (total == 0) print "0 0 0.000000"
            else printf "%d %d %.6f", total, assigned, 100*assigned/total
        }
    ' "$STREAMLINE_ASSIGNMENTS"
)"
assignment_status="$(
    awk -v value="$assignment_percentage" \
        'BEGIN {print value >= 80 ? "PASS" : "WARN"}'
)"
record_qc tractography assigned_streamline_percentage \
    "$assignment_percentage" "$assignment_status" \
    ">=80% heuristic; inspect endpoints and parcellation"

read -r sift2_count sift2_sum sift2_mean sift2_min sift2_max <<< "$(
    awk '
        /^[[:space:]]*#/ {next}
        {
            for (field_index=1; field_index<=NF; field_index++) {
                value=$field_index+0
                if (count == 0 || value < minimum) minimum=value
                if (count == 0 || value > maximum) maximum=value
                sum += value
                count++
            }
        }
        END {
            if (count == 0) print "0 0 0 0 0"
            else printf "%d %.10g %.10g %.10g %.10g", \
                count, sum, sum/count, minimum, maximum
        }
    ' "$SIFT2_WEIGHTS"
)"
record_qc sift2 weight_count "$sift2_count" \
    "$([[ "$sift2_count" == "$STREAMLINES" ]] && echo PASS || echo WARN)" \
    "one weight per streamline"
record_qc sift2 weight_sum "$sift2_sum" REVIEW \
    "compare with cohort"
record_qc sift2 weight_mean "$sift2_mean" REVIEW \
    "compare with cohort"
record_qc sift2 weight_range "${sift2_min}..${sift2_max}" REVIEW \
    "non-negative; compare with cohort"

if grep -q $'\tFAIL\t' "$QC_METRICS"; then
    AUTOMATIC_QC_STATUS=FAIL
elif grep -q $'\tWARN\t' "$QC_METRICS"; then
    AUTOMATIC_QC_STATUS=WARN
else
    AUTOMATIC_QC_STATUS=PASS
fi

cat > "$OUTPUT/qc/QC_GUIDE.txt" <<QC_GUIDE
Structural connectivity QC guide
================================

Automatic QC status: $AUTOMATIC_QC_STATUS

PASS means the programmed criterion was met.
WARN requires inspection but does not automatically invalidate the subject.
REVIEW has no universal single-subject threshold and must be compared with
the images and, once available, the cohort distribution.

Required manual review:
1. Open qc/eddy/quad/index.html; inspect motion, outliers, CNR and residuals.
2. Inspect raw_to_preprocessed_b0.gif, both b0 images and dwi_mask.nii.gz.
3. Inspect freesurfer_brainmask.gif and FreeSurfer brain/aseg/wmparc
   against freesurfer_T1.nii.gz.
   eeg_scalp_surface must be PASS before source-space EEG analysis.
4. Inspect mni_to_t1_registration.gif and overlay nodes_in_t1_space.nii.gz
   on freesurfer_brain_t1.nii.gz.
5. Inspect t1_to_dwi_registration.gif and overlay nodes_in_dwi_space.nii.gz
   on preprocessed_mean_b0.nii.gz.
6. Inspect 5tt_in_dwi_space.nii.gz, gmwmi_seed.nii.gz and wm_fod_l0.nii.gz.
7. Inspect representative streamlines in MRview using the WSL work directory.
8. Review qc_metrics.tsv for isolated nodes, matrix density, streamline
   assignment percentage and cohort outliers.

Processing completion does not replace visual QC. A subject should only be
included after these manual checks are recorded.
QC_GUIDE

complete_transient_step 20t_compute_qc_metrics

cat > "$OUTPUT/summary.json" <<SUMMARY
{
  "subject": "$SUBJECT",
  "processing_status": "complete",
  "automatic_qc_status": "$AUTOMATIC_QC_STATUS",
  "manual_qc_required": true,
  "atlas": "AAL-90 (original AAL/SPM12 cerebrum labels)",
  "atlas_nodes": 90,
  "preprocessing": "$PREPROCESSING_MODE",
  "bias_correction": "$BIAS_MODE",
  "freesurfer_recon_mode": "$FS_RECON_MODE",
  "freesurfer_subjects_dir_wsl": "$SUBJECTS_DIR",
  "freesurfer_subject_id": "$FS_SUBJECT",
  "freesurfer_scalp_surface_wsl": "$FS_SUBJECT_DIR/surf/lh.seghead",
  "freesurfer_scalp_surface_lowres_wsl": "$FS_SUBJECT_DIR/bem/outer_skin.surf",
  "eeg_source_model_inputs_ready": true,
  "streamlines": $STREAMLINES,
  "connectome_sift2": "connectome_sift2.csv",
  "connectome_streamline_count": "connectome_streamline_count.csv",
  "connectome_mean_length_mm": "connectome_mean_length_mm.csv",
  "node_table": "nodes.tsv",
  "qc_directory": "qc",
  "qc_metrics": "qc/qc_metrics.tsv",
  "qc_guide": "qc/QC_GUIDE.txt",
  "eddy_qc": "qc/eddy/quad/index.html",
  "pipeline_state": "pipeline_state.json",
  "pipeline_events": "pipeline_events.tsv",
  "checkpoints": "checkpoints",
  "resumable": true,
  "work_directory_wsl": "$WORK",
  "intermediate_directory": "intermediates",
  "scratch_directory": "intermediates/scratch",
  "tractogram": "intermediates/tracks_${STREAMLINES}.tck"
}
SUMMARY

record_event STAGE_COMPLETED qc '' ''
record_event RUN_COMPLETED qc '' \
    "automatic_qc_status=$AUTOMATIC_QC_STATUS"
write_state completed qc ''

echo "Pipeline completed: $(date --iso-8601=seconds)"
echo "Main result: $OUTPUT/connectome_sift2.csv"
echo "Automatic QC status: $AUTOMATIC_QC_STATUS"
echo "Manual QC guide: $OUTPUT/qc/QC_GUIDE.txt"
"""


def run_pipeline(script: str, distro: str, output_directory: Path, preflight: bool) -> int:
    name = "preflight" if preflight else "pipeline"
    script_path = output_directory / f"{name}.sh"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    if os.name == "nt":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl:
            raise RuntimeError("WSL is required when launching from Windows.")
        command = [wsl, "--distribution", distro, "--exec", "bash", "-s"]
    else:
        bash = shutil.which("bash")
        if not bash:
            raise RuntimeError("Cannot find bash.")
        command = [bash, "-s"]
    log_path = output_directory / f"{name}_launcher.log"
    with script_path.open("rb") as script_file, log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n{datetime.now(timezone.utc).isoformat()}\n")
        log_file.flush()
        with subprocess.Popen(
            command,
            stdin=script_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(output_directory),
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            if process.stdout is None:
                raise RuntimeError("Cannot capture pipeline output.")
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return process.wait()


def _check_connectome_export_dependencies() -> tuple[str, str]:
    try:
        import matplotlib
        import numpy
    except ImportError as error:
        raise RuntimeError("NPY/PNG export requires NumPy and Matplotlib in the current Python environment.") from error
    return numpy.__version__, matplotlib.__version__


def export_connectome_derivatives(output_directory: Path) -> dict[str, Path]:

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix_files = {
        "sift2": output_directory / "connectome_sift2.csv",
        "streamline_count": output_directory / "connectome_streamline_count.csv",
        "mean_length_mm": output_directory / "connectome_mean_length_mm.csv",
    }
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, Path] = {}
    for matrix_name, csv_path in matrix_files.items():
        if not csv_path.is_file():
            raise RuntimeError(f"无法导出 NPY，缺少连接矩阵：{csv_path}")
        matrix = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        if matrix.shape != (90, 90):
            raise RuntimeError(f"{csv_path.name} 应为 90×90，实际为 {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{csv_path.name} 包含 NaN 或无穷值")
        if (matrix < -1e-8).any():
            raise RuntimeError(f"{csv_path.name} 包含负连接权重")
        if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
            raise RuntimeError(f"{csv_path.name} 不是对称矩阵")

        npy_path = csv_path.with_suffix(".npy")
        np.save(npy_path, matrix, allow_pickle=False)
        matrices[matrix_name] = matrix
        artifacts[f"connectome_{matrix_name}_npy"] = npy_path

    sift2_matrix = matrices["sift2"]
    display_matrix = np.log1p(np.maximum(sift2_matrix, 0))
    positive_values = display_matrix[display_matrix > 0]
    if positive_values.size:
        color_maximum = float(np.quantile(positive_values, 0.99))
        if color_maximum <= 0:
            color_maximum = float(positive_values.max())
    else:
        color_maximum = 1.0

    dark_blue = "#073563"
    figure, axis = plt.subplots(figsize=(10, 9), dpi=200)
    figure.patch.set_facecolor(dark_blue)
    axis.set_facecolor(dark_blue)
    image = axis.imshow(
        display_matrix,
        cmap="RdBu_r",
        vmin=0,
        vmax=color_maximum,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_title(
        "AAL-90 Structural Connectivity (SIFT2)",
        color="white",
        pad=14,
    )
    axis.set_xlabel("AAL-90 node index", color="white")
    axis.set_ylabel("AAL-90 node index", color="white")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label(
        "log1p(SIFT2 weight), clipped at positive 99th percentile",
        color="white",
    )
    colorbar.ax.tick_params(colors="white")
    colorbar.outline.set_edgecolor("white")

    png_path = output_directory / "connectome_sift2_heatmap.png"
    figure.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    artifacts["connectome_sift2_heatmap"] = png_path

    summary_path = output_directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "connectome_sift2_npy": "connectome_sift2.npy",
                "connectome_streamline_count_npy": ("connectome_streamline_count.npy"),
                "connectome_mean_length_mm_npy": ("connectome_mean_length_mm.npy"),
                "connectome_sift2_heatmap": "connectome_sift2_heatmap.png",
                "heatmap_display_transform": (
                    "log1p; color maximum = positive 99th percentile"
                ),
            }
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return artifacts


def show_pipeline_status(output_directory: Path) -> int:
    state_path = output_directory / "pipeline_state.json"
    if not state_path.is_file():
        print("[状态] 尚未找到流水线状态文件；正式计算可能还没有启动")
        print(f"[预期位置] {state_path}")
        return 0

    state = json.loads(state_path.read_text(encoding="utf-8"))
    print(f"[状态] {state.get('status', 'unknown')}")
    print(f"[受试者] {state.get('subject', 'unknown')}")
    print(f"[当前阶段] {state.get('current_stage', '')}")
    print(f"[当前步骤] {state.get('current_step', '') or '(无)'}")
    print(f"[已完成检查点] {state.get('completed_checkpoints', 0)}")
    print(f"[更新时间] {state.get('updated_at', '')}")
    print(f"[WSL 工作目录] {state.get('work_directory_wsl', '')}")

    event_path = output_directory / "pipeline_events.tsv"
    if event_path.is_file():
        event_lines = event_path.read_text(encoding="utf-8").splitlines()
        recent_events = event_lines[-8:]
        if recent_events:
            print("[最近事件]")
            for event_line in recent_events:
                print(f"  {event_line}")
    return 0


def normalize_subject_id(value: str) -> str:
    value = value.strip()
    label = value[4:] if value.startswith("sub-") else value
    if not label or not label.isascii() or not label.isalnum():
        raise ValueError(f"Invalid BIDS subject ID: {value!r}")
    return f"sub-{label}"


def session_directories(
    subject_directory: Path,
    session: str | None = None,
) -> list[Path]:
    if not subject_directory.is_dir():
        raise RuntimeError(f"MRI subject directory does not exist: {subject_directory}")
    if session is not None:
        label = session[4:] if session.startswith("ses-") else session
        if not label or not label.isascii() or not label.isalnum():
            raise ValueError(f"Invalid BIDS session label: {session!r}")
        session_directory = subject_directory / f"ses-{label}"
        if not session_directory.is_dir():
            raise RuntimeError(f"Requested MRI session does not exist: {session_directory}")
        return [session_directory]
    directories = sorted(
        path
        for path in subject_directory.glob("ses-*")
        if path.is_dir()
    )
    if (subject_directory / "anat").is_dir() or (subject_directory / "dwi").is_dir():
        directories.insert(0, subject_directory)
    if not directories:
        raise RuntimeError(f"No BIDS MRI sessions found in {subject_directory}")
    return directories


def select_subject_inputs(
    arguments: argparse.Namespace,
    subject: str,
) -> tuple[Path, InputFiles]:
    subject_directory = arguments.data_dir / "MRI" / subject
    candidates: list[tuple[Path, InputFiles]] = []
    errors: list[str] = []
    for session_directory in session_directories(subject_directory, arguments.session):
        try:
            inputs = discover_inputs(
                session_directory,
                t1w_name=arguments.t1w_name,
                dwi_name=arguments.dwi_name,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            errors.append(f"{session_directory.name}: {error}")
        else:
            candidates.append((session_directory, inputs))
    if not candidates:
        details = "\n".join(errors)
        raise RuntimeError(f"No usable MRI session for {subject}.\n{details}")
    if len(candidates) > 1:
        sessions = ", ".join(directory.name for directory, _ in candidates)
        raise RuntimeError(
            f"Multiple usable MRI sessions for {subject}: {sessions}. "
            "Select one with --session."
        )
    return candidates[0]


def select_subjects(arguments: argparse.Namespace) -> list[str]:
    if arguments.subject:
        subjects = list(
            dict.fromkeys(
                normalize_subject_id(value)
                for group in arguments.subject
                for value in group
            )
        )
    elif arguments.status:
        subjects = [
            normalize_subject_id(path.name)
            for path in sorted(DEFAULT_OUTPUT_DIRECTORY.glob("sub-*"))
            if (path / "structural_connectivity").is_dir()
        ]
    else:
        manifest_path = arguments.data_dir / DOWNLOAD_MANIFEST_FILENAME
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"bids_subject_id", "mri_status"}.issubset(reader.fieldnames or []):
                    raise RuntimeError(
                        f"Download manifest requires bids_subject_id and mri_status columns: {manifest_path}"
                    )
                subjects = list(
                    dict.fromkeys(
                        normalize_subject_id(row.get("bids_subject_id") or "")
                        for row in reader
                        if (row.get("mri_status") or "").strip().lower() == "success"
                    )
                )
        else:
            mri_directory = arguments.data_dir / "MRI"
            if not mri_directory.is_dir():
                raise RuntimeError(f"MRI download directory does not exist: {mri_directory}")
            subjects = []
            for subject_directory in sorted(mri_directory.glob("sub-*")):
                if not subject_directory.is_dir():
                    continue
                try:
                    subject = normalize_subject_id(subject_directory.name)
                    sessions = session_directories(subject_directory)
                except (OSError, RuntimeError, ValueError):
                    continue
                for session_directory in sessions:
                    has_t1w = any(
                        _is_downloaded_image(path)
                        for path in (session_directory / "anat").glob("*_T1w.nii.gz")
                    )
                    has_dwi = any(
                        _is_downloaded_image(path)
                        for path in (session_directory / "dwi").glob("*_dwi.nii.gz")
                    )
                    if has_t1w and has_dwi:
                        subjects.append(subject)
                        break
    if not subjects:
        if arguments.status:
            raise RuntimeError(f"No structural-connectivity results found in {DEFAULT_OUTPUT_DIRECTORY}")
        raise RuntimeError(
            "No eligible MRI subjects found. The download manifest must contain "
            "mri_status=success, or specify --subject to validate existing MRI files."
        )
    return subjects


def save_subject_record(output_directory: Path, record: dict[str, str]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "run_status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def save_batch_records(records: list[dict[str, str]]) -> None:
    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = DEFAULT_OUTPUT_DIRECTORY / "structural_connectivity_manifest.csv"
    fieldnames = [
        "bids_subject_id",
        "session",
        "status",
        "exit_code",
        "output_directory",
        "input_session_directory",
        "t1w",
        "dwi",
        "started_at",
        "finished_at",
        "automatic_qc_status",
        "error",
    ]
    merged: dict[str, dict[str, str]] = {}
    if destination.is_file():
        with destination.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                subject = row.get("bids_subject_id") or ""
                if subject:
                    merged[subject] = row
    for record in records:
        merged[record["bids_subject_id"]] = record
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged[subject] for subject in sorted(merged))
    os.replace(temporary, destination)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HBN structural connectivity from downloaded Data/MRI into SFC."
    )
    parser.add_argument("--subject", nargs="+", action="append")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--session")
    parser.add_argument("--t1w-name")
    parser.add_argument("--dwi-name")
    parser.add_argument("--aal-atlas", type=Path, default=DEFAULT_AAL_ATLAS)
    parser.add_argument("--aal-mapping", type=Path, default=DEFAULT_AAL_MAPPING)
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--mrtrix-bin")
    parser.add_argument("--freesurfer-home")
    parser.add_argument("--freesurfer-license")
    parser.add_argument("--preprocessing", choices=("auto", "fsl"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--streamlines", type=int, default=2_000_000)
    parser.add_argument("--until-stage", choices=PIPELINE_STAGES, default="qc")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    arguments = parser.parse_args()
    if arguments.threads <= 0:
        parser.error("--threads must be greater than zero")
    if arguments.streamlines <= 0:
        parser.error("--streamlines must be greater than zero")
    for name in ("t1w_name", "dwi_name"):
        value = getattr(arguments, name)
        if value is not None and (
            not value or "/" in value or "\\" in value or value in {".", ".."}
        ):
            parser.error(f"--{name.replace('_', '-')} must be an exact file name")
    arguments.data_dir = arguments.data_dir.expanduser().resolve()
    arguments.aal_atlas = arguments.aal_atlas.expanduser().resolve()
    arguments.aal_mapping = arguments.aal_mapping.expanduser().resolve()
    return arguments


def main() -> int:
    arguments = parse_arguments()
    try:
        subjects = select_subjects(arguments)
    except (OSError, RuntimeError, ValueError, csv.Error) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    if arguments.status:
        failed = False
        for subject in subjects:
            output_directory = DEFAULT_OUTPUT_DIRECTORY / subject / "structural_connectivity"
            print(f"[SUBJECT] {subject}")
            try:
                if show_pipeline_status(output_directory) != 0:
                    failed = True
                run_status = output_directory / "run_status.json"
                if run_status.is_file():
                    record = json.loads(run_status.read_text(encoding="utf-8"))
                    print(f"[RUN STATUS] {record.get('status', 'unknown')}")
                    if record.get("error"):
                        print(f"[RUN ERROR] {record['error']}")
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                print(f"[ERROR] {subject}: {error}", file=sys.stderr)
                failed = True
        return int(failed)
    try:
        _check_connectome_export_dependencies()
        _validate_aal90(arguments.aal_atlas, arguments.aal_mapping)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    records: list[dict[str, str]] = []
    failed = False
    for index, subject in enumerate(subjects, start=1):
        output_directory = DEFAULT_OUTPUT_DIRECTORY / subject / "structural_connectivity"
        record = {
            "bids_subject_id": subject,
            "session": "",
            "status": "running",
            "exit_code": "",
            "output_directory": str(output_directory),
            "input_session_directory": "",
            "t1w": "",
            "dwi": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": "",
            "automatic_qc_status": "",
            "error": "",
        }
        print(f"[SUBJECT] ({index}/{len(subjects)}) {subject}")
        try:
            save_subject_record(output_directory, record)
            session_directory, inputs = select_subject_inputs(arguments, subject)
            record.update(
                {
                    "session": session_directory.name if session_directory.name.startswith("ses-") else "",
                    "input_session_directory": str(session_directory),
                    "t1w": str(inputs.t1w),
                    "dwi": str(inputs.dwi),
                }
            )
            save_subject_record(output_directory, record)
            script = build_wsl_script(
                inputs=inputs,
                output_directory=output_directory,
                aal_atlas=arguments.aal_atlas,
                aal_mapping=arguments.aal_mapping,
                subject=subject,
                distro=arguments.distro,
                mrtrix_bin=arguments.mrtrix_bin,
                freesurfer_home=arguments.freesurfer_home,
                freesurfer_license=arguments.freesurfer_license,
                preprocessing=arguments.preprocessing,
                threads=arguments.threads,
                streamlines=arguments.streamlines,
                preflight=arguments.preflight,
                until_stage=arguments.until_stage,
            )
            return_code = run_pipeline(
                script,
                arguments.distro,
                output_directory,
                arguments.preflight,
            )
            record["exit_code"] = str(return_code)
            if return_code == PIPELINE_PAUSED_EXIT_CODE:
                record["status"] = "paused"
            elif return_code != 0:
                raise RuntimeError(f"Pipeline exited with code {return_code}; inspect the launcher log and pipeline.log in {output_directory}")
            elif arguments.preflight:
                record["status"] = "preflight_passed"
            else:
                export_connectome_derivatives(output_directory)
                summary_path = output_directory / "summary.json"
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    record["automatic_qc_status"] = str(summary.get("automatic_qc_status", ""))
                record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            if not record["exit_code"] or record["exit_code"] == "0":
                record["exit_code"] = "1"
            failed = True
            print(f"[ERROR] {subject}: {error}", file=sys.stderr)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        try:
            save_subject_record(output_directory, record)
        except OSError as error:
            failed = True
            print(f"[ERROR] Could not save status for {subject}: {error}", file=sys.stderr)
        try:
            save_batch_records(records)
        except (OSError, ValueError, csv.Error) as error:
            failed = True
            print(f"[ERROR] Could not save batch manifest: {error}", file=sys.stderr)
        print(f"[RESULT] {subject}: {record['status']}")
    print(f"[MANIFEST] {DEFAULT_OUTPUT_DIRECTORY / 'structural_connectivity_manifest.csv'}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
