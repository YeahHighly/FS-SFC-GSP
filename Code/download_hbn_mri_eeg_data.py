from __future__ import annotations

import argparse
import csv
import os
import shutil
import site
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Data"
DEFAULT_SUBJECT_CSV = DATA_DIR / "xxxxx.csv"
MRI_SOURCE_ROOT = DATA_DIR / "MRI"
MRI_SHORT_CLONE_ROOT = DATA_DIR / "_mri"
EEG_SOURCE_ROOT = DATA_DIR / "EEG"
EEG_STAGE_ROOT = EEG_SOURCE_ROOT / "_dataset_staging"
MANIFEST_PATH = DATA_DIR / "download_hbn_mri_eeg_data_manifest.csv"
RBC_REPO_URL = "https://github.com/ReproBrainChart/HBN_BIDS.git"
HBN_EEG_ALL_DATASET_IDS = [
    "ds005505", "ds005506", "ds005507", "ds005508",
    "ds005509", "ds005510", "ds005511", "ds005512",
    "ds005513", "ds005514", "ds005515", "ds005516",
]
OPENNEURO_MAX_RETRIES = 8
OPENNEURO_METADATA_TIMEOUT = 60.0
OPENNEURO_MAX_CONCURRENT_DOWNLOADS = 3
OPENNEURO_SUBJECT_CHUNK_SIZE = 80
DATALAD_GET_CHUNK_SIZE = 120
DATALAD_CLONE_MAX_RETRIES = 5
DATALAD_CLONE_RETRY_SLEEP_SECONDS = 20


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("\n[CMD]", " ".join(str(value) for value in cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{detail}")
    return result


def chunked(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def to_bids_subject_id(subject_id: str) -> str:
    value = str(subject_id).strip().strip('"').strip("'")
    label = value[4:] if value.startswith("sub-") else value
    if not label or not label.isascii() or not label.isalnum():
        raise ValueError("Subject IDs must contain only ASCII letters and digits, with an optional sub- prefix.")
    return f"sub-{label}"


def normalize_subject_aliases(subject: str) -> list[str]:
    normalized = to_bids_subject_id(subject)
    return [normalized]


def read_subject_ids_from_csv(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find subject CSV: {csv_path}. Provide --subject-csv with your private list.")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        columns = {column.strip(): column for column in reader.fieldnames}
        candidates = ["subject_id", "Subject_ID", "subject", "Subject", "participant_id", "Identifiers", "id", "ID"]
        subject_col = next((columns[name] for name in candidates if name in columns), reader.fieldnames[0])
        subjects: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            value = (row.get(subject_col) or "").strip()
            if not value:
                continue
            try:
                subject = to_bids_subject_id(value)
            except ValueError as exc:
                raise ValueError(f"Invalid subject ID in CSV row {row_number}.") from exc
            if subject not in seen:
                subjects.append(subject)
                seen.add(subject)
    if not subjects:
        raise RuntimeError(f"No subject IDs found in CSV: {csv_path}")
    print(f"Read {len(subjects)} unique subjects from {csv_path.name}")
    return subjects


def write_manifest(records: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def check_datalad_available() -> None:
    try:
        run_command(["datalad", "--version"], check=True)
    except Exception as exc:
        raise RuntimeError(
            "DataLad is not installed or not available.\n"
            "Please install first, e.g.:\n"
            "conda install -c conda-forge datalad git-annex -y\n"
            f"Original error: {exc}"
        ) from exc


def clone_hbn_bids_if_needed() -> Path:
    candidates = [MRI_SOURCE_ROOT.resolve(), MRI_SHORT_CLONE_ROOT.resolve()]
    for candidate in candidates:
        if (candidate / ".datalad").is_dir() and (candidate / ".git").exists():
            return candidate
    last_error: Exception | None = None
    for target_root in candidates:
        if target_root.exists():
            last_error = RuntimeError(f"MRI clone target exists but is not a DataLad dataset: {target_root}")
            print(f"[WARN] {last_error}")
            continue
        target_root.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["datalad", "-c", "core.longpaths=true", "clone", RBC_REPO_URL, str(target_root)]
        for attempt in range(1, DATALAD_CLONE_MAX_RETRIES + 1):
            try:
                run_command(cmd)
                return target_root
            except Exception as exc:
                last_error = exc
                print(f"[WARN] MRI clone attempt {attempt}/{DATALAD_CLONE_MAX_RETRIES} failed: {exc}")
                if target_root.exists():
                    resolved_target = target_root.resolve()
                    resolved_data = DATA_DIR.resolve()
                    if target_root.is_symlink() or resolved_target == resolved_data or not resolved_target.is_relative_to(resolved_data):
                        raise RuntimeError(f"Cannot clean up clone outside Data: {target_root}") from exc
                    shutil.rmtree(target_root)
                if attempt < DATALAD_CLONE_MAX_RETRIES:
                    time.sleep(DATALAD_CLONE_RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"Failed to clone MRI dataset: {last_error}") from last_error


def scan_one_subject_mri_files(subject_dir: Path) -> dict[str, list[Path]]:
    t1w_nii_files = sorted(set(subject_dir.rglob("*T1w.nii.gz")))
    t1w_json_files = sorted(set(subject_dir.rglob("*T1w.json")))

    dwi_nii_files = sorted(set(subject_dir.rglob("*dwi.nii.gz")))
    dwi_json_files = sorted(set(subject_dir.rglob("*dwi.json")))
    dwi_bval_files = sorted(set(subject_dir.rglob("*dwi.bval")))
    dwi_bvec_files = sorted(set(subject_dir.rglob("*dwi.bvec")))

    ap_fmap_nii_files = sorted(set(
        list(subject_dir.rglob("*acq-dwi*dir-AP*_epi.nii.gz")) +
        list(subject_dir.rglob("*acq-DWI*dir-AP*_epi.nii.gz")) +
        list(subject_dir.rglob("*dwi*AP*_epi.nii.gz")) +
        list(subject_dir.rglob("*DWI*AP*_epi.nii.gz"))
    ))
    ap_fmap_json_files = sorted(set(
        list(subject_dir.rglob("*acq-dwi*dir-AP*_epi.json")) +
        list(subject_dir.rglob("*acq-DWI*dir-AP*_epi.json")) +
        list(subject_dir.rglob("*dwi*AP*_epi.json")) +
        list(subject_dir.rglob("*DWI*AP*_epi.json"))
    ))

    pa_fmap_nii_files = sorted(set(
        list(subject_dir.rglob("*acq-dwi*dir-PA*_epi.nii.gz")) +
        list(subject_dir.rglob("*acq-DWI*dir-PA*_epi.nii.gz")) +
        list(subject_dir.rglob("*dwi*PA*_epi.nii.gz")) +
        list(subject_dir.rglob("*DWI*PA*_epi.nii.gz"))
    ))
    pa_fmap_json_files = sorted(set(
        list(subject_dir.rglob("*acq-dwi*dir-PA*_epi.json")) +
        list(subject_dir.rglob("*acq-DWI*dir-PA*_epi.json")) +
        list(subject_dir.rglob("*dwi*PA*_epi.json")) +
        list(subject_dir.rglob("*DWI*PA*_epi.json"))
    ))

    return {
        "T1w_nii": t1w_nii_files,
        "T1w_json": t1w_json_files,
        "DWI_nii": dwi_nii_files,
        "DWI_json": dwi_json_files,
        "DWI_bval": dwi_bval_files,
        "DWI_bvec": dwi_bvec_files,
        "DWI_AP_fmap_nii": ap_fmap_nii_files,
        "DWI_AP_fmap_json": ap_fmap_json_files,
        "DWI_PA_fmap_nii": pa_fmap_nii_files,
        "DWI_PA_fmap_json": pa_fmap_json_files,
    }


def has_required_mri(files_dict: dict[str, list[Path]]) -> bool:
    return (
        len(files_dict.get("T1w_nii", [])) > 0
        and len(files_dict.get("DWI_nii", [])) > 0
        and len(files_dict.get("DWI_AP_fmap_nii", [])) > 0
        and len(files_dict.get("DWI_PA_fmap_nii", [])) > 0
    )


def datalad_get_files(files: list[Path], mri_source_root: Path) -> None:
    if not files:
        return

    relative_files = [str(p.relative_to(mri_source_root)) for p in files]
    for chunk in chunked(relative_files, DATALAD_GET_CHUNK_SIZE):
        run_command(["datalad", "get"] + chunk, cwd=mri_source_root, check=True)


def copy_mri_files_preserve_layout(
    files: list[Path],
    mri_clone_root: Path,
    mri_output_root: Path,
) -> int:
    copied_count = 0
    for src_file in files:
        relative = src_file.relative_to(mri_clone_root)
        dst_file = mri_output_root / relative
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied_count += 1
    return copied_count


def ensure_openneuro_installed() -> None:
    try:
        import openneuro
        print("openneuro-py is available in current Python environment.")
    except Exception:
        run_command([sys.executable, "-m", "pip", "install", "openneuro-py"], check=True)


def _can_run_help(cmd_base: list[str]) -> bool:
    try:
        result = subprocess.run(
            cmd_base + ["--help"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_openneuro_cli() -> list[str]:
    for name in ("openneuro-py", "openneuro_py"):
        path = shutil.which(name)
        if path:
            return [path]

    version_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
    candidate_dirs = {
        Path(sys.executable).resolve().parent,
        Path(sysconfig.get_path("scripts")),
        Path(site.getuserbase()) / ("Scripts" if os.name == "nt" else "bin"),
        Path.home() / "AppData" / "Roaming" / "Python" / version_tag / "Scripts",
    }

    script_names = [
        "openneuro-py.exe",
        "openneuro-py",
        "openneuro_py.exe",
        "openneuro_py",
    ]
    for directory in candidate_dirs:
        for name in script_names:
            candidate = directory / name
            if candidate.exists():
                return [str(candidate)]

    module_bases = [
        [sys.executable, "-m", "openneuro"],
        [sys.executable, "-m", "openneuro_py"],
    ]
    for base in module_bases:
        if _can_run_help(base):
            return base

    searched = "\n".join(f"- {p}" for p in sorted(str(x) for x in candidate_dirs))
    raise FileNotFoundError(
        "Cannot find openneuro-py executable after installation.\n"
        f"Searched script dirs:\n{searched}"
    )


def detect_target_dir_flag(openneuro_cmd_base: list[str]) -> str:
    result = subprocess.run(
        openneuro_cmd_base + ["download", "--help"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if "--target-dir" in help_text:
        return "--target-dir"
    if "--target_dir" in help_text:
        return "--target_dir"
    return "--target-dir"


def find_subject_eeg_dirs(eeg_source_root: Path, subject: str) -> list[Path]:
    eeg_dirs: list[Path] = []

    for alias in normalize_subject_aliases(subject):
        sub_dir = eeg_source_root / alias
        if not sub_dir.exists():
            continue

        root_eeg = sub_dir / "eeg"
        if root_eeg.is_dir():
            eeg_dirs.append(root_eeg)

        for ses_dir in sorted(p for p in sub_dir.glob("ses-*") if p.is_dir()):
            ses_eeg = ses_dir / "eeg"
            if ses_eeg.is_dir():
                eeg_dirs.append(ses_eeg)

    unique: dict[str, Path] = {}
    for eeg_dir in eeg_dirs:
        unique[str(eeg_dir)] = eeg_dir
    return [unique[k] for k in sorted(unique)]


def count_eeg_files(eeg_dirs: list[Path]) -> int:
    return sum(1 for directory in eeg_dirs for path in directory.rglob("*") if path.is_file())


def copy_eeg_files(subject: str, dataset_root: Path, eeg_dirs: list[Path]) -> int:
    subject_root = dataset_root / subject
    for source_dir in eeg_dirs:
        destination = EEG_SOURCE_ROOT / subject / source_dir.relative_to(subject_root)
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
    return count_eeg_files(eeg_dirs)


def download_subject_subset(
    openneuro_cmd_base: list[str],
    dataset_id: str,
    subjects: list[str],
    target_root: Path,
    target_dir_flag: str,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)

    for subject_chunk in chunked(subjects, OPENNEURO_SUBJECT_CHUNK_SIZE):
        cmd = openneuro_cmd_base + [
            "download",
            f"--dataset={dataset_id}",
            target_dir_flag,
            str(target_root),
            "--max-retries",
            str(OPENNEURO_MAX_RETRIES),
            "--metadata-timeout",
            str(OPENNEURO_METADATA_TIMEOUT),
            "--max-concurrent-downloads",
            str(OPENNEURO_MAX_CONCURRENT_DOWNLOADS),
        ]
        include_values: list[str] = []
        for subject in subject_chunk:
            include_values.extend(normalize_subject_aliases(subject))
        for include_value in sorted(set(include_values)):
            cmd.extend(["--include", include_value])
        run_command(cmd, check=True)


def download_dataset_with_fallback(
    openneuro_cmd_base: list[str],
    dataset_id: str,
    subjects: list[str],
    target_root: Path,
    target_dir_flag: str,
) -> dict[str, str]:
    if not subjects:
        return {}
    try:
        download_subject_subset(openneuro_cmd_base, dataset_id, subjects, target_root, target_dir_flag)
        return {}
    except Exception as exc:
        print(f"[WARN] Batch download failed for {dataset_id}, retrying each subject: {exc}")
    errors: dict[str, str] = {}
    for subject in subjects:
        try:
            download_subject_subset(openneuro_cmd_base, dataset_id, [subject], target_root, target_dir_flag)
        except Exception as exc:
            errors[subject] = str(exc)
    return errors


def download_mri(subjects: list[str], records: dict[str, dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        clone_root = next(
            (path.resolve() for path in (MRI_SOURCE_ROOT, MRI_SHORT_CLONE_ROOT)
             if (path / ".datalad").is_dir() and (path / ".git").exists()),
            None,
        )
        if clone_root is None:
            for record in records.values():
                record["mri_status"] = "dry_run_pending_download"
            return
    else:
        try:
            check_datalad_available()
            clone_root = clone_hbn_bids_if_needed()
        except Exception as exc:
            for record in records.values():
                record["mri_status"] = "failed"
                record["mri_error"] = str(exc)
            return
    for index, subject in enumerate(subjects, start=1):
        print(f"[MRI] ({index}/{len(subjects)}) {subject}")
        record = records[subject]
        try:
            subject_dir = clone_root / subject
            if not subject_dir.is_dir():
                record["mri_status"] = "missing_subject_dir"
                continue
            files_by_type = scan_one_subject_mri_files(subject_dir)
            files = sorted({path for paths in files_by_type.values() for path in paths})
            if not has_required_mri(files_by_type):
                record["mri_status"] = "missing_required_mri"
                continue
            if dry_run:
                record["mri_status"] = "dry_run_ready"
                continue
            datalad_get_files(files, clone_root)
            if not all(path.is_file() for path in files):
                raise RuntimeError("Some MRI files are unavailable after datalad get.")
            if clone_root != MRI_SOURCE_ROOT.resolve():
                file_count = copy_mri_files_preserve_layout(files, clone_root, MRI_SOURCE_ROOT)
            else:
                file_count = len(files)
            record["mri_num_files"] = str(file_count)
            record["mri_status"] = "success"
        except Exception as exc:
            record["mri_status"] = "failed"
            record["mri_error"] = str(exc)


def download_eeg(
    subjects: list[str],
    records: dict[str, dict[str, str]],
    dry_run: bool,
    force_redownload: bool,
) -> None:
    pending: list[str] = []
    for subject in subjects:
        record = records[subject]
        existing_dirs = find_subject_eeg_dirs(EEG_SOURCE_ROOT, subject)
        existing_count = count_eeg_files(existing_dirs)
        if existing_count and not force_redownload:
            record["eeg_status"] = "existing_local_eeg"
            record["eeg_num_source_dirs"] = str(len(existing_dirs))
            record["eeg_num_files"] = str(existing_count)
        else:
            pending.append(subject)
    if not pending:
        return
    openneuro_cmd_base: list[str] = []
    target_dir_flag = "--target-dir"
    if not dry_run:
        try:
            ensure_openneuro_installed()
            openneuro_cmd_base = find_openneuro_cli()
            target_dir_flag = detect_target_dir_flag(openneuro_cmd_base)
        except Exception as exc:
            for subject in pending:
                records[subject]["eeg_status"] = "failed"
                records[subject]["eeg_error"] = str(exc)
            return
    for dataset_id in HBN_EEG_ALL_DATASET_IDS:
        if not pending:
            break
        print(f"[EEG] {dataset_id}: {len(pending)} subjects remaining")
        dataset_root = EEG_STAGE_ROOT / dataset_id
        errors: dict[str, str] = {}
        if not dry_run:
            errors = download_dataset_with_fallback(
                openneuro_cmd_base, dataset_id, pending, dataset_root, target_dir_flag,
            )
        unresolved: list[str] = []
        for subject in pending:
            record = records[subject]
            tried = record["eeg_tried_datasets"].split("|") if record["eeg_tried_datasets"] else []
            tried.append(dataset_id)
            record["eeg_tried_datasets"] = "|".join(tried)
            if subject in errors:
                record["eeg_error"] = f"{dataset_id}: {errors[subject]}"
                unresolved.append(subject)
                continue
            try:
                eeg_dirs = find_subject_eeg_dirs(dataset_root, subject)
                file_count = count_eeg_files(eeg_dirs)
                if not file_count:
                    unresolved.append(subject)
                    continue
                if dry_run:
                    record["eeg_status"] = "dry_run_found_in_stage"
                else:
                    file_count = copy_eeg_files(subject, dataset_root, eeg_dirs)
                    record["eeg_status"] = "success"
                record["eeg_resolved_dataset_id"] = dataset_id
                record["eeg_num_source_dirs"] = str(len(eeg_dirs))
                record["eeg_num_files"] = str(file_count)
                record["eeg_error"] = ""
            except Exception as exc:
                record["eeg_error"] = f"{dataset_id}: {exc}"
                unresolved.append(subject)
        pending = unresolved
    for subject in pending:
        record = records[subject]
        if dry_run:
            record["eeg_status"] = "dry_run_pending_download"
        elif record["eeg_error"]:
            record["eeg_status"] = "failed"
        else:
            record["eeg_status"] = "not_found_any_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HBN MRI and EEG for subjects in one private CSV into Data.")
    parser.add_argument("--subject-csv", "--subject-file", dest="subject_csv", type=Path, default=DEFAULT_SUBJECT_CSV)
    parser.add_argument("--skip-mri", action="store_true")
    parser.add_argument("--skip-eeg", action="store_true")
    parser.add_argument("--force-redownload-eeg", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    subject_csv = args.subject_csv.expanduser().resolve()
    subjects = read_subject_ids_from_csv(subject_csv)
    if args.limit:
        subjects = subjects[:args.limit]
    records: list[dict[str, str]] = [
        {
            "bids_subject_id": subject,
            "mri_status": "pending",
            "mri_num_files": "0",
            "mri_error": "",
            "eeg_resolved_dataset_id": "",
            "eeg_tried_datasets": "",
            "eeg_status": "pending",
            "eeg_num_source_dirs": "0",
            "eeg_num_files": "0",
            "eeg_error": "",
        }
        for subject in subjects
    ]
    records_by_subject = {record["bids_subject_id"]: record for record in records}
    print(f"Subjects to process: {len(subjects)}")
    print(f"Subject CSV: {subject_csv}")
    print(f"MRI output: {MRI_SOURCE_ROOT}")
    print(f"EEG output: {EEG_SOURCE_ROOT}")
    if args.skip_mri:
        for record in records:
            record["mri_status"] = "skipped"
    else:
        download_mri(subjects, records_by_subject, args.dry_run)
    write_manifest(records)
    if args.skip_eeg:
        for record in records:
            record["eeg_status"] = "skipped"
    else:
        download_eeg(subjects, records_by_subject, args.dry_run, args.force_redownload_eeg)
    write_manifest(records)
    print(f"Manifest: {MANIFEST_PATH}")
    for modality in ("mri", "eeg"):
        counts: dict[str, int] = {}
        for record in records:
            status = record[f"{modality}_status"]
            counts[status] = counts.get(status, 0) + 1
        print(f"{modality.upper()} status: {counts}")


if __name__ == "__main__":
    main()
