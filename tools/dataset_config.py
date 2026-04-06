"""
Project-level dataset configuration and archive helpers.

The runtime still reads from `data/arxiv_202502_cs_cl/`, but when that folder is
missing we can safely restore it from a local `.tar` archive, a local `.tar.gz`
archive, or Git-tracked split `.tar.gz.partXX` chunks.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import List
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_LABEL = "arXiv 2025-02 cs.CL"
DATASET_NAME = "arxiv_202502_cs_cl"
DATASET_PARENT_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATASET_PARENT_DIR / DATASET_NAME
DATASET_RELATIVE_PATH = DATASET_DIR.relative_to(PROJECT_ROOT).as_posix()

BUNDLED_DATA_DIR = PROJECT_ROOT / "bundled_data"
DATASET_TAR_GZ_PATH = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar.gz"
DATASET_TAR_PATH = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar"
DATASET_TAR_GZ_PART_PREFIX = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar.gz.part"
DATASET_TAR_GZ_PART_GLOB = f"{DATASET_NAME}.tar.gz.part*"
DATASET_RELEASE_TAG = "dataset-20260407"
DATASET_RELEASE_ASSET_URL = (
    f"https://github.com/wangzekun6/paper_search_app/releases/download/"
    f"{DATASET_RELEASE_TAG}/{DATASET_NAME}.tar.gz"
)
DATASET_ARCHIVE_CANDIDATES = (DATASET_TAR_PATH, DATASET_TAR_GZ_PATH)


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _archive_candidates() -> List[Path]:
    return [path for path in DATASET_ARCHIVE_CANDIDATES if path.exists()]


def _split_archive_candidates() -> List[Path]:
    return sorted(BUNDLED_DATA_DIR.glob(DATASET_TAR_GZ_PART_GLOB))


def dataset_archive_paths() -> List[Path]:
    return list(DATASET_ARCHIVE_CANDIDATES) + _split_archive_candidates()


def dataset_archive_relative_paths() -> List[str]:
    relative_paths = [path.relative_to(PROJECT_ROOT).as_posix() for path in DATASET_ARCHIVE_CANDIDATES]
    relative_paths.append(f"{DATASET_TAR_GZ_PART_PREFIX.relative_to(PROJECT_ROOT).as_posix()}*")
    return relative_paths


def available_dataset_archive_paths() -> List[Path]:
    archives = _archive_candidates()
    if archives:
        return archives
    return _split_archive_candidates()


def _safe_extract_tar(archive_path: Path, destination_dir: Path) -> None:
    destination_root = destination_dir.resolve(strict=False)
    destination_root_str = str(destination_root)

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_name = member.name.strip()
            if not member_name:
                continue
            if member.islnk() or member.issym():
                raise ValueError(f"Refusing to extract link entry from archive: {member.name}")

            target_path = (destination_root / member_name).resolve(strict=False)
            if os.path.commonpath([destination_root_str, str(target_path)]) != destination_root_str:
                raise ValueError(f"Archive entry escapes extraction directory: {member.name}")

        tar.extractall(path=destination_root)


def _extract_split_archive(split_parts: List[Path], destination_dir: Path) -> None:
    if not split_parts:
        raise FileNotFoundError("No split dataset archive parts were provided for extraction.")

    with tempfile.TemporaryDirectory(prefix=f"{DATASET_NAME}_", suffix="_extract") as temp_dir:
        merged_archive = Path(temp_dir) / f"{DATASET_NAME}.tar.gz"
        with merged_archive.open("wb") as destination_handle:
            for part_path in split_parts:
                with part_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        _safe_extract_tar(merged_archive, destination_dir)


def _download_release_archive(destination_path: Path) -> Path:
    BUNDLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{DATASET_NAME}_", suffix=".tar.gz.download", delete=False
    ) as temp_handle:
        temp_path = Path(temp_handle.name)

    request = Request(
        DATASET_RELEASE_ASSET_URL,
        headers={"User-Agent": "PaperCompass dataset bootstrap"},
    )
    try:
        with urlopen(request) as response, temp_path.open("wb") as destination_handle:
            shutil.copyfileobj(response, destination_handle)
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return destination_path


def ensure_default_dataset_available() -> Path:
    if DATASET_DIR.exists():
        return DATASET_DIR.resolve()

    archives = _archive_candidates()
    split_archives = _split_archive_candidates()
    if not archives and not split_archives:
        try:
            _download_release_archive(DATASET_TAR_GZ_PATH)
        except Exception as exc:
            archive_list = ", ".join(dataset_archive_relative_paths())
            raise FileNotFoundError(
                f"Dataset directory not found: {DATASET_DIR}. "
                f"Expected a local archive at one of: {archive_list}, "
                f"or a downloadable release asset at: {DATASET_RELEASE_ASSET_URL}."
            ) from exc
        archives = _archive_candidates()

    DATASET_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    if archives:
        _safe_extract_tar(archives[0], DATASET_PARENT_DIR)
    else:
        _extract_split_archive(split_archives, DATASET_PARENT_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Bundled dataset archive was extracted, but dataset directory is still missing: {DATASET_DIR}"
        )
    return DATASET_DIR.resolve()


def resolve_dataset_root(data_root: str | Path = DATASET_DIR) -> Path:
    requested = _normalized_path(data_root)
    default_dataset = _normalized_path(DATASET_DIR)
    if requested.exists():
        return requested
    if requested == default_dataset:
        return ensure_default_dataset_available()
    raise FileNotFoundError(f"Dataset path not found: {requested}")
