#!/usr/bin/env python
"""
Create an upload-ready ZIP archive of the project.

Usage:
    python create_zip.py

Output:
    <project-root-name>.zip (e.g., Project-2.zip)

The script keeps core code/artifacts and excludes environment/data/cache files.
"""

import fnmatch
import os
import zipfile

# Portable base directory (works on Windows/Linux/Kaggle/Colab)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = os.path.basename(os.path.normpath(BASE_DIR))
ZIP_NAME = f"{PROJECT_NAME}.zip"
ZIP_PATH = os.path.join(BASE_DIR, ZIP_NAME)

# Keep these top-level files
INCLUDE_FILES = {
    "README.md",
    "requirements.txt",
    "run_train.py",
    "run_evaluate.py",
    "run_cgan_augment.py",
    "run_inference.py",
    "create_zip.py",
    ".gitignore",
}

# Keep these directories
INCLUDE_DIR_PREFIXES = {
    "src",
    "notebooks",
    "scripts",
    "app",
    "docs",
    "results/models",
}

# Only keep checkpoint-like files under results/models
CHECKPOINT_EXTENSIONS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
}

# Exclude these directories globally
EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".ipynb_checkpoints",
    "data",
    "backups",
}

# Exclude these file patterns globally
EXCLUDE_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.log",
}


def to_posix(path_str: str) -> str:
    return path_str.replace("\\", "/")


def is_under(rel_path: str, prefix: str) -> bool:
    return rel_path == prefix or rel_path.startswith(prefix + "/")


def should_include(rel_path: str) -> bool:
    if rel_path in INCLUDE_FILES:
        return True

    for prefix in INCLUDE_DIR_PREFIXES:
        if is_under(rel_path, prefix):
            if prefix == "results/models":
                ext = os.path.splitext(rel_path)[1].lower()
                return ext in CHECKPOINT_EXTENSIONS
            return True

    return False


def should_exclude(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True

    base_name = os.path.basename(rel_path)
    if base_name == ZIP_NAME:
        return True

    for pattern in EXCLUDE_FILE_PATTERNS:
        if fnmatch.fnmatch(base_name, pattern):
            return True

    return False


def create_zip() -> None:
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    files_added = 0

    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Prune excluded directories early for speed and cleanliness
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]

            for file_name in files:
                abs_path = os.path.join(root, file_name)
                rel_path = to_posix(os.path.relpath(abs_path, BASE_DIR))

                if should_exclude(rel_path):
                    continue
                if not should_include(rel_path):
                    continue

                zf.write(abs_path, rel_path)
                files_added += 1

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"ZIP created successfully: {ZIP_PATH}")
    print(f"Files included: {files_added}")
    print(f"Archive size : {zip_size_mb:.2f} MB")


if __name__ == "__main__":
    create_zip()
