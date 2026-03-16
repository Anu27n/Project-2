"""
Backup notebooks and training results to prevent loss.
Run this after training to save your work with outputs preserved.
"""
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
BACKUP_DIR = PROJECT_ROOT / "backups"


def backup():
    """Create timestamped backup of notebooks and key results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)

    # Backup notebooks (with outputs preserved)
    notebooks_src = PROJECT_ROOT / "notebooks"
    notebooks_dst = backup_path / "notebooks"
    if notebooks_src.exists():
        shutil.copytree(notebooks_src, notebooks_dst, dirs_exist_ok=True)
        print(f"  Copied notebooks/ -> {notebooks_dst}")

    # Backup key results (metrics, training history - not .pth to save space)
    results_src = PROJECT_ROOT / "results"
    results_dst = backup_path / "results"
    if results_src.exists():
        results_dst.mkdir(exist_ok=True)
        for sub in ["metrics", "models"]:
            src_sub = results_src / sub
            dst_sub = results_dst / sub
            if src_sub.exists():
                dst_sub.mkdir(exist_ok=True)
                for f in src_sub.iterdir():
                    if f.suffix != ".pth":  # Skip large model files
                        shutil.copy2(f, dst_sub / f.name)
                        print(f"  Copied {f.relative_to(PROJECT_ROOT)}")
        # Copy figures
        figs_src = results_src / "figures"
        if figs_src.exists():
            shutil.copytree(figs_src, results_dst / "figures", dirs_exist_ok=True)
            print(f"  Copied results/figures/")

    # Create zip for easy download
    zip_path = BACKUP_DIR / f"notebooks_backup_{timestamp}"
    shutil.make_archive(str(zip_path), "zip", backup_path)
    print(f"\nCreated backup:")
    print(f"  Folder: {backup_path}")
    print(f"  Zip:    {zip_path}.zip")
    print(f"\nTo preserve notebook outputs in git: run 'git add notebooks/' then 'git commit'")
    return backup_path


if __name__ == "__main__":
    print("Backing up notebooks and results...")
    BACKUP_DIR.mkdir(exist_ok=True)
    backup()
    print("Done.")
