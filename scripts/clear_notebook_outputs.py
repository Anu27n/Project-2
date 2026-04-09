"""Clear all code cell outputs for a clean nbconvert --execute."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_Model_Training.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Cleared outputs in", NB)


if __name__ == "__main__":
    main()
