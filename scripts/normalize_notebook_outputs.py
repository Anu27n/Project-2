"""Add missing stream 'name' field for nbformat validation."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_Model_Training.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "name" not in out:
                out["name"] = "stdout"
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Normalized", NB)


if __name__ == "__main__":
    main()
