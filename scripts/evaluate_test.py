import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def main() -> None:
    print("Evaluation entrypoint placeholder")


if __name__ == "__main__":
    main()
