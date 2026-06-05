import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.config import DEFAULTS
from src import model_attention, model_cnn, model_cnn_attention  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dsp_acl")
    parser.add_argument("--config", default="configs/config.py")
    args = parser.parse_args()
    print(f"Training entrypoint placeholder: model={args.model}, config={args.config}, labels={DEFAULTS['num_labels']}")


if __name__ == "__main__":
    main()
