from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_lmx_pruning_export import PRESET_NAMES, materialize


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one fixed training-selected pruning preset to a StaffOMR page-artifact view.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESET_NAMES, required=True)
    args = parser.parse_args()
    pages = materialize(args.source_root, args.out_root, args.preset)
    print(json.dumps({"source_root": str(args.source_root), "out_root": str(args.out_root), "preset": args.preset, "pages": pages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
