from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


REQUIRED = (
    Path("notes/notes_v2_1.json"),
    Path("symbols/symbols_v2_1_shapes.json"),
    Path("symbols/symbols_v2_1_shapes_crop_fused.json"),
)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified page view containing pre-pruning StaffOMR artifacts.")
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    seen = set()
    for root in args.source_roots:
        for page_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if page_dir.name in seen:
                raise ValueError(f"Duplicate page name across roots: {page_dir.name}")
            sources = [page_dir / relative for relative in REQUIRED]
            if not sources[0].exists() or not any(path.exists() for path in sources[1:]):
                continue
            seen.add(page_dir.name)
            copied = []
            for relative, source in zip(REQUIRED, sources):
                if source.exists():
                    link_or_copy(source, args.out_root / page_dir.name / relative)
                    copied.append(str(relative))
            rows.append({"sample": page_dir.name, "source": str(page_dir), "artifacts": copied})
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"pages": len(rows), "source_roots": [str(path) for path in args.source_roots], "rows": rows}
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "pages": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
