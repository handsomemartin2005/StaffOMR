from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a Zeus pickle dataset into flat PNG/LMX/MusicXML files for StaffOMR experiments.")
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prefix", default="sample")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = pickle.loads(args.pickle.read_bytes())
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows):
        sample = f"{args.prefix}_{index:05d}"
        image = row["image"]
        if not isinstance(image, (bytes, bytearray)):
            raise TypeError(f"Expected PNG bytes for {sample}, got {type(image)!r}")
        (args.out_root / f"{sample}.png").write_bytes(bytes(image))
        (args.out_root / f"{sample}.lmx.txt").write_text(str(row["lmx"]).strip() + "\n", encoding="utf-8")
        (args.out_root / f"{sample}.musicxml").write_text(str(row["musicxml"]), encoding="utf-8")
        manifest.append({"sample": sample, "source_path": row.get("path"), "lmx_tokens": len(str(row["lmx"]).split())})
    payload = {"source_pickle": str(args.pickle), "samples": len(manifest), "rows": manifest}
    (args.out_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "samples": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
