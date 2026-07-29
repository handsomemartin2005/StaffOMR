from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Polish Scores HuggingFace split to flat PNG/eKern files.")
    parser.add_argument("--dataset", default="antoniorv6/polish-scores")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prefix", default="train")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import datasets

    split = datasets.load_dataset(args.dataset, split=args.split, trust_remote_code=False)
    if args.limit is not None:
        split = split.select(range(min(args.limit, len(split))))
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(split):
        name = f"{args.prefix}_{index:04d}"
        sample["image"].convert("RGB").save(args.out_root / f"{name}.png")
        transcription = str(sample["transcription"])
        (args.out_root / f"{name}.ekern.txt").write_text(transcription.rstrip() + "\n", encoding="utf-8")
        rows.append({"sample": name, "source_id": sample.get("id") or sample.get("name"), "characters": len(transcription)})
    payload = {"dataset": args.dataset, "split": args.split, "samples": len(rows), "rows": rows}
    (args.out_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "samples": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
