from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def export_fix_eval(source: Path, gt: Path, tag: str, xtol: float, conf: float, out_root: Path) -> dict:
    raw_name = f"score_{tag}.musicxml"
    fixed_name = f"score_{tag}_chordfix.musicxml"
    export_json = out_root / f"{tag}_export.json"
    fix_json = out_root / f"{tag}_chordfix.json"
    eval_json = out_root / f"{tag}_ser.json"
    eval_md = out_root / f"{tag}_ser.md"
    run([sys.executable, "tools/export_v2_1_official_piano_semantics.py", "--source-root", str(source), "--semantics-relpath", "semantics/semantics_v2_1.json", "--output-name", raw_name, "--out-summary", str(export_json), "--x-tolerance", str(xtol), "--backup-policy", "fixed", "--fixed-measure-units", "64", "--min-note-confidence", str(conf)])
    run([sys.executable, "tools/postprocess_musicxml_chords.py", "--source-root", str(source), "--prediction-relpath", f"semantics/{raw_name}", "--output-name", fixed_name, "--out-summary", str(fix_json)])
    run([sys.executable, "tools/evaluate_olimpic_official_metrics.py", "--source-root", str(source), "--gt-root", str(gt), "--prediction-relpath", f"semantics/{fixed_name}", "--out-json", str(eval_json), "--out-md", str(eval_md), "--skip-tedn", "--failure-policy", "empty"])
    metric = json.loads(eval_json.read_text(encoding="utf-8"))["summary"]
    return {"tag": tag, "x_tolerance": xtol, "confidence": conf, "samples": metric.get("samples"), "SER": metric["official_SER"], "SERnotuplets": metric["official_SERnotuplets"], "failures": metric["failures"], "prediction_relpath": f"semantics/{fixed_name}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select official-LMX export parameters only on a target training split, then apply them unchanged to a test split.")
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-gt", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--test-gt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--x-tolerances", type=float, nargs="+", default=[4, 6, 8])
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.60, 0.65, 0.70])
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for xtol in args.x_tolerances:
        for conf in args.confidences:
            tag = f"{args.name}_x{xtol:g}_c{int(round(conf*100)):02d}"
            rows.append(export_fix_eval(args.train_source, args.train_gt, tag, xtol, conf, args.out_root))
    best = min(rows, key=lambda row: float(row["SER"]))
    test_tag = f"{args.name}_selected_x{best['x_tolerance']:g}_c{int(round(best['confidence']*100)):02d}_test"
    test = export_fix_eval(args.test_source, args.test_gt, test_tag, float(best["x_tolerance"]), float(best["confidence"]), args.out_root)
    result = {
        "protocol": "Parameters selected exclusively on train-source/train-gt; test metrics never used for selection.",
        "name": args.name,
        "candidates": rows,
        "best_train": best,
        "test": test,
    }
    selection = args.out_root / f"{args.name}_selection.json"
    selection.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selection": str(selection), "best_train": best, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
