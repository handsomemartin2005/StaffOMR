from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_dirs(source_root: Path, prediction_relpath: str) -> list[Path]:
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / prediction_relpath).exists()
    )


def gold_cost_from_lmx(text: str) -> int:
    return len(text.split())


def load_olimpic_modules(olimpic_root: Path):
    sys.path.insert(0, str(olimpic_root.resolve()))
    from app.linearization.Linearizer import Linearizer
    from app.symbolic.MxlFile import MxlFile
    from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
    from zeus.ser_metric import ser_metric

    return Linearizer, MxlFile, TEDn_lmx_xml, ser_metric


def linearize_musicxml(path: Path, Linearizer, MxlFile) -> str:
    tree = ET.ElementTree(ET.fromstring(path.read_text(encoding="utf-8")))
    mxl = MxlFile(tree)
    try:
        part = mxl.get_piano_part()
    except Exception:
        part = mxl.tree.find("part")
    if part is None or part.tag != "part":
        return ""
    linearizer = Linearizer(errout=sys.stderr)
    linearizer.process_part(part)
    return " ".join(linearizer.output_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StaffOMR outputs with OLiMPiC official LMX SER and TEDn.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--olimpic-root", type=Path, default=Path(".local-tools/olimpic-icdar24"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--prediction-relpath", default="semantics/score_v2_1.musicxml")
    parser.add_argument("--pred-lmx-dir", type=Path)
    parser.add_argument("--tedn-flavors", nargs="*", default=["lmx"], choices=["lmx", "full"])
    parser.add_argument("--skip-tedn", action="store_true")
    parser.add_argument(
        "--failure-policy",
        choices=["empty", "skip"],
        default="empty",
        help="How to handle predictions that cannot be linearized. 'empty' counts all gold tokens as errors.",
    )
    args = parser.parse_args()

    Linearizer, MxlFile, TEDn_lmx_xml, ser_metric = load_olimpic_modules(args.olimpic_root)
    pred_lmx_dir = args.pred_lmx_dir or (args.out_json.parent / f"{args.out_json.stem}_pred_lmx")
    pred_lmx_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    gold_lmx_items: list[str] = []
    pred_lmx_items: list[str] = []

    for sample_dir in sample_dirs(args.source_root, args.prediction_relpath):
        sample = sample_dir.name
        pred_xml = sample_dir / args.prediction_relpath
        gold_xml = args.gt_root / f"{sample}.musicxml"
        gold_lmx = args.gt_root / f"{sample}.lmx.txt"
        if not gold_xml.exists() or not gold_lmx.exists():
            failures.append({"sample": sample, "reason": "missing gold musicxml or lmx"})
            continue
        gold_lmx_text = gold_lmx.read_text(encoding="utf-8").strip()
        try:
            pred_lmx = linearize_musicxml(pred_xml, Linearizer, MxlFile)
            pred_lmx_path = pred_lmx_dir / f"{sample}.lmx"
            pred_lmx_path.write_text(pred_lmx + "\n", encoding="utf-8")
            official_ser = ser_metric([gold_lmx_text], [pred_lmx])
            row: dict[str, Any] = {
                "sample": sample,
                "prediction_musicxml": str(pred_xml),
                "reference_musicxml": str(gold_xml),
                "prediction_lmx": str(pred_lmx_path),
                "reference_lmx": str(gold_lmx),
                "pred_lmx_tokens": len(pred_lmx.split()),
                "gold_lmx_tokens": len(gold_lmx_text.split()),
                "SER": official_ser["SER"],
                "SERnotuplets": official_ser["SERnotuplets"],
            }
            if not args.skip_tedn:
                gold_xml_text = gold_xml.read_text(encoding="utf-8")
                for flavor in args.tedn_flavors:
                    result = TEDn_lmx_xml(pred_lmx, gold_xml_text, flavor=flavor, errout=sys.stderr)
                    row[f"TEDn_{flavor}"] = 100.0 * result.edit_cost / result.gold_cost if result.gold_cost else 0.0
                    row[f"TEDn_{flavor}_edit_cost"] = result.edit_cost
                    row[f"TEDn_{flavor}_gold_cost"] = result.gold_cost
            rows.append(row)
            gold_lmx_items.append(gold_lmx_text)
            pred_lmx_items.append(pred_lmx)
        except Exception as exc:
            failure = {"sample": sample, "reason": repr(exc)}
            failures.append(failure)
            if args.failure_policy == "skip":
                continue
            pred_lmx_path = pred_lmx_dir / f"{sample}.lmx"
            pred_lmx_path.write_text("\n", encoding="utf-8")
            official_ser = ser_metric([gold_lmx_text], [""])
            rows.append(
                {
                    "sample": sample,
                    "prediction_musicxml": str(pred_xml),
                    "reference_musicxml": str(gold_xml),
                    "prediction_lmx": str(pred_lmx_path),
                    "reference_lmx": str(gold_lmx),
                    "pred_lmx_tokens": 0,
                    "gold_lmx_tokens": gold_cost_from_lmx(gold_lmx_text),
                    "SER": official_ser["SER"],
                    "SERnotuplets": official_ser["SERnotuplets"],
                    "failure": failure["reason"],
                }
            )
            gold_lmx_items.append(gold_lmx_text)
            pred_lmx_items.append("")

    aggregate_ser = ser_metric(gold_lmx_items, pred_lmx_items) if rows else {"SER": None, "SERnotuplets": None}
    summary: dict[str, Any] = {
        "samples": len(rows),
        "failures": len(failures),
        "failure_policy": args.failure_policy,
        "official_SER": aggregate_ser["SER"],
        "official_SERnotuplets": aggregate_ser["SERnotuplets"],
    }
    for flavor in args.tedn_flavors:
        values = [row[f"TEDn_{flavor}"] for row in rows if f"TEDn_{flavor}" in row]
        if values:
            edit_cost = sum(float(row[f"TEDn_{flavor}_edit_cost"]) for row in rows if f"TEDn_{flavor}_edit_cost" in row)
            gold_cost = sum(float(row[f"TEDn_{flavor}_gold_cost"]) for row in rows if f"TEDn_{flavor}_gold_cost" in row)
            summary[f"TEDn_{flavor}"] = 100.0 * edit_cost / gold_cost if gold_cost else 0.0

    payload = {
        "source_root": str(args.source_root),
        "gt_root": str(args.gt_root),
        "olimpic_root": str(args.olimpic_root),
        "summary": summary,
        "rows": rows,
        "failures": failures,
    }
    write_json(args.out_json, payload)

    lines = [
        "# OLiMPiC Official Metric Evaluation",
        "",
        f"Source root: `{args.source_root}`",
        f"GT root: `{args.gt_root}`",
        f"Samples: {summary['samples']}",
        f"Failures: {summary['failures']} (`{summary['failure_policy']}` policy)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| SER | {summary['official_SER']:.6f} |" if summary["official_SER"] is not None else "| SER | n/a |",
        f"| SERnotuplets | {summary['official_SERnotuplets']:.6f} |" if summary["official_SERnotuplets"] is not None else "| SERnotuplets | n/a |",
    ]
    for flavor in args.tedn_flavors:
        if f"TEDn_{flavor}" in summary:
            lines.append(f"| TEDn {flavor} | {summary[f'TEDn_{flavor}']:.6f} |")
    lines.extend(
        [
            "",
            "## Per Sample",
            "",
            "| Sample | SER | SER no tuplets | TEDn lmx | TEDn full | Pred/Gold LMX tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {sample} | {ser:.6f} | {ser_nt:.6f} | {tedn_lmx} | {tedn_full} | {pred}/{gold} |".format(
                sample=row["sample"],
                ser=row["SER"],
                ser_nt=row["SERnotuplets"],
                tedn_lmx=f"{row['TEDn_lmx']:.6f}" if "TEDn_lmx" in row else "-",
                tedn_full=f"{row['TEDn_full']:.6f}" if "TEDn_full" in row else "-",
                pred=row["pred_lmx_tokens"],
                gold=row["gold_lmx_tokens"],
            )
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['sample']}`: {failure['reason']}")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "summary": summary, "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
