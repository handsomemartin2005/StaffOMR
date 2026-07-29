from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sweep_v2_1_symbol_classifier_fusion import (
    apply_family_fusion,
    classify_symbols,
    load_classifier,
    read_json,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply family-constrained crop classification to V2.1 shapes.")
    parser.add_argument("--shapes-json", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--conditional-threshold", type=float, default=0.55)
    parser.add_argument("--family-mass-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enabled-families", nargs="+", choices=("notehead", "accidental"), default=["notehead", "accidental"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model, class_names, transform, checkpoint = load_classifier(args.classifier, device)
    shapes = read_json(args.shapes_json)
    predictions = classify_symbols(
        shapes,
        model,
        class_names,
        transform,
        checkpoint,
        device,
        args.batch_size,
    )
    fused, stats = apply_family_fusion(
        shapes,
        predictions,
        conditional_threshold=args.conditional_threshold,
        family_mass_threshold=args.family_mass_threshold,
        enabled_families=set(args.enabled_families),
    )
    write_json(args.out_json, fused)
    summary = {
        "input": str(args.shapes_json),
        "classifier": str(args.classifier),
        "output": str(args.out_json),
        "conditional_threshold": args.conditional_threshold,
        "family_mass_threshold": args.family_mass_threshold,
        "device": str(device),
        "enabled_families": args.enabled_families,
        **stats,
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
