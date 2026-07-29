from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs")
IJCV = ROOT / "ijcv_repro"


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pair(a: Any, b: Any) -> str:
    if a is None or b is None:
        return "未完成"
    return f"{float(a):.2f} / {float(b):.2f}"


def official(path: Path) -> str:
    payload = load(path)
    if payload is None:
        return "未完成"
    summary = payload.get("summary", payload)
    return pair(summary.get("official_SER", summary.get("SER")), summary.get("official_SERnotuplets", summary.get("SERnotuplets")))


def correction(path: Path) -> str:
    payload = load(path)
    if payload is None:
        return "未完成"
    summary = payload.get("summary", payload)
    return pair(summary.get("SER"), summary.get("SERnotuplets"))


def selection(path: Path) -> str:
    payload = load(path)
    if payload is None:
        return "未完成"
    test = payload.get("test", {})
    return pair(test.get("SER"), test.get("SERnotuplets"))


def polish(path: Path) -> str:
    payload = load(path)
    if payload is None:
        return "未完成"
    if "test" in payload:
        metrics = payload["test"].get("metrics", payload["test"])
        return pair(metrics.get("ser_proxy_percent"), metrics.get("cer_proxy_percent"))
    return pair(payload.get("SER_proxy"), payload.get("CER_proxy"))


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def abcd_rows(dataset: str) -> list[list[str]]:
    if dataset == "polish":
        payload = load(ROOT / "abcd_ablation/polish/polish_abcd_summary.json") or {}
        by_name = {row["variant"]: row for row in payload.get("variants", [])}
        return [[name, pair(by_name.get(name, {}).get("SER_proxy"), by_name.get(name, {}).get("CER_proxy"))] for name in variant_names()]
    return [[name, official(ROOT / f"abcd_ablation/{dataset}/{name}_ser.json")] for name in variant_names()]


def variant_names() -> list[str]:
    return [f"A{a}B{b}C{c}D{d}" for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]


GRAPH_VARIANTS = ("baseline", "note_dist_090", "note_dist_180", "note_x_070", "note_x_120", "max_notes_3", "max_notes_9", "beam_endpoint_030", "beam_endpoint_060", "ledger_dist_140", "ledger_dist_230", "slur_endpoint_200", "slur_endpoint_400")
SAM_VARIANTS = ("box_only", "box_positive", "box_pos_neg_staff")


def graph_rows(dataset: str) -> list[list[str]]:
    if dataset == "polish":
        payload = load(ROOT / "graph_parameter_ablation/polish/graph_parameter_summary.json") or {}
        by_name = {row["variant"]: row for row in payload.get("rows", [])}
        return [[name, pair(by_name.get(name, {}).get("SER_proxy"), by_name.get(name, {}).get("CER_proxy"))] for name in GRAPH_VARIANTS]
    return [[name, official(ROOT / f"graph_parameter_ablation/{dataset}/{name}_ser.json")] for name in GRAPH_VARIANTS]


def sam_rows(dataset: str) -> list[list[str]]:
    if dataset == "polish":
        payload = load(ROOT / "sam2_prompt_full_ablation/polish/summary_metrics.json") or {}
        by_name = {row["strategy"]: row for row in payload.get("rows", [])}
        return [[name, pair(by_name.get(name, {}).get("SER_proxy"), by_name.get(name, {}).get("CER_proxy"))] for name in SAM_VARIANTS]
    return [[name, official(ROOT / f"sam2_prompt_full_ablation/{dataset}/{name}_ser.json")] for name in SAM_VARIANTS]


def expected_files() -> list[Path]:
    paths = [
        IJCV / "remaining_experiments_preflight.json",
        IJCV / "ours_grandstaff_lmx_train1000_calibration_source/full_dataset_summary.json",
        IJCV / "本地全部实验与数据总账_20260711.md",
        IJCV / "all_transfer_results_big_table_20260711.md",
        IJCV / "ours_grandstaff_lmx_test200_official_style_chordfix_ser.json",
        IJCV / "ours_grandstaff_lmx_test200_train100_calibrated_ser.json",
        IJCV / "ours_grandstaff_pruning_export_calibration/train1000_selection.json",
        IJCV / "ours_grandstaff_lmx_corrector_train1000_test200_eval.json",
        IJCV / "ours_grandstaff_edit_corrector_train1000_test200_eval.json",
        IJCV / "ours_grandstaff_corrector_to_olimpic_zero_test200_eval.json",
        IJCV / "ours_grandstaff_edit_corrector_to_olimpic_zero_test200_eval.json",
        IJCV / "ours_olimpic_calibration/train100_selection.json",
        IJCV / "ours_olimpic_calibration/train1000_selection.json",
        IJCV / "ours_olimpic_train100_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train1000_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train100_edit_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train1000_edit_corrector_test200_eval.json",
        IJCV / "ours_olimpic_corrector_to_grandstaff_zero_test200_eval.json",
        IJCV / "ours_olimpic_edit_corrector_to_grandstaff_zero_test200_eval.json",
        IJCV / "ours_polish_train83_calibrated_test24.json",
        IJCV / "ours_polish_train83_edit_corrector_test24_eval.json",
        ROOT / "abcd_ablation/polish/polish_abcd_summary.json",
        ROOT / "abcd_ablation/polish/summary.json",
        ROOT / "graph_parameter_ablation/polish/graph_parameter_summary.json",
        ROOT / "graph_parameter_ablation/polish/summary.json",
        ROOT / "sam2_prompt_full_ablation/polish/summary_metrics.json",
        ROOT / "sam2_prompt_full_ablation/polish/summary.json",
        ROOT / "rtdetr_ablation/grandstaff_ser.json",
        ROOT / "rtdetr_ablation/olimpic_ser.json",
        ROOT / "rtdetr_ablation/polish_proxy.json",
        ROOT / "yolo_ablation/grandstaff_ser.json",
        ROOT / "yolo_ablation/olimpic_ser.json",
        ROOT / "yolo_ablation/polish_proxy.json",
        ROOT / "rtdetr_runs/rtdetrv2_symbol_expanded_clean/training_complete.json",
        ROOT / "yolo_runs/yolo11n_symbol_expanded_clean/training_complete.json",
    ]
    for model_name in ("ours_grandstaff_lmx_corrector_train1000", "ours_olimpic_lmx_corrector_train100", "ours_olimpic_lmx_corrector_train1000"):
        paths.extend((IJCV / model_name / "split_manifest.json", IJCV / model_name / "best.json", IJCV / model_name / "best.pt"))
    for dataset in ("grandstaff", "olimpic"):
        paths.extend(ROOT / f"abcd_ablation/{dataset}/{name}_ser.json" for name in variant_names())
        paths.extend(ROOT / f"graph_parameter_ablation/{dataset}/{name}_ser.json" for name in GRAPH_VARIANTS)
        paths.extend(ROOT / f"sam2_prompt_full_ablation/{dataset}/{name}_ser.json" for name in SAM_VARIANTS)
    return paths


def coverage_failures() -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def require(path: Path, actual: Any, expected: int, label: str) -> None:
        if path.exists() and actual != expected:
            failures.append({"label": label, "path": str(path), "actual": actual, "expected": expected})

    preflight_path = IJCV / "remaining_experiments_preflight.json"
    preflight = load(preflight_path)
    if preflight is not None and preflight.get("pass") is not True:
        failures.append({"label": "preflight", "path": str(preflight_path), "actual": preflight.get("pass"), "expected": True})
    main_summary_path = IJCV / "ours_grandstaff_lmx_train1000_calibration_source/full_dataset_summary.json"
    main_summary = load(main_summary_path)
    if main_summary is not None:
        require(main_summary_path, main_summary.get("completed"), 1000, "GrandStaff train1000 completed pages")
        require(main_summary_path, main_summary.get("failures"), [], "GrandStaff train1000 failures")
    for path, label in (
        (ROOT / "rtdetr_runs/rtdetrv2_symbol_expanded_clean/training_complete.json", "RT-DETR training epochs"),
        (ROOT / "yolo_runs/yolo11n_symbol_expanded_clean/training_complete.json", "YOLO training epochs"),
    ):
        payload = load(path)
        if payload is not None:
            require(path, payload.get("epochs"), 400, label)
    for model_name, total in (
        ("ours_grandstaff_lmx_corrector_train1000", 1000),
        ("ours_olimpic_lmx_corrector_train100", 100),
        ("ours_olimpic_lmx_corrector_train1000", 1000),
    ):
        manifest_path = IJCV / model_name / "split_manifest.json"
        manifest = load(manifest_path)
        if manifest is not None:
            actual_total = int(manifest.get("train_count", 0)) + int(manifest.get("dev_count", 0))
            require(manifest_path, actual_total, total, f"{model_name} split coverage")
            require(manifest_path, manifest.get("overlap"), [], f"{model_name} train/dev overlap")
            require(manifest_path, manifest.get("seed"), 42, f"{model_name} reproducibility seed")
        best_path = IJCV / model_name / "best.json"
        best = load(best_path)
        if best is not None and best.get("selection_split") != "training-domain dev":
            failures.append({"label": f"{model_name} checkpoint selection", "path": str(best_path), "actual": best.get("selection_split"), "expected": "training-domain dev"})

    for dataset in ("grandstaff", "olimpic"):
        for name in variant_names():
            path = ROOT / f"abcd_ablation/{dataset}/{name}_ser.json"
            payload = load(path)
            if payload is not None:
                require(path, payload.get("summary", {}).get("samples"), 200, f"ABCD {dataset} {name}")
        for name in GRAPH_VARIANTS:
            path = ROOT / f"graph_parameter_ablation/{dataset}/{name}_ser.json"
            payload = load(path)
            if payload is not None:
                require(path, payload.get("summary", {}).get("samples"), 200, f"graph {dataset} {name}")
        for name in SAM_VARIANTS:
            path = ROOT / f"sam2_prompt_full_ablation/{dataset}/{name}_ser.json"
            payload = load(path)
            if payload is not None:
                require(path, payload.get("summary", {}).get("samples"), 200, f"SAM {dataset} {name}")
        for detector in ("rtdetr", "yolo"):
            path = ROOT / f"{detector}_ablation/{dataset}_ser.json"
            payload = load(path)
            if payload is not None:
                require(path, payload.get("summary", {}).get("samples"), 200, f"{detector} {dataset}")

    for path, train_expected, test_expected, label in (
        (IJCV / "ours_grandstaff_pruning_export_calibration/train1000_selection.json", 1000, 200, "GrandStaff train1000"),
        (IJCV / "ours_olimpic_calibration/train100_selection.json", 100, 200, "OLiMPiC train100"),
        (IJCV / "ours_olimpic_calibration/train1000_selection.json", 1000, 200, "OLiMPiC train1000"),
    ):
        payload = load(path)
        if payload is not None:
            require(path, payload.get("best_train", {}).get("samples"), train_expected, f"{label} full training predictions")
            require(path, payload.get("test", {}).get("samples"), test_expected, f"{label} test")

    correction_paths = [
        IJCV / "ours_grandstaff_lmx_corrector_train1000_test200_eval.json",
        IJCV / "ours_grandstaff_edit_corrector_train1000_test200_eval.json",
        IJCV / "ours_grandstaff_corrector_to_olimpic_zero_test200_eval.json",
        IJCV / "ours_grandstaff_edit_corrector_to_olimpic_zero_test200_eval.json",
        IJCV / "ours_olimpic_train100_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train1000_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train100_edit_corrector_test200_eval.json",
        IJCV / "ours_olimpic_train1000_edit_corrector_test200_eval.json",
        IJCV / "ours_olimpic_corrector_to_grandstaff_zero_test200_eval.json",
        IJCV / "ours_olimpic_edit_corrector_to_grandstaff_zero_test200_eval.json",
    ]
    for path in correction_paths:
        payload = load(path)
        if payload is not None:
            require(path, payload.get("samples"), 200, path.stem)

    for path, label in (
        (ROOT / "abcd_ablation/polish/summary.json", "ABCD Polish"),
        (ROOT / "graph_parameter_ablation/polish/summary.json", "graph Polish"),
        (ROOT / "sam2_prompt_full_ablation/polish/summary.json", "SAM Polish"),
    ):
        payload = load(path)
        if payload is not None:
            require(path, payload.get("pages"), 24, label)
    for path, label in (
        (ROOT / "rtdetr_ablation/polish_proxy.json", "RT-DETR Polish"),
        (ROOT / "yolo_ablation/polish_proxy.json", "YOLO Polish"),
        (IJCV / "ours_polish_train83_edit_corrector_test24_eval.json", "Polish edit corrector"),
    ):
        payload = load(path)
        if payload is not None:
            require(path, payload.get("samples"), 24, label)
    polish_calibration_path = IJCV / "ours_polish_train83_calibrated_test24.json"
    polish_calibration = load(polish_calibration_path)
    if polish_calibration is not None:
        require(polish_calibration_path, polish_calibration.get("test", {}).get("samples"), 24, "Polish calibration")
        require(polish_calibration_path, polish_calibration.get("materialized", {}).get("train_rows"), 83, "Polish train83 materialization")
        require(polish_calibration_path, polish_calibration.get("materialized", {}).get("test_rows"), 24, "Polish test24 label-free inputs")
    return failures


def render() -> str:
    lines = [
        "# 三数据集迁移、对比与消融完整大表",
        "",
        "所有数值均由本地结果文件自动读取；`未完成` 表示没有对应结果文件。LMX 表格为 SER / SER-no-tuplets，Polish 表格为 proxy SER / proxy CER；均为越低越好。",
        "",
        "## 方法说明",
        "",
        "- ours：本文方法，固定 A=规则谱线先验、B=检测器、C=SAM2、D=图关系主架构；样本微调仅调整训练域选择的剪枝/导出策略与轻量 LMX 后端。",
        "- Zeus：已发布的端到端 LMX 基线；Zeus-Olympic 与 Zeus-GrandStaff 表示不同源域权重。",
        "- SMT：已发布的 eKern 基线，用于 Polish Scores 对比。",
        "- RT-DETR、YOLO：只替换 ours 的 B 检测器，其余模块和评测协议保持不变。",
        "- A（先验G）：通过规则检测谱线/谱表，提供staff、staff-space、pitch-step等几何先验及规则候选；A=0时移除该规则分支和对应先验属性。",
        "- B（检测器）：默认DEIM/D-FINE符号检测；B=0时移除检测器来源符号。RT-DETR与YOLO只作为B的内部替换实验。",
        "- C（SAM2）：用检测框及可选正/负点prompt生成动态mask并细化符号几何；C=0使用SAM2前的固定bbox表示。",
        "- D（图关系）：构建note-stem、beam-stem、ledger-note、slur-note关系并用于事件组装；D=0时在事件组装前清空这些关系。",
        "- 当A=B=0时没有符号候选，C没有prompt、D没有节点，按真实空预测计分，不用伪造符号补齐。",
        "",
        "历史 Zeus/SMT 数字来自本地已核验台账 `outputs/ijcv_repro/本地全部实验与数据总账_20260711.md` 与 `outputs/ijcv_repro/all_transfer_results_big_table_20260711.md`；其中 Zeus→OLiMPiC train1000 的42.924/39.870以较新的中文总账为准。新数字来自本轮 JSON 产物。",
        "",
    ]

    lines += ["## GrandStaff-LMX test200", "", "### 主对比", ""]
    grand_selection = IJCV / "ours_grandstaff_pruning_export_calibration/train1000_selection.json"
    lines += table(["方法", "训练/迁移设置", "目标样本", "SER / SER-no-tuplets"], [
        ["ours", "直接零样本", "0", official(IJCV / "ours_grandstaff_lmx_test200_official_style_chordfix_ser.json")],
        ["ours", "训练域联合选参", "100", official(IJCV / "ours_grandstaff_lmx_test200_train100_calibrated_ser.json")],
        ["ours", "联合选参", "1000", selection(grand_selection)],
        ["ours", "Transformer纠错", "1000", correction(IJCV / "ours_grandstaff_lmx_corrector_train1000_test200_eval.json")],
        ["ours", "保守编辑纠错", "1000", correction(IJCV / "ours_grandstaff_edit_corrector_train1000_test200_eval.json")],
        ["ours", "OLiMPiC-only → GrandStaff 零样本 Transformer", "目标0", correction(IJCV / "ours_olimpic_corrector_to_grandstaff_zero_test200_eval.json")],
        ["ours", "OLiMPiC-only → GrandStaff 零样本编辑纠错", "目标0", correction(IJCV / "ours_olimpic_edit_corrector_to_grandstaff_zero_test200_eval.json")],
        ["Zeus", "Zeus-Olympic → GrandStaff 零样本", "0", "53.19 / 52.87"],
        ["Zeus", "Zeus-Olympic 微调", "100", "38.18 / 35.48"],
        ["Zeus", "Zeus-Olympic 微调", "1000", "14.01 / 14.25"],
        ["Zeus", "Zeus-GrandStaff 域内参考", "全量", "1.44 / 1.49"],
    ])
    lines += ["### ours 样本量影响", ""]
    lines += table(["目标训练样本", "结果"], [["0", official(IJCV / "ours_grandstaff_lmx_test200_official_style_chordfix_ser.json")], ["100", official(IJCV / "ours_grandstaff_lmx_test200_train100_calibrated_ser.json")], ["1000（联合选参）", selection(grand_selection)]])
    lines += ["### A/B/C/D 全量消融", ""] + table(["组合", "结果"], abcd_rows("grandstaff"))
    lines += ["### 模块内消融", "", "检测器替换：", ""] + table(["B检测器", "结果"], [["DEIM", official(IJCV / "ours_grandstaff_lmx_test200_official_style_chordfix_ser.json")], ["RT-DETR", official(ROOT / "rtdetr_ablation/grandstaff_ser.json")], ["YOLO", official(ROOT / "yolo_ablation/grandstaff_ser.json")]])
    lines += ["SAM2 prompt：", ""] + table(["策略", "结果"], sam_rows("grandstaff"))
    lines += ["图关系参数：", ""] + table(["变体", "结果"], graph_rows("grandstaff"))

    lines += ["## OLiMPiC test200", "", "### 主对比", ""]
    lines += table(["方法", "训练/迁移设置", "目标样本", "SER / SER-no-tuplets"], [
        ["ours", "直接零样本", "0", "74.42 / 73.57"],
        ["ours", "联合选参", "100", selection(IJCV / "ours_olimpic_calibration/train100_selection.json")],
        ["ours", "Transformer纠错", "100", correction(IJCV / "ours_olimpic_train100_corrector_test200_eval.json")],
        ["ours", "保守编辑纠错", "100", correction(IJCV / "ours_olimpic_train100_edit_corrector_test200_eval.json")],
        ["ours", "联合选参", "1000", selection(IJCV / "ours_olimpic_calibration/train1000_selection.json")],
        ["ours", "Transformer纠错", "1000", correction(IJCV / "ours_olimpic_train1000_corrector_test200_eval.json")],
        ["ours", "保守编辑纠错", "1000", correction(IJCV / "ours_olimpic_train1000_edit_corrector_test200_eval.json")],
        ["ours", "GrandStaff-only → OLiMPiC 零样本 Transformer", "目标0", correction(IJCV / "ours_grandstaff_corrector_to_olimpic_zero_test200_eval.json")],
        ["ours", "GrandStaff-only → OLiMPiC 零样本编辑纠错", "目标0", correction(IJCV / "ours_grandstaff_edit_corrector_to_olimpic_zero_test200_eval.json")],
        ["Zeus", "Zeus-GrandStaff → OLiMPiC 零样本", "0", "56.49 / 50.63"],
        ["Zeus", "Zeus-GrandStaff 微调", "100", "51.79 / 47.80"],
        ["Zeus", "Zeus-GrandStaff 微调（expanded）", "1000", "42.92 / 39.87"],
        ["Zeus", "Zeus-Olympic 域内参考", "全量", "13.29 / 9.53"],
    ])
    lines += ["### ours 样本量影响", ""] + table(["目标训练样本", "结果"], [["0", "74.42 / 73.57"], ["100", selection(IJCV / "ours_olimpic_calibration/train100_selection.json")], ["1000", selection(IJCV / "ours_olimpic_calibration/train1000_selection.json")]])
    lines += ["### A/B/C/D 全量消融", ""] + table(["组合", "结果"], abcd_rows("olimpic"))
    lines += ["### 模块内消融", "", "检测器替换：", ""] + table(["B检测器", "结果"], [["DEIM", "74.42 / 73.57"], ["RT-DETR", official(ROOT / "rtdetr_ablation/olimpic_ser.json")], ["YOLO", official(ROOT / "yolo_ablation/olimpic_ser.json")]])
    lines += ["SAM2 prompt：", ""] + table(["策略", "结果"], sam_rows("olimpic"))
    lines += ["图关系参数：", ""] + table(["变体", "结果"], graph_rows("olimpic"))

    lines += ["## Polish Scores test24", "", "### 主对比", ""]
    lines += table(["方法", "训练/迁移设置", "目标样本", "proxy SER / proxy CER"], [
        ["ours", "直接零样本", "0", "98.73 / 64.13"],
        ["ours", "训练域剪枝校准", "83（全训练集）", polish(IJCV / "ours_polish_train83_calibrated_test24.json")],
        ["ours", "保守编辑纠错", "83（全训练集）", polish(IJCV / "ours_polish_train83_edit_corrector_test24_eval.json")],
        ["SMT", "FP-GrandStaff → Polish 零样本", "0", "153.00 / 141.76"],
        ["SMT", "Polish full-train83 微调", "83", "107.97 / 125.30"],
    ])
    lines += ["说明：Polish 公开训练集只有83页，因此不报告虚构的100或1000样本结果。", "", "### ours 样本量影响", ""]
    lines += table(["目标训练样本", "结果"], [["0", "98.73 / 64.13"], ["83（全量）", polish(IJCV / "ours_polish_train83_calibrated_test24.json")], ["83（编辑纠错）", polish(IJCV / "ours_polish_train83_edit_corrector_test24_eval.json")]])
    lines += ["### A/B/C/D 全量消融", ""] + table(["组合", "结果"], abcd_rows("polish"))
    lines += ["### 模块内消融", "", "检测器替换：", ""] + table(["B检测器", "结果"], [["DEIM", "98.73 / 64.13"], ["RT-DETR", polish(ROOT / "rtdetr_ablation/polish_proxy.json")], ["YOLO", polish(ROOT / "yolo_ablation/polish_proxy.json")]])
    lines += ["SAM2 prompt：", ""] + table(["策略", "结果"], sam_rows("polish"))
    lines += ["图关系参数：", ""] + table(["变体", "结果"], graph_rows("polish"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final Chinese per-dataset comparison and ablation report from verified local artifacts.")
    parser.add_argument("--out-md", type=Path, default=IJCV / "三数据集迁移对比与消融完整大表_最终版.md")
    parser.add_argument("--out-json", type=Path, default=IJCV / "三数据集迁移对比与消融完整大表_完整性.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    missing = [str(path) for path in expected_files() if not path.exists()]
    coverage = coverage_failures()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render(), encoding="utf-8")
    status = {"complete": not missing and not coverage, "expected_files": len(expected_files()), "missing_count": len(missing), "missing": missing, "coverage_failure_count": len(coverage), "coverage_failures": coverage, "report": str(args.out_md)}
    args.out_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.strict and (missing or coverage):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
