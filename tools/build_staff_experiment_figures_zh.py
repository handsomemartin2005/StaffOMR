from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IJCV = ROOT / "outputs" / "ijcv_repro"
OUT = ROOT / "outputs" / "paper_reports" / "staff_experiments_zh" / "figures"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def annotate_bars(ax, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    polish = load(IJCV / "order_invariant_event_f1" / "polish_test24.json")
    debussy = load(IJCV / "order_invariant_event_f1" / "debussy_transfer24.json")
    olimpic = load(IJCV / "unified_transfer_olimpic_test200" / "metrics.json")
    ablation = load(ROOT / "outputs" / "debussy_abcd_ablation" / "summary.json")

    methods = ["STAFF（本文）", "SMT", "Zeus"]
    keys = ["staff", "smt", "zeus"]
    colors = ["#c83e4d", "#4c78a8", "#59a14f"]

    # Figure 1: primary zero-shot Event-F1.
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    datasets = ["Polish Scores\ntest24", "Debussy OMR\ntransfer24"]
    x = np.arange(len(datasets))
    width = 0.24
    for i, (method, key, color) in enumerate(zip(methods, keys, colors)):
        vals = [polish["summary"][key]["f1"], debussy["summary"][key]["f1"]]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=method, color=color)
        annotate_bars(ax, bars)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("无序音高–时值 Event-F1（%，越高越好）")
    ax.set_title("核心零样本跨域对比")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "核心零样本_EventF1.png", dpi=240)
    plt.close(fig)

    # Figure 2: precision/recall/F1 facets.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, name, payload in zip(axes, ["Polish Scores test24", "Debussy OMR transfer24"], [polish, debussy]):
        metric_names = ["Precision", "Recall", "Event-F1"]
        xm = np.arange(3)
        for i, (method, key, color) in enumerate(zip(methods, keys, colors)):
            row = payload["summary"][key]
            vals = [row["precision"], row["recall"], row["f1"]]
            bars = ax.bar(xm + (i - 1) * width, vals, width, label=method, color=color)
            annotate_bars(ax, bars, "{:.2f}")
        ax.set_xticks(xm, metric_names)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("指标（%，越高越好）")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("核心零样本实验：Precision、Recall 与 Event-F1", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "核心零样本_P_R_F1.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: paired bootstrap confidence intervals.
    comparisons = [
        ("Polish：STAFF−SMT", polish["paired_bootstrap"]["staff_minus_smt"]),
        ("Polish：STAFF−Zeus", polish["paired_bootstrap"]["staff_minus_zeus"]),
        ("Debussy：STAFF−SMT", debussy["paired_bootstrap"]["staff_minus_smt"]),
        ("Debussy：STAFF−Zeus", debussy["paired_bootstrap"]["staff_minus_zeus"]),
    ]
    y = np.arange(len(comparisons))
    delta = np.array([row[1]["staff_minus_baseline"] for row in comparisons])
    low = np.array([row[1]["ci95_low"] for row in comparisons])
    high = np.array([row[1]["ci95_high"] for row in comparisons])
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.errorbar(delta, y, xerr=[delta - low, high - delta], fmt="o", color="#c83e4d", capsize=5, markersize=7)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, [row[0] for row in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("STAFF 相对基线的 Event-F1 差值（百分点）")
    ax.set_title("10,000 次逐页配对 bootstrap 的 95% 置信区间")
    ax.grid(axis="x", alpha=0.22)
    for yi, d, lo, hi in zip(y, delta, low, high):
        ax.text(hi + 0.25, yi, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "核心零样本_显著性.png", dpi=240)
    plt.close(fig)

    # Figure 4: output-to-gold ratio diagnosis.
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(2)
    for i, (method, key, color) in enumerate(zip(methods, keys, colors)):
        vals = []
        for payload in [polish, debussy]:
            row = payload["summary"][key]
            vals.append(100.0 * row["predicted_events"] / row["gold_events"])
        bars = ax.bar(x + (i - 1) * width, vals, width, label=method, color=color)
        annotate_bars(ax, bars, "{:.1f}%")
    ax.axhline(100, color="#333333", linestyle="--", linewidth=1, label="预测数=真值数")
    ax.set_xticks(x, ["Polish Scores", "Debussy OMR"])
    ax.set_ylabel("预测事件数 / 真值事件数（%）")
    ax.set_title("输出长度诊断：欠生成与过生成")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "核心零样本_输出长度比.png", dpi=240)
    plt.close(fig)

    # Figure 5: OLiMPiC boundary result.
    metric_specs = [("SER", "SER", False), ("TEDn", "TEDn-lmx", False), ("Event-F1", "Event-F1", True)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (title, field, higher) in zip(axes, metric_specs):
        vals = []
        for key in keys:
            row = olimpic["methods"][key]
            if field == "SER":
                vals.append(row["SER"]["SER"])
            elif field == "Event-F1":
                vals.append(row["Event-F1"]["f1"])
            else:
                vals.append(row[field])
        bars = ax.bar(methods, vals, color=colors)
        annotate_bars(ax, bars, "{:.2f}")
        ax.set_title(f"{title}（{'越高越好' if higher else '越低越好'}）")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("OLiMPiC test200：严格结构转录的边界结果", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "OLiMPiC_结构边界.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: Debussy leave-one-out with confidence intervals.
    loo = ablation["leave_one_out"]
    module_labels = ["A：谱表/规则先验", "B：DEIM 检测", "C：SAM2", "D：关系图"]
    module_keys = ["A", "B", "C", "D"]
    delta = np.array([loo[k]["bootstrap"]["full_minus_ablated"] for k in module_keys])
    low = np.array([loo[k]["bootstrap"]["ci95_low"] for k in module_keys])
    high = np.array([loo[k]["bootstrap"]["ci95_high"] for k in module_keys])
    y = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    point_colors = ["#59a14f" if x > 0 else "#c83e4d" for x in delta]
    for yi, d, lo, hi, color in zip(y, delta, low, high, point_colors):
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=color, capsize=5, markersize=8)
        ax.text(hi + 0.18, yi, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, module_labels)
    ax.invert_yaxis()
    ax.set_xlabel("完整模型 − 去模块模型 Event-F1（百分点）")
    ax.set_title("Debussy transfer24：模块 leave-one-out 贡献")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "Debussy_模块贡献.png", dpi=240)
    plt.close(fig)

    manifest = {
        "sources": {
            "polish": str(IJCV / "order_invariant_event_f1" / "polish_test24.json"),
            "debussy": str(IJCV / "order_invariant_event_f1" / "debussy_transfer24.json"),
            "olimpic": str(IJCV / "unified_transfer_olimpic_test200" / "metrics.json"),
            "debussy_ablation": str(ROOT / "outputs" / "debussy_abcd_ablation" / "summary.json"),
        },
        "figures": [str(path) for path in sorted(OUT.glob("*.png"))],
    }
    (OUT.parent / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
