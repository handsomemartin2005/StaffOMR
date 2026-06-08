# V2 方法设计文档：面向 OMR 的神经符号识别

## 1. 版本定位

V1 是当前项目中的可解释传统 OMR baseline。它依赖二值化、五线谱水平投影、几何过滤、连通域、密度规则和调试可视化来识别谱线、符头、符杆、连桁、小节线、休止符、变音记号、文字区域、加线以及简单的连音线/延音线候选。

V2 的目标不是完全替换 V1，而是保留 V1 中稳定、可解释的几何能力，把最脆弱的符号识别规则替换成神经网络模块。简单说：

```text
V1 继续负责坐标系和几何约束
V2 用神经模型识别符号、细化 mask、辅助标注和聚类
```

V2 当前限定在“符号级识别”，即：

```text
输入乐谱图像
-> 定位五线谱坐标系
-> 检测基础视觉符号
-> 必要时细化符号 mask
-> 输出符号级 JSON 和可视化结果
```

V2 暂时不解决完整乐谱语义问题，包括：

- 不做完整 MusicXML 解码
- 不做声部拆分
- 不做小节级节奏合法性校验
- 不做跨谱表音乐关系推理
- 不做端到端 page-to-score 转录

因此，V2 是从传统规则系统过渡到神经 OMR 系统的中间层：它把页面上的可见符号更稳定地找出来，并保留足够结构化的信息，供后续关系建模、MusicXML 解码或 LLM 音乐教育反馈使用。

## 2. 核心方法总览

V2 当前采用的核心组合是：

```text
DEIM-D-FINE + SAM2 + DINOv2 + RapidOCR + V1 geometry
```

整体推理流程：

```text
V1 staff / staff_space 定位
-> 页面或 staff crop 生成
-> DEIM-D-FINE 检测乐谱符号框
-> V1 规则候选和神经检测结果融合
-> SAM2 用检测框作为 prompt 细化 mask
-> RapidOCR 识别 text_region 内文字
-> DINOv2 提供 crop embedding，用于标注清洗、聚类、检索和重复框处理
-> 输出 V2 symbol JSON、overlay、metrics 和重建图
```

其中，DEIM-D-FINE 是主检测器；SAM2 不是一级检测器，只在已有候选框后做 mask refinement；DINOv2 也不是主检测器，而是为标注和质量控制提供视觉 embedding。

## 3. 模块职责

| 模块 | 模型/方法 | 主要输出 | 在 V2 中的作用 |
|---|---|---|---|
| V1 几何规则 | 传统图像处理 | staff lines、staff space、staff regions | 建立坐标系、约束符号位置、提供弱标签和 fallback |
| DEIM-D-FINE | DETR-style object detection | class + bbox + confidence | 主符号检测器 |
| SAM2 | promptable segmentation | pixel mask | 根据检测框细化符号 mask |
| DINOv2 | self-supervised visual features | embedding vector | 聚类、检索、弱标签清洗、重复检测合并 |
| RapidOCR | OCR | text string + confidence | 识别 `To Coda`、`D.S. al Coda`、`8va`、`Pno.` 等文字 |
| 评估脚本 | DeepScores / V1-V2 metrics | CER、SER、LER、per-class precision/recall/F1 | 量化 V1 与 V2 的符号级差异 |

## 4. V1 中保留的能力

以下 V1 能力在 V2 中继续保留，因为它们稳定、可解释，并且能减少神经模型要学习的问题规模：

| V1 组件 | 是否保留 | 原因 |
|---|---:|---|
| 灰度化 / 二值化 | 是 | 提供调试视图，也可作为后续模型输入变体 |
| 谱线检测 | 是 | 对 printed score 当前较稳定 |
| staff grouping | 是 | 提供 staff index 和垂直坐标系 |
| staff_space 估计 | 是 | 归一化不同页面和字体下的符号尺度 |
| staff crop 生成 | 是 | 避免整页缩放导致小符号丢失 |
| pitch-grid 坐标 | 是 | 符头检测后用于 pitch step 赋值 |
| symbol-to-staff assignment | 是 | 基于几何的后处理仍然可靠 |
| overlay / reconstruction | 是 | 用于定位神经模型误检、漏检和分类错误 |

以下 V1 规则不再作为主识别器，而是变成弱标签、fallback 或评估辅助：

| 原 V1 规则 | V2 替代方向 |
|---|---|
| 符头椭圆密度打分 | DEIM-D-FINE 符头检测 |
| filled/open 中心密度规则 | DEIM 类别预测，后续可加 crop classifier |
| 变音记号连通域规则 | DEIM-D-FINE accidental 检测 |
| 休止符形状启发式 | DEIM-D-FINE rest 检测 |
| 符杆竖向 run 检测 | DEIM 检测 + SAM2 mask + 几何检查 |
| 连桁 corridor scoring | DEIM 检测 + SAM2 mask |
| 加线短横线检测 | DEIM 检测 + notehead 关系约束 |
| 文字区域连通域聚类 | DEIM text_region + OCR |

## 5. 目标符号类别

V2 第一版使用较小的基础视觉符号集合：

```text
filled_notehead
open_notehead
stem
beam
barline
ledger_line
sharp
flat
natural
rest
treble_clef
bass_clef
text_region
slur_or_tie
```

第一版避免过早拆分太多细类。类别越细，标注成本越高，弱标签噪声越难控制，训练和评估也更难稳定。

后续稳定后可以拆分：

```text
quarter_rest / eighth_rest / half_rest / whole_rest
double_barline / repeat_barline
beam_primary / beam_secondary
grace_notehead
ornament
dynamic_text
```

## 6. 数据与标注策略

V2 不从纯手工标注开始，而是采用弱监督 bootstrap：

```text
V1 规则生成弱框
-> 导出 COCO detection labels
-> 裁剪 symbol crops
-> DINOv2 生成 embedding
-> 聚类、检索、发现离群样本
-> 人工修正一小批关键样本
-> 训练 DEIM-D-FINE
-> 用 DEIM 预测 + SAM2 mask 做 active learning
-> 迭代修正和重训
```

当前工程中对应脚本：

| 阶段 | 脚本 |
|---|---|
| 导出 V1 弱标签 | `tools/export_v1_weak_labels.py` |
| 裁剪符号候选 | `tools/build_symbol_crops.py` |
| DINOv2 embedding / clustering | `tools/embed_symbol_crops_dinov2.py` |
| DeepScores 转 V2 COCO | `tools/prepare_deepscores_v2_dataset.py` |
| 生成 DEIM config | `tools/create_deim_symbol_config.py` |
| DEIM 推理导出 | `tools/export_deim_predictions.py` |

COCO detection 标注格式：

```json
{
  "images": [],
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1,
      "area": 123,
      "iscrowd": 0
    }
  ],
  "categories": []
}
```

mask 标注可以后置；第一阶段优先保证 bbox 和 class。SAM2 生成的 mask 先作为辅助结果和可视化调试信息，不直接当作完全可信的真值。

## 7. 训练设计

### 7.1 数据集准备

训练样本来源包括：

- 当前 demo 页面和本地输出
- 本地 DeepScoresV2 / dense printed-score 数据
- 后续人工修正过的符号框
- 渲染出的不同字体、不同缩放和不同页面布局样本

完整页面上的符号很小，不能只把整页压到固定输入尺寸训练。推荐训练单元优先级：

```text
staff crop
-> grand-staff / system crop
-> overlapping page tile
-> full page as ablation
```

staff crop 推荐范围：

```text
x = staff 横向全宽
y = staff_y0 - 3.5 * staff_space
    到 staff_y1 + 3.5 * staff_space
overlap = 10% 到 20%
```

### 7.2 DEIM-D-FINE 训练

DEIM-D-FINE 负责输出：

```json
{
  "bbox": [x0, y0, x1, y1],
  "class": "filled_notehead",
  "confidence": 0.93,
  "source": "deim_dfine"
}
```

训练初期可以先使用较粗类别：

```text
notehead
stem
beam
barline
ledger_line
accidental
rest
clef
text_region
slur_or_tie
```

当数据质量足够稳定后，再拆成：

```text
filled_notehead / open_notehead
sharp / flat / natural
treble_clef / bass_clef
```

### 7.3 checkpoint 选择

训练脚本会输出多个 checkpoint。当前导出逻辑优先使用：

```text
last.pth
best_stg2.pth
best_stg1.pth
checkpoint*.pth
```

实际评估时不应只看最后一个 epoch，还要比较 validation AP 和目标页面上的符号级 precision/recall/F1。

## 8. 推理与融合策略

V2 推理有两种主要输出策略。

### 8.1 Detector-only

只使用 DEIM-D-FINE 检测结果，再用 V1 几何补充 staff、pitch_step 等字段。

优点：

- precision 通常更高
- 输出更干净
- 误检少，适合后续语义图构建

缺点：

- recall 可能偏低
- 对训练集中覆盖不足的类别容易漏检
- ledger_line、stem 等细长目标可能缺失

### 8.2 Hybrid / rule fallback

把 DEIM 检测结果和 V1 rule candidate 融合。检测器优先或规则优先由参数控制：

```text
--merge-rule-classes
--rule-fallback-iou
--fusion-preference detector|rule
```

优点：

- recall 更高
- 能补上 detector 漏掉的一部分细小或几何清晰符号
- 适合作为 active learning 的候选集合

缺点：

- V1 弱规则会带入误检
- 输出更密，需要后处理或人工审查
- 对 stem、ledger_line、open_notehead 这类容易过检的类别要额外控制

工程上建议同时保留 detector-only 和 hybrid 两份结果：前者作为高精度结构输入，后者作为查漏和迭代标注输入。

## 9. SAM2 mask refinement

SAM2 的使用方式：

```text
image crop + DEIM/V1 bbox prompt
-> SAM2 mask
-> class-specific geometry sanity check
-> 写入 symbol JSON
```

SAM2 对以下类别尤其有用：

- stem
- beam
- ledger_line
- slur_or_tie
- barline
- notehead outline

但 SAM2 mask 不能无条件信任。需要继续使用 V1 几何检查：

```text
stem mask 应主要是竖向细长结构
beam mask 应是横向或斜向长条，并靠近符杆端点
ledger_line 应接近符头并短于 staff line
slur_or_tie 应有弧形特征，不能和 staff line 重合
barline 应跨 staff 高度且近似竖直
```

不通过检查的 mask 可以保留 bbox，但不写入高可信 mask 字段。

## 10. OCR 文字识别

V2 将文字识别限定在 `text_region` 上，不让 OCR 直接扫整页。流程：

```text
DEIM/V1 找到 text_region
-> 裁剪 text crop
-> RapidOCR 识别文字
-> 把 text、confidence、source 写回 symbol JSON
```

目标文字包括：

```text
To Coda
D.S. al Coda
8va
Pno.
动态记号和其他辅助文本
```

文字识别结果暂时作为附加信息，不直接参与乐谱节奏或音高解码。

## 11. DINOv2 的使用方式

DINOv2 生成每个 symbol crop 的视觉 embedding。它不负责直接分类最终符号，而是用于：

1. 聚类辅助标注
   - 相似 crop 聚成一组
   - 人工一次性审查一个 cluster

2. 弱标签清洗
   - 同一类别内部找 outlier
   - 发现 V1 规则误标样本

3. 检索
   - 用户标记一个真实 `natural`
   - 系统找出视觉相似的候选

4. 重复框合并
   - overlapping tiles 可能重复检测同一符号
   - 同时比较 IoU、中心距离和 embedding similarity

DINOv2 embedding 适合用于“找相似”和“找异常”，不应直接替代检测器。

## 12. 输出 JSON 契约

V2 输出应保持结构化，并能和 V1 结果对齐比较。

示例：

```json
{
  "version": "v2",
  "input": "page.png",
  "staff_geometry_source": "v1_rules",
  "symbols": [
    {
      "id": "sym_000001",
      "class": "filled_notehead",
      "bbox": [120, 88, 132, 99],
      "confidence": 0.94,
      "staff": 0,
      "pitch_step": 4,
      "detector": "deim_dfine",
      "mask": null,
      "embedding_id": "dinov2_000001",
      "source": "v2_detector"
    }
  ]
}
```

必要字段：

- `id`
- `class`
- `bbox`
- `confidence`
- `staff`
- `source`

建议字段：

- `pitch_step`
- `normalized_width`
- `normalized_height`
- `mask`
- `mask_score`
- `detector`
- `embedding_id`
- `ocr_text`
- `postprocess_decisions`

## 13. 评估设计

V2 评估以符号级为主。

检测指标：

- AP / AP50 / AP75
- per-class precision
- per-class recall
- per-class F1
- 每页 false positives
- 每个 staff 的 missed symbols

结构化 OMR 指标：

- CER：class sequence edit rate
- SER：symbol edit rate
- LER：staff line / layout error rate

mask 指标在有可靠 mask 标注后再加入：

- mask IoU
- stem / beam / ledger_line / slur 的 skeleton 或 centerline error

V1 与 V2 的比较重点：

| 符号 | V1 常见问题 | V2 预期改进 |
|---|---|---|
| notehead | 密集和弦下密度阈值不稳 | detector 学习局部视觉上下文 |
| filled/open | 去谱线可能破坏符头中心 | class head 使用原始 crop 视觉信息 |
| accidental | 连通域规则受字体影响 | detector 学习形状变化 |
| rest | 启发式形状规则脆弱 | detector 学习休止符外观 |
| beam | corridor 规则容易过连或漏连 | detector/mask 处理局部形状 |
| ledger_line | 短横线容易和噪声混淆 | detector/mask + notehead 关系约束 |
| text_region | 模板窄，泛化差 | detector 找文字区域，OCR 只处理局部 crop |

## 14. 当前工程落地文件

主要入口：

```text
tools/run_v2_full_pipeline.py
```

核心脚本：

| 文件 | 作用 |
|---|---|
| `tools/omr_v2_common.py` | V2 类别、bbox、JSON、overlay 和 COCO 工具 |
| `tools/export_v1_weak_labels.py` | V1 输出转弱标签 |
| `tools/build_symbol_crops.py` | 构建 symbol crop 数据 |
| `tools/embed_symbol_crops_dinov2.py` | DINOv2 embedding、nearest neighbors、clusters |
| `tools/prepare_deepscores_v2_dataset.py` | DeepScores dense 标注转 V2 COCO |
| `tools/create_deim_symbol_config.py` | 生成 DEIM-D-FINE 训练配置 |
| `tools/export_deim_predictions.py` | DEIM checkpoint 推理并导出预测 |
| `tools/infer_neural_symbols_v2.py` | V2 symbol fusion、staff assignment、overlay |
| `tools/refine_masks_sam2.py` | SAM2 mask refinement |
| `tools/ocr_score_text.py` | text_region OCR |
| `tools/evaluate_v1_v2_metrics.py` | 单页 V1/V2 指标 |
| `tools/evaluate_v1_v2_dataset_metrics.py` | 数据集级 V1/V2 指标 |
| `tools/reconstruct_v2_symbols.py` | 从 symbol JSON 重建可视化 |

典型输出：

```text
outputs/<run>/deim/predictions.json
outputs/<run>/symbols/symbols_v2_with_sam2.json
outputs/<run>/ocr/symbols_v2_ocr.json
outputs/<run>/ocr/ocr_overlay.png
outputs/<run>/metrics/evaluation_v1_v2.json
outputs/<run>/visuals/input_vs_v2_reconstructed.png
```

## 15. 推荐运行命令

安装依赖：

```bash
pip install -r requirements-v2.txt
```

下载 DEIM 预训练权重：

```bash
python -m gdown 1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e -O outputs/models/deim/deim_hgnetv2_n_coco.pth
```

运行完整目标页 V2 pipeline：

```bash
python tools/run_v2_full_pipeline.py \
  --out-root outputs/v2_deim_full_tuned100 \
  --deim-dataset-root outputs/v2_deim_ds_target_tuned100 \
  --deim-run-dir outputs/v2_deim_runs/v2_symbol_target100_tuned \
  --deim-epochs 100 \
  --deim-train-limit 1 \
  --deim-val-limit 1 \
  --deim-tuning-checkpoint outputs/models/deim/deim_hgnetv2_n_coco.pth
```

如果使用本地完整 DeepScores split：

```bash
python tools/run_v2_full_pipeline.py \
  --out-root outputs/v2_deim_full_all \
  --deim-dataset-root outputs/v2_deim_ds_all \
  --deim-run-dir outputs/v2_deim_runs/v2_symbol_all \
  --deim-epochs 60 \
  --deim-train-limit -1 \
  --deim-val-limit -1 \
  --deim-tuning-checkpoint outputs/models/deim/deim_hgnetv2_n_coco.pth
```

如果只想跑旧的 V1-weak V2 shell：

```bash
python tools/run_v2_full_pipeline.py --detector weak
```

## 16. 风险与控制

| 风险 | 控制方式 |
|---|---|
| V1 弱标签本身有噪声 | 用 DINOv2 聚类和人工 review 清洗 |
| 小符号在整页缩放时丢失 | 训练和推理优先使用 staff crop / overlapping tile |
| 密集和弦中 bbox 重叠严重 | 保持高分辨率 crop，单独评估 notehead recall |
| SAM2 mask 吃进谱线或邻近符号 | 加 class-specific geometry sanity checks |
| detector 只学会单一字体 | 混合 DeepScores、local score、渲染变体 |
| 类别过多导致训练不稳 | 先用粗类别，稳定后再拆细 |
| hybrid fallback 误检过多 | 分开保留 detector-only 与 hybrid，并按用途选择 |
| OCR 误识别音乐符号 | OCR 只在 text_region crop 内运行 |

## 17. 后续迭代重点

短期优先级：

1. 对 detector-only 和 hybrid 输出分别做可视化审查，明确哪些类别该用 detector，哪些类别该保留 rule fallback。
2. 针对 stem、ledger_line、open_notehead 这些高误检或低召回类别做专门清洗。
3. 用 DINOv2 nearest neighbors 找出弱标签中的 outlier，建立人工修正小样本集。
4. 把 SAM2 mask 的几何检查结果写入 `postprocess_decisions`，便于排查 mask 错误。
5. 增加 dataset-level evaluation，避免只对单页过拟合。

中期方向：

1. 引入 crop classifier 修正 filled/open、sharp/natural/flat 等易混类别。
2. 建立 notehead-stem-beam 的关系图，为节奏和声部推理做准备。
3. 在符号级 JSON 基础上生成 MusicXML 或 linearized score 的中间表示。
4. 将结构化识别结果接入 LLM 音乐教育反馈，例如错音定位、节奏讲解和练习建议。
