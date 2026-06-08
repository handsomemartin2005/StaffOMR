# V2.1 方法设计文档：五线谱几何约束下的 Box-to-Mask 符号结构细化

## 1. V2.1 的定位

V2.1 是 V2 神经符号识别方法上的一次形状表达升级。

V2 的核心目标是：

```text
V1 staff geometry
-> DEIM-D-FINE 符号检测
-> SAM2 mask refinement
-> DINOv2 / OCR / metrics
-> symbol-level JSON
```

V2.1 不改变这个主链路，也不把目标检测器从矩形框检测器改成所谓的“非矩形检测框检测器”。V2.1 要强调的是：

```text
矩形检测框 bbox 仍然是粗定位结果
bbox 进一步作为 prompt
在五线谱几何约束下生成符号级非矩形 mask
再把 mask 转成 polygon / skeleton
用于更精细的符号结构分析
```

因此，V2.1 的准确表述不是：

```text
矩形检测框 -> 非矩形检测框
```

而是：

```text
矩形候选框 bbox
-> staff-aware prompt refinement
-> SAM2 符号 mask
-> staff-line-aware filtering
-> polygon / skeleton / shape descriptor
-> fine-grained symbol structure analysis
```

这个模块可以称为：

```text
Staff-aware Box-to-Mask Refinement
```

中文名称：

```text
五线谱感知的框到掩膜细化模块
```

或者更强调约束：

```text
Geometry-Constrained Box-to-Mask Refinement
五线谱几何约束的框到掩膜细化模块
```

## 2. 为什么 V2 需要升级到 V2.1

在普通目标检测任务中，bbox 通常已经能表达目标的大致位置。例如检测人、车、猫时，矩形框虽然不精确，但目标和背景通常比较分离。

乐谱图像不同。一个乐谱符号经常和五线谱线、符杆、连桁、加线或邻近符号粘连。比如一个符头的 bbox 中可能同时包含：

```text
符头本体
穿过符头的五线谱线
连接符头的符杆
靠近符头的加线
邻近的升号、还原号或其他符号
背景空白
```

bbox 只能回答：

```text
这里大概有一个符号
```

但不能回答：

```text
哪些像素属于这个符号
哪些像素属于五线谱背景
哪些像素是粘连的其他符号
符号真实轮廓是什么
符杆中心线在哪里
连桁和哪些符杆连接
加线是否穿过某个符头
```

V2.1 的新增价值就在这里：它把 V2 的检测框进一步转成可分析的非矩形符号表示。

## 3. 核心思想

V2.1 的核心思想是：

```text
DEIM-D-FINE / V1 先给出 bbox
V1 staff geometry 给出 staff line、staff_space、pitch grid
根据 symbol class 调整 prompt
SAM2 在 prompt 内生成初始 mask
再用五线谱几何和类别先验过滤 mask
最后输出 polygon / skeleton / shape descriptor
```

也就是说，矩形框不是最终结构，而是一个 prompt。SAM2 也不是单独使用，而是被五线谱几何约束引导。

V2.1 的输入：

```text
original image or crop
symbol class
detector bbox or V1 proposal bbox
confidence
staff index
staff line y coordinates
staff_space
pitch grid
nearby symbols
```

V2.1 的输出：

```text
refined mask
polygon contour
skeleton centerline
rotated bbox or oriented shape descriptor
geometry_check_result
postprocess_decisions
```

## 4. V2.1 总流程

完整流程如下：

```text
Input score image
-> V1 staff geometry extraction
-> DEIM-D-FINE symbol bbox detection
-> optional V1 proposal fallback
-> class-specific prompt adjustment
-> SAM2 box / point prompt segmentation
-> staff-line-aware mask filtering
-> class-specific shape sanity check
-> mask to polygon / skeleton conversion
-> fine-grained symbol relation analysis
-> V2.1 JSON output
```

其中，V2.1 相比 V2 的新增部分是：

```text
class-specific prompt adjustment
staff-line-aware mask filtering
mask to polygon
mask to skeleton
shape and relation descriptors
```

## 5. 矩形框在 V2.1 中的角色

V2.1 不否定 bbox。bbox 仍然很重要：

- 用于快速定位候选符号
- 用于 COCO detection 训练和 AP 评估
- 用于 NMS、重复框合并和检索
- 用于给 SAM2 提供 box prompt
- 用于和 V1 / DeepScores 标注对齐

但 bbox 不再是最终的符号几何表达。最终用于结构分析的是：

```text
mask
polygon
skeleton
symbol-specific geometry attributes
```

推荐同时保留 bbox 和非矩形表示：

```json
{
  "bbox": [100, 52, 116, 64],
  "mask": {...},
  "polygon": [...],
  "skeleton": [...]
}
```

这样既保留检测任务的可评估性，也获得更精细的符号结构。

## 6. Class-specific Prompt Adjustment

不同符号不应该使用同一种 prompt 策略。V2.1 使用类别自适应 prompt。

| 类别 | prompt 策略 | 关键约束 |
|---|---|---|
| `filled_notehead` / `open_notehead` | bbox 略微扩大，允许 staff line 穿过 | mask 应接近椭圆，中心靠近 pitch grid |
| `stem` | 保持窄竖框，可加入竖向正点 | mask 主方向应近似竖直 |
| `beam` | 使用较长框，允许斜率 | mask 应为较粗长条，靠近 stem 端点 |
| `ledger_line` | 使用短横框，并关联 nearby notehead | 长度短于 staff line，靠近或穿过符头 |
| `barline` | 使用高竖框 | 应跨 staff 高度，近似竖直 |
| `slur_or_tie` | 使用长弧形区域，可加正负点 | skeleton 应为弧线，不应等同 staff line |
| `text_region` | 使用文本区域框 | mask 可以粗略，后续主要交给 OCR |
| `clef` / `rest` / `accidental` | bbox 适度扩张 | 保留整体轮廓，避免被 staff line 截断 |

prompt 调整示例：

```text
notehead:
  expand bbox by 10% to 20%
  positive point = bbox center
  negative points = bbox 外延伸的 staff line 区域

stem:
  keep narrow vertical bbox
  positive points = vertical center samples
  negative points = nearby horizontal staff line pixels

beam:
  expand along beam direction if slope can be estimated
  positive points = thick bar centerline
  negative points = nearby standard staff line y positions
```

## 7. Staff-line-aware Mask Filtering

五线谱线是 V2.1 的特殊难点。它既是背景，又可能和符号视觉上重叠。

不能简单执行：

```text
从所有 mask 中删除 staff_line_mask
```

因为这会把符头切断，也可能误删加线或连桁。V2.1 需要按类别处理。

| 类别 | staff line 处理策略 |
|---|---|
| notehead | 允许符头内部和 staff line 重叠，只删除轮廓外延伸线 |
| stem | staff line 大多是干扰，应删除横向长线成分 |
| beam | 不能按水平线盲删，要结合厚度、位置和与 stem 的连接 |
| ledger_line | 本身就是短横线，不能因为像谱线就删除 |
| barline | 保留跨 staff 的竖线，不受 staff line 删除影响 |
| slur_or_tie | staff line 通常是负样本，可用负点或过滤排除 |
| clef/rest/accidental | 允许局部 staff line 穿过，但限制 mask 不能延展成整条谱线 |

这种过滤的核心是：

```text
同样是黑色像素
在不同 symbol class 下语义不同
必须结合 staff geometry 和 class prior 判断
```

## 8. 类别级处理规则

### 8.1 Notehead

符头经常被五线谱线穿过，因此不能简单删除符头内部的谱线像素。

处理策略：

```text
1. 使用 DEIM bbox 或 V1 notehead proposal。
2. 根据 staff_space 扩框 10% 到 20%。
3. 用 bbox center 或椭圆中心作为 positive point。
4. SAM2 生成初始 mask。
5. 检查 mask 面积、宽高比、椭圆度和连通性。
6. 允许 mask 内部与 staff line 重叠。
7. 删除 mask 轮廓外沿 staff line 延伸出去的细长水平成分。
8. 输出 notehead polygon。
```

检查条件：

```text
area in expected range
aspect ratio close to notehead prior
center close to pitch grid
does not become a long horizontal staff line
keeps connected notehead body
```

### 8.2 Stem

符杆是竖直细长结构，skeleton 比 polygon 更重要。

处理策略：

```text
1. 使用 detector stem bbox 或 V1 vertical-run proposal。
2. SAM2 生成 mask。
3. 删除与 staff line y 坐标重合且横向过长的成分。
4. 保留竖向主连通域。
5. 提取 skeleton 作为 stem centerline。
6. 记录 stem top / bottom / direction。
```

检查条件：

```text
principal direction near vertical
width small relative to staff_space
height several staff_space units
skeleton length sufficient
```

### 8.3 Beam

连桁是横向或斜向粗条，不能简单用“水平线删除”处理。

处理策略：

```text
1. 使用 detector beam bbox 或 V1 beam corridor proposal。
2. 根据相邻 stem 端点调整 prompt。
3. SAM2 生成 mask。
4. 检查厚度是否大于 staff line。
5. 检查位置是否远离标准 staff line y。
6. 提取 polygon 和中心 skeleton。
7. 记录连接的 stem ids。
```

检查条件：

```text
elongated component
thickness > staff_line_thickness
connected or near multiple stems
not exactly one of the five staff lines
```

### 8.4 Ledger line

加线和五线谱线形状相似，但长度更短，并且必须靠近符头。

处理策略：

```text
1. 从 notehead 周围生成短横 proposal。
2. 或使用 detector ledger_line bbox。
3. SAM2 细化短横线 mask。
4. 判断长度、位置和 nearby notehead 关系。
5. 输出 polygon 或短线 skeleton。
```

检查条件：

```text
shorter than normal staff line segment
near or crossing a notehead
aligned with pitch grid extension
not part of full staff line
```

### 8.5 Slur or tie

连音线/延音线需要弧形 skeleton，而不是简单 bbox。

处理策略：

```text
1. 使用 detector bbox 或 V1 slur/tie proposal。
2. 在弧线候选上采 positive points。
3. 在 staff lines 上采 negative points。
4. SAM2 生成 mask。
5. 提取 skeleton。
6. 检查曲率、长度和端点位置。
```

检查条件：

```text
curved skeleton
not collinear with staff lines
endpoints near noteheads or note groups
reasonable arc height
```

## 9. Mask 到 Polygon

polygon 用于表达符号轮廓，适合：

- notehead outline
- rest / clef / accidental 的外形
- beam / barline / text region 的区域边界
- COCO segmentation 输出
- 可视化 overlay

转换流程：

```text
binary mask
-> remove tiny components
-> choose main component or class-specific components
-> contour extraction
-> contour simplification
-> polygon validity check
-> write polygon coordinates
```

推荐保存：

```json
{
  "mask": {
    "type": "polygon",
    "points": [[103, 53], [110, 52], [116, 57], [113, 63], [105, 64], [100, 59]]
  },
  "mask_source": "sam2_box_prompt",
  "mask_score": 0.88
}
```

## 10. Mask 到 Skeleton

skeleton 用于表达细长结构和连接关系，适合：

- stem centerline
- beam centerline
- ledger_line centerline
- slur_or_tie curve
- barline centerline

转换流程：

```text
binary mask
-> thinning / skeletonization
-> graph extraction
-> prune tiny branches
-> fit line or curve
-> record endpoints and centerline points
```

推荐保存：

```json
{
  "skeleton": {
    "type": "polyline",
    "points": [[118, 39], [118, 40], [118, 41], [118, 42], [118, 43]],
    "length": 37.2,
    "orientation": "vertical"
  }
}
```

skeleton 对关系分析尤其重要。例如：

```text
stem skeleton endpoint -> connected notehead
beam skeleton endpoint -> connected stem group
ledger_line skeleton -> nearby notehead pitch extension
slur skeleton endpoint -> notehead group relation
```

## 11. Iterative Prompt Refinement

V2.1 可以在推理阶段进行 prompt 迭代，不必更新模型参数。

流程：

```text
initial bbox
-> SAM2 mask
-> geometry check
-> if failed:
     adjust bbox
     add positive points
     add negative points
     rerun SAM2
-> stop when pass or max_iter reached
```

例子：

```text
stem 第一次 mask 吃进 staff line
-> 在 staff line 上加 negative points
-> 重新跑 SAM2
-> 竖直细长检查通过
```

另一个例子：

```text
notehead mask 被 staff line 切碎
-> 允许符头内部 staff overlap
-> 扩大 bbox 并加中心 positive point
-> 重新生成 mask
```

推荐记录每次迭代：

```json
{
  "postprocess_decisions": [
    "initial_mask_failed_horizontal_leak",
    "added_negative_points_on_staff_lines",
    "rerun_sam2_iter_2",
    "vertical_thin_component_pass"
  ]
}
```

## 12. V2.1 JSON 输出契约

V2.1 应在 V2 JSON 基础上增加 shape representation 字段。

示例：

```json
{
  "version": "v2.1",
  "input": "page.png",
  "staff_geometry_source": "v1_rules",
  "symbols": [
    {
      "id": "sym_00031",
      "class": "filled_notehead",
      "bbox": [100, 52, 116, 64],
      "prompt_box": [98, 50, 118, 66],
      "confidence": 0.91,
      "staff": 0,
      "pitch_step": 4,
      "detector": "deim_dfine",
      "mask": {
        "type": "polygon",
        "points": [[103, 53], [110, 52], [116, 57], [113, 63], [105, 64], [100, 59]]
      },
      "skeleton": null,
      "shape": {
        "area": 92,
        "aspect_ratio": 1.33,
        "staff_overlap_policy": "allowed_inside_symbol"
      },
      "mask_source": "sam2_box_prompt",
      "geometry_check": "notehead_ellipse_check_pass",
      "postprocess_decisions": [
        "bbox_expanded_15_percent",
        "staff_overlap_allowed",
        "external_staff_line_removed"
      ]
    }
  ]
}
```

stem 示例：

```json
{
  "version": "v2.1",
  "symbols": [
    {
      "id": "sym_00044",
      "class": "stem",
      "bbox": [115, 38, 121, 75],
      "confidence": 0.88,
      "staff": 0,
      "mask": {
        "type": "rle",
        "id": "stem_mask_00044"
      },
      "skeleton": {
        "type": "polyline",
        "points": [[118, 39], [118, 40], [118, 41], [118, 42], [118, 43]],
        "orientation": "vertical"
      },
      "geometry_check": "vertical_thin_component_pass"
    }
  ]
}
```

## 13. 结构分析用途

V2.1 的 mask / polygon / skeleton 不是只为了画得更好看，而是为了支持更细的符号结构分析。

可以支持：

- 判断 stem 连接哪个 notehead
- 判断 beam 连接哪些 stems
- 判断 ledger_line 是否属于某个 notehead
- 判断 slur_or_tie 的端点落在哪两个 note group 附近
- 判断 notehead 内部 staff line overlap 是否正常
- 区分 beam 和 staff line
- 区分 ledger_line 和 staff line
- 生成更可靠的 relation graph

关系图示例：

```text
notehead polygon center
-> nearest stem skeleton endpoint
-> beam skeleton / polygon contact
-> note group
-> rhythm candidate
```

V2.1 因此是从“符号检测”走向“符号结构理解”的一步。

## 14. 评估设计

V2.1 除了继承 V2 的 detection 和 symbol metrics，还应增加 shape-level 指标。

继承 V2 指标：

- AP / AP50 / AP75
- CER
- SER
- LER
- per-class precision / recall / F1

新增 mask 指标：

- mask IoU
- polygon IoU
- contour distance
- area error

新增 skeleton 指标：

- centerline distance
- endpoint localization error
- orientation error
- branch count error

新增结构关系指标：

- notehead-stem attachment accuracy
- stem-beam attachment accuracy
- ledger_line-notehead association accuracy
- slur/tie endpoint association accuracy

如果短期没有人工 mask 真值，可以先使用：

```text
geometry sanity pass rate
manual visual review score
relation consistency score
```

作为弱评估。

## 15. 当前工程改造建议

当前 V2 已有：

```text
tools/refine_masks_sam2.py
tools/infer_neural_symbols_v2.py
tools/reconstruct_v2_symbols.py
tools/evaluate_v1_v2_metrics.py
```

V2.1 建议新增或扩展：

| 文件 | 建议作用 |
|---|---|
| `tools/refine_masks_sam2.py` | 增加 class-specific prompt 和 staff-line-aware filtering |
| `tools/omr_v2_common.py` | 增加 polygon、skeleton、shape descriptor 的 JSON 工具 |
| `tools/extract_symbol_shapes_v2_1.py` | 从 SAM2 mask 提取 polygon / skeleton |
| `tools/evaluate_v2_1_shape_metrics.py` | 评估 mask / skeleton / relation 指标 |
| `tools/visualize_v2_1_shapes.py` | 单独画 polygon、skeleton 和关系图 |

推荐产物：

```text
outputs/<run>/symbols/symbols_v2_1_shapes.json
outputs/<run>/shapes/masks/
outputs/<run>/shapes/polygons.json
outputs/<run>/shapes/skeletons.json
outputs/<run>/visuals/v2_1_polygon_overlay.png
outputs/<run>/visuals/v2_1_skeleton_overlay.png
outputs/<run>/metrics/evaluation_v2_1_shapes.json
```

## 16. 风险与控制

| 风险 | 控制 |
|---|---|
| SAM2 把 staff line 吃进 mask | 使用 staff-line-aware negative points 和类别过滤 |
| 直接删 staff line 导致 notehead 断裂 | notehead 内部允许 staff overlap，只删外部延伸线 |
| beam 被误删成 staff line | 按厚度、位置、斜率和 stem 连接关系判断 |
| ledger_line 被当成 staff line 删除 | ledger_line 以 notehead 邻近关系作为核心约束 |
| skeleton 分叉太多 | skeleton pruning 和 class-specific branch constraints |
| polygon 太复杂 | contour simplification，保存简化 polygon 和原始 mask |
| 没有 mask 真值 | 先用几何通过率、人工 visual review 和关系一致性评估 |

## 17. 论文式表述

V2.1 可以这样写成方法贡献：

```text
We do not replace rectangular detection boxes with non-rectangular detection boxes.
Instead, we use rectangular boxes as prompts and refine them into staff-aware symbol masks.
The masks are constrained by staff-line geometry, staff spacing, pitch grids, and class-specific shape priors.
They are further converted into polygons and skeletons for fine-grained symbol structure analysis.
```

中文表述：

```text
本方法并不是把矩形检测框改造成非矩形检测框，而是以矩形框作为提示，在五线谱几何约束下生成符号级非矩形 mask，并进一步转化为 polygon 和 skeleton，用于更精细的符号结构分析。
```

这使 V2.1 不只是“用了 SAM2”，而是把 SAM2 变成了一个与 OMR 场景强绑定的几何引导模块。

## 18. 一句话总结

V2.1 的核心是：

```text
DEIM-D-FINE bbox
+ V1 staff geometry
+ class-specific prompt
-> SAM2 mask
-> staff-aware filtering
-> polygon / skeleton
-> fine-grained symbol structure analysis
```

也就是：

```text
bbox 用于定位
mask 用于边界
polygon 用于轮廓
skeleton 用于连接关系
staff geometry 用于约束和纠错
```
