# RIGOR-OMR 当前实现技术细节核验（代码事实版）

> 核验日期：2026-07-22  
> 目的：回答方法正文与补充材料所需的实现细节。以下内容按当前仓库、冻结命令日志、配置和可执行字节码的真实行为整理；没有实现的模块直接写“未使用”。  
> 状态约定：**主结果**＝冻结主运行中实际启用；**部分运行**＝仅特定数据集或变体启用；**实验实现**＝仓库中存在但不能写成主结果默认设置；**未使用**＝当前主流程没有该机制。

## 最需要先改正的表述

1. 主检测并不是“full-page 与 staff-crop 两路检测”。实际是整页的 `full / tile / auto` 推理；`tile` 是页面滑窗，不是谱表裁剪。
2. 冻结主运行的 SAM2 提示策略是 `box_only`：使用原始融合框，不输入正点或负点。正负点规则属于提示消融，不能画成主结果默认机制。
3. 通用的符号 crop classifier 不是所有主结果的默认模块。可确认进入冻结运行的是独立的四类谱号 ROI 分类器；另有四谱号 crop classifier 变体，但它也不是全类别分类器。
4. 实际关系图只有四类显式边；变音记号和附点通过事件装配阶段的最近邻规则绑定，并不在关系图中建边。
5. 当前关系解析不是全局图优化、二分图匹配或学习式统一打分，而是按固定顺序执行的局部启发式规则。
6. SAM2 会影响掩码描述、细长符号骨架和部分关系距离，但现有端到端证据未证明其稳定提高 Event-F1。因此正文应把它写成辅助几何模块，而不是核心收益来源。

---

# 0. 总体设置

## 0.1 主结果固定流水线

冻结主运行对应的模块化流水线是：

```text
图像/PDF
→ V1 谱表几何与规则候选
→ DEIM-D-FINE 符号检测
→ detector/rule 融合
→ 可选谱号 ROI 补充
→ SAM2.1 掩码细化
→ V2.1 掩码描述与细长符号骨架
→ 四类启发式关系边
→ note/rest 事件装配
→ 过识别剪枝
→ semantic JSON
→ draft MusicXML / 数据集代理序列
```

- **SAM2：主结果中开启。**冻结日志使用 `--prompt-strategy box_only`。但从功能定位和现有 Event-F1 消融看，它应写成辅助模块。
- **通用 crop classifier：不是统一默认开启。**Debussy/GrandStaff 的可核验冻结日志使用独立谱号 ROI 分类器，而非全符号 crop classifier。仓库另有四谱号 crop classifier 变体，仅在特定 crop-fusion 运行启用。
- **检测后的 detector/rule 融合：开启。**可核验命令为 `--merge-rule-classes all --rule-fallback-iou 0.5 --fusion-preference detector`。
- **过识别剪枝：开启。**代表性冻结运行使用 `balanced` 预设。

## 0.2 Polish、Debussy、GrandStaff、OLiMPiC 是否完全同参

**不是完全相同。**共享的是主干检测器、58 类表和大部分几何/关系代码；差异包含运行规模、页面推理方式、最大检测数、谱号补充模块、SAM2 提示消融设置，以及输出代理格式。

| 项目 | 可核验实际设置/差异 | 性质 |
| --- | --- | --- |
| DEIM 主干 | 同一 OMR checkpoint，输入尺寸 640，常用置信度 0.05、NMS IoU 0.5 | 共享主干 |
| GrandStaff 冻结示例 | `tile`，tile 1280、overlap 0.25、max detections 2000；谱号 ROI；SAM2 `box_only`；balanced prune | 运行配置 |
| Debussy 代表运行 | `auto`，max detections 1000；谱号 ROI；SAM2 `box_only`；balanced prune | 运行配置 |
| Polish | 使用 Polish/eKern 规范化序列化；仓库存在 target-train calibration source 产物 | 输出适配并含校准差异 |
| OLiMPiC | 使用 LMX/官方风格评测适配；提示实验中存在 `box_positive` 等变体；仓库存在 OLiMPiC-selected threshold 产物 | 输出适配并含校准差异 |

不能把这些运行统称为“完全零样本、完全同参”。其中 eKern/LMX 的差异属于输出适配；目标域校准源、固定阈值选择和部分推理参数差异属于数据集相关设置，必须在结果表或补充材料逐项披露。

## 0.3 阈值来源

目前证据是**混合来源，不能统一声称全部来自源域验证集**：

- DEIM 在 DeepScores 风格的 OMR 训练/验证拆分上训练和选择 checkpoint。
- 检测主阈值通常为 0.05，逐类 NMS IoU 为 0.5。
- 仓库存在 Polish train-83 与 OLiMPiC train-100 calibration-source 产物，也存在明确写为 “OLiMPiC-selected thresholds” 的结果表。
- 未发现主结果使用目标**测试标签**直接调参的证据；但目标域训练/校准数据确实在部分配置中使用。
- 因而论文应按每个结果报告阈值来源。若某个表要标为 zero-shot，必须先锁定对应冻结配置并确认它没有使用目标域校准产物。

---

# 1. Staff Geometry Estimation

## 1.1 图像预处理

- 输入转为 RGB，随后生成灰度图。
- PDF 默认只渲染第一页，默认 220 DPI。
- 二值化不是纯 Otsu：

  ```text
  threshold = min(212, max(170, otsu(gray) + 35))
  foreground = gray < threshold
  ```

- 对二值前景做 8 邻域连通域去噪，删除面积小于 3 像素的连通域。
- “反色”：没有单独反色步骤；仅把深色像素定义为前景布尔值。
- 形态学开闭、膨胀/腐蚀：**未使用**。
- 倾斜校正、Hough deskew：**未使用**。

## 1.2 谱线检测

当前使用**二值前景的水平投影**，不是 Hough line，也不是学习式谱线模型。

实际步骤如下：

1. 对每一行计算前景像素数 `projection[y] = mask[y, :].sum()`。
2. 计算投影的 99.5 百分位数 `strong`。
3. 行候选阈值为：

   ```text
   row_threshold = max(0.22 × image_width, 0.62 × strong)
   ```

4. 把相邻或间隔不超过 1 行的候选行合并为一个行带。
5. 候选行必须包含长度不小于 `0.35 × image_width` 的连续水平前景段。
6. 每条谱线的 y 坐标是该行带内按水平投影加权的均值，不是中位数。

没有 Hough、线段拟合或曲线跟踪。断裂、弯曲和倾斜仅能在其水平投影仍形成足够强峰值时被容忍；基本算法没有显式修复机制。困难运行中出现过 spacing fallback/template recovery，但不能把它写成上述基础检测器的默认步骤。

## 1.3 五条线如何组成 staff

1. 收集相邻候选谱线的 y 间隔，只保留 `[6, 80]` 像素的间隔。
2. 对四舍五入后的间隔取众数，得到页面级主间距；保留与众数相差不超过 2 像素的间隔，再取中位数作为全局 staff spacing。
3. 从上到下滑动检查连续五条候选线。
4. 五线组的四个相邻间隔都必须满足：

   ```text
   abs(gap - global_spacing) <= max(2.4 px, 0.22 × global_spacing)
   ```

5. 接受后索引前进 5；不接受则前进 1。

因此分组是**贪心滑窗**，不是动态规划，也不是枚举全部五线组合后全局筛选。多余水平线通过滑窗失败后继续尝试处理；断裂、弯曲和倾斜没有专门的组合修复规则。

## 1.4 staff spacing

题目中的

```text
s_k = median_j(line[k,j+1] - line[k,j])
```

**不是基础谱线检测阶段的真实计算。**该阶段先估计一个页面级全局 spacing：相邻候选线间隔的众数筛选后再取中位数。每个已接受 staff 保存该全局 spacing。后续若对象中已有单独的 `spacing` 字段则直接使用。

## 1.5 staff region

正式的扩展区域

```text
[top_line - alpha_top × s, bottom_line + alpha_bottom × s]
```

**未使用**，因此没有统一的 `alpha_top` 和 `alpha_bottom`。staff 保存顶线、底线和五条线坐标；符号分配使用中心点到谱表顶线、底线和中线的最小垂直距离，不用扩展区域重叠率。

## 1.6 pitch grid

V2.1 事件装配中的真实规则是：

```text
cx, cy = notehead bbox center
b       = bottom staff line
step    = round((b - cy) / (0.5 × s))
y(step) = b - step × 0.5 × s
```

- 参考点：**底线**。
- 每个离散步长：半个 staff spacing，即线与间交替。
- 图像 y 向下增加，因此 `step > 0` 表示音符在底线上方，`step < 0` 表示在底线下方。
- 使用 Python `round`，精确 `.5` 情况采用 ties-to-even，不是始终远离零的传统四舍五入。
- 若符号属性已经提供 `pitch_step`，则优先使用该属性。
- notehead 的位置来自 bbox 中心，不是 SAM mask 质心。

## 1.7 clef 与 pitch

- 检测类别支持 treble、bass、C-alto、C-tenor 四种谱号。
- 神经主干由检测器识别；可选谱号 ROI 模型也输出这四类。
- 传统 V1 弱标签路径不做视觉谱号分类，而是按 staff 奇偶交替赋 treble/bass。
- 绝对音高导出实际上只完整支持 treble 和 bass：底线的 diatonic 基准分别为 30（E4）和 18（G2）。
- 未检测到谱号时默认 treble；未知谱号和 C clef 在当前导出器中也回退到 treble，因此不能声称已正确支持 C clef 音高映射。
- staff 内谱号变化及其作用域：**未使用**。
- ledger line 不参与网格步数计算；超出五线谱范围时仍直接外推 `step`。
- 临时升、降、还原记号通过最近左侧匹配改变半音。
- 调号、同小节临时变音持续规则、octave clef：**未使用**。

---

# 2. Symbol Detection and Staff-Aware Prompt Generation

## 2.1 DEIM-D-FINE 配置

| 项目 | 实际配置 |
| --- | --- |
| 模型 | DEIM + D-FINE transformer decoder |
| Backbone | HGNetv2-N |
| Encoder | HybridEncoder |
| 特征层数 | 3 |
| Decoder layers | 6 |
| Queries | 300 |
| 输入尺寸 | 640 × 640 |
| 类别数 | 58 |
| 配置文件 | `.local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded_clean_5070ti.yml` |
| checkpoint | `outputs/v2_deim_runs/v2_symbol_all_expanded_clean_5070ti/checkpoint0399.pth` |
| 训练数据 | `ds2_dense`，DeepScores 风格 COCO；train 1362 页/872,416 instances，val 352 页/239,796 instances |
| OMR 微调 | 是。该 checkpoint 是在 OMR 符号数据上训练的模型，不是直接使用通用 COCO 权重 |

完整 58 类为：

```text
notehead_black, notehead_half, notehead_whole,
stem, beam, ledger_line, augmentation_dot, repeat_dot,
flag_8th, flag_16th, flag_32nd, flag_64th,
treble_clef, bass_clef, c_clef_alto, c_clef_tenor,
accidental_sharp, accidental_flat, accidental_natural,
rest_whole, rest_half, rest_quarter, rest_8th, rest_16th,
slur_or_tie,
time_sig_0, time_sig_1, time_sig_2, time_sig_3, time_sig_4,
time_sig_5, time_sig_6, time_sig_7, time_sig_8, time_sig_9,
time_sig_common, time_sig_cut_common,
dynamic_p, dynamic_f, dynamic_m, dynamic_s, dynamic_z,
dynamic_crescendo_hairpin, dynamic_diminuendo_hairpin,
artic_staccato, artic_accent, artic_tenuto, artic_marcato,
fermata, tuplet_3, tuplet_6, fingering,
pedal_mark, pedal_up, ornament_mordent, ornament_turn,
arpeggiato, brace
```

类别表和计数见 `outputs/v2_deim_ds_all_expanded_clean/summary.json` 及对应 COCO annotation。

## 2.2 full-page 与 staff-crop 检测

**主结果没有运行 full-page + staff-crop 两种尺度。staff-crop 检测未使用。**

实际检测入口支持：

- `full`：整页送入模型的 resize/letterbox 流程；模型输入 640 × 640。
- `tile`：对页面做 1280 × 1280 左右的滑窗 tile，默认 overlap 0.25；每个 tile 再送入 640 模型。
- `auto`：当页面足够宽且长宽比达到约 2.2 时选择 tile，否则 full。

tile 是页面坐标中的滑窗，不根据 staff region 裁剪，也不包含“上下相邻 staff”的定义。tile 预测框通过加回 tile 左上角偏移映射到原图，随后统一合并。

仓库中存在 full/crop fusion 实验工具，但不能写成当前主结果的默认模块。

## 2.3 检测过滤与合并

- confidence threshold：主运行通常为 **0.05**。
- NMS IoU threshold：**0.5**。
- NMS：**逐类别 NMS**。
- NMS 后按置信度降序排序。
- 最大保留数：不同运行为 1000 或 2000；GrandStaff 冻结日志示例为 2000。
- tile 的重叠预测使用相同的逐类 NMS 合并。
- weighted box fusion：**未使用**。
- 因为 NMS 逐类执行，同一图像位置可能保留多个不同类别的重叠框。
- detector 与规则候选的融合另有 IoU 0.5 的 fallback/conflict 逻辑，偏好 detector；这不是 WBF。

## 2.4 staff assignment

每个符号取 bbox 中心 `(cx, cy)`，选择使以下值最小的 staff：

```text
min(|cy - top_line|,
    |cy - bottom_line|,
    |cy - (top_line + bottom_line)/2|)
```

- 不使用 box 与 staff region 的最大重叠。
- 不使用二维中心距离，实际只比较垂直方向。
- 没有拒绝阈值；只要存在 staff，就总会分给最近的一个 staff。

## 2.5 crop classifier

### 主结果结论

**全类别 crop classifier 未作为统一主模块使用。**可核验冻结日志实际启用的是谱号 ROI 分类器：

```text
checkpoint: outputs/v2_1_clef_roi_classifier_scan_pseudo/symbol_crop_classifier.pt
classes: treble, bass, C-alto, C-tenor
threshold: 0.4
min margin: 0.03
ROI mode: fixed
```

另一个特定 crop-fusion 变体使用：

```text
architecture: SmallCropCNN
input: 128 × 128
pad ratio: 1.2
pad pixels: 12
classes: treble, bass, C-alto, C-tenor
training set: same DeepScores-style COCO train/val crops
train/val crop counts: 5629 / 1422
classifier confidence threshold: 0.55
family-mass threshold: 0.50
```

它只有一个谱号家族，满足阈值时只会在四种谱号间重标，不会重标 notehead、stem、beam 等其他符号。因此正文不应描述为“family-constrained general symbol crop classifier”；若主结果表没有使用对应 crop-fusion 冻结配置，应从主方法删掉，仅放补充材料的变体说明。

## 2.6 哪些类别进入 SAM2

冻结主运行 `box_only` 会遍历融合符号 JSON 中的**全部符号实例**，不是只选 notehead/stem/beam。没有主类白名单。

## 2.7 box prompt

- 冻结主运行 `box_only`：使用符号的原始融合 bbox，不扩边。
- 没有固定像素、框比例或 staff spacing 扩边。
- 仓库的 `method_aligned` 实验策略实现了按类别的扩框和局部 crop，但它不是冻结主结果设置。

`method_aligned` 的实验扩框为：

| 类别 | x 扩边 | y 扩边 |
| --- | ---: | ---: |
| notehead | bbox width × 0.18 | bbox height × 0.18 |
| stem | 0.25s | 0.08s |
| beam | 0.20s | 0.30s |
| ledger line | 0.30s | 0.20s |
| barline | 0.20s | 0.10s |
| slur/tie | 0.35s | 0.35s |
| accidental/clef/rest | bbox width × 0.12 | bbox height × 0.12 |
| 其他 | 0.08s | 0.08s |

## 2.8 positive points

### 冻结主结果

`box_only` **没有 positive points**。

### 提示消融中的 `box_positive` / `box_pos_neg_staff`

| 类别族 | 正点规则 |
| --- | --- |
| stem、barline | bbox 竖直中心线上 3 点，y 分数为 0.28、0.50、0.72 |
| beam、ledger line、slur/tie | bbox 水平中心线上 3 点，x 分数为 0.25、0.50、0.75 |
| notehead、accidental、rest、clef 和其他 | bbox 中心 1 点 |

这些点由 bbox 几何产生，不使用检测置信度峰值或 mask 峰值。

## 2.9 negative points

### 冻结主结果

`box_only` **没有 negative points**。因此若主文方法图画了 staff-line negatives，必须标成提示消融/实验策略，不能标成主结果默认提示。

### `box_pos_neg_staff` 消融规则

只考虑位于 `[top_line - 0.25s, bottom_line + 0.25s]` 内、且与目标框有关系的 staff lines；点至少离图像边界 1.5 px，并在最后做取整、裁剪和去重。

| 类别族 | 正点 | 负点数量与位置 | box 扩张 |
| --- | --- | --- | --- |
| notehead（实心/空心） | bbox 中心 | 0 | 无 |
| stem | 中心线上 3 正点 | 对每条穿过目标垂直范围的谱线，在 `x0 - 0.55s` 与 `x1 + 0.55s` 各放 1 点 | 无 |
| beam | 水平中心线上 3 正点 | 对每条穿过 bbox 的谱线，在 bbox 内 x=0.25/0.50/0.75 处各 1 点 | 无 |
| ledger line | 水平中心线上 3 正点 | 0 | 无 |
| barline | 竖直中心线上 3 正点 | 0 | 无 |
| slur/tie | 水平中心线上 3 正点 | 对每条穿过 bbox 的谱线，在 bbox 内 x=0.25/0.50/0.75 处各 1 点 | 无 |
| accidental/rest/clef/其他 | bbox 中心 | 对每条穿过 bbox 的谱线，在 bbox 内 x=0.25/0.50/0.75 处各 1 点 | 无 |

负点总数随穿过目标区域的谱线数变化，不是固定数量。特别地，当前代码**不会给 notehead 放 staff-line negative**。

---

# 3. Prompt-Guided Mask Refinement

## 3.1 SAM2 版本

| 项目 | 实际设置 |
| --- | --- |
| 版本 | SAM2.1 |
| 规模 | Hiera-Tiny |
| checkpoint | `outputs/models/sam2/sam2.1_hiera_tiny.pt` |
| config | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| 是否冻结 | 是，仅推理 |
| OMR 训练/微调 | **未使用** |

## 3.2 输入方式

### 冻结主结果

- 输入整页图像，predictor 对整页执行一次 `set_image`。
- 每个符号在同一整页坐标系下输入 bbox。
- 不裁局部 crop，因此没有 crop resize 和坐标映射问题。

### `method_aligned` 实验实现

- 使用局部 crop。
- crop context margin 为：

  ```text
  context_scale × max(s, 0.25 × prompt_dimension)
  ```

  默认 `context_scale=1.0`。
- 不再显式 resize 到固定尺寸；predictor 接收原生 crop。
- box/point 坐标通过减去 crop 左上角映射到局部坐标。

该策略是实验实现，不是冻结主结果。

## 3.3 mask 输出选择

- 默认 `multimask_output=False`，因此主结果只请求单个 mask。
- 若实验中开启多 mask，则按 SAM 返回的 predicted score 最大值选择，即 `argmax(scores)`。
- stability score、与检测框重叠最大、自定义综合分：**未使用**。

## 3.4 mask 后处理

SAM refiner 本身把 SAM 输出转为布尔 mask，没有额外的概率阈值、填洞、最大连通域或正负点覆盖验证。

V2.1 shape extractor 随后执行：

1. 从文件读取 mask，非零像素视为前景。
2. 将 mask 限制到对应的类专用 prompt box。
3. 做 8 邻域连通域分析。
4. 一般类别删除面积小于 `max(4, floor(0.015s²))` 的连通域。
5. stem/barline/ledger 等细长类别删除面积小于 `max(3, floor(0.01s²))` 的连通域。
6. 保留所有达到面积阈值且与 prompt box 相交的连通域，不是只保留最大域。
7. 若过滤后为空，可回退到 prompt mask。

填洞、要求覆盖正点、显式排除负点：**未使用**。

## 3.5 失败回退

- 某实例没有可用 mask 时，用该实例 bbox 生成矩形 mask。
- 连通域过滤后为空时可回退到 prompt mask。
- 不会仅因 SAM mask 缺失而丢弃实例。
- legacy 整页主路径没有完善的逐实例 OOM 降级；`method_aligned` 路径会对 OOM 重试，耗尽后仍抛出异常。
- “失败后使用原始二值 crop”：**未使用**。

## 3.6 SAM2 的实际用途

mask 影响：

- 面积、mask bbox、填充率、长宽比、质心、PCA 方向和 staff-overlap 描述；
- stem、beam、barline、ledger line、slur/tie 的 Zhang-Suen 骨架；
- stem 的端点，以及 stem-note、stem-beam、slur-note 等部分关系距离；
- 部分细长符号的方向和可视化轮廓。

mask **不影响**：

- notehead 的最终中心，仍使用 bbox center；
- clef 的绝对音高映射规则；
- 关系解析的全局最优化，因为该模块不存在。

骨架 crop 面积大于 180,000 像素或骨架点少于 2 时，使用 bbox 中心线回退。现有端到端因子实验没有验证 SAM2 对 Event-F1 的稳定增益，因此主文应称其为“辅助的几何细化”，不应把它写成主要性能来源。

---

# 4. Relation-Aware Event Assembly and Structured Reconstruction

## 4.1 几何基元

### notehead

- 中心：bbox center，不是 mask centroid。
- 实心/空心：由检测类别直接给出；`notehead_black` 为 filled，`notehead_half/whole` 为 open。
- V1 规则候选存在密度启发式，但神经主结果优先使用 detector class。

### stem

- 从 SAM/bbox mask 做 Zhang-Suen thinning。
- 对骨架点做 PCA，沿主轴排序得到 polyline 和两端点。
- Hough line：**未使用**。
- 显式小分支剪枝：**未使用**；只有连通域面积过滤和 thinning。
- 骨架失败时用 bbox 竖直中心线。

### beam

- 方向来自 mask 骨架/PCA 或 bbox 回退。
- beam 数量优先读取检测属性 `beam_count / beam_level / parallel_beams`，限制到 1–5。
- 没有属性时按 beam bbox 高度与 spacing 比值：

  ```text
  h/s >= 3.40 → 4
  h/s >= 2.70 → 3
  h/s >= 2.15 → 2
  otherwise   → 1
  ```

- 一条粗 beam 不是通过 mask 中显式分离平行条带来计数，主要是 bbox 高度启发式。

### ledger line

- 使用 mask/bbox 的细长骨架描述。
- 通过显式 `ledger_line_notehead_attachment` 边绑定 notehead。

### 其他符号

| 类别 | 是否进入后续 | 实际用途 |
| --- | --- | --- |
| accidental | 是 | 同 staff、note 左侧最近邻；改变半音 |
| augmentation dot | 是 | note/rest 右侧最近未使用 dot；改变时值 |
| repeat dot | 否 | 不作为 augmentation dot |
| flag | 检测但事件时值中未使用 | **未使用**于 duration |
| rest | 是 | 由 detector rest class 直接产生 rest event |
| clef | 是 | staff 级 treble/bass pitch 映射 |
| barline | 是 | 按 staff 划分 measure |
| slur/tie | 部分 | 单一类别，绑定两个端点并导出 slur 标记；不区分 tie |
| tuplet | 检测但不解码 | **未使用**于 duration/MusicXML tuplet |
| double/final/repeat barline | 未单独解析 | **未使用** |

## 4.2 代码中实际存在的关系边

只有以下四类显式边：

1. `notehead_stem_attachment`
2. `beam_stem_group`
3. `ledger_line_notehead_attachment`
4. `slur_tie_notehead_endpoints`

下列关系边**未使用/不存在**：

- accidental-notehead edge；
- dot-notehead/rest edge；
- flag-stem edge；
- barline-staff edge；
- clef-staff edge。

Event-F1 只评估最终 pitch-duration 多重集。`notehead_stem_attachment` 和 `beam_stem_group` 会通过时值影响 Event-F1；ledger 边通常不改变当前 pitch 外推；slur/tie 边只影响结构化/MusicXML 输出，不进入 Event-F1。accidental 和 dot 虽不建图边，但其装配结果会改变 pitch/duration，因此会进入 Event-F1。

## 4.3 候选边条件

### 4.3.1 notehead–stem

对同一 staff 的每个 stem：

```text
dist = min(distance(stem_endpoint_1, note_bbox),
           distance(stem_endpoint_2, note_bbox))
x_dist = abs(stem_center_x - note_center_x)
vertical_cover = note_center_y in [stem_y0 - 0.45s, stem_y1 + 0.45s]
side_touch = min(abs(note_x0 - stem_center_x),
                 abs(note_x1 - stem_center_x)) <= 0.95s
threshold = max(8 px, 1.35s)
```

若满足以下任一条件则成为候选：

```text
dist <= threshold
or (vertical_cover and x_dist <= 0.95s)
or (vertical_cover and side_touch)
```

候选按 `(dist, x_dist, note_center_y)` 升序；每个 stem 最多保留 6 个 notehead。边分数：

```text
score = 1 - min(1, dist / threshold)
```

- 一个 stem 可连多个 notehead。
- 原始关系图中一个 notehead 可连多个 stem，因为约束只在每个 stem 的局部候选列表内。
- 事件装配时每个 notehead 最终只选择一个最高分 stem。

### 4.3.2 stem–beam

对同 staff 的 beam：

```text
expanded_beam_box:
  x padding = 0.35s
  y padding = 0.55s

dist = min(distance(stem_endpoint_1, expanded_beam_box),
           distance(stem_endpoint_2, expanded_beam_box))
threshold = max(5 px, 0.45s)
```

若 `dist <= threshold` 或 stem bbox 与 expanded beam bbox 相交，则 stem 是候选。每个 beam 至少绑定 2 个 stem 才写入 `beam_stem_group`；所有接受的 stem 按中心 x 排序，没有数量上限。边分数：

```text
score = 1 - min(1, mean(endpoint_distances) / max(1, 0.65s))
```

- 同时检查两个端点，取更近的一个，不固定只用上端或下端。
- 显式方向/斜率一致性约束：**未使用**。

### 4.3.3 ledger line–notehead

同 staff 内，ledger line 与 notehead 需满足：

```text
horizontal overlap after padding note bbox by 0.75s on each side >= 0
abs(ledger_center_y - note_center_y) <= 1.65s
```

候选按中心欧氏距离排序；每条 ledger line 只绑定最近的一个 notehead。一个 notehead 可以绑定多条 ledger line。分数中的归一化阈值为 `max(8 px, 1.85s)`，但该阈值不额外参与候选拒绝。

### 4.3.4 accidental–notehead

这不是图边，而是每个 note 事件的局部最近邻：

```text
dx = note_center_x - accidental_center_x
0 <= dx <= 3.2s
abs(dy) <= 1.25s
```

在满足条件的同 staff accidental 中取欧氏距离最小者。没有一对一约束，同一个 accidental 可被多个 note 复用。

### 4.3.5 dot–note/rest

只搜索 `augmentation_dot`，不把 `artic_staccato` 或 `repeat_dot` 当附点：

```text
dx = dot_center_x - event_center_x
0.05s <= dx <= 1.85s
abs(dy) <= 0.75s
```

取距离最近且尚未使用的 dot。一个 dot 最多使用一次，每个当前事件最多绑定一个 dot。

### 4.3.6 slur/tie–notehead

- 不区分 slur 与 tie，检测类别只有 `slur_or_tie`。
- 对骨架的两个端点分别搜索同 staff 最近 note bbox。
- 端点到 note bbox 距离必须不超过 `3.0s`。
- 去重后必须得到两个不同 notehead，保留前两个目标。
- 分数：

  ```text
  score = 1 - min(1, mean(endpoint_distances) / max(1, 3s))
  ```

- 没有“tie 两端音高必须相同”的约束。
- 不进入 Event-F1，只用于 slur 类结构输出。

## 4.4 多候选冲突解决

实际实现是**固定执行顺序 + 局部启发式选择**，不是全局优化。

### 关系构建顺序

```text
1. notehead–stem
2. beam–stem
3. ledger–notehead
4. slur/tie endpoints–notehead
```

### 事件装配顺序

```text
1. 对每个 notehead 选择 score 最高的 stem
2. 通过已选 stem 查找 beam group，得到 beam_count
3. 在 note 左侧选择最近 accidental
4. 在 note/rest 右侧选择最近的尚未使用 augmentation dot
```

没有统一的加权总公式。每类边有自己的几何分数。

- notehead 最终 stem：按 `score` 最大选择；相同 score 时 Python `max` 保留先出现的关系。原始关系顺序来自确定性的候选排序。
- detection confidence 不参与关系边候选分数。
- note event 的最终 confidence 是 notehead detector confidence 与所选 stem relation score 的平均值。
- 跨 staff 关系：**未使用**。staff 已知时要求相同 staff。
- 符号复用：stem 可连接多个 note；accidental 可复用；beam group 可含多个 stem；dot 不可复用；note 可在原始图中有多个 stem 边，但最终事件只保留一个 stem。
- 二分图匹配、Hungarian、整数规划、消息传递、全局图神经网络：**未使用**。

## 4.5 音高推断

- notehead 位置：bbox center。
- 网格量化：`round((bottom_line - cy)/(0.5s))`，使用 Python ties-to-even。
- 超出五线谱：直接外推网格，不要求先检测到 ledger line。
- treble：底线对应 E4；bass：底线对应 G2。
- 未知/C clef 当前回退到 treble；缺失 pitch 数据时 MusicXML 回退 C4。
- sharp/flat/natural 分别施加 `+1/-1/0` 半音 alter。
- key signature：**未使用**。
- 小节内临时变音持续与还原状态机：**未使用**。
- octave clef：**未使用**。

## 4.6 时值推断

### note 时值规则表

| notehead 类型 | stem | beam/flag 数 | 输出时值 |
| --- | ---: | ---: | --- |
| open | 否 | 0 | whole |
| open | 是 | 0 | half |
| filled | 是 | 0 | quarter |
| filled | 任意 | 1 beam | eighth |
| filled | 任意 | 2 beams | sixteenth |
| filled | 任意 | 3 beams | 32nd |
| filled | 任意 | ≥4 beams | 64th |
| filled | 否 | 0 | `notehead_only_unknown`，导出时回退 quarter |
| 未知 | 任意 | 任意 | quarter 回退 |

补充说明：

- beam 数优先来自 beam relation 数量和 beam 属性提示的最大值；几何回退按 beam 高度/spacing 估计。
- duration 映射最多到 4 beams，即 64th；即便属性出现 5，也在时值函数中截到 4。
- flag 检测类别存在，但 flag-stem 关系和 flag 计数：**未使用**。
- hollow/filled 由 detector notehead class 决定。
- augmentation dot：当前 assembler 每个事件最多 1 个，因此输出 `1.5D`。导出器本身用 `D × (1 + 1/2 + 1/4 + ...)` 的通用循环支持多个 dot_count，但当前装配不会产生多个。
- tuplet：**未使用**。
- rest 时值由 `rest_whole/half/quarter/8th/16th` 类别直接映射；更短 rest 类没有完整独立支持。

## 4.7 chord 与 onset

### 主 MusicXML 导出

- 没有单独建立全局 onset 图。
- 在每个物理 staff/part、每个 measure 内，事件按 x/y/id 排序。
- 当前 note 的 x 与前一个 note 的 x 相差不超过 **1.5 原图像素**时写 `<chord/>`。
- 不要求共享 stem。
- 因此同 x、不同 stem 的 note 也可能组成 chord。
- 不通过累积时值推断 onset，也不按 staff spacing 归一化阈值。

### Polish/eKern 代理序列

- 按 x 做顺序聚类，默认 tolerance **4 原图像素**。
- 若同 staff 且共享 stem，允许与当前组中心相差最多 `2 × tolerance = 8 px`。
- 这是固定像素阈值，不按 staff spacing 归一化。

## 4.8 system 与 measure

### 主 semantic/MusicXML

- 每个物理 staff 直接成为一个 MusicXML part。
- piano system grouping：**未使用**。
- 每个 staff 使用自己的 barline x 列表。
- 事件的小节号为其左侧可用 barline 数量加 1。
- 没检测到 barline 时，整个 staff 作为 measure 1。

### Polish/eKern 代理序列

- 物理 staff 先按 staff index 自上而下排序。
- 默认把相邻两个 staff 固定配成一个 system；最后剩一个则保留为单 staff system。
- 如果上 staff 被识别为 bass、下 staff 被识别为 treble，代码不会交换它们，而是把上 staff 单独成组，然后从下 staff 继续配对。
- 不使用 brace 或垂直 barline 来学习 system。
- 双 staff system 中，barline 只有在两侧 staff 都有观测、且 x 在 **6 px** 内可聚类时才保留；只出现于一侧的 barline 被丢弃。
- 单 staff system 保留该 staff 的全部 barline。
- double/final/repeat barline 不做独立语义识别。
- 没有 barline 时，所有事件进入 local measure 0，序列化为一个 measure。

## 4.9 reading order

不能把所有输出统一写成

```text
system → measure → onset → staff
```

### 主 semantic/MusicXML 的真实顺序

```text
physical staff/part index
→ per-staff measure index
→ event x
→ event y
→ event id
```

它没有跨 staff 的 system/onset 统一排序。

### Polish canonical eKern 的真实顺序

```text
1. physical staff index（缺失时用事件 median y，再用 part id）
2. 相邻 staff 组成的 system index
3. local measure index
4. onset group，按 x、staff_slot、pitch 排序后顺序聚类
5. 输出 spine 顺序：lower/bass slot 1 在前，upper/treble slot 0 在后
6. chord 内：pitch、x、event id
```

注意：这是 Polish 代理适配器的顺序，不是主 MusicXML 解码器的通用顺序。

## 4.10 输出

### 内部 note event 主要字段

```text
id, type, staff, center,
notehead_id, notehead_class,
stem_id,
beam_ids, beam_count,
ledger_line_ids,
accidental_id, accidental,
slur_or_tie_ids,
dot_id, dot_count,
pitch_step, pitch_error, pitch_source,
duration_hint,
confidence,
source_symbol_ids
```

rest/mark event 使用相应的 `rest_id / kind / text / bbox` 等字段；semantic 层再补充 pitch、duration_units、MusicXML beam/slur 状态。

### Event-F1 实际使用字段

Event-F1 从生成 MusicXML 解析事件，只比较**无序多重集**：

```text
(pitch token, normalized duration)
```

- pitch token 为 `step + alter + octave`，rest 为 `R`。
- duration 使用 MusicXML `<duration>` 除以 `<divisions>` 后的有理数。
- 不比较 measure、staff、voice、reading order、chord 标记、relation id 或 slur。

### MusicXML 与代理序列

- MusicXML 是真实写出的 `score_v2_1.musicxml`，不是只有 proxy sequence。
- 但导出器明确标为 **draft semantic exporter**，不是完整排版/语义级 OMR 解码器。
- Polish 使用 semantic JSON 到 canonical eKern 的适配器。
- OLiMPiC/GrandStaff 的 LMX 评测使用相应数据集适配/官方风格评测入口；不能把这些代理序列描述为同一个通用 decoder 的原生输出。

### 未识别属性的默认值

| 属性 | 当前默认/行为 |
| --- | --- |
| divisions | 16 |
| key | fifths = 0，即 C major/A minor 默认 |
| time | 4/4 |
| voice | 1 |
| clef | bass→F4，其他/未知→G2 |
| missing pitch | C4 |
| missing duration | quarter，duration units = 16 |
| tuplets | **未使用**，不导出 tuplet |
| tie | **未区分/未导出 tie**；`slur_or_tie` 只导出 slur notation |
| measure duration balancing | **未使用**；不插入 `<backup>` 或 `<forward>`，不保证小节时值和为拍号长度 |
| chord | 相邻 note x 差 ≤1.5 px 时写 `<chord/>` |

---

# 5. 可用于正文与补充材料的措辞边界

## 正文可以写

- 页面级谱表几何为符号分配、pitch grid 和关系阈值提供 staff-aware normalization。
- DEIM-D-FINE 提供 58 类 OMR 符号候选。
- 显式关系解析覆盖 notehead–stem、stem–beam、ledger–notehead 与 slur endpoint–notehead。
- 事件装配使用确定性的局部几何约束，并从 notehead/stem/beam/accidental/dot 恢复 pitch-duration 事件。
- SAM2.1 提供可选的掩码与骨架几何细化。

## 正文不能写成既成事实

- full-page + staff-crop 双尺度融合；
- 所有数据集完全同参且全部阈值仅由源域确定；
- 主结果使用 staff-line positive/negative prompt；
- 通用全符号 crop classifier 是默认模块；
- 关系图包含 accidental、dot、flag、barline 等全部符号边；
- 全局图优化或学习式统一关系打分；
- C clef、key signature、临时变音持续、tuplet、tie、voice、跨 staff onset 的完整支持；
- SAM2 已被证明是 Event-F1 提升的核心来源。

---

# 6. 主要代码与证据索引

| 内容 | 文件/产物 |
| --- | --- |
| 谱表预处理、谱线和传统几何 | `tools/traditional_omr_demo.py` |
| SAM2 prompt 与推理 | `tools/refine_masks_sam2.py` |
| Polish canonical eKern | `tools/polish_canonical_serialization.py` |
| 关系边消融 | `tools/run_relation_edge_ablation.py`、`tools/run_cross_dataset_relation_edge_ablation.py` |
| Event-F1 解析 | `tools/paper_evidence_metrics.py` |
| DEIM 配置 | `.local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded_clean_5070ti.yml` |
| 58 类数据摘要 | `outputs/v2_deim_ds_all_expanded_clean/summary.json` |
| 冻结 GrandStaff 命令日志 | `outputs/ijcv_repro/fp_grandstaff_test24_full_run_logs/ours.log` |
| 冻结 MusicXML 示例 | `outputs/ijcv_dwd_dataset_runs/fp_grandstaff_test0000_v2_1_sam2_neural_clef_roi_rl_balance_test24/semantics/score_v2_1.musicxml` |
| 当前方法实现说明 | `docs/staff_method_implementation_v2_1_zh.md` |
| 归档 shape/relation 实现 | `tools_py311/extract_symbol_shapes_v2_1.pyc` |
| 归档事件装配实现 | `tools_py311/assemble_v2_1_notes.pyc` |
| 归档 semantic/MusicXML 实现 | `tools_py311/export_v2_1_semantics.pyc` |
| 归档 staff assignment 等公共实现 | `tools_py311/omr_v2_common.pyc` |

> 注意：部分核心 `.py` 源文件在当前工作区只保留了 Python 3.11 `.pyc`。本文对应规则由可执行字节码反汇编并结合冻结 JSON/XML 产物交叉核验。提交复现包前，建议恢复这些源文件；否则补充材料虽能说明算法，却不能满足完整源码复现要求。
