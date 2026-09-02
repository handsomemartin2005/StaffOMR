# STAFF AAAI-27 初稿状态

## 已完成

- 论文已拆分为 `main.tex`、`head.tex`、`sections/*.tex`、`appendix.tex` 和 `reference.bib`；`staff_aaai27.tex` 保留为兼容入口。
- 使用用户提供的 AAAI 2027 Author Kit 建立匿名 LaTeX 稿。
- 完成 Abstract、Introduction、Related Work、Protocol、Method、Experiments、Ablation、Analysis、Limitations、Conclusion。
- 写入仓库中已经验证的跨域实验结果，不虚构实验数字。
- 区分外部发布模型、我们的本地适配实验和 StaffOMR-SAM 自研模块。
- 加入 3 张图、6 张表、26 条实际引用。
- 生成可编辑方法图 `staff_overview.drawio`，并提供论文用 PNG/SVG。
- 完成 PDF 编译、逐页渲染、字体和匿名检查。

## 当前主张边界

- 可以主张：发布的 OMR 序列模型存在显著跨域退化；100/1000 页目标域适配形成明确预算曲线；StaffOMR-SAM 的主要瓶颈在语义解码而不只是符号定位。
- 不应主张：StaffOMR-SAM 达到 SOTA；Zeus/SMT 架构由本文提出；Polish eKern proxy 与官方 LMX SER 可直接比较；1000 页结果证明通用域适配算法。

## 下一轮必须补齐

- 复核每条 BibTeX 的作者、会议/期刊、页码和 DOI。
- 完成 AAAI reproducibility checklist 和匿名补充材料。
- 补充 OLiMPiC 反向 1000 页适配或说明无法完成的预算原因。
- 增加结构约束混合模型的真实实验；当前 hybrid 只作为下一步设计，不作为完成贡献。
- 固定开发集并避免对 test20 做规则选择；headline 保留 test200 原始结果。
- 最终提交前再次核对 AAAI 官网是否更新页数、匿名或生成式 AI 政策。
