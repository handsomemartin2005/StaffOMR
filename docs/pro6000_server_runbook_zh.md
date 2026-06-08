# RTX PRO 6000 服务器训练手册

目标：在租到单卡 RTX PRO 6000 Blackwell 96GB 后，尽量少手工配置，直接训练 V2 expanded detector。

## 推荐机器

- Ubuntu 22.04/24.04
- Python 3.10 或 3.11
- NVIDIA driver 能正常运行 `nvidia-smi`
- 1 张 RTX PRO 6000 Blackwell 96GB
- 32 vCPU、128GB RAM、500GB 以上本地 NVMe

不要把数据集放在网络盘上。这个任务每页标注很多，数据读取慢会直接拖低 GPU 利用率。

## 代码和数据目录

服务器上 clone 仓库后，数据放成下面结构：

```text
StaffOMR/
  ds2_dense/
    ds2_dense/
      deepscores_train.json
      deepscores_test.json
      images/
```

训练输出不会进 Git，默认写到：

```text
outputs/v2_deim_ds_all_expanded
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000
```

## 一键配置环境

```bash
bash scripts/setup_pro6000_server.sh
```

这个脚本会做这些事：

- 创建 `.venv-v2`
- 安装 CUDA 版 PyTorch
- 安装 V2 项目依赖
- clone `ShihuaHuang95/DEIM` 到 `.local-tools/DEIM-main`
- 给 DEIM `train.py` 应用本项目需要的 CUDA 显存限制补丁
- 下载 `deim_hgnetv2_n_coco.pth`
- 检查 CUDA、checkpoint、DEIM 入口是否可用

默认 PyTorch wheel 使用：

```bash
https://download.pytorch.org/whl/cu128
```

如果服务器镜像已经有可用的 CUDA PyTorch 环境，可以跳过 venv：

```bash
CREATE_VENV=0 INSTALL_TORCH=0 bash scripts/setup_pro6000_server.sh
```

## 开始训练

```bash
bash scripts/run_pro6000_v2_expanded.sh
```

默认会：

- 使用 expanded taxonomy，81 类
- 生成全量 DeepScores COCO 数据
- 生成 DEIM config
- 从 COCO 预训练 checkpoint tuning
- 开 AMP
- 自动按显存选择 batch。96GB 卡默认 train batch 为 24，val batch 为 4
- 训练 400 epoch

手动拉高 batch：

```bash
bash scripts/run_pro6000_v2_expanded.sh --train-batch-size 32 --val-batch-size 4
```

如果 32 OOM，退回：

```bash
bash scripts/run_pro6000_v2_expanded.sh --train-batch-size 24 --val-batch-size 4
```

## 跑 V2.1 重构对比

训练完成后直接运行：

```bash
bash scripts/run_v2_1_after_training.sh
```

脚本会自动选择：

```text
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/best_stg2.pth
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/best_stg1.pth
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/last.pth
```

优先级从上到下。默认样例输入是：

```text
ds2_dense/ds2_dense/images/lg-2267728-aug-beethoven--page-2.png
```

输出位置：

```text
outputs/v2_1_runs/pro6000_sample/summary.json
outputs/v2_1_runs/pro6000_sample/visuals/input_vs_v2_reconstructed.png
outputs/v2_1_runs/pro6000_sample/visuals/v2_1_relation_overlay.png
outputs/v2_1_runs/pro6000_sample/symbols/symbols_v2_1_shapes.json
```

如果要换页面：

```bash
INPUT=ds2_dense/ds2_dense/images/your-page.png \
OUT_ROOT=outputs/v2_1_runs/your-page \
bash scripts/run_v2_1_after_training.sh
```

如果服务器上没有 SAM2 checkpoint，脚本不会中断，会自动用 bbox fallback masks 跑 V2.1。等 SAM2 checkpoint 放到 `outputs/models/sam2/sam2.1_hiera_tiny.pt` 后，会自动启用 SAM2。

## 中断后续训

```bash
bash scripts/run_pro6000_v2_expanded.sh \
  --skip-dataset \
  --skip-config \
  --resume outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/last.pth
```

## 只验证 checkpoint

```bash
bash scripts/run_pro6000_v2_expanded.sh \
  --skip-dataset \
  --skip-config \
  --resume outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/last.pth \
  --test-only
```

## 关键日志

```text
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/train_launch_info.json
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/train_console.log
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/log.txt
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/best_stg1.pth
outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000/last.pth
```

`train_launch_info.json` 记录了 batch、worker、CUDA 信息和实际训练命令。出错时先看这个文件和 `train_console.log`。
