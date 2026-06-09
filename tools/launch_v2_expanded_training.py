from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run_checked(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def run_streamed(command: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"\n# started_at={datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"torch_import_error": repr(exc), "cuda_available": False}

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device_index": device,
                "device_name": props.name,
                "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return info


def auto_train_batch_size(total_memory_mb: int | None) -> int:
    if total_memory_mb is None:
        return 1
    if total_memory_mb >= 90000:
        return 24
    if total_memory_mb >= 78000:
        return 20
    if total_memory_mb >= 46000:
        return 12
    if total_memory_mb >= 23000:
        return 6
    if total_memory_mb >= 15000:
        return 3
    return 1


def auto_val_batch_size(total_memory_mb: int | None) -> int:
    if total_memory_mb is None:
        return 1
    if total_memory_mb >= 78000:
        return 4
    if total_memory_mb >= 46000:
        return 2
    return 1


def auto_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def require_path(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. {hint}")


def write_launch_info(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def raise_nofile_limit(target: int = 65535) -> dict[str, int | str]:
    if os.name == "nt":
        return {"status": "unsupported_on_windows"}
    try:
        import resource
    except ImportError:
        return {"status": "resource_module_unavailable"}

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(max(soft, target), hard)
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    final_soft, final_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {"soft": int(final_soft), "hard": int(final_hard)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the V2 expanded DEIM detector on a Linux GPU server.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--train-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_train.json"))
    parser.add_argument("--val-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_test.json"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/v2_deim_ds_all_expanded"))
    parser.add_argument("--config", type=Path, default=Path(".local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded_pro6000.yml"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000"))
    parser.add_argument("--tuning-checkpoint", type=Path, default=Path("outputs/models/deim/deim_hgnetv2_n_coco.pth"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--val-limit", type=int, default=-1)
    parser.add_argument("--train-batch-size", type=int, default=0, help="0 means auto-select from GPU memory.")
    parser.add_argument("--val-batch-size", type=int, default=0, help="0 means auto-select from GPU memory.")
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 means auto-select from CPU count.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, help="Optional PYTORCH_CUDA_MEMORY_FRACTION value, e.g. 0.90.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.workdir.resolve()
    deim_root = resolve_under(root, args.deim_root).resolve()
    train_json = resolve_under(root, args.train_json).resolve()
    val_json = resolve_under(root, args.val_json).resolve()
    image_root = resolve_under(root, args.image_root).resolve()
    dataset_root = resolve_under(root, args.dataset_root).resolve()
    config = resolve_under(root, args.config).resolve()
    run_dir = resolve_under(root, args.run_dir).resolve()
    tuning_checkpoint = resolve_under(root, args.tuning_checkpoint).resolve() if args.tuning_checkpoint else None
    resume = resolve_under(root, args.resume).resolve() if args.resume else None

    require_path(root / "tools" / "prepare_deepscores_v2_dataset.py", "Run this from the StaffOMR repository root.")
    require_path(deim_root / "train.py", "Run scripts/setup_pro6000_server.sh first to install DEIM.")
    if not args.skip_dataset:
        require_path(train_json, "Copy ds2_dense/deepscores_train.json into the repository first.")
        require_path(val_json, "Copy ds2_dense/deepscores_test.json into the repository first.")
        require_path(image_root, "Copy ds2_dense/images into the repository first.")
    elif not args.skip_config:
        require_path(dataset_root / "annotations" / "instances_train.json", "Run once without --skip-dataset first.")
        require_path(dataset_root / "annotations" / "instances_val.json", "Run once without --skip-dataset first.")
        require_path(image_root, "Copy ds2_dense/images into the repository first.")
    else:
        require_path(config, "Run once without --skip-config first, or provide --config.")
    if resume is None and tuning_checkpoint is not None:
        require_path(tuning_checkpoint, "Run scripts/setup_pro6000_server.sh or provide --tuning-checkpoint.")

    cuda_info = torch_cuda_info()
    if args.device.startswith("cuda") and not cuda_info.get("cuda_available"):
        raise RuntimeError(f"CUDA is not available in the active Python environment: {cuda_info}")
    total_memory_mb = cuda_info.get("total_memory_mb")
    total_memory = int(total_memory_mb) if total_memory_mb is not None else None
    train_batch_size = args.train_batch_size if args.train_batch_size > 0 else auto_train_batch_size(total_memory)
    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else auto_val_batch_size(total_memory)
    num_workers = args.num_workers if args.num_workers >= 0 else auto_workers()

    run_dir.mkdir(parents=True, exist_ok=True)
    nofile_limit = raise_nofile_limit()
    launch_info = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "workdir": str(root),
        "deim_root": str(deim_root),
        "taxonomy": "expanded",
        "num_classes": 81,
        "epochs": args.epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "use_amp": not args.no_amp,
        "cuda": cuda_info,
        "dataset_root": str(dataset_root),
        "config": str(config),
        "run_dir": str(run_dir),
        "tuning_checkpoint": str(tuning_checkpoint) if tuning_checkpoint else None,
        "resume": str(resume) if resume else None,
        "nofile_limit": nofile_limit,
    }
    write_launch_info(run_dir / "train_launch_info.json", launch_info)
    print(json.dumps(launch_info, ensure_ascii=False, indent=2), flush=True)

    if not args.skip_dataset:
        run_checked(
            [
                sys.executable,
                "tools/prepare_deepscores_v2_dataset.py",
                "--taxonomy",
                "expanded",
                "--train-json",
                str(train_json),
                "--val-json",
                str(val_json),
                "--image-root",
                str(image_root),
                "--out-root",
                str(dataset_root),
                "--train-limit",
                str(args.train_limit),
                "--val-limit",
                str(args.val_limit),
                "--skip-class",
                "text_region",
            ],
            root,
        )

    if not args.skip_config:
        run_checked(
            [
                sys.executable,
                "tools/create_deim_symbol_config.py",
                "--taxonomy",
                "expanded",
                "--dataset-root",
                str(dataset_root),
                "--image-root",
                str(image_root),
                "--out-config",
                str(config),
                "--deim-output-dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--train-batch-size",
                str(train_batch_size),
                "--val-batch-size",
                str(val_batch_size),
                "--num-workers",
                str(num_workers),
                "--checkpoint-freq",
                "10",
            ],
            root,
        )

    train_cmd = [
        sys.executable,
        "train.py",
        "-c",
        str(config),
        "-d",
        args.device,
        "--output-dir",
        str(run_dir),
    ]
    if not args.no_amp:
        train_cmd.append("--use-amp")
    if args.test_only:
        train_cmd.append("--test-only")
    if resume:
        train_cmd.extend(["-r", str(resume)])
    elif tuning_checkpoint:
        train_cmd.extend(["-t", str(tuning_checkpoint)])

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    env.setdefault("PYTORCH_SHARING_STRATEGY", "file_system")
    if args.memory_fraction is not None:
        env["PYTORCH_CUDA_MEMORY_FRACTION"] = str(args.memory_fraction)

    launch_info["train_command"] = train_cmd
    launch_info["env"] = {
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_CUDA_MEMORY_FRACTION": env.get("PYTORCH_CUDA_MEMORY_FRACTION"),
        "PYTORCH_SHARING_STRATEGY": env.get("PYTORCH_SHARING_STRATEGY"),
    }
    write_launch_info(run_dir / "train_launch_info.json", launch_info)

    if args.dry_run:
        print("dry_run: training command was not executed")
        return

    run_streamed(train_cmd, deim_root, run_dir / "train_console.log", env=env)


if __name__ == "__main__":
    main()
