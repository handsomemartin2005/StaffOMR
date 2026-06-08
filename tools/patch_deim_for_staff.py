from __future__ import annotations

import argparse
from pathlib import Path


CUDA_LIMIT_HELPER = r'''

def configure_cuda_memory_limit() -> None:
    """Optionally cap this process' CUDA allocator fraction via env var."""
    fraction = os.environ.get('PYTORCH_CUDA_MEMORY_FRACTION')
    if not fraction:
        return
    try:
        value = float(fraction)
    except ValueError:
        print(f'Ignore invalid PYTORCH_CUDA_MEMORY_FRACTION={fraction!r}')
        return
    if value <= 0 or value > 1:
        print(f'Ignore out-of-range PYTORCH_CUDA_MEMORY_FRACTION={fraction!r}')
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(value)
        print(f'Set CUDA memory fraction limit: {value:.3f}')
'''


def patch_train_py(train_py: Path) -> bool:
    text = train_py.read_text(encoding="utf-8")
    original = text

    if "def configure_cuda_memory_limit()" not in text:
        marker = "\ndef main(args, ) -> None:\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find DEIM main() marker in {train_py}")
        text = text.replace(marker, CUDA_LIMIT_HELPER + marker, 1)

    call = "\n    configure_cuda_memory_limit()\n"
    if call not in text:
        docstring_marker = '    """main\n    """\n'
        if docstring_marker in text:
            text = text.replace(docstring_marker, docstring_marker + "    configure_cuda_memory_limit()\n", 1)
        else:
            main_marker = "def main(args, ) -> None:\n"
            text = text.replace(main_marker, main_marker + "    configure_cuda_memory_limit()\n", 1)

    if text == original:
        return False
    train_py.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Staff OMR compatibility patches to a local DEIM checkout.")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    args = parser.parse_args()

    train_py = args.deim_root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"DEIM train.py not found: {train_py}")

    changed = patch_train_py(train_py)
    status = "patched" if changed else "already_patched"
    print(f"{status}: {train_py}")


if __name__ == "__main__":
    main()
