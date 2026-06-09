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

TORCH_MP_HELPER = r'''

def configure_torch_multiprocessing() -> None:
    """Use a robust tensor sharing strategy for DataLoader worker processes."""
    strategy = os.environ.get('PYTORCH_SHARING_STRATEGY', 'file_system')
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy(strategy)
        print(f'Set torch multiprocessing sharing strategy: {strategy}')
    except Exception as exc:
        print(f'Could not set torch multiprocessing sharing strategy {strategy!r}: {exc!r}')
'''

COMPOSE_APPLY_HELPER = r'''
    @staticmethod
    def apply_transform(transform, sample):
        if not isinstance(sample, tuple):
            return transform(sample)
        # DEIM passes (image, target, dataset) so policies can inspect dataset.epoch.
        # torchvision v2 transforms should only see transformable inputs.
        tail = ()
        inputs = sample
        if len(sample) >= 3 and hasattr(sample[-1], 'epoch'):
            inputs = sample[:-1]
            tail = (sample[-1],)
        output = transform(*inputs)
        if not isinstance(output, tuple):
            output = (output,)
        return output + tail

'''

CONVERT_BOXES_CLASS = r'''@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.transform(inpt, params)


'''

CONVERT_PIL_IMAGE_CLASS = r'''@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
        Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, PIL.Image.Image):
            inpt = F.pil_to_tensor(inpt)
        else:
            inpt = inpt.as_subclass(torch.Tensor)

        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            if not torch.is_floating_point(inpt) or bool(inpt.detach().max() > 1):
                inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.transform(inpt, params)
'''


def patch_train_py(train_py: Path) -> bool:
    text = train_py.read_text(encoding="utf-8")
    original = text

    if "def configure_cuda_memory_limit()" not in text:
        marker = "\ndef main(args, ) -> None:\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find DEIM main() marker in {train_py}")
        text = text.replace(marker, CUDA_LIMIT_HELPER + marker, 1)

    if "def configure_torch_multiprocessing()" not in text:
        marker = "\ndef main(args, ) -> None:\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find DEIM main() marker in {train_py}")
        text = text.replace(marker, TORCH_MP_HELPER + marker, 1)

    cuda_call = "    configure_cuda_memory_limit()\n"
    if cuda_call not in text:
        docstring_marker = '    """main\n    """\n'
        if docstring_marker in text:
            text = text.replace(docstring_marker, docstring_marker + cuda_call, 1)
        else:
            main_marker = "def main(args, ) -> None:\n"
            text = text.replace(main_marker, main_marker + cuda_call, 1)

    mp_call = "    configure_torch_multiprocessing()\n"
    if mp_call not in text:
        if cuda_call in text:
            text = text.replace(cuda_call, cuda_call + mp_call, 1)
        else:
            docstring_marker = '    """main\n    """\n'
            text = text.replace(docstring_marker, docstring_marker + mp_call, 1)

    if text == original:
        return False
    train_py.write_text(text, encoding="utf-8")
    return True


def patch_compose_container(container_py: Path) -> bool:
    text = container_py.read_text(encoding="utf-8")
    original = text

    if "def apply_transform(transform, sample):" not in text:
        marker = "    def default_forward(self, *inputs: Any) -> Any:\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find Compose.default_forward marker in {container_py}")
        text = text.replace(marker, COMPOSE_APPLY_HELPER + marker, 1)

    text = text.replace("sample = transform(sample)", "sample = self.apply_transform(transform, sample)")

    if text == original:
        return False
    container_py.write_text(text, encoding="utf-8")
    return True


def replace_class(text: str, class_name: str, replacement: str, end_marker: str | None = None) -> str:
    start_marker = f"@register()\nclass {class_name}"
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Cannot find {class_name} class")
    if end_marker is None:
        end = len(text)
    else:
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            raise RuntimeError(f"Cannot find end marker for {class_name}")
    return text[:start] + replacement + text[end:]


def patch_transforms_py(transforms_py: Path) -> bool:
    text = transforms_py.read_text(encoding="utf-8")
    original = text

    old_pad = (
        "    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:\n"
        "        fill = self._fill[type(inpt)]\n"
        "        padding = params['padding']\n"
        "        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]\n"
    )
    new_pad = (
        "    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:\n"
        "        fill = self._fill[type(inpt)]\n"
        "        padding = params['padding']\n"
        "        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]\n\n"
        "    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:\n"
        "        return self.transform(inpt, params)\n"
    )
    if old_pad in text and new_pad not in text:
        text = text.replace(old_pad, new_pad, 1)

    text = replace_class(text, "ConvertBoxes", CONVERT_BOXES_CLASS, "@register()\nclass ConvertPILImage")
    text = replace_class(text, "ConvertPILImage", CONVERT_PIL_IMAGE_CLASS)

    if text == original:
        return False
    transforms_py.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Staff OMR compatibility patches to a local DEIM checkout.")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    args = parser.parse_args()

    train_py = args.deim_root / "train.py"
    container_py = args.deim_root / "engine" / "data" / "transforms" / "container.py"
    transforms_py = args.deim_root / "engine" / "data" / "transforms" / "_transforms.py"
    for path in (train_py, container_py, transforms_py):
        if not path.exists():
            raise FileNotFoundError(f"DEIM file not found: {path}")

    changed = False
    changed |= patch_train_py(train_py)
    changed |= patch_compose_container(container_py)
    changed |= patch_transforms_py(transforms_py)
    status = "patched" if changed else "already_patched"
    print(f"{status}: {args.deim_root}")


if __name__ == "__main__":
    main()
