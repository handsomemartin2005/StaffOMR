from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
TOOLS311 = REPO / "tools_py311"


def _redirect(value):
    if isinstance(value, types.CodeType):
        return value.replace(co_consts=tuple(_redirect(item) for item in value.co_consts))
    if isinstance(value, tuple):
        redirected = tuple(_redirect(item) for item in value)
        if redirected == ("box_only", "box_positive", "box_pos_neg_staff"):
            return (*redirected, "method_aligned")
        return redirected
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        # Keep the pipeline wrapper as a real Python entrypoint. It loads the
        # archived bytecode and rewrites that bytecode's own tool references.
        # Redirecting this outer call straight to .pyc would skip the nested
        # rewrite and make restored tool wrappers invisible.
        if normalized == "tools/run_v2_full_pipeline.py":
            return value
        # This module has been restored as maintainable source so that the
        # method-aligned crop/box/point prompt path can be exercised directly.
        if normalized == "tools/refine_masks_sam2.py" and (REPO / normalized).exists():
            return value
        if normalized.startswith("tools/") and normalized.endswith(".py"):
            candidate = TOOLS311 / (Path(normalized).stem + ".pyc")
            if candidate.exists():
                return candidate.relative_to(REPO).as_posix()
    return value


def run(pyc_name: str, display_file: str) -> None:
    pyc_path = TOOLS311 / pyc_name
    with pyc_path.open("rb") as handle:
        handle.read(16)
        code = marshal.load(handle)
    code = _redirect(code)
    if str(TOOLS311) not in sys.path:
        sys.path.insert(0, str(TOOLS311))
    namespace = {
        "__name__": "__main__",
        "__file__": str(REPO / display_file),
        "__package__": None,
        "__cached__": str(pyc_path),
    }
    exec(code, namespace, namespace)
