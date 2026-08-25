"""Artifact dispatch between Semantic JSON, Torch Prims, and raw TTIR."""

import json
from pathlib import Path

from ..model import FrontendSpec, InputError, Program
from ..prims import FORMAT_PRIMS, load_prims_program
from ..schema import load_program
from .libtriton import parse_ttir
from .lift import lift_ttir


def load_program_artifact(path: Path, frontend: FrontendSpec = FrontendSpec()) -> Program:
    path = Path(path).resolve()
    kind = frontend.kind
    if kind == "auto":
        if path.suffix == ".json":
            try:
                declared_format = json.loads(path.read_text(encoding="utf-8")).get(
                    "format"
                )
            except (OSError, ValueError, AttributeError):
                declared_format = None
            kind = "prims" if declared_format == FORMAT_PRIMS else "semantic_json"
        elif path.suffix in {".ttir", ".mlir"}:
            kind = "ttir"
        else:
            raise InputError(f"cannot infer frontend from extension: {path}", "FRONTEND_REQUIRED")
    if kind == "semantic_json":
        return load_program(path)
    if kind == "prims":
        return load_prims_program(path)
    if kind != "ttir":
        raise InputError(f"unknown frontend kind {kind!r}")
    if frontend.programs is None:
        raise InputError(
            "raw TTIR verification requires frontends.<side>.programs because launch grids are not encoded in TTIR",
            "TTIR_LAUNCH_REQUIRED",
        )
    return lift_ttir(parse_ttir(path, function=frontend.function), frontend.programs)
