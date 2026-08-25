"""Production frontend for lifting raw TTIR into ETV's semantic IR."""

from pathlib import Path

from ..ir import Program
from ..model import FrontendSpec, InputError
from .libtriton import parse_ttir
from .lift import lift_ttir


def load_program_artifact(path: Path, frontend: FrontendSpec = FrontendSpec()) -> Program:
    path = Path(path).resolve()
    if frontend.kind != "ttir":
        raise InputError(
            "ETV verification accepts raw TTIR on both sides; non-TTIR frontends "
            "are not part of the production verification boundary",
            "TTIR_PAIR_REQUIRED",
        )
    if path.suffix not in {".ttir", ".mlir"}:
        raise InputError(
            f"ETV verification inputs must be raw .ttir or .mlir files: {path}",
            "TTIR_PAIR_REQUIRED",
        )
    if frontend.programs is None:
        raise InputError(
            "raw TTIR verification requires frontends.<side>.programs because launch "
            "grids are not encoded in TTIR",
            "TTIR_LAUNCH_REQUIRED",
        )
    return lift_ttir(parse_ttir(path, function=frontend.function), frontend.programs)
