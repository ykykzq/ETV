"""Build the read-only adapter against the exact headers used by libtriton."""

import argparse
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pybind11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-include", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("triton").split("+")[0] != "3.7.1":
        parser.error("adapter requires Triton 3.7.1")
    if not (args.llvm_include / "mlir/IR/BuiltinOps.h").is_file():
        parser.error("LLVM include directory is missing MLIR headers")
    spec = importlib.util.find_spec("triton._C.libtriton")
    if spec is None or spec.origin is None:
        parser.error("install pinned libtriton first")
    root = Path(__file__).resolve().parents[1]
    output = root / "src/etv" / ("_mlir_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC"]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    for include in (
        sysconfig.get_path("include"),
        pybind11.get_include(),
        args.llvm_include,
    ):
        command.append("-I" + str(include))
    command += [str(root / "native/snapshot.cpp"), spec.origin, "-o", str(output)]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
