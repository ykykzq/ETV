"""Download the LLVM headers used by the pinned official Triton build."""

import argparse
from pathlib import Path
import platform
import tarfile
from urllib.request import urlopen

LLVM_REVISION = "1f126a6dea50d185c0781743a667390037ae88bd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/llvm"))
    args = parser.parse_args()
    arch = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
    system = {"Linux": "ubuntu", "Darwin": "macos"}.get(platform.system())
    if not arch or not system:
        parser.error("supply matching LLVM headers manually on this platform")
    name = f"llvm-{LLVM_REVISION[:8]}-{system}-{arch}-1"
    target = args.out.resolve()
    marker = target / "revision.txt"
    if marker.exists() and marker.read_text().strip() == LLVM_REVISION:
        print(target / name / "include")
        return
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    with (
        urlopen(url, timeout=120) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            if member.name.startswith(name + "/include/"):
                archive.extract(member, target, filter="data")
    if not (target / name / "include/mlir/IR/BuiltinOps.h").is_file():
        raise RuntimeError("upstream archive has an unexpected header layout")
    marker.write_text(LLVM_REVISION + "\n")
    print(target / name / "include")


if __name__ == "__main__":
    main()
