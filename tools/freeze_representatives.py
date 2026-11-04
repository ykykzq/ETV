"""Seal the fixed representatives, including all failures and partial TTIR captures."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    # Refuse to replace an existing version; changing a corpus needs a new name.
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.lock, args.out / "suite.lock.json")
    rows = []
    for nodeid in lock["selected"]:
        case_id = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        source = args.captures / case_id
        if not (source / "process.json").is_file():
            raise ValueError("representative has not run: " + nodeid)
        target = args.out / case_id
        target.mkdir()
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if not path.is_file():
                continue
            if path.suffix == ".ttir" or str(relative) in (
                "capture.json",
                "case_status.json",
                "process.json",
                "upstream.json",
                "pytest.log",
                "pytest.xml",
            ):
                destination = target / relative
                destination.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(path, destination)
        rows.append(
            {
                "nodeid": nodeid,
                "case_id": case_id,
                "ttir_files": len(list(target.rglob("*.ttir"))),
                "has_capture_contract": (target / "capture.json").is_file(),
            }
        )
    repo = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob("*")) if p.is_file()
    }
    sources = {
        str(p.relative_to(repo)): digest(p)
        for folder in ("src", "tools", "tests")
        for p in sorted((repo / folder).rglob("*.py"))
    }
    result = {
        "format": "etv-frozen-representatives-v1",
        "source_commit": lock["source_commit"],
        "selected": len(rows),
        "cases": rows,
        "files": files,
        "implementation_files": sources,
        "uv_lock_sha256": digest(repo / "uv.lock"),
    }
    (args.out / "corpus.lock.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cases": len(rows),
                "ttir_files": sum(r["ttir_files"] for r in rows),
                "lock_sha256": digest(args.out / "corpus.lock.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
