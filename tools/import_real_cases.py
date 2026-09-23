"""Import the small archived multilaunch corpus using captured storage evidence."""

import argparse
import json
from pathlib import Path

from etv_bench.migrate import migrate_v2
from build_examples import contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "examples/real"
    for name in ("argsort", "rms_norm", "batch_norm", "quantile"):
        source = args.source / name
        mapping = json.loads((source / "mapping.json").read_text())
        pair = json.loads((source / "pair.json").read_text())
        roles = {}
        for component in pair["metadata"]["launches"]["rhs"]:
            inventories = [
                item
                for item in mapping["launch_pointer_inventories"]
                if item["launch_id"] == component["id"]
            ]
            if len(inventories) != 1:
                raise ValueError("missing unique captured pointer inventory")
            pointers = [
                p for p in inventories[0]["pointers"] if p["semantic_role"] == "selected_output"
            ]
            if len(pointers) != 1:
                raise ValueError("missing unique selected store pointer")
            candidates = [
                role
                for role, endpoint in component["abi"].items()
                if endpoint["name"] == pointers[0]["name"]
            ]
            if len(candidates) != 1:
                raise ValueError("ambiguous selected store role")
            roles["rhs:" + component["id"]] = candidates[0]
        output = migrate_v2(source / "pair.json", target / name, roles)
        migrated = json.loads(output.read_text())
        contract(
            output.parent,
            migrated,
            {
                "status": "UNKNOWN",
                "allowed_reasons": ["TTIR_REGION_UNSUPPORTED", "TTIR_OP_UNSUPPORTED"],
            },
        )
        (output.parent / "capture_mapping.json").write_text(
            json.dumps(mapping, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
