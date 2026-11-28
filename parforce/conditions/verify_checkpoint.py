"""Verify the pinned offline Qwen snapshot and write its local receipt."""

from hashlib import file_digest
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    spec_file = Path(__file__).with_name("policies") / "qwen-weights.json"
    spec = json.loads(spec_file.read_text())
    size_total = 0
    for name, expected in spec["files"].items():
        path = args.model / name
        size = path.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"Incorrect size for {name}")
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"Incorrect SHA-256 for {name}")
        size_total += size
    receipt = {"repo": spec["repo"], "commit": spec["commit"], "ready": True,
               "total_bytes": size_total, "all_weights_sha256_verified": True}
    temporary = args.model / "COMPLETE.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(args.model / "COMPLETE.json")


if __name__ == "__main__":
    main()
