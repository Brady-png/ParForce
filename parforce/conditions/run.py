"""Build particle and surface depth conditions from first-frame perception."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
import traceback

from .firstview_v2 import digest
from .pipeline import run
from .quality import POLICY, check_case


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def worker(row, root):
    case = row["case"]
    if not isinstance(case, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", case):
        raise ValueError(f"Invalid case identifier: {case!r}")
    output = Path(root) / "cases" / case
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        if row["material"] != "rigid":
            raise ValueError("Unsupported material or actuation route for the rigid backend")
        pointmap = Path(row["depth_file"])
        receipt = json.loads((pointmap.parent / "RECEIPT.json").read_text())
        frame_hash = receipt.get("source_frame_sha256", receipt.get("first_frame_sha256"))
        if frame_hash != row["first_frame_sha256"]:
            raise ValueError("Depth receipt is bound to a different first frame")
        if digest(pointmap) != receipt["pointmap_sha256"]:
            raise ValueError("Pointmap checksum differs from the depth receipt")
        run(row, root)
        gate = check_case(row, output)
        result = {"case": case, "completed": True, "gate": gate}
    except Exception as error:
        result = {
            "case": case,
            "completed": False,
            "gate": {"passed": False, "reasons": [str(error)]},
            "error": traceback.format_exc(),
        }
    result["seconds"] = time.monotonic() - start
    write_json(output / "PIPELINE_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        parser.error("Manifest must contain a nonempty JSON array of cases")
    cases = [row.get("case") for row in rows]
    if any(not isinstance(case, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", case) for case in cases):
        parser.error("Case identifiers must contain only letters, digits, underscores or hyphens")
    if len(set(cases)) != len(cases):
        parser.error("Each case identifier must be unique within the manifest")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.output / "GATE_POLICY.json").exists() or any(
        (args.output / "cases" / case).exists() for case in cases
    ):
        parser.error("Use a fresh output directory for another reconstruction attempt")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "GATE_POLICY.json").open("x") as gate:
        json.dump(POLICY, gate, indent=2)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(worker, row, str(args.output)) for row in rows]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            write_json(
                args.output / "PROGRESS.json",
                {
                    "total": len(rows),
                    "completed": len(results),
                    "gate_passed": sum(item["gate"]["passed"] for item in results),
                    "results": results,
                },
            )
            print(f"CASE {result['case']} {result['gate']}", flush=True)
    write_json(args.output / "ALL_PIPELINE_RESULTS.json", results)
    return int(any(not result["completed"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
