"""Re-evaluate stored generations and verify the immutable artifact hashes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import check_new_outputs, read_jsonl, write_json
from src.evaluate import evaluate_records


def compare(actual, expected, location=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"Chaves diferentes em {location}"
        for key in expected:
            if key != "input_file":
                compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, (int, float)):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), location
    else:
        assert actual == expected, location


def verify_hashes():
    manifest = json.loads((ROOT / "docs/original-artifacts.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Artefato alterado: {name}"
    return len(manifest["artifacts"])


def audit(output_dir):
    check_new_outputs(output_dir)
    preserved = verify_hashes()
    contracts = {c["family"]: c for c in json.loads((ROOT / "contracts/families.json").read_text(encoding="utf-8"))["families"]}
    reports = []
    for path in sorted((ROOT / "results").glob("*_summary.json")):
        expected = json.loads(path.read_text(encoding="utf-8"))
        input_path = ROOT / expected["input_file"]
        evaluated, actual = evaluate_records(read_jsonl(input_path), contracts, expected["input_file"])
        compare(actual, expected)
        historical_details = read_jsonl(input_path.with_name(input_path.stem + "_evaluated.jsonl"))
        assert evaluated == historical_details, f"Detalhes divergentes: {input_path.name}"
        write_json(output_dir / path.name, actual)
        reports.append({"file": path.name, "status": "PASS", "examples": actual["examples"], "details_match": True})
        print(f"PASS: {path.name}", flush=True)
    assert verify_hashes() == preserved
    report = {"status": "PASS", "immutable_artifacts": preserved, "results": reports, "note": "Respostas salvas reavaliadas; nenhuma inferência nova no test set."}
    write_json(output_dir / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args.output_dir)
    except (AssertionError, ValueError, OSError) as error:
        print(f"AUDIT FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
