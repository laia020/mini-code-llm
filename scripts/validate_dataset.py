#!/usr/bin/env python3
"""Validate Mini Code LLM Dataset V1 with only the Python standard library."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PER_FAMILY = {"train": 60, "validation": 10, "test": 10}
REQUIRED_FIELDS = {"id", "family", "difficulty", "instruction", "prompt", "code", "text"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def comparable(value):
    if isinstance(value, tuple):
        return [comparable(item) for item in value]
    if isinstance(value, list):
        return [comparable(item) for item in value]
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    return value


def validate_contract(contract: dict) -> None:
    tree = ast.parse(contract["code"])
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert len(definitions) == 1, f"{contract['family']}: canonical code must contain one function"
    definition = definitions[0]
    assert definition.name == contract["family"], f"{contract['family']}: function name mismatch"

    namespace = {}
    exec(compile(tree, f"<{contract['family']}>", "exec"), {"__builtins__": __builtins__}, namespace)
    function = namespace[contract["family"]]
    assert list(inspect.signature(function).parameters) == contract["parameters"], (
        f"{contract['family']}: parameter mismatch"
    )
    for case_number, case in enumerate(contract["tests"], start=1):
        result = function(*case["args"])
        if case.get("expected_type") == "tuple":
            assert isinstance(result, tuple), f"{contract['family']} test {case_number}: expected tuple"
        assert comparable(result) == comparable(case["expected"]), (
            f"{contract['family']} test {case_number}: {result!r} != {case['expected']!r}"
        )


def validate(root: Path) -> dict:
    contracts_data = json.loads((root / "contracts" / "families.json").read_text(encoding="utf-8"))
    contracts = {item["family"]: item for item in contracts_data["families"]}
    assert len(contracts) == 12, "exactly 12 family contracts are required"
    for contract in contracts.values():
        validate_contract(contract)

    all_ids = set()
    all_instructions = {}
    all_prompts = {}
    report = {"splits": {}, "families": {}}

    for split, expected_count in EXPECTED_PER_FAMILY.items():
        path = root / "data" / f"{split}.jsonl"
        rows = load_jsonl(path)
        counts = Counter(row.get("family") for row in rows)
        assert len(rows) == expected_count * len(contracts), f"{split}: wrong total record count"
        assert set(counts) == set(contracts), f"{split}: missing or unknown families"
        assert all(count == expected_count for count in counts.values()), f"{split}: unbalanced families"

        for line_number, row in enumerate(rows, start=1):
            location = f"{path}:{line_number}"
            assert set(row) == REQUIRED_FIELDS, f"{location}: schema mismatch"
            assert row["id"].startswith(f"{split}-{row['family']}-"), f"{location}: invalid id"
            assert row["id"] not in all_ids, f"{location}: duplicate id"
            all_ids.add(row["id"])
            assert row["family"] in contracts, f"{location}: unknown family"
            contract = contracts[row["family"]]
            assert row["difficulty"] == contract["difficulty"], f"{location}: difficulty mismatch"
            assert row["code"] == contract["code"], f"{location}: code differs from contract"
            expected_prompt = f"### Task:\n{row['instruction']}\n\n### Code:\n"
            assert row["prompt"] == expected_prompt, f"{location}: prompt format mismatch"
            assert row["text"] == row["prompt"] + row["code"], f"{location}: text mismatch"
            ast.parse(row["code"])

            normalized_instruction = normalize(row["instruction"])
            normalized_prompt = normalize(row["prompt"])
            assert normalized_instruction not in all_instructions, (
                f"{location}: normalized instruction duplicates {all_instructions[normalized_instruction]}"
            )
            assert normalized_prompt not in all_prompts, (
                f"{location}: normalized prompt duplicates {all_prompts[normalized_prompt]}"
            )
            all_instructions[normalized_instruction] = location
            all_prompts[normalized_prompt] = location

        report["splits"][split] = {"records": len(rows), "per_family": dict(sorted(counts.items()))}

    for family, contract in contracts.items():
        report["families"][family] = {
            "difficulty": contract["difficulty"],
            "tests": len(contract["tests"]),
            "records": sum(
                split_report["per_family"][family] for split_report in report["splits"].values()
            ),
        }

    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_records"] == len(all_ids), "manifest total mismatch"
    for split in EXPECTED_PER_FAMILY:
        actual_hash = sha256(root / "data" / f"{split}.jsonl")
        assert manifest["sha256"][f"{split}.jsonl"] == actual_hash, f"{split}: SHA-256 mismatch"
    assert manifest["contracts_sha256"] == sha256(root / "contracts" / "families.json"), (
        "contracts SHA-256 mismatch"
    )

    report["summary"] = {
        "status": "PASS",
        "total_records": len(all_ids),
        "unique_ids": len(all_ids),
        "unique_normalized_instructions": len(all_instructions),
        "cross_split_instruction_duplicates": 0,
        "contract_tests": sum(len(contract["tests"]) for contract in contracts.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        report = validate(args.root.resolve())
    except (AssertionError, FileNotFoundError, KeyError, SyntaxError) as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
