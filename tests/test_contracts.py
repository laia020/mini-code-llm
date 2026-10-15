#!/usr/bin/env python3
"""Regression tests for every family contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def comparable(value):
    if isinstance(value, tuple):
        return [comparable(item) for item in value]
    if isinstance(value, list):
        return [comparable(item) for item in value]
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    return value


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads((ROOT / "contracts" / "families.json").read_text(encoding="utf-8"))
        cls.contracts = data["families"]

    def test_all_canonical_implementations(self):
        for contract in self.contracts:
            with self.subTest(family=contract["family"]):
                namespace = {}
                exec(contract["code"], {"__builtins__": __builtins__}, namespace)
                function = namespace[contract["family"]]
                for case in contract["tests"]:
                    result = function(*case["args"])
                    if case.get("expected_type") == "tuple":
                        self.assertIsInstance(result, tuple)
                    self.assertEqual(comparable(result), comparable(case["expected"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
