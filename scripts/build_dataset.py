#!/usr/bin/env python3
"""Build the deterministic Mini Code LLM Dataset V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONTRACTS_DIR = ROOT / "contracts"


FAMILIES = [
    {
        "family": "add_numbers",
        "difficulty": 1,
        "signature": "add_numbers(a, b)",
        "parameters": ["a", "b"],
        "behavior": "return the sum of two numbers",
        "behavior_alt": "add its two numeric arguments and return the result",
        "contract": "Accept two numbers and return their sum.",
        "constraints": ["The result must be returned, not printed."],
        "code": "def add_numbers(a, b):\n    return a + b",
        "tests": [
            {"args": [2, 3], "expected": 5},
            {"args": [-2, 5], "expected": 3},
            {"args": [0, 0], "expected": 0},
            {"args": [1.5, 2.25], "expected": 3.75},
        ],
    },
    {
        "family": "is_even",
        "difficulty": 1,
        "signature": "is_even(number)",
        "parameters": ["number"],
        "behavior": "return True when an integer is even and False otherwise",
        "behavior_alt": "determine whether the given integer is divisible by two",
        "contract": "Accept an integer and return a Boolean indicating whether it is even.",
        "constraints": ["Zero and negative even integers count as even."],
        "code": "def is_even(number):\n    return number % 2 == 0",
        "tests": [
            {"args": [2], "expected": True},
            {"args": [7], "expected": False},
            {"args": [0], "expected": True},
            {"args": [-4], "expected": True},
            {"args": [-3], "expected": False},
        ],
    },
    {
        "family": "get_largest",
        "difficulty": 1,
        "signature": "get_largest(a, b)",
        "parameters": ["a", "b"],
        "behavior": "return the larger of two numbers",
        "behavior_alt": "compare two numeric arguments and return the greatest value",
        "contract": "Accept two numbers and return the larger value; equal inputs return that value.",
        "constraints": ["Equal inputs must return their shared value."],
        "code": "def get_largest(a, b):\n    return max(a, b)",
        "tests": [
            {"args": [2, 9], "expected": 9},
            {"args": [-2, -7], "expected": -2},
            {"args": [4, 4], "expected": 4},
            {"args": [3.5, 3.2], "expected": 3.5},
        ],
    },
    {
        "family": "reverse_string",
        "difficulty": 1,
        "signature": "reverse_string(text)",
        "parameters": ["text"],
        "behavior": "return a string with its characters in reverse order",
        "behavior_alt": "reverse the character order of the supplied string",
        "contract": "Accept a string and return all of its characters in reverse order.",
        "constraints": ["Preserve every character, including spaces and punctuation."],
        "code": "def reverse_string(text):\n    return text[::-1]",
        "tests": [
            {"args": ["python"], "expected": "nohtyp"},
            {"args": [""], "expected": ""},
            {"args": ["a"], "expected": "a"},
            {"args": ["a b!"], "expected": "!b a"},
        ],
    },
    {
        "family": "calculate_average",
        "difficulty": 2,
        "signature": "calculate_average(numbers)",
        "parameters": ["numbers"],
        "behavior": "return the arithmetic mean of a non-empty list of numbers",
        "behavior_alt": "compute the average of all values in a non-empty numeric list",
        "contract": "Accept a non-empty list of numbers and return its arithmetic mean.",
        "constraints": ["The input list is guaranteed to contain at least one number."],
        "code": "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
        "tests": [
            {"args": [[2, 4, 6]], "expected": 4.0},
            {"args": [[5]], "expected": 5.0},
            {"args": [[-2, 2]], "expected": 0.0},
            {"args": [[1.5, 2.5, 5.0]], "expected": 3.0},
        ],
    },
    {
        "family": "count_vowels",
        "difficulty": 2,
        "signature": "count_vowels(text)",
        "parameters": ["text"],
        "behavior": "count the English vowels in a string without regard to letter case",
        "behavior_alt": "return how many a, e, i, o, or u characters occur in the text, ignoring case",
        "contract": "Count ASCII vowels a, e, i, o, and u case-insensitively.",
        "constraints": ["Accented characters are not vowels in V1."],
        "code": "def count_vowels(text):\n    return sum(1 for char in text.lower() if char in \"aeiou\")",
        "tests": [
            {"args": ["Programming"], "expected": 3},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["rhythm"], "expected": 0},
            {"args": [""], "expected": 0},
            {"args": ["café Á"], "expected": 1},
        ],
    },
    {
        "family": "get_min_max",
        "difficulty": 2,
        "signature": "get_min_max(numbers)",
        "parameters": ["numbers"],
        "behavior": "return the minimum and maximum values from a non-empty list as a tuple",
        "behavior_alt": "find both extremes in a non-empty numeric list and return (minimum, maximum)",
        "contract": "Accept a non-empty list and return a tuple containing minimum first and maximum second.",
        "constraints": ["The input is non-empty.", "Return order is (minimum, maximum)."],
        "code": "def get_min_max(numbers):\n    return min(numbers), max(numbers)",
        "tests": [
            {"args": [[7, 2, 10, 4]], "expected": [2, 10], "expected_type": "tuple"},
            {"args": [[5]], "expected": [5, 5], "expected_type": "tuple"},
            {"args": [[-8, -2, -5]], "expected": [-8, -2], "expected_type": "tuple"},
            {"args": [[1.5, 4.2, 0.5]], "expected": [0.5, 4.2], "expected_type": "tuple"},
        ],
    },
    {
        "family": "is_palindrome",
        "difficulty": 2,
        "signature": "is_palindrome(text)",
        "parameters": ["text"],
        "behavior": "return whether a string reads exactly the same forward and backward",
        "behavior_alt": "check if the supplied text is an exact palindrome",
        "contract": "Return True only when the string exactly equals its reversal.",
        "constraints": ["Comparison is case-sensitive.", "Do not remove spaces or punctuation."],
        "code": "def is_palindrome(text):\n    return text == text[::-1]",
        "tests": [
            {"args": ["radar"], "expected": True},
            {"args": ["python"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["Level"], "expected": False},
            {"args": ["nurses run"], "expected": False},
        ],
    },
    {
        "family": "get_even_numbers",
        "difficulty": 2,
        "signature": "get_even_numbers(numbers)",
        "parameters": ["numbers"],
        "behavior": "return only the even numbers from a list while preserving their order",
        "behavior_alt": "filter a list of integers so that only even values remain in original order",
        "contract": "Return a new list containing each even integer in its original order.",
        "constraints": ["Preserve order and repeated even values."],
        "code": "def get_even_numbers(numbers):\n    return [number for number in numbers if number % 2 == 0]",
        "tests": [
            {"args": [[1, 2, 3, 4]], "expected": [2, 4]},
            {"args": [[2, 2, -4, 5]], "expected": [2, 2, -4]},
            {"args": [[1, 3, 5]], "expected": []},
            {"args": [[]], "expected": []},
            {"args": [[0, -1]], "expected": [0]},
        ],
    },
    {
        "family": "count_occurrences",
        "difficulty": 3,
        "signature": "count_occurrences(items, target)",
        "parameters": ["items", "target"],
        "behavior": "return how many times a target value appears in a list",
        "behavior_alt": "count occurrences of the requested value in the supplied list",
        "contract": "Accept a list and a target value, then return the number of equal list elements.",
        "constraints": ["Return zero when the target is absent."],
        "code": "def count_occurrences(items, target):\n    return items.count(target)",
        "tests": [
            {"args": [[1, 2, 2, 3, 2], 2], "expected": 3},
            {"args": [["a", "b", "a"], "a"], "expected": 2},
            {"args": [[], 1], "expected": 0},
            {"args": [[True, False, True], True], "expected": 2},
        ],
    },
    {
        "family": "remove_duplicates",
        "difficulty": 3,
        "signature": "remove_duplicates(items)",
        "parameters": ["items"],
        "behavior": "remove duplicate list values while preserving the first occurrence order",
        "behavior_alt": "return a list of unique items in their original order",
        "contract": "Keep the first occurrence of each value and preserve input order.",
        "constraints": ["Do not reorder the items.", "Items may be unhashable, so a set is not required."],
        "code": "def remove_duplicates(items):\n    result = []\n\n    for item in items:\n        if item not in result:\n            result.append(item)\n\n    return result",
        "tests": [
            {"args": [[3, 1, 3, 2, 1]], "expected": [3, 1, 2]},
            {"args": [["a", "a", "b"]], "expected": ["a", "b"]},
            {"args": [[]], "expected": []},
            {"args": [[[1], [1], [2]]], "expected": [[1], [2]]},
            {"args": [[0, False, 1, True]], "expected": [0, 1]},
        ],
    },
    {
        "family": "word_frequency",
        "difficulty": 3,
        "signature": "word_frequency(words)",
        "parameters": ["words"],
        "behavior": "return a dictionary mapping each word in a list to its occurrence count",
        "behavior_alt": "build a frequency dictionary for the words in a list",
        "contract": "Count each exact, case-sensitive word and return a word-to-count dictionary.",
        "constraints": ["Matching is case-sensitive.", "An empty list returns an empty dictionary."],
        "code": "def word_frequency(words):\n    frequency = {}\n\n    for word in words:\n        frequency[word] = frequency.get(word, 0) + 1\n\n    return frequency",
        "tests": [
            {"args": [["python", "java", "python", "csharp"]], "expected": {"python": 2, "java": 1, "csharp": 1}},
            {"args": [[]], "expected": {}},
            {"args": [["a", "A", "a"]], "expected": {"a": 2, "A": 1}},
            {"args": [["one"]], "expected": {"one": 1}},
        ],
    },
]


TRAIN_PATTERNS = [
    "Write a Python function named `{signature}` that will {behavior}.",
    "Create `{signature}` so it can {behavior}.",
    "Implement the Python function `{signature}` to {behavior}.",
    "Define `{signature}`; it should {behavior}.",
    "Build a function called `{signature}` that must {behavior}.",
    "Complete `{signature}` so that it will {behavior}.",
    "Provide an implementation of `{signature}` which should {behavior}.",
    "Develop `{signature}` in Python to {behavior}.",
    "Write the body of `{signature}` so the function can {behavior}.",
    "Produce a Python definition for `{signature}` that will {behavior}.",
    "Implement `{signature}` with the purpose of making it {behavior}.",
    "Create the Python callable `{signature}` and have it {behavior}.",
]

TRAIN_ENDINGS = [
    " Use the exact function and parameter names shown.",
    " Return the result rather than printing it.",
    " Follow the stated contract exactly.",
    " Keep the result deterministic for the same inputs.",
    " Output only the requested function implementation.",
]

VALIDATION_PATTERNS = [
    "Implement `{signature}` in Python. Its job is to {behavior}.",
    "Can you define `{signature}` to {behavior}? Return the value instead of printing it.",
    "Supply the code for `{signature}`, a function that must {behavior}.",
    "The function `{signature}` is needed to {behavior}. Write its Python implementation.",
    "Create an exact implementation of `{signature}` whose result will {behavior}.",
    "Write `{signature}` as a Python function designed to {behavior}.",
    "Implement the requested callable `{signature}` so it will {behavior}.",
    "Provide only a function named `{signature}` that can {behavior}.",
    "Define the Python function `{signature}` according to this requirement: {contract}",
    "Turn this requirement into `{signature}`: {contract}",
]

TEST_PATTERNS = [
    "Write Python code for `{signature}`. The function needs to {behavior_alt}.",
    "How would you implement `{signature}` so it can {behavior_alt}?",
    "Produce the definition of `{signature}` for this behavior: {contract}",
    "Implement a function with the signature `{signature}`. Requirement: {contract}",
    "Create `{signature}` and make sure it can {behavior_alt}.",
    "Return an implementation of `{signature}` which will {behavior_alt}.",
    "Code `{signature}` in Python under this contract: {contract}",
    "Provide a working `{signature}` function. It must {behavior_alt}.",
    "Use the exact signature `{signature}` to solve this task: {contract}",
    "Define `{signature}` as a deterministic Python function that will {behavior_alt}.",
]


def build_instructions(family: dict) -> dict[str, list[str]]:
    values = {
        "signature": family["signature"],
        "behavior": family["behavior"],
        "behavior_alt": family["behavior_alt"],
        "contract": family["contract"],
    }
    train = [
        pattern.format(**values) + ending
        for pattern in TRAIN_PATTERNS
        for ending in TRAIN_ENDINGS
    ]
    validation = [pattern.format(**values) for pattern in VALIDATION_PATTERNS]
    test = [pattern.format(**values) for pattern in TEST_PATTERNS]
    return {"train": train, "validation": validation, "test": test}


def make_record(family: dict, split: str, index: int, instruction: str) -> dict:
    prompt = f"### Task:\n{instruction}\n\n### Code:\n"
    code = family["code"]
    return {
        "id": f"{split}-{family['family']}-{index:03d}",
        "family": family["family"],
        "difficulty": family["difficulty"],
        "instruction": instruction,
        "prompt": prompt,
        "code": code,
        "text": prompt + code,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    global DATA_DIR, CONTRACTS_DIR
    DATA_DIR = args.output_root.resolve() / "data"
    CONTRACTS_DIR = args.output_root.resolve() / "contracts"
    targets = [DATA_DIR / f"{split}.jsonl" for split in ("train", "validation", "test")]
    targets += [DATA_DIR / "manifest.json", CONTRACTS_DIR / "families.json"]
    for target in targets:
        if target.exists():
            parser.error(f"Arquivo já existe; use --output-root com outro diretório: {target}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": [], "validation": [], "test": []}
    for family in FAMILIES:
        instructions = build_instructions(family)
        for split, items in instructions.items():
            split_rows[split].extend(
                make_record(family, split, index, instruction)
                for index, instruction in enumerate(items, start=1)
            )

    family_order = {family["family"]: index for index, family in enumerate(FAMILIES)}
    for rows in split_rows.values():
        rows.sort(key=lambda row: (int(row["id"].rsplit("-", 1)[1]), family_order[row["family"]]))

    for split, rows in split_rows.items():
        write_jsonl(DATA_DIR / f"{split}.jsonl", rows)

    contracts = {
        "dataset": "Mini Code LLM Dataset V1",
        "version": "1.0.0",
        "prompt_format": "### Task:\\n{instruction}\\n\\n### Code:\\n",
        "families": [
            {
                key: value
                for key, value in family.items()
                if key not in {"behavior", "behavior_alt"}
            }
            for family in FAMILIES
        ],
    }
    contracts_path = CONTRACTS_DIR / "families.json"
    contracts_path.write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset": "Mini Code LLM Dataset V1",
        "version": "1.0.0",
        "records": {split: len(rows) for split, rows in split_rows.items()},
        "total_records": sum(len(rows) for rows in split_rows.values()),
        "per_family": {"train": 60, "validation": 10, "test": 10},
        "sha256": {
            f"{split}.jsonl": sha256(DATA_DIR / f"{split}.jsonl")
            for split in split_rows
        },
        "contracts_sha256": sha256(contracts_path),
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
