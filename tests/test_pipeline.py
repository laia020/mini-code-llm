"""Regression coverage without downloading a model."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import build_prompt, check_new_outputs
from src.evaluate import evaluate_records, extract_python_code, run_contract_tests
from src.inference import CodeGenerator, select_device
from src.training import collate_batch, prepare_example


class TinyTokenizer:
    eos_token_id = 9
    pad_token_id = 9

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class DataTests(unittest.TestCase):
    def test_prompt_format(self):
        self.assertEqual(build_prompt("  Do something  "), "### Task:\nDo something\n\n### Code:\n")

    def test_empty_instruction(self):
        with self.assertRaises(ValueError):
            build_prompt("  ")

    def test_labels_mask_only_prompt(self):
        item = prepare_example({"prompt": "AB", "code": "C"}, TinyTokenizer())
        self.assertEqual(item["input_ids"], [65, 66, 67, 9])
        self.assertEqual(item["labels"], [-100, -100, 67, 9])

    def test_eos_survives_exact_limit(self):
        item = prepare_example({"prompt": "AB", "code": "C"}, TinyTokenizer(), 4)
        self.assertEqual(item["labels"][-1], 9)

    def test_long_example_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_example({"prompt": "AB", "code": "CD"}, TinyTokenizer(), 4)

    def test_missing_eos_is_rejected(self):
        tokenizer = TinyTokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaises(ValueError):
            prepare_example({"prompt": "A", "code": "B"}, tokenizer)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_padding_masks_eos_only_when_padding(self):
        batch = collate_batch([
            {"input_ids": [1, 2, 9], "labels": [-100, 2, 9]},
            {"input_ids": [1, 9], "labels": [-100, 9]},
        ], 9)
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 9], [-100, 9, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_existing_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("original")
            with self.assertRaises(FileExistsError):
                check_new_outputs(path)
            self.assertEqual(path.read_text(), "original")

    def test_input_cannot_be_output(self):
        with self.assertRaises(ValueError):
            check_new_outputs("same", inputs=["same"])

    def test_output_paths_must_differ(self):
        with self.assertRaises(ValueError):
            check_new_outputs("same", "same")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = {c["family"]: c for c in json.loads((ROOT / "contracts/families.json").read_text(encoding="utf-8"))["families"]}

    def evaluate_code(self, code, family="is_even"):
        contract = self.contracts[family]
        records, summary = evaluate_records([{
            "id": "unit", "family": family, "difficulty": contract["difficulty"],
            "instruction": "unit", "expected_code": contract["code"], "generated": code,
        }], self.contracts)
        return records[0], summary

    def test_alternative_implementation_passes_without_exact_match(self):
        item, _ = self.evaluate_code("def is_even(number):\n    if number % 2 == 0:\n        return True\n    return False")
        self.assertTrue(item["task_passed"])
        self.assertFalse(item["exact_match"])

    def test_additional_prose_is_measured(self):
        item, _ = self.evaluate_code("```python\ndef is_even(number):\n    return number % 2 == 0\n```\nExplanation: example")
        self.assertTrue(item["task_passed"])
        self.assertFalse(item["code_only_output"])

    def test_incorrect_signature(self):
        item, _ = self.evaluate_code("def is_even(value):\n    return value % 2 == 0")
        self.assertFalse(item["correct_signature"])
        self.assertFalse(item["task_passed"])

    def test_invalid_python(self):
        item, _ = self.evaluate_code("def is_even(")
        self.assertFalse(item["valid_python"])

    def test_bad_behavior(self):
        item, _ = self.evaluate_code("def is_even(number):\n    return True")
        self.assertFalse(item["task_passed"])

    def test_disallowed_import(self):
        item, _ = self.evaluate_code("def is_even(number):\n    import os\n    return True")
        self.assertFalse(item["safe_to_execute"])

    def test_worker_timeout(self):
        code = "def is_even(number):\n    for i in range(100000000000):\n        number += i\n    return True"
        passed, total, details = run_contract_tests(code, self.contracts["is_even"], timeout=0.2)
        self.assertEqual(passed, 0)
        self.assertEqual(total, 5)
        self.assertIn("Timeout", details[0])

    def test_tuple_return_contract(self):
        item, _ = self.evaluate_code(self.contracts["get_min_max"]["code"], "get_min_max")
        self.assertTrue(item["task_passed"])

    def test_extraction_preserves_code(self):
        self.assertEqual(extract_python_code("def f():\n    return 1\n\nExplanation follows."), "def f():\n    return 1")

    def test_annotation_like_prose_is_only_removed_for_display(self):
        from src.inference import clean_output
        raw = "def f():\n    return 1\n\nExplanation: hello"
        self.assertEqual(extract_python_code(raw), raw)
        self.assertEqual(clean_output(raw), "def f():\n    return 1")


class InferenceTests(unittest.TestCase):
    def test_auto_device(self):
        self.assertEqual(select_device("auto", True), "cuda")
        self.assertEqual(select_device("auto", False), "cpu")
        self.assertEqual(select_device("cpu", True), "cpu")

    def test_unavailable_cuda(self):
        with self.assertRaises(RuntimeError):
            select_device("cuda", False)

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            select_device("tpu", False)

    def test_cleaning_is_presentation_only(self):
        generator = CodeGenerator()
        raw = "```python\ndef f():\n    return 1\n```\nExplanation"
        with patch.object(generator, "generate_prompt", return_value=raw):
            self.assertEqual(generator.generate("Create f"), "def f():\n    return 1")
            self.assertEqual(generator.generate_prompt("formatted"), raw)

    def test_local_model_has_no_hub_subfolder(self):
        generator = CodeGenerator("models/example")
        self.assertIsNone(generator.subfolder)
        self.assertIsNone(generator.revision)

    def test_imports_are_inert_without_ml_packages(self):
        code = '''
import importlib, pathlib, sys
class BlockML:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'gradio', 'huggingface_hub'}:
            raise AssertionError('Heavy import during module import: ' + fullname)
sys.meta_path.insert(0, BlockML())
for p in pathlib.Path('src').glob('*.py'):
    importlib.import_module('src.' + p.stem)
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_generated_tokens_exclude_prompt(self):
        import torch
        from types import SimpleNamespace
        class Inputs(dict):
            def to(self, device):
                return self
        class Tokenizer:
            pad_token_id, eos_token_id, model_max_length = 0, 9, 128
            def __call__(self, prompt, **kwargs):
                return Inputs(input_ids=torch.tensor([[11, 12]]))
            def decode(self, ids, **kwargs):
                return str(ids.tolist())
        class Model:
            config = SimpleNamespace(max_position_embeddings=128)
            def generate(self, **kwargs):
                assert kwargs['do_sample'] is False
                return torch.tensor([[11, 12, 40, 41]])
        generator = CodeGenerator()
        generator.model, generator.tokenizer, generator.device = Model(), Tokenizer(), "cpu"
        self.assertEqual(generator.generate_prompt("prompt", 4), "[40, 41]")
        with self.assertRaises(ValueError):
            generator.generate_prompt("prompt", 128)


class CommandProtectionTests(unittest.TestCase):
    def test_builder_refuses_frozen_dataset(self):
        result = subprocess.run([sys.executable, "scripts/build_dataset.py"], cwd=ROOT, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"--output-root", result.stderr)

    def test_generator_refuses_existing_output_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.jsonl"
            path.write_text("original")
            result = subprocess.run([sys.executable, "src/generate.py", "--input", "missing.jsonl", "--output", str(path)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), "original")

    def test_evaluator_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.jsonl"
            path.write_text("original")
            result = subprocess.run([sys.executable, "src/evaluate.py", "--input", "results/final_test.jsonl", "--output", str(path), "--summary", str(Path(folder) / "summary.json")], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), "original")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_training_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "src/train.py", "--output-dir", folder], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
