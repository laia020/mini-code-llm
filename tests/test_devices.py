"""Offline device coverage and tiny local CPU/MPS training round trips."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.inference import CodeGenerator, select_device


class DeviceTests(unittest.TestCase):
    def test_historical_windows_paths_are_portable(self):
        from scripts.audit_results import ROOT, historical_input_path
        expected = ROOT / 'results' / 'baseline_validation.jsonl'
        self.assertEqual(historical_input_path(r'results\baseline_validation.jsonl'), expected)
        self.assertEqual(historical_input_path('results/baseline_validation.jsonl'), expected)

    def test_cli_device_options(self):
        root = Path(__file__).resolve().parents[1]
        for script in ('src/train.py', 'src/inference.py', 'src/generate.py',
                       'src/baseline_single.py', 'src/inspect_loss.py',
                       'src/app.py', 'scripts/smoke_test.py', 'iniciar.py'):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, script, '--help'], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('mps', result.stdout)

    def test_priority_matrix(self):
        for cuda in (False, True):
            for mps in (False, True):
                with self.subTest(cuda=cuda, mps=mps):
                    self.assertEqual(select_device('auto', cuda, mps),
                                     'cuda' if cuda else 'mps' if mps else 'cpu')
                    self.assertEqual(select_device('cpu', cuda, mps), 'cpu')
        self.assertEqual(select_device('mps', True, True), 'mps')
        self.assertEqual(select_device('cuda', True, True), 'cuda')

    def test_explicit_unavailable(self):
        for device in ('cuda', 'mps'):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                select_device(device, False, False)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_backend_detection(self):
        import torch
        with patch.object(torch.cuda, 'is_available', return_value=False), patch.object(torch.backends.mps, 'is_available', return_value=True):
            self.assertEqual(select_device(), 'mps')
        with patch.object(torch.cuda, 'is_available', return_value=False), patch.object(torch.backends, 'mps', None):
            self.assertEqual(select_device(), 'cpu')


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'requires transformers')
class TinyModelTests(unittest.TestCase):
    def round_trip(self, device):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from src.training import train
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            backend = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, '[PAD]': 2, 'task': 3, 'code': 4}, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]', pad_token='[PAD]', model_max_length=32)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32, eos_token_id=1, pad_token_id=2))
            initial_weights = model.model.embed_tokens.weight.detach().clone()
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            data = root / 'data.jsonl'
            data.write_text(json.dumps({'prompt': 'task', 'code': 'code'}) + '\n')
            metadata = train(SimpleNamespace(output_dir=root / 'checkpoint', experiment='experiment-3', model=str(source), train_file=data, validation_file=data, learning_rate=None, batch_size=1, epochs=1, max_length=32, seed=42, device=device, max_steps=1))
            self.assertEqual(metadata['device'], device)
            self.assertTrue(torch.isfinite(torch.tensor(metadata['best_validation_loss'])))
            generator = CodeGenerator(str(root / 'checkpoint'), device=device).load()
            self.assertFalse(torch.equal(initial_weights, generator.model.model.embed_tokens.weight.detach().cpu()))
            self.assertEqual(next(generator.model.parameters()).dtype, torch.float32)
            self.assertEqual(next(generator.model.parameters()).device.type, device)
            self.assertIsInstance(generator.generate_prompt('task', 2), str)
            cpu = CodeGenerator(str(root / 'checkpoint'), device='cpu').load()
            self.assertEqual(next(cpu.model.parameters()).device.type, 'cpu')
            self.assertIsInstance(cpu.generate_prompt('task', 2), str)

    def test_cpu_round_trip(self):
        self.round_trip('cpu')

    def test_mps_round_trip(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable in this process')
        self.round_trip('mps')
