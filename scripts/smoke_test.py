"""Explicit online integration check: real Hub inference, Gradio and one optimizer step."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import MODEL_ID, MODEL_REVISION, build_prompt, check_new_outputs, read_jsonl, write_json
from src.evaluate import evaluate_records
from src.inference import CodeGenerator


def run(output, skip_training=False):
    check_new_outputs(output)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import torch
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from src.app import create_app
    from gradio_client import Client
    generator = CodeGenerator(device="cpu")
    contracts = {c["family"]: c for c in json.loads((ROOT / "contracts/families.json").read_text(encoding="utf-8"))["families"]}
    prompts = {
        "add_numbers": "Create a Python function called add_numbers(a, b) that returns their sum.",
        "count_vowels": "Create count_vowels(text) that counts vowels case-insensitively.",
        "remove_duplicates": "Create remove_duplicates(items) that removes duplicates while preserving order.",
    }
    records = []
    for family, instruction in prompts.items():
        raw = generator.generate_prompt(build_prompt(instruction))
        records.append({"id": f"smoke-{family}", "family": family, "difficulty": contracts[family]["difficulty"], "instruction": instruction, "expected_code": contracts[family]["code"], "generated": raw})
        print(f"Generated: {family}", flush=True)
    evaluated, summary = evaluate_records(records, contracts)
    assert summary["tasks_passed"]["count"] == 3, evaluated
    assert summary["code_only_output"]["count"] == 3, evaluated
    report = {
        "status": "PASS", "platform": platform.system(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "gradio", "huggingface_hub"]},
        "model": MODEL_ID, "revision": MODEL_REVISION, "subfolder": "experiment-3",
        "anonymous_hub_access": True, "cpu_inference": "PASS", "cuda_available": torch.cuda.is_available(),
        "smoke_examples": evaluated, "smoke_summary": summary,
    }
    if torch.cuda.is_available():
        cuda_generator = CodeGenerator(device="cuda")
        cuda_records = []
        for record in records:
            cuda_records.append({**record, "generated": cuda_generator.generate_prompt(build_prompt(record["instruction"]))})
        _, cuda_summary = evaluate_records(cuda_records, contracts)
        assert cuda_summary["tasks_passed"]["count"] == 3
        report["cuda_inference"] = "PASS"
        del cuda_generator
    else:
        report["cuda_inference"] = "NOT_AVAILABLE"

    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    demo = create_app(generator)
    try:
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True, quiet=True)
        with urllib.request.urlopen(url, timeout=15) as response:
            assert response.status == 200
        client = Client(url, verbose=False)
        code, status = client.predict(prompts["add_numbers"], api_name="/generate")
        item, _ = evaluate_records([{**records[0], "generated": code}], contracts)
        assert item[0]["task_passed"], code
        cleared = client.predict(api_name="/clear")
        assert cleared[0] == "" and cleared[1] == ""
        try:
            client.predict(" ", api_name="/generate")
        except Exception as error:
            assert "instru" in str(error).lower(), str(error)
        else:
            raise AssertionError("Empty instruction did not produce a user-facing error")
        report["gradio"] = {"http": 200, "generation": "PASS", "clear": "PASS", "empty_input_error": "PASS", "shutdown": "PASS"}
        print("PASS: Gradio HTTP, generation, clear and error handling", flush=True)
    finally:
        demo.close()

    if not skip_training:
        from src.training import train
        config = hf_hub_download(MODEL_ID, "config.json", subfolder="experiment-3", revision=MODEL_REVISION, token=False)
        local_model = str(Path(config).parent)
        with tempfile.TemporaryDirectory(prefix="mini-code-llm-smoke-") as folder:
            root = Path(folder)
            for split in ("train", "validation"):
                rows = read_jsonl(ROOT / "data" / f"{split}.jsonl")[:2]
                (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metadata = train(SimpleNamespace(
                output_dir=root / "checkpoint", experiment="experiment-3", model=local_model,
                train_file=root / "train.jsonl", validation_file=root / "validation.jsonl",
                learning_rate=None, batch_size=2, epochs=1, max_length=256,
                seed=42, device="cpu", max_steps=1,
            ))
            assert (root / "checkpoint/model.safetensors").is_file()
            # Reload what the training loop saved, including its tokenizer.
            reloaded = CodeGenerator(str(root / "checkpoint"), device="cpu").load()
            assert reloaded.generate(prompts["add_numbers"])
            report["training_smoke"] = {
                "status": "PASS", "optimizer_steps": 1, "checkpoint_reload": "PASS",
                "start_model": "Published experiment-3, for pipeline verification only",
                "history": metadata["history"], "temporary_checkpoint_deleted": True,
            }
            del reloaded
        print("PASS: forward, backward, optimizer, checkpoint save and reload", flush=True)
    else:
        report["training_smoke"] = {"status": "SKIPPED"}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    run(args.output, args.skip_training)


if __name__ == "__main__":
    main()
