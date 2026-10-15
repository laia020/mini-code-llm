"""Full supervised fine-tuning of pretrained SmolLM2."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
from pathlib import Path

if __package__:
    from .common import BASE_MODEL, ROOT, read_jsonl
    from .inference import select_device
else:
    from common import BASE_MODEL, ROOT, read_jsonl
    from inference import select_device

PRESETS = {
    "experiment-1": {"learning_rate": 5e-5, "batch_size": 2, "epochs": 5},
    "experiment-2": {"learning_rate": 1e-5, "batch_size": 1, "epochs": 3},
    "experiment-3": {"learning_rate": 1e-4, "batch_size": 2, "epochs": 5},
}


def prepare_example(example, tokenizer, max_length=256):
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer sem EOS.")
    prompt = tokenizer.encode(example["prompt"], add_special_tokens=False)
    code = tokenizer.encode(example["code"], add_special_tokens=False)
    input_ids = prompt + code + [tokenizer.eos_token_id]
    if len(input_ids) > max_length:
        raise ValueError(f"Exemplo {example.get('id', '?')} tem {len(input_ids)} tokens; limite {max_length}.")
    return {"input_ids": input_ids, "labels": [-100] * len(prompt) + code + [tokenizer.eos_token_id]}


class CodeDataset:
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = [prepare_example(item, tokenizer, max_length) for item in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id):
    import torch
    length = max(len(item["input_ids"]) for item in batch)
    tensors = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        padding = length - len(item["input_ids"])
        tensors["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        tensors["labels"].append(item["labels"] + [-100] * padding)
        tensors["attention_mask"].append([1] * len(item["input_ids"]) + [0] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in tensors.items()}


def validation_loss(model, loader, device):
    import torch
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            losses.append(model(**{k: v.to(device) for k, v in batch.items()}).loss.item())
    model.train()
    # Original protocol: unweighted mean of batch losses.
    return sum(losses) / len(losses) if losses else math.inf


def train(args):
    import torch
    from functools import partial
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Diretório já existe; escolha outro --output-dir: {output}")
    preset = PRESETS[args.experiment].copy()
    for key in preset:
        if getattr(args, key) is not None:
            preset[key] = getattr(args, key)
    if preset["epochs"] < 1 or preset["batch_size"] < 1 or preset["learning_rate"] <= 0:
        raise ValueError("Epochs, batch e learning rate precisam ser positivos.")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps precisa ser positivo.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = read_jsonl(args.train_file)
    validation = read_jsonl(args.validation_file)
    if not examples or not validation:
        raise ValueError("Train e validation não podem estar vazios.")
    train_dataset = CodeDataset(examples, tokenizer, args.max_length)
    val_dataset = CodeDataset(validation, tokenizer, args.max_length)
    collate = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=preset["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=preset["batch_size"], shuffle=False, collate_fn=collate)
    model = AutoModelForCausalLM.from_pretrained(args.model, token=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=preset["learning_rate"])
    metadata = {
        "model": args.model, "experiment": args.experiment, **preset,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "data_sha256": {"train": hashlib.sha256(args.train_file.read_bytes()).hexdigest(), "validation": hashlib.sha256(args.validation_file.read_bytes()).hexdigest()},
        "seed": args.seed, "device": device, "max_length": args.max_length,
        "gradient_clip_max_norm": 1.0, "optimizer": "AdamW", "weight_decay": 0.01,
        "train_file": str(args.train_file), "validation_file": str(args.validation_file),
        "train_examples": len(examples), "validation_examples": len(validation),
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "transformers", "huggingface_hub")},
        "python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "max_steps": args.max_steps, "smoke_run": args.max_steps is not None,
        "note": "Nova execução; não reconstrói seed ou ambiente dos experimentos históricos.",
        "history": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    best, steps = math.inf, 0
    for epoch in range(1, preset["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita; execução interrompida.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            steps += 1
            if steps % 25 == 0:
                print(f"Epoch {epoch} | Step {steps} | Loss {loss.item():.4f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        val_loss = validation_loss(model, val_loader, device)
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss não finita.")
        metadata["history"].append({"epoch": epoch, "steps": steps, "train_loss": sum(losses) / len(losses), "validation_loss": val_loss})
        print(f"Epoch {epoch}: train={sum(losses)/len(losses):.6f}, validation={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            metadata["best_epoch"] = epoch
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
        metadata["best_validation_loss"] = best
        (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.max_steps and steps >= args.max_steps:
            break
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(PRESETS), default="experiment-3")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=ROOT / "data/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, help="Smoke run only; not a complete experiment.")
    args = parser.parse_args(argv)
    try:
        train(args)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Falha no treinamento: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
