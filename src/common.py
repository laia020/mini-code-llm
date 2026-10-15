"""Shared paths and conservative output handling (standard library only)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "laia020/mini-code-llm"
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "044a7de4030020a593eedb2417d11aecd38e5268"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(instruction):
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Digite uma instrução antes de gerar código.")
    return f"### Task:\n{instruction}\n\n### Code:\n"


def check_new_outputs(*paths, inputs=()):
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Os caminhos de saída devem ser diferentes.")
    for path in resolved:
        if path in {Path(item).resolve() for item in inputs}:
            raise ValueError(f"A saída não pode substituir a entrada: {path}")
        if path.exists():
            raise FileExistsError(f"Saída já existe; escolha outro caminho: {path}")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
