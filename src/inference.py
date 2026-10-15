"""CPU/CUDA inference shared by the command line, batch generator and Gradio."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .common import MODEL_ID, MODEL_REVISION, build_prompt
else:
    from common import MODEL_ID, MODEL_REVISION, build_prompt


def select_device(requested="auto", cuda_available=None):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Dispositivo deve ser auto, cpu ou cuda.")
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA solicitada, mas indisponível neste ambiente.")
    return ("cuda" if cuda_available else "cpu") if requested == "auto" else requested


def clean_output(text):
    # Presentation only: evaluation always uses the unmodified raw response.
    if __package__:
        from .evaluate import extract_python_code
    else:
        from evaluate import extract_python_code
    candidate = extract_python_code(text)
    # A prose label such as "Explanation: text" can parse as a Python annotation.
    # For display, retain only the first generated function; do not alter the evaluator.
    import ast
    try:
        tree = ast.parse(candidate)
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if function is not None:
            start = min([function.lineno] + [node.lineno for node in function.decorator_list]) - 1
            return "\n".join(candidate.splitlines()[start:function.end_lineno]).strip()
    except SyntaxError:
        pass
    return candidate


class CodeGenerator:
    def __init__(self, model=MODEL_ID, subfolder=None, revision=None, device="auto"):
        self.model_id = model
        self.subfolder = ("experiment-3" if model == MODEL_ID else None) if subfolder is None else subfolder
        self.revision = revision or (MODEL_REVISION if model == MODEL_ID else None)
        self.requested_device = device
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        if self.model is not None:
            return self
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = select_device(self.requested_device)
        kwargs = {"token": False}
        if self.subfolder:
            kwargs["subfolder"] = self.subfolder
        if self.revision and not Path(self.model_id).is_dir():
            kwargs["revision"] = self.revision
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        if tokenizer.eos_token_id is None:
            raise ValueError("O tokenizer precisa definir EOS.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model.to(self.device)
        model.eval()
        self.tokenizer, self.model = tokenizer, model
        return self

    def generate_prompt(self, prompt, max_new_tokens=128):
        if not prompt.strip():
            raise ValueError("Prompt vazio.")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens deve ser um inteiro positivo.")
        self.load()
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", self.tokenizer.model_max_length)
        if input_length + max_new_tokens > limit:
            raise ValueError(f"Instrução longa demais: entrada e saída devem caber em {limit} tokens.")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)

    def generate(self, instruction, max_new_tokens=128):
        raw = self.generate_prompt(build_prompt(instruction), max_new_tokens)
        return clean_output(raw)


def add_model_arguments(parser):
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    try:
        generator = CodeGenerator(args.model, args.subfolder, args.revision, args.device)
        print(generator.generate(args.instruction, args.max_new_tokens))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Falha na inferência: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
