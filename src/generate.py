"""Generate raw benchmark responses without changing historical results."""

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .common import check_new_outputs, read_jsonl, write_json
    from .inference import CodeGenerator, add_model_arguments
else:
    from common import check_new_outputs, read_jsonl, write_json
    from inference import CodeGenerator, add_model_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = args.output.with_suffix(".metadata.json")
    try:
        check_new_outputs(args.output, metadata, inputs=[args.input])
        examples = read_jsonl(args.input)
        generator = CodeGenerator(args.model, args.subfolder, args.revision, args.device).load()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            for index, example in enumerate(examples, 1):
                generated = generator.generate_prompt(example["prompt"], args.max_new_tokens)
                result = {key: example[key] for key in ("id", "family", "difficulty", "instruction")}
                result.update(expected_code=example["code"], generated=generated)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(examples)}] {example['family']}")
        write_json(metadata, {
            "model": args.model, "subfolder": generator.subfolder, "revision": generator.revision,
            "device": generator.device, "max_new_tokens": args.max_new_tokens, "do_sample": False,
            "input": str(args.input), "examples": len(examples),
            "note": "Reprodução; não substitui os resultados históricos.",
        })
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Falha na geração: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
