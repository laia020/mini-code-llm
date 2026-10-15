"""Compatibility wrapper; new outputs go to results/reproductions/."""
import sys
if __package__:
    from .generate import main
else:
    from generate import main

if __name__ == "__main__":
    raise SystemExit(main(['--model', 'HuggingFaceTB/SmolLM2-135M', '--input', 'data/validation.jsonl', '--output', 'results/reproductions/baseline_validation.jsonl'] + sys.argv[1:]))
