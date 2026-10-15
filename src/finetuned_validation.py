"""Compatibility wrapper; new outputs go to results/reproductions/."""
import sys
if __package__:
    from .generate import main
else:
    from generate import main

if __name__ == "__main__":
    raise SystemExit(main(['--model', 'laia020/mini-code-llm', '--input', 'data/validation.jsonl', '--output', 'results/reproductions/finetuned_validation.jsonl', '--subfolder', 'experiment-1'] + sys.argv[1:]))
