"""Compatibility wrapper; new outputs go to results/reproductions/."""
import sys
if __package__:
    from .generate import main
else:
    from generate import main

if __name__ == "__main__":
    raise SystemExit(main(['--model', 'laia020/mini-code-llm', '--input', 'data/test.jsonl', '--output', 'results/reproductions/final_test.jsonl', '--subfolder', 'experiment-3'] + sys.argv[1:]))
