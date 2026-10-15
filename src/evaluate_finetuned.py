"""Compatibility wrapper around the shared evaluator."""
import sys
if __package__:
    from .evaluate import main
else:
    from evaluate import main

if __name__ == "__main__":
    raise SystemExit(main(['--input', 'results/finetuned_validation.jsonl', '--output', 'results/reproductions/finetuned_validation_evaluated.jsonl', '--summary', 'results/reproductions/finetuned_validation_summary.json'] + sys.argv[1:]))
