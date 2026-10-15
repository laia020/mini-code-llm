"""Compatibility wrapper around the shared evaluator."""
import sys
if __package__:
    from .evaluate import main
else:
    from evaluate import main

if __name__ == "__main__":
    raise SystemExit(main(['--input', 'results/final_test.jsonl', '--output', 'results/reproductions/final_test_evaluated.jsonl', '--summary', 'results/reproductions/final_test_summary.json'] + sys.argv[1:]))
