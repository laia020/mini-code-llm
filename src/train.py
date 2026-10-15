"""CLI entry point for full supervised fine-tuning."""
if __package__:
    from .training import main
else:
    from training import main

if __name__ == "__main__":
    raise SystemExit(main())
