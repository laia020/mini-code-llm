"""Educational demonstration; run explicitly from the project root."""

def main():
    import json
    from pathlib import Path

    from transformers import AutoTokenizer


    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    TRAIN_FILE = Path("data/train.jsonl")

    MAX_LENGTH = 256


    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    # ---------------------------------------------------------
    # Load one training example
    # ---------------------------------------------------------

    with TRAIN_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        example = json.loads(
            next(line for line in file if line.strip())
        )


    prompt = example["prompt"]
    code = example["code"]


    # ---------------------------------------------------------
    # Tokenize prompt and target separately
    # ---------------------------------------------------------

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    code_ids = tokenizer.encode(
        code,
        add_special_tokens=False
    )


    # ---------------------------------------------------------
    # Add EOS
    # ---------------------------------------------------------

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer does not have an EOS token."
        )


    input_ids = (
        prompt_ids
        + code_ids
        + [tokenizer.eos_token_id]
    )


    # ---------------------------------------------------------
    # Create labels
    # ---------------------------------------------------------

    labels = (
        [-100] * len(prompt_ids)
        + code_ids
        + [tokenizer.eos_token_id]
    )


    # ---------------------------------------------------------
    # Length check
    # ---------------------------------------------------------

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Example has {len(input_ids)} tokens, "
            f"but MAX_LENGTH is {MAX_LENGTH}."
        )


    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    print("\n=== TRAINING EXAMPLE ===")

    print(f"\nID: {example['id']}")
    print(f"Family: {example['family']}")
    print(f"Difficulty: {example['difficulty']}")


    print("\n--- PROMPT ---")
    print(prompt)


    print("\n--- TARGET CODE ---")
    print(code)


    print("\n--- TOKEN COUNTS ---")

    print(
        f"Prompt tokens: {len(prompt_ids)}"
    )

    print(
        f"Code tokens: {len(code_ids)}"
    )

    print(
        f"Total tokens (+ EOS): {len(input_ids)}"
    )


    supervised_tokens = sum(
        label != -100
        for label in labels
    )

    ignored_tokens = sum(
        label == -100
        for label in labels
    )


    print(
        f"Tokens used in loss: {supervised_tokens}"
    )

    print(
        f"Tokens ignored in loss: {ignored_tokens}"
    )


    print("\n--- INPUT IDS ---")

    print(input_ids)


    print("\n--- LABELS ---")

    print(labels)


    print("\n--- FULL INPUT DECODED ---")

    print(
        tokenizer.decode(
            input_ids,
            skip_special_tokens=False
        )
    )


    print("\n--- TOKENS THAT CONTRIBUTE TO LOSS ---")

    target_ids = [
        label
        for label in labels
        if label != -100
    ]

    print(
        tokenizer.decode(
            target_ids,
            skip_special_tokens=False
        )
    )


if __name__ == "__main__":
    main()
