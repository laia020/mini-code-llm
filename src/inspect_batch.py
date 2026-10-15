"""Educational demonstration; run explicitly from the project root."""

def main():
    import json
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer


    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    TRAIN_FILE = Path("data/train.jsonl")

    MAX_LENGTH = 256
    BATCH_SIZE = 4


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Alguns tokenizers de modelos causais não possuem PAD separado.
    # Nesse caso usamos EOS como valor de preenchimento.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        examples = [
            json.loads(line)
            for line in file
            if line.strip()
        ]


    # ---------------------------------------------------------
    # Convert one example into input_ids + labels
    # ---------------------------------------------------------

    def prepare_example(example):

        prompt_ids = tokenizer.encode(
            example["prompt"],
            add_special_tokens=False,
        )

        code_ids = tokenizer.encode(
            example["code"],
            add_special_tokens=False,
        )

        input_ids = (
            prompt_ids
            + code_ids
            + [tokenizer.eos_token_id]
        )

        labels = (
            [-100] * len(prompt_ids)
            + code_ids
            + [tokenizer.eos_token_id]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


    # ---------------------------------------------------------
    # Check lengths of ALL training examples
    # ---------------------------------------------------------

    prepared_examples = [
        prepare_example(example)
        for example in examples
    ]

    lengths = [
        len(example["input_ids"])
        for example in prepared_examples
    ]


    print("=== DATASET LENGTHS ===")
    print(f"Training examples: {len(lengths)}")
    print(f"Shortest example: {min(lengths)} tokens")
    print(f"Longest example: {max(lengths)} tokens")
    print(
        f"Average length: "
        f"{sum(lengths) / len(lengths):.1f} tokens"
    )


    too_long = [
        length
        for length in lengths
        if length > MAX_LENGTH
    ]

    print(
        f"Examples above {MAX_LENGTH}: "
        f"{len(too_long)}"
    )


    if too_long:
        raise ValueError(
            "Some examples are longer than MAX_LENGTH."
        )


    # ---------------------------------------------------------
    # Create a small batch
    # ---------------------------------------------------------

    batch_examples = prepared_examples[:BATCH_SIZE]

    batch_max_length = max(
        len(example["input_ids"])
        for example in batch_examples
    )


    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []


    for example in batch_examples:

        input_ids = example["input_ids"]
        labels = example["labels"]

        padding_length = (
            batch_max_length
            - len(input_ids)
        )


        padded_input_ids = (
            input_ids
            + [tokenizer.pad_token_id] * padding_length
        )


        padded_labels = (
            labels
            + [-100] * padding_length
        )


        attention_mask = (
            [1] * len(input_ids)
            + [0] * padding_length
        )


        batch_input_ids.append(
            padded_input_ids
        )

        batch_labels.append(
            padded_labels
        )

        batch_attention_mask.append(
            attention_mask
        )


    # ---------------------------------------------------------
    # Convert lists into PyTorch tensors
    # ---------------------------------------------------------

    batch_input_ids = torch.tensor(
        batch_input_ids,
        dtype=torch.long,
    )

    batch_labels = torch.tensor(
        batch_labels,
        dtype=torch.long,
    )

    batch_attention_mask = torch.tensor(
        batch_attention_mask,
        dtype=torch.long,
    )


    # ---------------------------------------------------------
    # Inspect batch
    # ---------------------------------------------------------

    print("\n=== BATCH ===")

    print(
        f"input_ids shape: "
        f"{batch_input_ids.shape}"
    )

    print(
        f"labels shape: "
        f"{batch_labels.shape}"
    )

    print(
        f"attention_mask shape: "
        f"{batch_attention_mask.shape}"
    )


    print("\n--- FIRST ATTENTION MASK ---")
    print(
        batch_attention_mask[0]
    )


    print("\n--- FIRST LABELS ---")
    print(
        batch_labels[0]
    )


if __name__ == "__main__":
    main()
