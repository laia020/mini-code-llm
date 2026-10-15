"""Educational demonstration; run explicitly from the project root."""

def main():
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    TRAIN_FILE = Path("data/train.jsonl")

    BATCH_SIZE = 4
    MAX_LENGTH = 256


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")


    # ---------------------------------------------------------
    # Load tokenizer and model
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)

    # Agora estamos simulando treinamento, não inferência.
    model.train()


    # ---------------------------------------------------------
    # Load training examples
    # ---------------------------------------------------------

    with TRAIN_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        examples = [
            json.loads(line)
            for line in file
            if line.strip()
        ]


    # ---------------------------------------------------------
    # Prepare one example
    # ---------------------------------------------------------

    def prepare_example(example):

        prompt_ids = tokenizer.encode(
            example["prompt"],
            add_special_tokens=False
        )

        code_ids = tokenizer.encode(
            example["code"],
            add_special_tokens=False
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

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Example has {len(input_ids)} tokens."
            )

        return {
            "input_ids": input_ids,
            "labels": labels
        }


    # ---------------------------------------------------------
    # Prepare a small batch
    # ---------------------------------------------------------

    prepared = [
        prepare_example(example)
        for example in examples[:BATCH_SIZE]
    ]


    batch_max_length = max(
        len(example["input_ids"])
        for example in prepared
    )


    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []


    for example in prepared:

        input_ids = example["input_ids"]
        labels = example["labels"]

        padding_length = (
            batch_max_length
            - len(input_ids)
        )


        input_ids = (
            input_ids
            + [tokenizer.pad_token_id]
            * padding_length
        )


        labels = (
            labels
            + [-100]
            * padding_length
        )


        attention_mask = (
            [1] * (
                batch_max_length
                - padding_length
            )
            + [0] * padding_length
        )


        input_ids_batch.append(
            input_ids
        )

        labels_batch.append(
            labels
        )

        attention_mask_batch.append(
            attention_mask
        )


    # ---------------------------------------------------------
    # Convert to tensors
    # ---------------------------------------------------------

    input_ids_batch = torch.tensor(
        input_ids_batch,
        dtype=torch.long
    ).to(device)


    labels_batch = torch.tensor(
        labels_batch,
        dtype=torch.long
    ).to(device)


    attention_mask_batch = torch.tensor(
        attention_mask_batch,
        dtype=torch.long
    ).to(device)


    print(
        f"Batch shape: {input_ids_batch.shape}"
    )


    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    outputs = model(
        input_ids=input_ids_batch,
        attention_mask=attention_mask_batch,
        labels=labels_batch
    )


    loss = outputs.loss
    logits = outputs.logits


    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Loss: {loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
