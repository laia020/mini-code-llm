"""Educational demonstration; run explicitly from the project root."""

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")


    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")


    prompt = """### Task:
    Create a function that checks whether a number is even.

    ### Code:
    """


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )


    print("\n--- ORIGINAL TEXT ---")
    print(prompt)

    print("--- TOKENS ---")
    print(tokens)

    print("\n--- INPUT IDS ---")
    print(input_ids)

    print("\n--- TENSOR SHAPE ---")
    print(input_ids.shape)

    print("\n--- NUMBER OF TOKENS ---")
    print(input_ids.shape[1])

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    print("\n--- LOGITS SHAPE ---")
    print(logits.shape)


    last_token_logits = logits[0, -1, :]

    probabilities = torch.softmax(
        last_token_logits,
        dim=-1
    )


    top_probabilities, top_token_ids = torch.topk(
        probabilities,
        k=5
    )


    print("\n--- TOP 5 NEXT TOKEN PREDICTIONS ---")

    for probability, token_id in zip(
        top_probabilities,
        top_token_ids
    ):
        token = tokenizer.decode([token_id.item()])

        print(
            f"Token: {token!r} | "
            f"ID: {token_id.item()} | "
            f"Probability: {probability.item():.4f}"
        )




    print("\n--- GENERATED OUTPUT ---")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    generated_tokens = generated[0][input_ids.shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print(generated_text)


if __name__ == "__main__":
    main()
