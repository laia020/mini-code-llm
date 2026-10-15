"""Local Gradio interface: python src/app.py."""

import argparse

if __package__:
    from .common import build_prompt
    from .inference import CodeGenerator, select_device
else:
    from common import build_prompt
    from inference import CodeGenerator, select_device

EXAMPLES = [
    "Create a Python function called add_numbers(a, b) that returns their sum.",
    "Create count_vowels(text) that counts vowels case-insensitively.",
    "Create remove_duplicates(items) that removes duplicates while preserving order.",
    "Create is_even(number) that returns True if the integer is even and False otherwise.",
    "Create reverse_string(text) that returns all characters in reverse order, preserving spaces and punctuation.",
    "Create calculate_average(numbers) that returns the arithmetic mean of a non-empty list of numbers.",
]

EXAMPLE_LABELS = [
    "Somar números",
    "Contar vogais",
    "Remover duplicados",
    "Verificar paridade",
    "Inverter texto",
    "Calcular média",
]

FAMILIES_HELP = """Uma **família** é um tipo de tarefa usado no treinamento do modelo.

| Função | O que faz |
|---|---|
| `add_numbers` | Soma dois números. |
| `is_even` | Verifica se um número inteiro é par. |
| `get_largest` | Retorna o maior entre dois números. |
| `reverse_string` | Inverte um texto, preservando todos os caracteres. |
| `calculate_average` | Calcula a média de uma lista não vazia de números. |
| `count_vowels` | Conta as vogais a, e, i, o, u, sem diferenciar maiúsculas e minúsculas. |
| `get_min_max` | Retorna o menor e o maior valor de uma lista não vazia. |
| `is_palindrome` | Verifica se o texto é exatamente igual à sua inversão. |
| `get_even_numbers` | Seleciona os números pares de uma lista, mantendo a ordem. |
| `count_occurrences` | Conta quantas vezes um item aparece em uma lista. |
| `remove_duplicates` | Remove itens repetidos, mantendo a ordem da primeira ocorrência. |
| `word_frequency` | Conta cada palavra, diferenciando maiúsculas e minúsculas. |
"""


def create_app(generator=None):
    import gradio as gr
    generator = generator or CodeGenerator()

    def respond(instruction):
        try:
            build_prompt(instruction)
            result = generator.generate(instruction)
            return result, f"Dispositivo: {generator.device} · experiment-3"
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(f"Não foi possível gerar o código: {error}") from error

    def clear():
        return "", "", "Pronto para uma nova instrução."

    with gr.Blocks(title="Mini Code LLM") as demo:
        gr.Markdown("# Mini Code LLM\nFine-tuning supervisionado do SmolLM2-135M para pequenas funções Python.")
        gr.Markdown(
            "Especializado em 12 famílias conhecidas. Escreva a instrução em **inglês**, "
            "preferencialmente com o nome e os parâmetros da função. "
            "O prompt é formatado automaticamente. O código gerado é exibido para revisão."
        )
        with gr.Accordion("Quais são as 12 famílias de tarefas?", open=False):
            gr.Markdown(FAMILIES_HELP)
        instruction = gr.Textbox(label="Instrução", lines=4, placeholder=EXAMPLES[0], render=False)
        gr.Markdown("**Prompts prontos:** clique para preencher e depois em **Generate**.")
        for start in range(0, len(EXAMPLES), 3):
            with gr.Row():
                for label, prompt in zip(EXAMPLE_LABELS[start:start + 3], EXAMPLES[start:start + 3]):
                    button = gr.Button(label, scale=1, min_width=0)
                    button.click(lambda text=prompt: text, outputs=instruction, queue=False, api_name=False)
        instruction.render()
        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary")
            clear_button = gr.Button("Clear")
        output = gr.Code(label="Código Python", language="python", interactive=False)
        status = gr.Markdown(f"Dispositivo detectado: {select_device()} · Modelo carregado na primeira geração.")
        generate_button.click(respond, instruction, [output, status], api_name="generate", concurrency_id="model", concurrency_limit=1)
        clear_button.click(clear, outputs=[instruction, output, status], api_name="clear", queue=False)
        instruction.submit(respond, instruction, [output, status], concurrency_id="model", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open-browser", action="store_true", help="Abre o frontend no navegador.")
    args = parser.parse_args(argv)
    create_app().launch(server_name=args.host, server_port=args.port, share=False, inbrowser=args.open_browser)


if __name__ == "__main__":
    main()
