# Contexto acadêmico

O Mini Code LLM foi desenvolvido como atividade sobre fine-tuning de uma LLM e avaliação das respostas geradas. O objetivo é demonstrar a especialização de um modelo pré-treinado em um problema delimitado, com experimentos em GPU e uma interface para demonstração.

## Requisitos e evidências

| Requisito | Implementação ou evidência |
|---|---|
| Selecionar uma LLM | `HuggingFaceTB/SmolLM2-135M`, aproximadamente 135 milhões de parâmetros |
| Escolher um problema específico | Geração de pequenas funções Python, com 12 famílias de tarefas |
| Comparar parâmetros de treinamento | Três experimentos variando learning rate, batch size e epochs |
| Avaliar o modelo e suas respostas | Baseline, validation e teste final, com AST, assinatura e testes comportamentais |
| Executar os experimentos em GPU | Fluxo experimental original na instância Linux com RTX 4090 de 24 GB, PyTorch `2.13.0+cu130` e CUDA 13.0; o log preservado registra `Device: cuda` |
| Criar um frontend para prompts | Gradio com campo de instrução, seis exemplos clicáveis e geração de código |
| Apresentar print da GPU no VS Code | Screenshot original será entregue pelo autor em documento separado |
| Disponibilizar código e documentação no GitHub | [laia020/mini-code-llm](https://github.com/laia020/mini-code-llm); pesos disponíveis no Hugging Face |

Baseline, os três treinamentos, validações e teste final foram executados na instância GPU. O projeto foi posteriormente copiado para outra máquina para organização e verificações técnicas da aplicação. Os resultados dessas verificações posteriores não substituem os experimentos originais.

O trabalho realizou **full supervised fine-tuning de um modelo pré-treinado**. As métricas de 100% se referem às tarefas das 12 famílias conhecidas, com novas formulações de instrução; não demonstram capacidade geral de programação.

## Material de apresentação

- [Roteiro da apresentação](presentation.md): sequência da demonstração e perguntas frequentes.
- [Orientações das evidências visuais](images/README.md): conteúdo esperado dos screenshots reais.
- [Relatório de implementação](implementation-report.md): registro da organização e das verificações técnicas.
- [Resultados das verificações](verification.json): evidências estruturadas da finalização e do inicializador.

O print da GPU deve mostrar a janela do VS Code e a saída de `nvidia-smi`, identificando a RTX 4090. Cópias das imagens podem ser adicionadas a `docs/images/`; a entrega acadêmica dos screenshots foi definida em documento separado.
