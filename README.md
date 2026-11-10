# Mini Code LLM

LLM compacta especializada em gerar pequenas funções Python a partir de instruções em inglês. O projeto combina **full supervised fine-tuning do SmolLM2-135M**, avaliação por testes comportamentais e uma interface Gradio para experimentar o modelo localmente.

O benchmark cobre **12 famílias de tarefas**. Em validation, a correção funcional passou de **63,3% no modelo base para 100% nos experimentos 1 e 3**. O experimento 3 também acertou 120/120 tarefas no teste final. Esses resultados se limitam a novas formulações das famílias conhecidas.

[Como usar](#como-usar) · [Modelos no Hugging Face](https://huggingface.co/laia020/mini-code-llm) · [Resultados](#resultados) · [Metodologia e reprodução](docs/reproduction.md) · [Contexto acadêmico](docs/academic-context.md)

## Requisitos

O inicializador aceita **Python 3.11 ou superior**. O ambiente Apple Silicon validado usa Python **3.12.16**, PyTorch **2.15.0** e Python nativo ARM64. Outras versões aceitas pelo inicializador dependem da disponibilidade de wheels compatíveis das bibliotecas; não foram todas testadas.

As dependências diretas estão em [`requirements.txt`](requirements.txt):

| Dependência | Requisito | Uso |
|---|---|---|
| PyTorch | `torch>=2.5` | Tensores, inferência e treinamento |
| Transformers | `transformers==5.19.0` | Modelo e tokenizer |
| Gradio | `gradio>=5,<6` | Interface local |
| Hugging Face Hub | `huggingface_hub>=1,<2` | Acesso aos checkpoints |

O **uv** é usado para gerenciar o ambiente manualmente. **pytest** é uma dependência opcional de desenvolvimento; a suíte também pode ser executada com `unittest`. Não há lockfile: os requisitos acima não fixam todas as versões transitivas.

## Como usar

### Windows

1. Tenha **Python 3.11 ou superior** instalado.
2. [Baixe o projeto](https://github.com/laia020/mini-code-llm/archive/refs/heads/main.zip) e extraia o ZIP.
3. Dê dois cliques em **[`iniciar.bat`](iniciar.bat)**.

O inicializador prepara a `.venv`, instala as dependências e abre o aplicativo em `http://127.0.0.1:7860`. Não é necessário ativar o ambiente ou instalar pacotes manualmente. Mantenha a janela aberta enquanto usa a aplicação; **Ctrl+C** encerra o servidor.

### Linux/macOS

Com Python 3.11 ou superior instalado, execute na pasta do projeto:

```bash
python3 iniciar.py
```

A instalação inicial requer internet. O modelo é baixado do Hugging Face no primeiro **Generate** e fica em cache. Nas próximas execuções, o ambiente preparado é reutilizado. Se os requisitos ou as versões instaladas mudarem, o inicializador confere e ajusta as dependências dentro da `.venv`.

Novos ambientes Windows/Linux usam PyTorch CPU, suficiente para a demonstração. Uma instalação PyTorch CUDA existente e compatível é preservada. Treinamento e inferência selecionam CUDA → MPS → CPU. Também é possível escolher explicitamente `--device cuda`, `--device mps` ou `--device cpu`.

### Apple Silicon com ambiente existente (uv)

Reutilize sua `.venv`, sem reinstalar dependências:

```bash
source .venv/bin/activate
python iniciar.py --skip-setup --device mps
```

`--skip-setup` evita instalações, atualizações e gravação do marcador de setup. Sem essa opção, o inicializador mantém seu fluxo de preparação com pip. A interface e as CLIs usam `auto` por padrão. Veja [instruções e limitações do MPS](docs/reproduction.md#apple-silicon-mps).

Problemas com Python, porta ocupada ou download? Consulte [solução de problemas](docs/reproduction.md#problemas-comuns-na-inicialização).

### Instalação manual com uv — Windows, Linux e macOS

Em um clone novo, com uv e Python 3.12 instalados, crie o ambiente **somente se `.venv` ainda não existir**:

```bash
uv venv --python 3.12 --prompt venv-mini-code-llm .venv
```

A pasta permanece `.venv`; o padrão de identificação no terminal é **`(venv-nome-do-projeto)`**, neste projeto **`(venv-mini-code-llm)`**. O nome exibido não altera os caminhos usados pelo inicializador. Para ambientes existentes, pule a criação e preserve a configuração atual.

Ative no Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Ou no Linux/macOS (bash/zsh):

```bash
source .venv/bin/activate
```

Em um ambiente novo ou com dependências ainda ausentes, instale os requisitos. Em um ambiente já configurado, execute apenas a verificação:

```bash
uv pip install -r requirements.txt
uv pip check
python iniciar.py --skip-setup
```

Para NVIDIA no Windows/Linux, instale primeiro o PyTorch CUDA compatível com seu driver conforme o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/), e depois as demais dependências. No macOS Apple Silicon, a distribuição macOS do PyTorch oferece o backend MPS; CUDA não se aplica. O modo manual com uv deve usar `--skip-setup`, pois o fluxo automático do inicializador usa pip, que pode não estar instalado em ambientes criados com uv.

## Dispositivos e execução

Execute os comandos a partir da raiz, com a `.venv` ativa. O modo padrão `--device auto` escolhe **CUDA → MPS → CPU**. Uma escolha explícita indisponível gera erro, sem trocar silenciosamente para CPU.

| Dispositivo | Ambiente | Opção |
|---|---|---|
| CPU | Windows, Linux e macOS | `--device cpu` |
| CUDA | NVIDIA com PyTorch CUDA e driver compatível | `--device cuda` |
| MPS | macOS com GPU Metal compatível, incluindo Apple Silicon | `--device mps` |

Confira a seleção sem carregar pesos:

```bash
python -c "from src.inference import select_device; print(select_device())"
```

Inferência de uma instrução e geração em lote para validation:

```bash
python src/inference.py "Create is_even(number) that checks parity." --device auto
python src/generate.py --input data/validation.jsonl --output results/reproductions/validation-new.jsonl --device auto
```

Esses comandos usam o experimento 3 publicado e podem baixar pesos se ainda não estiverem em cache. Para usar apenas um checkpoint local, informe `--model /caminho/do/checkpoint` e defina `HF_HUB_OFFLINE=1` antes de iniciar o Python. `src/app.py`, `src/train.py`, `src/generate.py`, `src/inference.py`, `src/baseline_single.py`, `src/inspect_loss.py` e `scripts/smoke_test.py` aceitam a opção de dispositivo; o inicializador a encaminha à interface.

O comando de treinamento completo abaixo é uma instrução de uso, **não foi executado na adaptação ou na revisão para publicação**:

```bash
python src/train.py --experiment experiment-3 --output-dir models/experiment-3-new --device auto
```

O treinamento ajusta todos os pesos e pode demandar bastante tempo e memória. Aceita um checkpoint local com `--model` e uma execução limitada com `--max-steps 1`; mesmo nesse caso, a validation inteira é processada. Os presets dos três experimentos estão preservados.

A avaliação usa as respostas brutas salvas e executa os mesmos contratos em CPU, sem carregar um LLM nem receber `--device`:

```bash
python src/evaluate.py --input results/reproductions/validation-new.jsonl --output results/reproductions/validation-evaluated-new.jsonl --summary results/reproductions/validation-summary-new.json
```

Escolha caminhos novos para cada execução. Não substitua os dados, as gerações ou os resumos históricos. O test set continua reservado ao protocolo final, sem participar de ajustes ou seleção.

## Testes automatizados

Com pytest já instalado na `.venv`, execute:

```bash
python -m pytest tests/ -v
```

Se estiver preparando um ambiente de desenvolvimento novo, `uv pip install pytest` instala a ferramenta opcional. Para impedir acesso ao Hub durante os testes, configure antes `HF_HUB_OFFLINE=1` (`export HF_HUB_OFFLINE=1` em bash/zsh; `$env:HF_HUB_OFFLINE="1"` em PowerShell). Os testes criam o modelo e o tokenizer localmente, sem downloads de pesos.

Alternativa sem pytest e verificações complementares:

```bash
python -m unittest discover -s tests -v
uv pip check
python scripts/validate_dataset.py
python scripts/verify_repository.py
python src/train.py --help
python src/inference.py --help
```

## Exemplo de uso

Na interface, escreva uma instrução em inglês ou clique em um dos **seis prompts prontos**. Depois, selecione **Generate**. O botão **Clear** limpa os campos, e a seção expansível explica as 12 famílias.

Exemplo de instrução:

```text
Create a Python function called add_numbers(a, b) that returns their sum.
```

Resposta esperada:

```python
def add_numbers(a, b):
    return a + b
```

Informe o nome e os parâmetros da função para obter uma assinatura consistente. A aplicação monta o prompt Task/Code internamente e exibe o código para revisão. O frontend não executa a função gerada.

## Modelo e abordagem

O modelo base é **`HuggingFaceTB/SmolLM2-135M`**, um Causal Language Model com aproximadamente 135 milhões de parâmetros. O tamanho compacto permite estudar o pipeline de especialização e executar a demonstração em CPU.

Foi realizado **fine-tuning supervisionado**, ajustando todos os pesos do modelo pré-treinado. Não houve treinamento do zero, uso de LoRA ou quantização no protocolo documentado. A meta é melhorar a correção funcional, respeitar assinaturas e produzir somente o código solicitado.

Os três checkpoints estão em [laia020/mini-code-llm](https://huggingface.co/laia020/mini-code-llm), nas subpastas `experiment-1`, `experiment-2` e **`experiment-3`**, escolhido para a aplicação. O carregamento é público e não exige login. Os pesos ficam no Hugging Face; este repositório contém código, dados, contratos e resultados.

## Dataset

São **960 exemplos sintéticos** em inglês, distribuídos em 12 famílias:

| Dificuldade | Famílias e assinaturas |
|---|---|
| 1 | `add_numbers(a, b)`, `is_even(number)`, `get_largest(a, b)`, `reverse_string(text)` |
| 2 | `calculate_average(numbers)`, `count_vowels(text)`, `get_min_max(numbers)`, `is_palindrome(text)`, `get_even_numbers(numbers)` |
| 3 | `count_occurrences(items, target)`, `remove_duplicates(items)`, `word_frequency(words)` |

| Divisão | Exemplos | Por família | Finalidade |
|---|---:|---:|---|
| Train | 720 | 60 | Atualizar os pesos |
| Validation | 120 | 10 | Comparar configurações e selecionar o modelo |
| Test | 120 | 10 | Avaliação final após a seleção |

As instruções normalizadas não se repetem entre divisões. Entretanto, famílias, assinaturas e implementações canônicas são compartilhadas: o benchmark tem **12 implementações alvo**, com variações na forma de pedir cada tarefa.

Os [contratos](contracts/families.json) definem assinaturas, restrições e **53 casos comportamentais**. Aplicados aos dez exemplos de cada família, totalizam **530 verificações** em validation e em test. Os JSONL e seu [manifest de integridade](data/manifest.json) estão em `data/`.

## Resultados

### Comparação em validation

Todas as configurações abaixo foram avaliadas nos mesmos 120 exemplos de validation. LR significa learning rate; batch é a quantidade de exemplos processados por passo.

| Configuração | LR | Batch | Epochs | Melhor val loss | Tasks passed |
|---|---:|---:|---:|---:|---:|
| [Base](results/baseline_validation_summary.json) | — | — | — | — | 76/120 (63,3%) |
| [Experimento 1](results/finetuned_validation_summary.json) | 5e-5 | 2 | 5 | 0,0010¹ | 120/120 (100%) |
| [Experimento 2](results/experiment_2_validation_summary.json) | 1e-5 | 1 | 3 | 0,0901¹ | 82/120 (68,3%) |
| [**Experimento 3**](results/experiment_3_validation_summary.json) | **1e-4** | **2** | **5** | **0,0005²** | **120/120 (100%)** |

¹ Losses e configurações registradas pelo responsável pelos experimentos; os logs dos treinamentos 1 e 2 não estão versionados. Suas métricas funcionais estão nos resultados vinculados. ² Loss arredondada registrada no [log do experimento 3](results/experiment_3_train.log).

| Métrica | Base | Exp. 1 | Exp. 2 | Exp. 3 |
|---|---:|---:|---:|---:|
| Python válido | 118/120 (98,3%) | 100% | 100% | 100% |
| Assinatura correta | 117/120 (97,5%) | 100% | 100% | 100% |
| Tasks passed | 76/120 (63,3%) | 100% | 82/120 (68,3%) | 100% |
| Unit tests | 354/530 (66,8%) | 530/530 | 381/530 (71,9%) | 530/530 |
| Exact match | 38/120 (31,7%) | 100% | 71/120 (59,2%) | 100% |
| Code-only output | 0/120 (0%) | 100% | 119/120 (99,2%) | 100% |

| Dificuldade | Base: tasks passed | Exp. 2: tasks passed | Exp. 3: tasks passed |
|---|---:|---:|---:|
| 1 | 92,5% | 100% | 100% |
| 2 | 56,0% | 80,0% | 100% |
| 3 | 36,7% | 6,7% | 100% |

**O baseline já produzia sintaxe válida.** A diferença entre 98,3% de Python válido após extração e 63,3% de tarefas corretas evidencia falhas de lógica e aderência aos contratos. As famílias mais fracas foram `count_vowels` (0%), `get_min_max` (20%), `remove_duplicates` (0%) e `word_frequency` (10%). As respostas também incluíam explicações e conclusões adicionais, às vezes incompatíveis com o código.

**Aprender o formato não garantiu correção funcional.** O experimento 2 atingiu 99,2% de code-only, mas apenas 68,3% de tarefas corretas. Na dificuldade 3, ficou em 6,7%; `count_occurrences` atingiu 20%, enquanto `count_vowels`, `remove_duplicates` e `word_frequency` ficaram em 0%. Como LR, batch e epochs mudaram juntos em relação ao experimento 1, o estudo não isola o efeito de cada parâmetro.

**O experimento 3 foi selecionado em validation.** Os experimentos 1 e 3 empataram nas métricas funcionais. A menor validation loss foi o critério de desempate, antes do uso do test set. A comparação não demonstrou superioridade funcional do experimento 3 sobre o experimento 1 nesse benchmark.

### Teste final

Depois da escolha do experimento 3, foi realizada uma única avaliação final nos 120 exemplos de test, conforme o protocolo documentado.

| Métrica | Resultado |
|---|---:|
| Python válido | 120/120 (100%) |
| Assinatura correta | 120/120 (100%) |
| Tasks passed | 120/120 (100%) |
| Unit tests | 530/530 (100%) |
| Exact match | 120/120 (100%) |
| Code-only output | 120/120 (100%) |

As dificuldades 1, 2 e 3 atingiram, respectivamente, **40/40**, **50/50** e **30/30** tarefas corretas. Todas as 12 famílias atingiram 100%. Consulte o [resumo final](results/final_test_summary.json) e as [avaliações detalhadas](results/final_test_evaluated.jsonl).

O baseline de 63,3% refere-se a **validation**; os 100% do teste final referem-se a **test**. A comparação direta entre configurações é a tabela de validation acima.

## Como o código é avaliado

A métrica principal é **Tasks passed**: a função precisa passar em todos os casos comportamentais do seu contrato. O avaliador extrai o código, analisa sintaxe e AST, confere nome e parâmetros e executa os casos permitidos em subprocesso, com timeout padrão de cinco segundos por exemplo.

**Exact match** compara o código extraído ao alvo canônico; é secundário, pois implementações diferentes podem ter o mesmo comportamento. **Code-only** compara a resposta bruta com o código extraído. A limpeza visual do frontend não é aplicada para melhorar as métricas.

As restrições de AST e built-ins servem à avaliação controlada deste benchmark e não constituem um sandbox geral para código arbitrário. As gerações brutas, avaliações e resumos ficam em `results/`; o prefixo `finetuned_validation` corresponde ao experimento 1.

## Ambiente dos experimentos

Baseline, os três treinamentos, validações e teste final foram executados na instância Linux com **NVIDIA GeForce RTX 4090 de 24 GB**, acessada pelo VS Code via SSH.

| Componente | Ambiente documentado |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| PyTorch | `2.13.0+cu130` |
| `torch.version.cuda` | `13.0` |
| CUDA exibida pelo `nvidia-smi` | `13.0` |
| `torch.cuda.is_available()` | `True` |

O [log de treinamento preservado](results/experiment_3_train.log) registra execução em CUDA. A demonstração local também foi verificada em CPU, incluindo inferência real e interface Gradio. Os registros dessas checagens estão em [docs/verification.json](docs/verification.json).

## Validação e limites no Apple Silicon

A adaptação foi validada no Mac com testes automatizados de seleção de dispositivo e um **Llama minúsculo de pesos aleatórios criado localmente**. Em CPU e MPS, a suíte verifica forward/backward, um passo de AdamW, validation loss finita, alteração dos pesos, gravação em safetensors, recarga e geração. O teste MPS é pulado quando o processo não tem acesso à GPU.

**O modelo SmolLM2-135M e os checkpoints publicados ainda não foram validados em MPS nesta adaptação. Nenhum fine-tuning completo foi executado no MacBook.** Os resultados dos três experimentos e a verificação real anterior em CPU pertencem ao ambiente histórico descrito acima. A revisão para publicação não repetiu inferência desses checkpoints nem baixou pesos.

CUDA e os caminhos Windows/Linux foram preservados e cobertos por testes de seleção e regressão com simulação quando necessário. Não houve nova execução nativa em Windows/Linux ou em GPU NVIDIA nesta revisão.

No MPS, o carregamento usa FP32 explicitamente; IDs, labels e máscaras mantêm `torch.long`. A GPU compartilha memória com o sistema, não suporta FP64 e pode ter operações indisponíveis. Se faltar memória, reduza o batch; o código rejeita sequências acima de `--max-length` em vez de truncá-las. Desempenho superior à CPU e igualdade numérica entre backends não são garantidos.

Para uma operação sem implementação MPS, o fallback opcional `PYTORCH_ENABLE_MPS_FALLBACK=1`, definido antes de iniciar o Python, pode executar a operação em CPU, com custo de desempenho; não corrige tipos incompatíveis. A alternativa é `--device cpu`. Veja os detalhes em [Apple Silicon (MPS)](docs/reproduction.md#apple-silicon-mps).

## Metodologia e reprodução

O treinamento usa comprimento máximo 256, AdamW e gradient clipping de norma 1,0. Os tokens do prompt recebem labels `-100`, mantendo a instrução como contexto e concentrando a loss no código e no EOS. Padding também é mascarado. O checkpoint é selecionado pela menor validation loss.

O [guia de metodologia e reprodução](docs/reproduction.md) detalha tokenização, logits, geração autoregressiva, labels, loss e batches, além dos comandos para treinamento, inferência, avaliação, auditoria e testes. As novas execuções registram seed, configuração, versões e hashes dos dados e recusam sobrescrever saídas existentes.

Os hashes dos dados e dos resultados originais permitem verificar sua integridade. A auditoria reavalia respostas salvas, sem gerar novamente o test set. A suíte inclui testes unitários e de regressão; o inicializador também foi verificado em uma cópia sem `.venv`, com caminho contendo espaços e inicialização HTTP real do Gradio.

## Limitações e próximos passos

- Os 100% medem especialização em famílias conhecidas. As divisões compartilham assinaturas e códigos alvo, e o resultado é compatível com memorização das 12 implementações.
- Os contratos verificam um conjunto finito de casos; não provam correção para todas as entradas possíveis.
- O estudo não inclui repetições com múltiplas seeds nem uma estimativa de variabilidade. O ambiente original completo e a seed não estão integralmente registrados.
- Instruções em português, novas assinaturas e algoritmos fora das famílias não foram validados.
- Nomes implícitos podem gerar assinaturas diferentes: uma instrução genérica de paridade retornou `isEven(number)` em vez de `is_even(number)`. Informe a assinatura desejada no prompt.

Próximos passos incluem diversificar instruções e implementações, ampliar os contratos e avaliar famílias inéditas em outro benchmark, mantendo o test set atual congelado.

## Organização e documentação

```text
contracts/          contratos e casos comportamentais
data/               train, validation, test e manifest de integridade
results/            gerações e métricas dos experimentos
src/                frontend, inferência, geração, treinamento e avaliação
scripts/            construção, validação, auditoria e smoke tests
tests/              testes do pipeline, dispositivos e inicializador
docs/               metodologia, evidências e contexto acadêmico
iniciar.bat         inicializador Windows
iniciar.py          preparação do ambiente e abertura do frontend
requirements.txt    dependências diretas
```

- [Metodologia e reprodução](docs/reproduction.md)
- [Resultados das verificações](docs/verification.json)
- [Contexto acadêmico e evidências](docs/academic-context.md)
- [Roteiro da apresentação](docs/presentation.md)
