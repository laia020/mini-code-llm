# Metodologia e reprodução

Este guia reúne os detalhes do fine-tuning, os comandos de experimentação e as verificações. Para apenas usar o frontend, siga o inicializador descrito no [README](../README.md).

## Ambiente para os comandos

Use Python 3.11 e execute os comandos a partir da raiz do projeto, com o ambiente virtual ativo. O inicializador `iniciar.bat` ou `python3.11 iniciar.py` prepara a `.venv` automaticamente.

<details>
<summary>Instalação e ativação manual</summary>

No Windows, em PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Se a ativação estiver bloqueada, use diretamente `.\.venv\Scripts\python.exe` no Windows ou `.venv/bin/python` no Linux/macOS no lugar de `python`.

</details>

Para usar GPU, instale a distribuição CUDA do PyTorch compatível com o driver seguindo o [seletor oficial](https://pytorch.org/get-started/locally/). O inicializador usa CPU em novos ambientes Windows/Linux e preserva um PyTorch CUDA existente e compatível. Confirme o dispositivo:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Transformers está fixado em 5.19.0, versão registrada nos artefatos publicados. A versão 4.57.6 falhou ao ler o formato de `extra_special_tokens` do tokenizer. O projeto usa um único dispositivo com `.to(device)`; Accelerate não é necessário.

## Preparação para supervised fine-tuning

O SmolLM2 é um Causal Language Model: prevê o próximo token a partir dos anteriores. A tokenização transforma texto em IDs; os logits são os escores para os tokens candidatos. A geração autoregressiva incorpora cada token produzido ao contexto da próxima previsão.

O prompt interno termina com uma quebra de linha:

```text
### Task:
{instruction}

### Code:
```

O treino concatena prompt, código e EOS:

```text
input_ids: [tokens do prompt] [tokens do código] [EOS]
labels:    [-100 ... -100]   [tokens do código] [EOS]
```

- **Labels e masking:** o valor `-100` faz a loss ignorar o prompt, que permanece visível como contexto. O modelo faz o deslocamento causal dos labels internamente.
- **EOS:** é supervisionado para ensinar o término da resposta. O código atual rejeita sequências acima do limite, evitando truncamento silencioso do código ou EOS.
- **Batches e padding:** os exemplos são agrupados e preenchidos até o maior comprimento do batch. Padding recebe labels `-100` e attention mask zero; o EOS real continua supervisionado, mesmo compartilhando o ID usado no padding.
- **Loss:** entropia cruzada dos próximos tokens supervisionados. A validation loss usa a média das losses dos batches, conforme o protocolo original. Menor loss não garante correção funcional.
- **AdamW:** atualiza os pesos a partir dos gradientes com adaptação por parâmetro e weight decay desacoplado, usando o padrão 0,01. Gradient clipping limita a norma a 1,0.
- **Learning rate e epochs:** controlam, respectivamente, a escala dos ajustes e o número de passagens pelo treino. As configurações completas usam comprimento máximo de 256 tokens.

Todos os pesos do modelo pré-treinado são ajustados. Não foram usados LoRA ou quantização no protocolo documentado. O melhor checkpoint de cada execução é escolhido pela menor validation loss; o test set não participa da seleção.

## Novo treinamento

```powershell
python src/train.py --experiment experiment-3 --output-dir models/experiment-3-new --device cuda
```

Os presets são `experiment-1`, `experiment-2` e `experiment-3`. A CLI também aceita `--learning-rate`, `--batch-size`, `--epochs`, `--seed`, `--max-length`, `--train-file` e `--validation-file`. Com `--device auto`, seleciona CUDA quando disponível e CPU caso contrário; treinamento completo em CPU pode ser lento.

Cada nova execução registra configuração, seed padrão 42, versões, dispositivo, hashes dos dados, losses e melhor epoch em `training_metadata.json`. A seed 42 não é atribuída aos experimentos originais. O ambiente original completo e sua seed não estão integralmente registrados, portanto não se promete reprodução bit a bit.

## Geração e avaliação de validation

```powershell
python src/generate.py --model laia020/mini-code-llm --subfolder experiment-3 --input data/validation.jsonl --output results/reproductions/exp3_validation.jsonl
python src/evaluate.py --input results/reproductions/exp3_validation.jsonl --output results/reproductions/exp3_evaluated.jsonl --summary results/reproductions/exp3_summary.json
```

Para baseline, use `--model HuggingFaceTB/SmolLM2-135M`, sem subfolder. Para um checkpoint novo salvo localmente, use `--model models/experiment-3-new`. O gerador salva a resposta bruta e os metadados separadamente. São usados `do_sample=False` e até 128 novos tokens, como nas gerações históricas.

Os nomes antigos, como `evaluate_experiment_3.py`, continuam como atalhos para a implementação compartilhada; suas saídas ficam em `results/reproductions/`. Os comandos recusam arquivos e diretórios de saída existentes. Escolha nomes novos para cada execução.

`final_test.py` permanece disponível para reprodução explícita. O conjunto final é congelado e não deve ser usado para selecionar configurações ou ajustar o modelo.

## Carregamento direto do Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "laia020/mini-code-llm"
options = {
    "subfolder": "experiment-3",
    "revision": "044a7de4030020a593eedb2417d11aecd38e5268",
    "token": False,
}
tokenizer = AutoTokenizer.from_pretrained(repo_id, **options)
model = AutoModelForCausalLM.from_pretrained(repo_id, **options)
```

O repositório é público e a inferência não exige login. A revisão fixa mantém a origem dos arquivos estável; a CLI aceita `--revision` para escolher outra versão. Cada experimento possui configuração, tokenizer e `model.safetensors`. Os pesos ficam no Hugging Face, fora do histórico Git deste projeto.

## Verificações

Checks locais:

```powershell
python -m compileall -q iniciar.py src scripts tests
python -m unittest discover -s tests -v
python scripts/validate_dataset.py
python scripts/verify_repository.py
```

Auditoria das 600 respostas históricas, sem nova geração no test set:

```powershell
python scripts/audit_results.py --output-dir results/verification/audit-new
```

Verificação online opcional do pipeline:

```powershell
python scripts/smoke_test.py --output results/verification/runtime-new.json
```

O smoke test verifica três instruções independentes dos arquivos de test, inferência em CPU, CUDA quando disponível e os callbacks do Gradio. Também executa um passo de otimização em checkpoint temporário, partindo do experimento 3 publicado, para verificar o código de treinamento. O checkpoint é descartado; essa checagem não representa outro experimento acadêmico.

Para reconstruir o dataset sem substituir o benchmark congelado:

```powershell
python scripts/build_dataset.py --output-root results/reproductions/dataset-copy
python scripts/validate_dataset.py --root results/reproductions/dataset-copy
```

Os scripts educacionais `inspect_training_example.py`, `inspect_batch.py`, `inspect_loss.py` e `baseline_single.py` demonstram labels, padding, logits e previsão do próximo token. Execute-os explicitamente com `python src/<script>.py`.

## Problemas comuns na inicialização

- **Python incompatível:** o inicializador requer Python 3.11. Uma `.venv` existente e incompatível é preservada, com uma mensagem de erro.
- **Porta ocupada:** encerre outra instância do frontend ou use `python iniciar.py --port 7861`.
- **Instalação interrompida:** confira a conexão e execute o inicializador novamente. Somente instalações concluídas são marcadas como prontas.
- **Download do modelo parado:** tente o transporte HTTP padrão no lugar do Xet, conforme abaixo.

No PowerShell:

```powershell
$env:HF_HUB_DISABLE_XET = "1"
python iniciar.py
```

No Linux/macOS:

```bash
HF_HUB_DISABLE_XET=1 python3.11 iniciar.py
```

No Windows sem suporte a symlinks, o Hub pode emitir um aviso e usar mais espaço de cache. Não é necessário executar como administrador apenas por esse aviso.
