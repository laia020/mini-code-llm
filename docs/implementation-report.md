# Relatório de finalização

Data: 15/10/2026. Este relatório descreve a preparação local; a publicação no GitHub permanece a cargo do autor.

**Procedência:** todo o trabalho experimental original foi realizado pelo autor na instância Linux com RTX 4090 de 24 GB: baseline, os três treinamentos, validações e teste final. A máquina atual recebeu uma cópia do projeto. Os testes técnicos de finalização descritos neste relatório foram executados posteriormente nesta cópia e não são a origem dos experimentos ou das métricas acadêmicas.

## O que foi implementado

- Frontend Gradio com exemplos, Generate/Clear, saída Python e indicação CPU/CUDA.
- Inferência compartilhada e carregamento anônimo do experimento 3 com revisão fixa do Hugging Face.
- Geração em lote preservando a resposta bruta e avaliador único com timeout por subprocesso.
- Treinamento parametrizado para os três experimentos, proteção contra sobrescrita, seed e metadados para novas execuções.
- Scripts educacionais preservados, sem carregar modelos ou executar treinamentos durante imports.
- Auditoria das respostas históricas, verificação de integridade e inspeção dos arquivos publicáveis.
- README, roteiro da apresentação e instruções para os screenshots reais.

## Arquivos criados

| Grupo | Arquivos |
|---|---|
| Aplicação e pipeline | `src/__init__.py`, `src/common.py`, `src/inference.py`, `src/app.py`, `src/generate.py`, `src/training.py` |
| Verificações | `scripts/audit_results.py`, `scripts/smoke_test.py`, `scripts/verify_repository.py`, `tests/test_pipeline.py` |
| Ambiente e Git | `requirements.txt`, `.gitignore`, `.gitattributes` |
| Inicialização simples | `iniciar.bat`, `iniciar.py` |
| Documentação | `README.md`, `docs/reproduction.md`, `docs/academic-context.md`, `docs/presentation.md`, `docs/images/README.md`, `docs/original-artifacts.json`, este relatório e `docs/verification.json` |

O ambiente `.venv/` e os resultados de verificações/reproduções ficam apenas localmente e são ignorados pelo Git. O resumo publicável das verificações está em `docs/verification.json`.

## Arquivos alterados, removidos ou movidos

- `src/train.py` tornou-se a entrada da CLI de treinamento; a implementação compartilhada está em `src/training.py`.
- `src/evaluate.py` concentra o protocolo histórico e a CLI. Os quatro `evaluate_*` são wrappers compatíveis.
- `src/baseline.py`, `finetuned_validation.py`, `experiment_2_validation.py`, `experiment_3_validation.py` e `final_test.py` delegam ao gerador compartilhado; escrevem em `results/reproductions/` por padrão.
- `src/baseline_single.py` e os três `inspect_*` preservam as demonstrações e agora só executam via `main`.
- `scripts/build_dataset.py` aceita `--output-root` e recusa substituir o dataset congelado.
- Nenhum arquivo original foi removido. Dados, contratos, resultados históricos e log foram preservados byte a byte. Os hashes originais do código estão registrados separadamente para rastreabilidade.

## Testes e resultados

| Verificação | Resultado |
|---|---|
| Compilação de `src`, `scripts` e `tests` | Aprovada |
| Testes unitários e regressões | 39 testes aprovados, incluindo sete verificações do inicializador, sem skips no ambiente virtual |
| Validação do dataset | 960 registros, 960 instruções normalizadas únicas, hashes válidos e 53 casos contratuais |
| Reconstrução em pasta separada | Dados, manifest e contratos idênticos aos originais |
| Auditoria histórica | Cinco resumos e os detalhes de 600 respostas reproduzidos; nenhuma nova geração no test set |
| Integridade | 21 artefatos originais preservados |
| Dependências | `pip check` sem conflitos |
| Análise estática | Ruff: regras F sem erros |
| Inferência real em CPU | 3/3 tarefas, 14/14 casos comportamentais, assinaturas corretas e saída somente código |
| Gradio | HTTP 200; geração real, Clear, erro de entrada vazia e encerramento aprovados |
| Treinamento mínimo em CPU | Forward, backward, AdamW, salvamento e recarregamento aprovados |
| CLI de geração e avaliação | 2/2 amostras de validation, 9/9 casos comportamentais |
| CUDA local | Indisponível nesta instalação; testes de seleção simulados aprovados |
| Inicializador automático | Instalação limpa sem `.venv` aprovada; segunda execução sem pip; `iniciar.bat` abriu HTTP 200 em uma pasta com espaços, a partir de outra pasta |

As novas verificações não substituem os resultados acadêmicos. Os arquivos locais incluem `results/verification/audit-initial/`, `dataset.json`, `cli_evaluated.jsonl`, `cli_summary.json` e `runtime.json`. Novas execuções devem escolher caminhos ainda inexistentes.

Ambiente verificado: Windows, Python 3.11.9, PyTorch 2.14.1 (CPU), Transformers 5.19.0, Gradio 5.50.0 e huggingface_hub 1.33.0. O limite mínimo de PyTorch 2.5 acompanha o requisito da versão escolhida de Transformers. A GPU histórica RTX 4090 não foi revalidada nesta máquina.

O autor confirmou que todo o fluxo experimental original foi executado na instância Linux com **RTX 4090 de 24 GB**, PyTorch **`2.13.0+cu130`**, `torch.version.cuda` igual a **`13.0`** e `nvidia-smi` exibindo **CUDA Version: 13.0**. CUDA estava disponível e foi usado no treinamento. A indicação de CUDA indisponível na tabela acima se aplica somente à verificação local posterior. O screenshot original continua pendente de inclusão nesta cópia.

A verificação técnica local do código de treinamento usou duas amostras de train, duas de validation e um passo de AdamW em CPU, partindo do experimento 3 já publicado para conferir forward, backward, salvamento e recarregamento. O checkpoint temporário foi descartado. Essa checagem posterior não treinou nenhum dos modelos entregues, não produziu as métricas acadêmicas e não reproduziu os treinamentos originais, todos realizados na instância.

## Instalar e executar

Para quem baixar o repositório, basta ter **Python 3.11** instalado e dar dois cliques em **`iniciar.bat`** no Windows. O inicializador cria ou reutiliza a `.venv`, instala as dependências e abre o navegador. Nas próximas execuções, reutiliza as dependências enquanto o requirements e as versões instaladas permanecerem iguais.

No Linux/macOS:

```bash
python3.11 iniciar.py
```

A interface abre em `http://127.0.0.1:7860`. A instalação inicial e o primeiro download do modelo precisam de internet. O inicializador instala PyTorch CPU em novos ambientes Windows/Linux e preserva uma instalação CUDA existente que satisfaça o requisito mínimo. Não instala Python nem drivers da GPU. As instruções manuais continuam em uma seção opcional do README.

Inferência individual:

```powershell
python src/inference.py "Create a function that checks whether a number is even."
```

Novo treinamento, sem substituir pesos existentes:

```powershell
python src/train.py --experiment experiment-3 --output-dir models/experiment-3-new --device cuda
```

Geração e avaliação de validation:

```powershell
python src/generate.py --model laia020/mini-code-llm --subfolder experiment-3 --input data/validation.jsonl --output results/reproductions/validation-new.jsonl
python src/evaluate.py --input results/reproductions/validation-new.jsonl --output results/reproductions/validation-new-evaluated.jsonl --summary results/reproductions/validation-new-summary.json
```

O [README](../README.md) apresenta o projeto, o inicializador, os resultados e suas limitações para o público do repositório. Os comandos avançados, instalação CUDA, carregamento com `AutoTokenizer`/`AutoModelForCausalLM.from_pretrained` e scripts educacionais estão no [guia de metodologia e reprodução](reproduction.md). O checklist da atividade está em [contexto acadêmico](academic-context.md).

## Compatibilidade e limites de reprodução

Os artefatos publicados registram Transformers **5.19.0**. O teste inicial com 4.57.6 falhou no tokenizer: `extra_special_tokens` é uma lista nos arquivos publicados, enquanto aquela versão esperava um dicionário nesse campo. A dependência foi fixada na versão dos artefatos; não foram alterados tokenizer, pesos ou configs remotos. O [guia oficial de migração para v5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) documenta a mudança de formato.

O acesso ao repositório público foi confirmado sem token. O transporte Xet apresentou lentidão/travamento na rede desta máquina; o arquivo de pesos foi obtido por HTTP público e conferido por tamanho e SHA-256 antes do carregamento pelo cache do Hub. Isso não modifica pesos nem métricas. O README descreve a alternativa padrão `HF_HUB_DISABLE_XET=1`.

Não há logs dos treinamentos 1 e 2 nesta cópia. Suas losses e configurações são atribuídas ao relato do autor; as métricas funcionais vêm dos resultados reais. O log do experimento 3 confirma CUDA e loss arredondada de 0,0005. PyTorch `2.13.0+cu130` e CUDA 13.0 foram confirmados pelo autor para a instância de treinamento. A seed e as demais versões históricas não estão integralmente registradas; os metadados das verificações locais não são atribuídos retroativamente ao treinamento.

## GitHub

Git foi inicializado localmente em **main**. O autor criou o repositório público [laia020/mini-code-llm](https://github.com/laia020/mini-code-llm), conectado como `origin`, e autorizou os commits e a publicação. O histórico separa configuração, dados e resultados, pipeline e frontend, e documentação. Nome, e-mail pessoal dos commits e preferência de autenticação foram configurados somente neste projeto. A remoção da lista de participantes do README pelo autor foi preservada. O `.gitignore` exclui pesos, ambientes, caches, segredos e reproduções locais; `.gitattributes` protege os bytes dos artefatos históricos contra conversão de finais de linha, conferidos também diretamente nos blobs dos commits.

Para publicar atualizações após revisar:

```powershell
git status --short
python scripts/verify_repository.py
git add .
git diff --cached --stat
git commit -m "Atualiza projeto Mini Code LLM"
git remote -v
git push -u origin main
```

O verificador inspeciona nomes, tamanho e padrões reconhecíveis de segredos, relatando somente os caminhos. A inspeção é heurística e não substitui a revisão pessoal antes da publicação.

## Pendências do autor

1. Entregar os screenshots reais ao professor em documento separado, conforme definido pelo autor. O print da GPU deve identificar a janela do VS Code e mostrar RTX 4090 e `nvidia-smi`.
2. Opcionalmente incluir cópias em `docs/images/`, identificando sua origem.
3. Conferir a entrega no repositório pessoal e manter o histórico atualizado para futuras alterações.

O resultado de 100% se limita às 12 famílias e às formulações sintéticas do benchmark. Não demonstra capacidade geral de programação, desempenho em português ou generalização a algoritmos inéditos. O frontend fornece código para revisão; a execução do avaliador é restrita ao experimento e não constitui sandbox geral.

Na execução real da CLI com `Create a function that checks whether a number is even.`, a lógica foi retornada como `def isEven(number)`, sem o nome canônico `is_even`. A instrução não exigia assinatura; a observação foi registrada como limitação, sem renomear artificialmente a saída. Os três smoke tests contratuais usam assinaturas explícitas.
