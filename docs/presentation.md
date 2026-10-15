# Roteiro de apresentação — Mini Code LLM

Roteiro sugerido de 10 a 12 minutos. A demonstração usa o experimento 3 já publicado e não exige novo treinamento ao vivo.

Contexto a apresentar: baseline, os três treinamentos, validações e teste final foram executados na instância Linux com RTX 4090 de 24 GB e CUDA 13.0. O projeto foi depois copiado para a máquina atual para finalizar código e documentação; as verificações técnicas locais posteriores não fazem parte dos resultados experimentais originais.

## 1. Problema e objetivo — 1 minuto

“Especializamos uma LLM compacta para transformar instruções em inglês em pequenas funções Python. O objetivo é demonstrar fine-tuning e avaliação objetiva em um domínio limitado.”

Mostre `is_even(number)`. Explique que a meta envolve lógica correta, assinatura e saída somente com código.

## 2. Modelo base — 1 minuto

SmolLM2-135M, um Causal Language Model com aproximadamente 135 milhões de parâmetros. O tamanho permite experimentos acessíveis. O modelo já sabe padrões de texto e código; o treinamento realizado foi **full supervised fine-tuning**, ajustando todos os pesos, e não pré-treinamento do zero.

Mostre o formato Task/Code. A interface monta os marcadores automaticamente.

## 3. Dataset e protocolo — 1 minuto

960 exemplos: 720 train, 120 validation, 120 test; 12 famílias em três dificuldades. Train atualiza pesos, validation orienta seleção, test confirma o desempenho após a escolha.

Ressalva: são reformulações sintéticas das mesmas 12 famílias, com as mesmas assinaturas e implementações canônicas. Não são 960 algoritmos diferentes.

## 4. Baseline — 1 minuto

Mostre no README os números de validation: 98,3% Python válido, mas 63,3% de tarefas corretas e 0% code-only. Sintaxe válida não garante lógica correta. O modelo também acrescentava explicações e conclusões.

Abra uma resposta de `results/baseline_validation.jsonl` e seu registro em `baseline_validation_evaluated.jsonl` para conectar a métrica ao código real.

## 5. Como treinamos — 2 minutos

Explique a sequência `[prompt][code][EOS]`. Labels do prompt recebem `-100`; o modelo vê a instrução como contexto e a loss supervisiona o código e EOS. Padding também recebe `-100`, mas o EOS verdadeiro continua sendo alvo.

O modelo produz logits para os próximos tokens; a loss mede o erro das previsões supervisionadas. AdamW atualiza pesos. Learning rate controla o tamanho dos ajustes; batch agrupa exemplos; epoch representa uma passagem pelos dados. O treino usa 256 tokens, clipping 1,0 e checkpoint de menor validation loss.

Se houver tempo, execute um script educacional, por exemplo `python src/inspect_training_example.py`. Não execute o treinamento completo durante a apresentação.

O treinamento foi realizado na instância com RTX 4090 de 24 GB. O autor confirmou PyTorch `2.13.0+cu130`, `torch.version.cuda == '13.0'` e `CUDA Version: 13.0` no `nvidia-smi`; o log preservado registra `Device: cuda`. Mostre o screenshot original no VS Code quando ele for inserido. Os smoke tests posteriores em CPU pertencem à máquina de finalização e não descrevem o ambiente do treinamento.

## 6. Experimentos e seleção — 1 minuto

Use a tabela do README. Experimento 1: 100%; experimento 2: 68,3%; experimento 3: 100%, todos em validation. O terceiro foi escolhido pelo empate funcional e menor validation loss.

No experimento 2 a dificuldade 3 atingiu apenas 6,7%. A configuração foi insuficiente, mas não podemos isolar a causa: LR, batch e epochs foram modificados simultaneamente. As losses dos experimentos 1 e 2 foram informadas pelo autor e não têm logs nesta cópia.

Destaque o contraste do experimento 2: **99,2% de respostas somente com código, mas 68,3% de tarefas corretas**. Aprender o formato da resposta não garante aprender o comportamento solicitado. Entre os hiperparâmetros declarados dos experimentos 1 e 3, batch, epochs e max length foram mantidos e o LR mudou; sem repetições com várias seeds, isso não comprova uma vantagem causal geral. A menor loss serviu como desempate, pois ambos atingiram 100% funcional em validation.

## 7. Teste final — 30 segundos

120/120 tarefas, 530/530 verificações, todas as famílias com 100%. O test set foi usado depois da escolha, segundo o protocolo informado pelo autor. A refatoração apenas reavaliou respostas históricas e não gerou novamente o teste final.

## 8. Demonstração do frontend — 1 a 2 minutos

Antes da apresentação, ative o ambiente, execute `python src/app.py` e gere um exemplo para aquecer o cache/modelo. Abra `http://127.0.0.1:7860`.

1. Use `Create a Python function called add_numbers(a, b) that returns their sum.` e clique em Generate.
2. Use `Create count_vowels(text) that counts vowels case-insensitively.`.
3. Use `Create remove_duplicates(items) that removes duplicates while preserving order.`.
4. Mostre Clear e a indicação CPU/CUDA.

Explique que o carregamento vem do Hugging Face, não de pesos no GitHub. A interface exibe a função para revisão; ela não executa código arbitrário do usuário.

Se não houver internet, aqueça previamente o cache e confirme a demonstração antes de apresentar. Se surgir uma falha, apresente os resultados salvos como evidência histórica, identificando-os como tal.

## 9. Conclusões — 30 segundos

O estudo demonstra melhor correção funcional e aderência de formato dentro das 12 famílias conhecidas. O resultado não prova capacidade geral de programação nem generalização a famílias novas. Próximos passos: tarefas mais diversas, famílias inéditas em benchmark separado e comparações com múltiplas seeds.

## Perguntas frequentes

| Pergunta | Resposta |
|---|---|
| Por que SmolLM2? | Modelo compacto pré-treinado, adequado para explorar o pipeline com recursos acessíveis. A escolha não afirma superioridade sobre modelos maiores. |
| Por que fine-tuning? | Aproveita conhecimentos já aprendidos e especializa comportamento, assinatura e formato com exemplos supervisionados. |
| Por que três divisões? | Evita usar o conjunto final para escolher configurações. Validation seleciona; test confirma depois. |
| Por que `-100`? | A loss ignora essas posições, concentrando a supervisão no código e no término, sem retirar o prompt do contexto. |
| O que é loss? | Erro de previsão dos tokens alvo; não mede diretamente a correção funcional. |
| O que faz AdamW? | Atualiza os pesos a partir dos gradientes com adaptação por parâmetro e weight decay desacoplado. |
| O que é learning rate? | Escala dos ajustes dos pesos a cada passo. |
| Por que unit tests? | Implementações diferentes podem ser corretas; os testes medem comportamento, enquanto exact match compara texto. |
| Por que 98,3% válido e só 63,3% correto? | O parser aceita sintaxe, mas não comprova semântica, casos de borda ou aderência ao contrato. |
| Por que o experimento 2 foi pior? | A configuração combinada aprendeu menos comportamento funcional, especialmente nas famílias difíceis; o estudo não isola cada parâmetro. |
| Por que escolher o experimento 3? | Empate funcional em validation com o experimento 1 e menor loss informada. |
| Por que 100% não significa programar qualquer coisa? | O benchmark tem apenas 12 alvos canônicos conhecidos, com novas formulações das instruções. Memorização é compatível com esse resultado. |
| Houve treinamento do zero? | Não. Todos os experimentos partiram do SmolLM2 pré-treinado, segundo o protocolo do projeto. |
| A nova seed 42 foi usada originalmente? | Não sabemos. Ela foi adicionada para novas execuções, sem atribuição retroativa. |

## Checklist antes da apresentação

- Inserir o print original da GPU no VS Code e conferir sua legibilidade.
- Executar os testes, aquecer o modelo e confirmar o frontend na máquina da apresentação.
- Abrir README, resultados reais e Hugging Face; confirmar a URL do GitHub após o push.
- Identificar limitações e pendências com clareza, sem apresentar novas execuções como evidências históricas.
