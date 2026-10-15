# Evidências visuais

O autor entregará os screenshots ao professor em documento separado. Esta pasta pode receber cópias opcionais, sempre de imagens reais. Esta cópia não contém os screenshots originais.

| Arquivo | O que deve mostrar | Origem |
|---|---|---|
| `gpu-vscode.png` | VS Code conectado por SSH à instância Linux; `nvidia-smi` com RTX 4090 de 24 GB e CUDA Version 13.0; idealmente PyTorch `2.13.0+cu130`, `torch.version.cuda` 13.0 e CUDA disponível | Ambiente confirmado pelo autor; inclusão do screenshot original pendente |
| `training.png` | Saída real de treinamento, epochs, train loss e validation loss | Captura original; não apresentar execução nova como histórica |
| `frontend.png` | Interface Gradio em execução, instrução e código retornado | Captura real após iniciar `python src/app.py` |

Antes de publicar, confira se os prints expõem tokens, endereços sensíveis ou outros dados privados. Oculte apenas esses dados, mantendo as evidências técnicas legíveis. Adicione as imagens ao README quando existirem; não use imagens substitutas de hardware ou sessões diferentes.
