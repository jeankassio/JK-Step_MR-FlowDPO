# Verificação desta versão

Data: 04/10/2026. Ambiente isolado, Windows, Python 3.12, PyTorch 2.10 CUDA 12.8, duas RTX 3090 de 24 GB (GPU 0 e 1). A instalação do ComfyUI não foi usada como ambiente do treinador.

Verificado:

- Download automático real do SFT XL do repositório `jeankassio/acestep_v1.5_sft_xl`, arquitetura oficial, VAE e Qwen3-Embedding-0.6B; reutilização dos arquivos completos.
- Checkpoint SFT XL real: loss/gradientes finitos, atualização LoRA e pesos base congelados. Teste curto de rank 2 usou pico de 7.89 GiB; esse número não estima uma música inteira nem rank 64.
- Dataset de teste com um tom stereo original de quatro segundos e uma degradação: VAE, texto, códigos semânticos compartilhados e tensores preparados corretamente.
- CLI real com rank 32/alpha 64: atualização, checkpoint, retomada para outra atualização e exportação native ComfyUI. O adapter exportado de aproximadamente 160 MB é apenas artefato de teste, não um LoRA musical para uso.
- Treinamento distribuído real do SFT XL nas duas RTX 3090, com backend `local` e buffers de gradiente em memória compartilhada: rank 2 e rank 32/alpha 64, divisão de pares, gradientes/loss finitos, atualização, validação, checkpoint com dois RNGs, retomada e exportação native. Os testes usam tensores sintéticos curtos; não medem velocidade em músicas completas. As duas GPUs recebem uma réplica do decoder, sem somar VRAM.
- Seis testes com dois processos CPU e PEFT verificam redução ponderada contra um cálculo independente, LoRAs idênticos nas réplicas, validação sem duplicação, parada iniciada pelo rank 1, escrita apenas pelo rank 0 e retomada bit a bit. A suíte completa passou: **73 testes**.
- CLAP musical e Audiobox Aesthetics reais: download dos avaliadores e cálculo dos dois eixos em um áudio de teste. A representação semântica privada do artigo não está disponível e não foi substituída por uma métrica inventada.
- Interface no Chrome headless em 1440×1000 e 390×844, sem erro de JavaScript ou overflow horizontal. Todos os campos de configuração e os controles de rank aparecem.
- Instalador PowerShell, build de wheel e ZIP de fonte sem ambientes, modelos, dados ou sessões.
- Testes automatizados cobrem loss/máscaras, ruído/tempo compartilhados, referência congelada, crop, retomada determinística, scaling de exportação, MRSD, validação/cache, downloads incompletos/conflitantes, autenticação, fila, cancelamento e editor de avaliações.

Comandos reproduzíveis para desenvolvimento:

```powershell
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe scripts/smoke_gpu.py --checkpoint-file checkpoints/acestep-v15-xl-sft/acestep_v1.5_sft_xl.safetensors --model-config-dir checkpoints/acestep-v15-xl-sft
.venv\Scripts\python.exe scripts/smoke_preprocess.py
.venv\Scripts\python.exe scripts/smoke_multigpu.py --checkpoint-file checkpoints/acestep-v15-xl-sft/acestep_v1.5_sft_xl.safetensors --model-config-dir checkpoints/acestep-v15-xl-sft --rank 32 --resume --export-comfyui
```

Não foi realizado um treinamento musical de produção ou comparação auditiva de um LoRA de qualidade. Estes resultados verificam que o software executa o método; composição, clareza vocal e fidelidade da letra dependem do dataset, hiperparâmetros e avaliação externa.

Neste build Windows, `is_gloo_available()` retorna verdadeiro, mas a criação de transporte falha (`unsupported gloo device`). O launcher usa processos locais autenticados e memória compartilhada em `auto`, sem modificar PyTorch/ComfyUI. O caminho NCCL/Linux permanece disponível, mas não foi executado neste ambiente.
