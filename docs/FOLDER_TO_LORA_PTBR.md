# Da pasta de músicas até LoRA: MR-FlowDPO automático ou SFT

[English](FOLDER_TO_LORA.md) · **Português (Brasil)** · [Español](FOLDER_TO_LORA_ES.md) · [README do projeto](../README_PTBR.md)

Coloque gravações existentes em uma pasta, selecione-a no JK-Step e prepare. O aplicativo produz descrições, transcrições, trechos de áudio/texto e JSON. Escolha **MR-FlowDPO automático** para criar preferências sintéticas controladas e codificar ambos os lados, ou **SFT normal** para um alvo por gravação. A UI prepara os manifests automaticamente: não exige músicas Turbo, JSON manual de pares, scores ou chave Genius/API.

As anotações automáticas são estimativas. O Whisper pode omitir, repetir ou inventar palavras cantadas; seus timestamps podem ser imprecisos. O Qwen pode reconhecer instrumentos ou voz incorretamente. Preparar o dataset e reduzir a loss não comprovam melhora audível nem dicção perfeita.

## Pela interface

Execute `start_ui.bat` no Windows ou `bash jk-step.sh gui` no Linux.

Escolha **English**, **Português** ou **Español** no seletor de idioma da interface; o navegador salva sua escolha. Essa preferência é separada do idioma das letras/transcrição, portanto a interface e as músicas podem usar idiomas diferentes.

1. Em **Dataset**, selecione o fluxo de pasta e seu objetivo: MR-FlowDPO (padrão da UI) ou SFT normal.
2. Escolha a **Pasta de músicas** ou cole seu caminho absoluto. Subpastas são verificadas. **Verificar pasta** mostra um inventário sem carregar modelos.
3. No conteúdo, selecione voz, instrumentais ou detecção automática. Deixe o idioma em `auto` ou indique o idioma conhecido, como `pt`, `en` ou `es`.
4. Clique em **Preparar dataset**. Modelos ausentes são baixados automaticamente quando habilitado. Acompanhe em **Preparar**: os modelos analisam os áudios localmente e o fluxo organiza trechos e tensores.
5. Use **Revisar amostras** e confira o relatório, principalmente itens excluídos. Ouça os trechos correspondentes ao conferir palavras e limites dos cortes.
6. Continue em **Treinar**. O manifesto correto e o preset do objetivo são preenchidos automaticamente. Escolha uma saída nova, ajuste as opções e inicie.

A preparação preserva os arquivos originais. Trechos e sidecars ficam em outra pasta. Não aplica normalização de ganho ou masterização; o pré-processamento ACE converte sua cópia de trabalho para a taxa de amostragem e canais exigidos pelo modelo.

Descrições/transcrição locais fazem parte da instalação essencial, inclusive com `-CoreOnly`. A decodificação usa SoundFile e, quando necessário, o executável `imageio-ffmpeg` fornecido; esse fluxo de pasta não exige configurar FFmpeg externo. Funções opcionais do toolkit legado podem ter outras dependências de decodificação.

Um teste funcional limitado concluiu entrada de pasta, anotação local Qwen/Whisper, JSON/tensores, treino SFT iniciado pela UI e exportação. Outro teste MR automático preparou três pares controlados de áudio em português, codificou ambos os lados, executou uma atualização Flow-DPO e exportou pesos finitos. Duas RTX 3090 foram usadas em outro teste de execução SFT. Isso verifica funcionamento do software, não melhora musical, precisão das palavras ou desempenho geral em datasets.

## O que é preparado

| Etapa | Resultado |
| --- | --- |
| Leitura | Inventário recursivo, sidecars existentes e hashes das fontes. |
| Letras | Transcrição local com Whisper large-v3 e estimativas dos tempos das frases. Letras revisadas existentes são preservadas quando há correspondência segura. |
| Trechos | Agrupa frases inteiras com tempos até a duração configurada, normalmente 30 segundos, anexando apenas as palavras dessas frases. Frases longas demais ou inutilizáveis podem ser excluídas. |
| Descrições | Qwen2.5-Omni-7B ouve cada trecho preparado e escreve uma descrição musical. Captions existentes compatíveis podem ser reutilizadas. |
| Dataset | JSON de áudio/texto, procedência, separação treino/validação por grupos e relatório de itens excluídos. |
| Preferências, somente MR | Compara o trecho original com cópias contendo defeitos controlados; cada par compartilha letra e caption. |
| Pré-processamento | Tensores VAE/texto/ACE: um alvo no SFT ou escolhido/rejeitado com condicionamento compartilhado no MR-FlowDPO. |

Os limites das frases vêm do reconhecimento automático, não de alinhamento revisado por uma pessoa. A letra completa de uma música não é anexada a um corte curto arbitrário. O preset mantém os trechos preparados inteiros com `max_latent_length=0`; o treinador rejeita um corte latente menor de uma amostra vocal em vez de manter palavras que já não correspondem ao áudio.

## MR-FlowDPO automático da pasta

Esse modo usa o trecho existente anotado como **escolhido** e uma cópia com degradação acústica controlada como **rejeitado**. Não certifica que o original é excelente: o rótulo diz que é preferível àquela cópia artificialmente danificada. Começo, duração, letra e caption ficam alinhados; os originais não mudam. Ambos os lados são codificados e entram no objetivo Flow-DPO real, com o SFT original congelado como referência.

| `pair_options` | Padrão / finalidade |
| --- | --- |
| `degradation` | `mixed`: lowpass, ruído e clipping; ou escolha `lowpass`, `noise`, `clipping`. |
| `variations_per_audio` | 3 no misto, 1 para um defeito selecionado. |
| `cutoff_hz` | Corte lowpass de 6000 Hz. |
| `noise_snr_db` | Relação sinal/ruído de 24 dB. |
| `clip_threshold` | Limite de amplitude 0.15. |
| `context_mode` | `chosen_semantic` ou `silence` para texto/letra padrão. |

São preferências acústicas sintéticas, **não os rewards MRSD do artigo** nem evidência de qualidade musical geral. O lowpass afeta a mix completa e não isola voz. Evitar esses defeitos não equivale a compor melhor, cantar mais naturalmente ou pronunciar tudo corretamente. Revise as comparações e teste músicas geradas antes de julgar o adapter.

Pares humanos e seleção MRSD por scores continuam no fluxo separado; não são substituídos por essa experiência automática. O SFT de pasta não cria gravações rejeitadas ou scores de preferência.

### Voz e instrumentais

- `content_mode="auto"`: usa reconhecimento e classificação de áudio juntos. Transcrição vazia não comprova música instrumental; exemplos incertos ou contraditórios ficam separados para revisão.
- `content_mode="vocal"`: prepara palavras cantadas e tempos. Uma falha de transcrição não recebe um rótulo `[Instrumental]` inventado.
- `content_mode="instrumental"`: o usuário declara explicitamente que a pasta é instrumental. A letra é `[Instrumental]`; uma letra vocal existente em conflito é rejeitada.

Use uma pasta consistente ao escolher somente voz ou somente instrumentais. A detecção automática permite pastas mistas. A classificação ainda pode errar, portanto confira os exemplos resultantes.

### Letras existentes ou corrigidas

Metadados são opcionais. Os sidecars reconhecidos incluem `musica.caption.txt`, `musica.lyrics.txt` ou `musica.txt` com campos chave/valor; `musica` precisa corresponder ao nome do áudio sem extensão. Trechos revisados que cabem no limite de preparação podem usar diretamente sua letra completa fornecida. Gravações longas exigem tempos; se a letra fornecida divergir da transcrição, o material é separado para revisão em vez de forçar texto sobre áudio incompatível.

Para corrigir com precisão, prepare um trecho de origem curto com as palavras realmente cantadas nele e o sidecar correspondente, então repita a preparação. A prévia permite revisão; não certifica nem corrige automaticamente as letras. Editar apenas os metadados de um tensor em cache não altera o condicionamento já codificado.

## Modelos e primeiro uso

O fluxo padrão usa SFT XL puro, arquitetura correspondente, VAE ACE e embedding Qwen3, além de **Qwen2.5-Omni-7B** para descrições e **Whisper large-v3** para letras. Os modelos ACE usam `checkpoint_dir`; os modelos de anotação usam o cache local Hugging Face. Downloads completos e anotações/tensores compatíveis são reutilizados. A primeira preparação pode gastar bastante tempo baixando modelos; as seguintes reutilizam os arquivos.

Reserve espaço além dos cerca de 22 GB do conjunto ACE: modelos de anotação, ambiente, trechos e caches ocupam mais. Depois de baixados, os modelos executam localmente. A preparação libera os modelos de anotação antes de codificar os tensores ACE e usa um dispositivo escolhido; a sincronização entre várias GPUs é uma opção de treinamento.

Fontes oficiais: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) e [modelos ACE-Step](https://huggingface.co/ACE-Step/Ace-Step1.5). O card Qwen identifica Apache 2.0. O card Hugging Face do Whisper identifica esse repositório como Apache 2.0; a [licença do software Whisper original](https://github.com/openai/whisper/blob/main/LICENSE) é MIT. Preserve as licenças efetivamente baixadas e a atribuição [CC BY-NC-SA 4.0](../LICENSE) do projeto ao redistribuir.

## Pela linha de comando

Use o ambiente ativado do projeto (`.venv\Scripts\Activate.ps1` no Windows ou `source .venv/bin/activate` no Linux). Também é possível substituir `python -m jk_step` por `jk-step.bat` ou `bash jk-step.sh`.

```powershell
# Apenas inventário: não carrega ou baixa modelos.
python -m jk_step dataset inspect --audio-dir "E:/Minhas músicas"

# MR automático: anotações, comparações controladas e tensores pareados.
python -m jk_step dataset prepare --audio-dir "E:/Minhas músicas" --output datasets/minhas_musicas_mr --objective flow_dpo
python -m jk_step train --preset conservative --pairs-manifest "<pairs_manifest retornado>" --max-latent-length 0 --output-dir output/meu_lora_mr

# Alternativamente, SFT normal: um alvo por trecho preparado.
python -m jk_step dataset prepare --audio-dir "E:/Minhas músicas" --output datasets/minhas_musicas --objective sft
python -m jk_step config --preset sft_lora --output configs/meu_sft.json
python -m jk_step train --config configs/meu_sft.json --dataset-manifest datasets/minhas_musicas/supervised_manifest.json --output-dir output/meu_lora_sft
```

A preparação retorna `objective`, `ready`, `dataset_json`, `tensor_dir` e `report`. **MR usa `pairs_manifest` retornado**; **SFT usa `dataset_manifest`**. Substitua o texto entre sinais de menor/maior pelo caminho real; a UI preenche automaticamente. Indique o objetivo na CLI: seu padrão histórico é SFT, enquanto a UI inicia com MR selecionado. `partial` pronto tem amostras válidas e algumas exclusões; confira o relatório. Preparação falha/cancelada não é pronta.

Para personalizar, salve este exemplo em `configs/minha_preparacao.json`:

```json
{
  "audio_dir": "E:/Minhas músicas",
  "output_dir": "datasets/minhas_musicas",
  "checkpoint_dir": "checkpoints",
  "model_variant": "xl-sft",
  "objective": "flow_dpo",
  "pair_options": {
    "degradation": "mixed",
    "variations_per_audio": 3,
    "cutoff_hz": 6000,
    "noise_snr_db": 24,
    "clip_threshold": 0.15
  },
  "caption_model": "Qwen/Qwen2.5-Omni-7B",
  "caption_tier": "auto",
  "lyrics_model": "openai/whisper-large-v3",
  "language": "auto",
  "content_mode": "auto",
  "clip_seconds": 30,
  "validation_fraction": 0.1,
  "seed": 42,
  "allow_download": true,
  "reuse": true,
  "preprocess": true,
  "normalize": "none"
}
```

```powershell
python -m jk_step dataset prepare --config configs/minha_preparacao.json
```

`--options` aceita objeto ou arquivo JSON. `{"language":"pt"}` fixa português; `{"objective":"flow_dpo","pair_options":{"degradation":"noise"}}` seleciona comparações com ruído. Use `objective=sft` para alvo único. `preprocess=false` produz anotações/comparações brutas sem dataset de tensores pronto; inclua pré-processamento antes de treinar. Veja argumentos com `--help`.

## Arquivos produzidos

| Caminho dentro da saída escolhida | Finalidade |
| --- | --- |
| `audio/` | Trechos preparados e sidecars correspondentes. |
| `annotation_cache/` | Reconhecimento/captions reutilizáveis conforme fontes e opções. |
| `dataset.json` | Áudios, captions, letras dos trechos, tempos e procedência para revisão/pré-processamento. |
| `sources.json` | Caminhos/hashes de origem e registro de preservação dos originais. |
| `tensors/<fingerprint>/` | Somente SFT: caches `.pt` compatíveis com Side-Step. |
| `supervised_manifest.json` | Somente SFT: caminhos/hashes de alvo único e splits por grupos. |
| `preferences/<fingerprint>/` | Somente MR: áudios de comparação, pares brutos e caches pareados. |
| `preparation_report.json` | Contagens, origem dos modelos, exclusões e avisos. |

`dataset.json` sozinho não é o cache de treino. SFT usa `dataset_manifest=supervised_manifest.json`; também aceita manifests/diretórios Side-Step. MR usa o `pairs_manifest` pré-processado retornado, não `pairs.json` bruto. Variantes agrupadas de gravação/artista ficam juntas no split e as repetições se aplicam somente ao treino.

## Opções editáveis do treino

MR automático usa `conservative`: rank 32/alpha 64, LR `1e-6`, beta 100, regularização FM do escolhido 0.1, batch 1, acumulação 8 e duas épocas. A UI de pasta mantém trechos inteiros (`max_latent_length=0`) e dropout CFG 0. `rank64` é o preset correspondente de preferências. São valores iniciais, não um ótimo de qualidade comprovado.

O preset `sft_lora` começa com:

| Opção | Valor inicial |
| --- | --- |
| Objetivo | `sft` |
| Rank / alpha | 32 / 64 |
| Learning rate / optimizer | `1e-5` / AdamW |
| Batch por GPU / acumulação | 1 / 4 |
| Épocas / passos de warmup | 100 / 50 |
| Dropout CFG | 0.1 |
| Corte latente adicional | 0: preserva cada trecho preparado |

São valores iniciais, não garantia de qualidade ou duração obrigatória do treinamento. Revise resultados externos e músicas geradas. Selecione `sft_rank64` para rank 64/alpha 128 ou altere qualquer opção suportada. Módulos/camadas, dropout, precisão, optimizer/scheduler, passos, checkpoints e hardware continuam configuráveis. `beta`, regularizações de referência/FM, label smoothing e pesos de pares não têm efeito no SFT.

Para duas GPUs:

```powershell
python -m jk_step train --config configs/meu_sft.json --dataset-manifest datasets/minhas_musicas/supervised_manifest.json --output-dir output/meu_sft_2gpu --multi-gpu --gpu-ids 0,1
```

O modelo e o batch local precisam caber em cada GPU; a VRAM não é somada. O batch efetivo é batch × acumulação × número de GPUs. Nesse preset, é 4 em uma GPU ou 8 em duas. Trechos longos, rank 64, alvos adicionais ou batch maior podem exigir mais memória. Veja as [notas multi-GPU](../README_PTBR.md#treinamento-multi-gpu-experimental).

## Parar, retomar e usar a LoRA

Parar na UI ou Ctrl+C na CLI solicita cancelamento na fronteira de atualização do optimizer e salva um checkpoint `stopped` retomável. Uma operação GPU ou um download pode precisar terminar antes da parada.

```powershell
python -m jk_step train --config output/meu_lora_sft/training_config.json --resume-from output/meu_lora_sft/stopped
```

Use o caminho real registrado em `latest.json`. A retomada restaura o estado e rejeita mudanças incompatíveis nos dados, objetivo ou opções. Use uma saída nova para um treino novo. Por padrão, o treino exporta `quality_comfyui.safetensors` e mantém o adapter PEFT em `final/`.

Aplique a exportação na mesma família SFT usada no treino. No loader LoRA SFT do ComfyUI, conecte a saída de modelo ao Generate; mantenha o text encoder original no TextEncode, porque os pesos CLIP/text encoder não foram treinados. Compare inicialmente forças de modelo como 0.5 e 1, com força de text encoder 0, mantendo seed, prompt, letra, CFG, sampler e códigos LM iguais. Avalie separadamente voz, palavras, instrumentos e estrutura musical; loss menor não comprova som melhor.

Para preferências MR humanas ou avaliadas, use o fluxo separado com `objective="flow_dpo"` e `pairs_manifest`. As comparações automáticas de pasta são explicitamente sintéticas e não presumem aqueles rótulos humanos/rewards. Consulte as [instruções de dataset por preferências](../README_PTBR.md#preparar-datasets-de-preferências).
