# Preparar datasets PT/EN de qualidade acústica

[English](QUALITY_DATASET_PREPARATION.md) · [Fontes de datasets](DATASETS.md)

O fluxo usa gravações existentes: músicas inteiras → grupos de linhas/seções completos e alinhados → amostras únicas PT/EN → pares com degradação controlada → pré-processar cada par único uma vez → balancear referências PT aos tensores em cache. Não gera músicas com Turbo/LM. As comparações lowpass são uma **adaptação experimental de preferências acústicas**, sem seleção MRSD completa por recompensas nem comprovação de melhora na composição, dicção ou clareza vocal.

Execute os comandos na raiz do repositório. Os exemplos usam PowerShell no Windows; no Linux substitua por `.venv/bin/python` e `./jk-step.sh`. A obtenção/importação das fontes está no [guia inicial dos datasets em português](PORTUGUESE_DATASETS.md). Preserve downloads originais, manifests de músicas inteiras, atribuições e relatórios.

## 1. Manter músicas inteiras e alinhar trechos em português

`datasets/jamendolyrics_pt/dataset.json` contém áudios originais e letras humanas da fonte, sem timestamps. Faça o alinhamento desse texto antes de recortar:

```powershell
.\.venv\Scripts\python.exe scripts/align_portuguese_lyrics.py --manifest datasets/jamendolyrics_pt/dataset.json --output datasets/jamendolyrics_pt_aligned --device cuda:0 --dtype float16 --model-cache .cache/mms_fa --min-seconds 8 --max-seconds 30 --max-gap 5 --padding 0.25 --min-line-score 0.35 --max-low-character-fraction 0.25 --max-word-seconds 8
```

Para inferência CPU, use `--device cpu --dtype float32`. O comando normal de alinhamento baixa o checkpoint MMS_FA fixado, de aproximadamente 1,26 GB, se estiver ausente, e confere tamanho/SHA256. O script mantém palavras, acentos e pontuação da fonte; uma cópia normalizada é usada somente no alinhamento CTC. A letra não é substituída por transcrição ASR. Marcadores de seção e letras numéricas que exigem pronúncia explícita vão para quarentena, sem remoção silenciosa.

MMS_FA é um modelo de fala aplicado experimentalmente ao canto. Seus scores CTC são **indicadores de suporte não calibrados**; não representam probabilidade de palavras corretas, alinhamento revisado por humanos ou qualidade musical. Os limites do exemplo rejeitam tempos fracos/implausíveis e preservam grupos de linhas inteiras, mas não certificam os trechos restantes. Ouça as palavras nos limites e revise `quarantine.json` e `alignment_report.json` antes de escolher positivos.

As saídas incluem `dataset.json`, `full_tracks.json`, `alignments/`, `quarantine.json` e `alignment_report.json`. O alinhador recebe uma cópia mono de 16 kHz apenas para inferência. Os recortes de treino mantêm ganho, sample rate e canais originais, em FLAC PCM24 ou WAV FLOAT32 quando os picos decodificados ultrapassam a faixa PCM inteira. Os arquivos originais permanecem no lugar.

Para reconstruir recortes usando um cache FP16 existente, sem inferência nem novo download de modelo:

```powershell
.\.venv\Scripts\python.exe scripts/align_portuguese_lyrics.py --manifest datasets/jamendolyrics_pt/dataset.json --output datasets/jamendolyrics_pt_aligned --model-cache .cache/mms_fa --cache-only --dtype float16 --min-seconds 8 --max-seconds 30 --min-line-score 0.35
```

O modelo fixado e o cache compatível com fonte/texto/inferência precisam estar presentes. `--dtype` deve corresponder ao cache; um cache ausente/incompatível interrompe a reconstrução. O alinhador não possui `--seed`. A seed 42 dos passos seguintes controla seleção/splits, sem calibrar a confiança do alinhamento.

O manifest separado `datasets/muljam_pt/dataset.json` já contém recortes de linhas inteiras alinhadas automaticamente pelo `import_muljam_pt.py`; ele pode ser combinado com os trechos PT aceitos. Preserve os relatórios de anotações/fontes e as licenças por faixa.

## 2. Selecionar seções inteiras do Muse em inglês

Parta do manifest local de músicas inteiras criado por `scripts/import_muse.py`, aqui `datasets/muse_english/dataset.json`:

```powershell
.\.venv\Scripts\python.exe scripts/segment_muse_dataset.py --input datasets/muse_english/dataset.json --output datasets/muse_english_sections_30s --max-samples 1000 --max-seconds 30 --min-seconds 8 --max-per-song 2 --seed 42 --validation-fraction 0.1
```

O script offline seleciona uma variante por `song_id`, no máximo dois recortes por música e uma ou duas seções consecutivas completas por recorte. Os intervalos não podem superar cinco segundos. Letras vazias e textos formados apenas por marcadores, como `[Instrumental]`, são excluídos. Uma seção acima de 30 segundos é relatada e excluída, sem truncamento. As seções selecionadas precisam caber no áudio decodificado e não podem cortar outra seção sobreposta que ficou fora do recorte.

As captions mantêm o estilo original e os `desc` conhecidos das seções selecionadas; a letra mantém o texto exato e seus rótulos. Não são inventados tempos por palavra nem scores de qualidade. As duas variantes da fonte e eventuais variantes degradadas permanecem no mesmo holdout por música. `crop_report.json` registra hashes de origem/recorte, limites por amostra, exclusões e formato final. A gravação lossless mantém as limitações do MP3; descrições/tempos automáticos do Muse e qualidade dos positivos seguem sem verificação.

## 3. Combinar amostras únicas antes dos pares e das repetições

```powershell
.\.venv\Scripts\python.exe scripts/combine_quality_datasets.py --pt-input datasets/jamendolyrics_pt_aligned/dataset.json --pt-input datasets/muljam_pt/dataset.json --en-input datasets/muse_english_sections_30s/dataset.json --output datasets/quality_pt_en_unique --validation-fraction 0.15 --seed 42 --no-balance
```

O combinador preserva caption/letra, registra os manifests de origem, deduplica o áudio atual por SHA256 e rejeita condicionamentos conflitantes. O holdout PT usa nomes de artista normalizados/IDs canônicos `artist:<hash>` entre fontes; EN usa o `song_id` original do Muse. O `group_id` do condicionamento é separado do holdout. Os splits são estratificados por idioma, portanto adicionar músicas EN não altera os artistas PT da validação.

O snapshot local contém **83 trechos PT + 1.000 trechos EN = 1.083 amostras únicas**: 53 recortes de alinhamento JamendoLyrics aceitos e 30 MulJam. São contagens observadas localmente, sem arquivos incluídos no repositório nem garantia de reproduzir esses totais com outros filtros/caches. Confira contagens e artistas/músicas em holdout no `combine_report.json`. Não repita linhas de amostras nesta etapa.

## 4. Criar pares acústicos controlados

Após revisar os originais:

```powershell
.\jk-step.bat pairs build --input datasets/quality_pt_en_unique/dataset.json --output datasets/quality_pt_en_unique_pairs --options configs/portuguese_acoustic_pairs.example.json
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_unique_pairs/pairs.json
```

O exemplo indicado usa seed 42 e uma variação lowpass de 6 kHz por original. `chosen` é o recorte existente; `rejected` é sua **mix completa** filtrada, com a mesma duração, caption e letra. Isso não isola a voz. Uma gravação ruim continua sendo um positivo ruim; a comparação não certifica a fonte nem inventa recompensas. Experiências apenas com o vocal exigem stems alinhados. MRSD natural exige candidatos comparáveis e scores medidos nos eixos de recompensa selecionados; este fluxo não os fornece.

As opções de pares do exemplo têm sua própria fração intermediária de validação. A combinação final do passo 6 define o split por artista/música usado no treino, com seed/fração explícitas; termine essa preparação antes de treinar a partir dele.

## 5. Pré-processar cada par único uma vez

```powershell
.\jk-step.bat pairs preprocess --manifest datasets/quality_pt_en_unique_pairs/pairs.json --checkpoint-dir checkpoints --model-variant xl-sft --output datasets/quality_pt_en_tensors --options configs/portuguese_preprocess.example.json
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_tensors/pairs.preprocessed.json --check-tensors
```

O exemplo usa a média do posterior VAE, `normalize=none` e `max_duration=0`, mantendo os trechos alinhados completos. O pré-processamento pode adaptar o áudio ao formato exigido pelo encoder ACE; essa etapa é separada da preservação dos arquivos originais/recortes. `context_mode=chosen_semantic` extrai um plano da gravação escolhida existente e o compartilha entre os dois lados, sem gerar música. Texto/letra também são compartilhados; a gravação rejeitada contribui com seu target VAE.

Codifique os pares únicos antes de balancear PT. Caches compatíveis são reutilizáveis; mudanças no áudio/condicionamento/pré-processamento exigem novos fingerprints. Não pré-processe novamente as linhas replicadas para treino.

## 6. Balancear somente referências aos tensores PT de treino

`--paired` exige manifests pré-processados separados por idioma. Se o passo 5 produziu um manifest misto, divida **somente seus registros JSON**, mantendo os mesmos caminhos de áudio, caminhos de tensores e metadata do pré-processamento:

```powershell
@'
import json
from pathlib import Path
root = Path("datasets/quality_pt_en_tensors")
raw = json.loads((root / "pairs.preprocessed.json").read_text(encoding="utf-8-sig"))
groups = {language: [] for language in ("pt", "en")}
for pair in raw["pairs"]:
    language = pair.get("language") or pair.get("metadata", {}).get("language")
    if language not in groups:
        raise ValueError(f"Unknown language for pair {pair['id']}")
    groups[language].append(pair)
for language, pairs in groups.items():
    if not pairs:
        raise ValueError(f"No {language} preprocessed pairs")
    target = root / f"pairs.{language}.json"
    target.write_text(json.dumps({**raw, "pairs": pairs}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
'@ | .\.venv\Scripts\python.exe -

.\.venv\Scripts\python.exe scripts/combine_quality_datasets.py --paired --pt-input datasets/quality_pt_en_tensors/pairs.pt.json --en-input datasets/quality_pt_en_tensors/pairs.en.json --output datasets/quality_pt_en_training --pt-fraction 0.6 --max-pt-repeat 20 --validation-fraction 0.15 --seed 42
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_training/pairs.combined.json --check-tensors
```

No Linux, execute o bloco Python com um here-document do shell ou salve-o em um `.py` temporário. Se PT/EN foram pré-processados separadamente uma vez, forneça diretamente esses dois manifests.

Somente linhas PT de **treino** são repetidas. As réplicas mantêm o mesmo `tensor_path`; áudios/tensores não são copiados nem gerados novamente, e a validação não é repetida. O fator inteiro fica limitado a 20, portanto a proporção PT alcançada pode diferir de 0,6; confira `combine_report.json`. Treine a partir do `pairs.combined.json` final. Com o balanceamento no manifest, mantenha `dataset_repeats=1`; use `max_latent_length=0` nestes trechos inteiros já alinhados para evitar um recorte arbitrário adicional acompanhado da letra completa do trecho.

## Licenças e redistribuição

Preserve licença, atribuição, URLs e restrições da fonte de cada gravação. Termos Jamendo/MTG e declarações por faixa precisam ser considerados individualmente; combinar ou recortar não concede novos direitos. O card do Muse declara MIT para a coleção publicada; preserve a proveniência sem tratar isso como autorização geral para todo uso derivado.

O checkpoint de alinhamento tem a [licença MMS CC-BY-NC 4.0](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md#license) separada. O alinhador registra publicador, SHA256 fixado e licença em `model.source.json`. Ele não vem incluído no JK-Step, e seus termos não comerciais permanecem separados da licença do treinador e das gravações. Os pacotes de distribuição incluem scripts/docs, sem gravações baixadas, pesos de alinhamento, caches de tensores ou adapters treinados.
