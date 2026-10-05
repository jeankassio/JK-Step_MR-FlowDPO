# Fontes e criação de datasets

Para os conjuntos em português com downloader limitado e importadores prontos, consulte [Portuguese starter datasets](PORTUGUESE_DATASETS.md). São 20 músicas JamendoLyrics PT e quatro faixas MulJam adicionais; os exemplos incluem trechos com letras alinhadas e um piloto acústico explicitamente experimental.

Você pode começar com gravações próprias/licenciadas e usar a ferramenta de dataset/sidecars do Side-Step para organizar caption, letra e metadata. Não é necessário gerar músicas com Turbo. O download de modelos do treinador é automático; downloads de datasets são seletivos porque algumas fontes ocupam centenas de GB.

| Fonte | Conteúdo | Uso possível |
| --- | --- | --- |
| [Muse](https://huggingface.co/datasets/bolshyC/Muse) | Mais de 116 mil músicas já geradas com Suno V5, letra/caption/estrutura | Diversidade e pares curados; não vem com preferências prontas. MIT declarada pelo card; conferir adequação da fonte. |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 150 músicas reais com stems de voz, bateria, baixo e outros | Pares com degradação apenas do vocal. Acesso acadêmico sob solicitação; precisa preencher letras/captions. |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | 55 mil faixas com tags musicais | Diversidade de gênero/instrumentos; usar arquivos stereo de qualidade completa. Pesquisa não comercial e licenças por faixa. |
| [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) | 79 músicas, palavras alinhadas, EN/FR/DE/ES | Holdout para avaliar dicção; preservar como avaliação se comparar nesse benchmark. Licenças por música. |

Para listar fontes:

```powershell
.\jk-step.bat sources
```

Para baixar apenas um arquivo HF (um arquivo Muse inglês é cerca de 11 GB):

```powershell
.\jk-step.bat sources --download bolshyC/Muse --file en_part01_of_35.tar --output datasets/downloads/muse
```

O downloader fixa a revisão e mostra os arquivos retornados. Não extrai archives automaticamente nem converte letras de qualquer schema sem conferir seu formato. Extraia apenas o que deseja e importe áudios/metadata pelo construtor Side-Step. Para fontes com acesso restrito, solicite acesso na própria fonte e configure autenticação HF quando necessário.

## Três maneiras de criar preferências

**Importação humana:** use `configs/pair_import.example.json`. `chosen` e `rejected` são arquivos alinhados da mesma intenção. Inclua `caption`, `lyrics`, `group_id` e opcional `holdout_group` por artista. A escolha pode ter peso `pair_weight`. Validação verifica duração e vazamento entre splits.

**MRSD:** use `pairs score --template` para criar um manifest com scores manuais/precalculados. Preencha scores e grupos e execute `pairs build` com `configs/mrsd.example.json`. Candidatos no mesmo grupo têm a mesma letra/caption. Scores ausentes não são inventados. Pode importar um arquivo adicional CSV/JSON de scores com `score_file`.

**Degradações controladas:** `mode=degraded` aceita lowpass, ruído, clipping e `vocal_lowpass`. Isso cria preferências acústicas artificiais rotuladas. O escolhido precisa ser bom de fato; o gerador não certifica musicalidade. Para `vocal_lowpass`, forneça `vocal_stems_dir`, stems alinhados de vocal e mixes; instrumentos permanecem na mix original. Resíduos de uma separação imperfeita podem contaminar a preferência.

## Recompensas opcionais

Instale `requirements-rewards.txt` (ou `install_windows.bat -Rewards`). CLAP usa o checkpoint musical; Audiobox utiliza Production Quality. Pesos desses avaliadores são downloads adicionais, autorizados na UI/opção `allow_download`; você também pode fornecer caminhos locais.

```json
{"providers":["clap","audiobox"],"allow_download":true,"device":"cuda:0"}
```

Salve essas opções em um JSON e use `pairs score --input candidatos.json --output scores.json --options opcoes.json`. CLAP não substitui a conferência de palavras. A recompensa semântica do artigo precisa de features de HuBERT musical e centroides compatíveis; é possível importar scores ou fornecer esses recursos próprios. Leia [a seção de método](TRAINING.md) antes de chamar qualquer score genérico de “qualidade musical”.
