# JK-Step MR-FlowDPO

[English](README.md) · [Português (Brasil)](README_PTBR.md) · **Español**

Entrenamiento LoRA supervisado y por preferencias para **ACE-Step 1.5 SFT y SFT XL**, con línea de comandos, interfaz local en el navegador y herramientas de dataset derivadas de [Side-Step](https://github.com/koda-dernet/Side-Step). El flujo de carpeta prepara automáticamente descripciones, letras, fragmentos, JSON y tensores de entrenamiento a partir de **una carpeta de música**.

El objetivo es investigar la calidad musical y acústica general conservando la fidelidad de la letra. Puedes crear datasets a partir de grabaciones existentes; **no necesitas generar canciones con Turbo**. El adapter modifica el modelo de generación, sin aplicar masterización, ecualización ni otro posprocesamiento a las canciones generadas.

![Interfaz de JK-Step MR-FlowDPO en el navegador](assets/Screenshots/JK-Step_MR-FlowDPO.png)

**Estado de la investigación:** el modo de preferencias es una adaptación experimental de [MR-FlowDPO](https://arxiv.org/abs/2512.10264) para ACE-Step. El artículo evaluó generación instrumental con otros modelos de flow matching. El modo SFT usa flow matching supervisado normal, sin comparaciones de preferencias. Los tests verifican que el entrenador ejecuta, actualiza el LoRA y congela el modelo base; no demuestran mejora musical ni garantizan la pronunciación. Consulta los [resultados de validación](docs/VALIDATION.md).

## Funciones incluidas

- Preparación de carpeta a LoRA: descripciones locales Qwen2.5-Omni-7B, letras/timestamps Whisper large-v3, fragmentos, JSON y tensores. Elige MR-FlowDPO automático o SFT normal. La UI no requiere canciones Turbo, Genius/API ni JSON escrito manualmente.
- Los pares MR-FlowDPO automáticos comparan cada fragmento existente con copias degradadas mediante lowpass, ruido o clipping. Estas comparaciones sintéticas enseñan a evitar defectos específicos; no establecen preferencias de calidad musical general ni reproducen las recompensas MRSD del artículo.
- SFT con un solo objetivo y pérdida de flow matching en FP32 con máscaras, separación por grupos antes de repetir datos y soporte de una o varias GPU en el mismo motor de checkpoints, reanudación y exportación.
- Objetivo Flow-DPO real, con el SFT original congelado como referencia, sin conservar una segunda copia del XL por GPU.
- Rank, alpha, dropout, rsLoRA, ajustes por módulo, capas del decoder, self/cross attention y objetivos MLP configurables.
- Selección de pares mediante varias recompensas, importación humana, degradaciones controladas, preprocesamiento compartido y cachés de tensores.
- Configuración por flags CLI, JSON o UI; supervisión, cancelación, checkpoints, reanudación y exportación ComfyUI.
- Organización de datasets, captions, letras, sidecars, análisis de audio, stems, PP++ y herramientas supervisadas de adapters heredadas de Side-Step. Algunas integraciones requieren paquetes opcionales, credenciales de API u otros modelos.
- Descarga automática del SFT XL puro y los modelos de preprocesamiento predeterminados, en el entorno y la carpeta de checkpoints del entrenador.

## Instalación

### Windows

1. Clona o extrae este repositorio en una carpeta donde puedas escribir.
2. Ejecuta `install_windows.bat`. Crea `.venv`, instala dependencias y descarga los modelos predeterminados.
3. Ejecuta `start_ui.bat`. La UI abre en `http://127.0.0.1:8771` con un token de sesión.

No requiere permisos de administrador ni una instalación de ComfyUI. El instalador usa su propio entorno Python y no actualiza paquetes de ComfyUI.

```powershell
.\jk-step.bat doctor
.\jk-step.bat --help
.\jk-step.bat train --help
```

El entorno verificado usa Python 3.12, PyTorch 2.10, CUDA 12.8 y TorchCodec 0.10. [constraints.txt](constraints.txt) fija las versiones comprobadas.

| Opción del instalador | Efecto |
| --- | --- |
| `install_windows.bat -CoreOnly` | Instala entrenador/UI y el flujo local de descripciones/transcripción, sin integraciones opcionales de APIs remotas, separación de stems y UI del terminal. |
| `install_windows.bat -Rewards` | Instala también los evaluadores opcionales CLAP y Audiobox. |
| `install_windows.bat -SkipModels` | Omite las descargas; usa modelos locales o ejecuta `models setup` después. |
| `install_windows.bat -Backend cpu` | Instala PyTorch CPU para desarrollo o tests pequeños. Entrenar XL en CPU es lento. |

La preparación de carpeta decodifica con SoundFile y recurre al ejecutable `imageio-ffmpeg` incluido cuando hace falta; este flujo no requiere configurar FFmpeg externo. Las funciones opcionales del toolkit heredado que usan TorchCodec pueden necesitar bibliotecas compartidas FFmpeg compatibles; consulta las [instrucciones oficiales de TorchCodec](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).

### Linux

Instala [uv](https://docs.astral.sh/uv/getting-started/installation/) y ejecuta:

```bash
bash install_linux.sh
bash jk-step.sh doctor
bash jk-step.sh gui
```

`JK_BACKEND=cpu` selecciona paquetes CPU; `JK_SKIP_MODELS=1` omite los modelos. Hay instaladores para Windows y Linux. El motor acepta dispositivos MPS, pero el entrenamiento XL no se ha validado en macOS.

### Almacenamiento y memoria GPU

Reserva unos **22 GB para los modelos ACE de generación/preprocesamiento**, además del entorno, audio, cachés, adapters y las descargas adicionales Qwen/Whisper para la preparación de carpeta. El SFT XL ocupa aproximadamente 20 GB. El nombre estándar Hugging Face se crea como hardlink cuando es posible; otros sistemas de archivos pueden necesitar otra copia de unos 20 GB para compatibilidad con el toolkit.

Se utilizó una RTX 3090 de 24 GB para tests de ejecución. La VRAM real depende de la duración de los fragmentos, batch, rank, capas y precisión. Empieza con batch 1, precisión mixta, offload y gradient checkpointing. El modo predeterminado selecciona GPU 0; puedes elegir otro dispositivo compatible, por ejemplo `--device cuda:1` o `--device cpu`.

## Modelos

```powershell
# El instalador predeterminado ya realiza este paso.
.\jk-step.bat models setup --checkpoint-dir checkpoints
```

| Componente | Origen |
| --- | --- |
| Pesos SFT XL puro | [jeankassio/acestep_v1.5_sft_xl](https://huggingface.co/jeankassio/acestep_v1.5_sft_xl), en la revisión fijada y registrada por la aplicación. |
| Arquitectura, configuración y latente de silencio correspondientes | [ACE-Step/acestep-v15-xl-sft](https://huggingface.co/ACE-Step/acestep-v15-xl-sft), fijados por separado. |
| VAE y Qwen3-Embedding-0.6B | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5), con la revisión resuelta registrada localmente. |
| Descripciones locales | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), descargado cuando es necesario para anotar audio. |
| Letras/timestamps locales | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), descargado cuando es necesario para transcribir. |

Las descargas se pueden reanudar y los archivos completos se reutilizan. La preparación descarga automáticamente los modelos ACE y de anotación que falten cuando está permitido. Este flujo no requiere un checkpoint Turbo/Merge, LM generador de música, token Genius ni API remota de captions/transcripción. Los modelos de evaluación/toolkit son descargas adicionales.

Para usar un safetensors existente, configura `checkpoint_file` y su `model_config_dir` correspondiente. Se verifican nombres y dimensiones antes de asignar pesos. `--no-auto-download` desactiva la descarga automática predeterminada. Los directorios Hugging Face pueden usar `checkpoint_dir` y `model_variant` (`xl-sft`, `sft`, `xl-base` o `base`). El motor no habilita entrenamiento por preferencias del Turbo destilado.

## Interfaz y herramientas de dataset

```powershell
.\jk-step.bat gui
# Opcional: cambia el puerto y no abre el navegador automáticamente.
.\jk-step.bat gui --port 8772 --no-browser
```

Selecciona **English**, **Português** o **Español** en el selector de idioma de la UI. El navegador guarda esta preferencia. Cambia los textos de la interfaz independientemente del idioma de las letras/transcripción (`auto`, `pt`, `en`, `es`, etc.); una UI en español puede preparar música en portugués.

La página principal abre la preparación de carpeta con MR-FlowDPO seleccionado; SFT normal es otra opción. Las pestañas de dataset, preparación y entrenamiento reúnen selección, preparación automática, revisión, configuración, logs y cancelación. Importar pares humanos y seleccionar candidatos evaluados sigue disponible por separado. El toolkit abre organización de audio, sidecars, captions/letras, análisis, stems, PP++ y herramientas heredadas de adapters.

```powershell
.\jk-step.bat toolkit --help
.\jk-step.bat toolkit dataset --help
.\jk-step.bat toolkit captions --help
.\jk-step.bat toolkit preprocess --help
```

`sidestep` conserva su función de alias de `toolkit`. [UPSTREAM.md](UPSTREAM.md) preserva la documentación histórica y atribución; [docs/toolkit](docs/toolkit) contiene guías heredadas. El paquete Python actual es `jk_engine`; la entrada opcional de la UI del terminal es `jk_step_tui.py`.

## Inicio rápido: carpeta de música a LoRA

1. Abre `start_ui.bat`. En dataset, elige tu carpeta y objetivo: **MR-FlowDPO** automático o **SFT** normal.
2. Inicia la preparación. La aplicación descarga modelos ausentes, anota localmente, crea fragmentos con la letra correspondiente y escribe dataset/tensores fuera de los originales.
3. En preparación, revisa captions/letras y los elementos excluidos. La transcripción del canto y los timestamps automáticos pueden ser incorrectos.
4. Continúa a entrenamiento. El manifiesto correcto y el preset se rellenan automáticamente; selecciona una salida nueva, ajusta las opciones e inicia el LoRA.

MR-FlowDPO usa `conservative`: rank 32, alpha 64, LR `1e-6`, beta 100, regularización FM de la elegida 0.1, batch 1, acumulación 8 y dos épocas. Conserva los fragmentos (`max_latent_length=0`, dropout CFG 0). Las comparaciones mixtas predeterminadas incluyen lowpass, ruido y clipping; son defectos artificiales conocidos, no valoraciones humanas de la composición.

SFT normal usa `sft_lora`: rank 32, alpha 64, LR `1e-5`, batch 1, acumulación 4, 100 épocas, warmup 50, dropout CFG 0.1 y `max_latent_length=0`. Son valores iniciales editables, no óptimos demostrados. `sft_rank64` ofrece rank 64/alpha 128. El contenido puede seleccionarse como vocal, instrumental o detección automática.

En el entorno activado del proyecto:

```powershell
python -m jk_step dataset inspect --audio-dir "E:/Mi música"

# MR-FlowDPO automático; usa pairs_manifest del resultado final.
python -m jk_step dataset prepare --audio-dir "E:/Mi música" --output datasets/mi_musica_mr --objective flow_dpo
python -m jk_step train --preset conservative --pairs-manifest "<pairs_manifest devuelto>" --max-latent-length 0 --output-dir output/mi_lora_mr

# Alternativamente, SFT normal con un único objetivo.
python -m jk_step dataset prepare --audio-dir "E:/Mi música" --output datasets/mi_musica --objective sft
python -m jk_step config --preset sft_lora --output configs/mi_sft.json
python -m jk_step train --config configs/mi_sft.json --dataset-manifest datasets/mi_musica/supervised_manifest.json --output-dir output/mi_lora_sft
```

Usa las rutas reales devueltas: **`pairs_manifest` para MR-FlowDPO**, **`dataset_manifest` para SFT**. Ambos escriben anotaciones en `dataset.json`; solo SFT escribe `supervised_manifest.json`, sin scores ni audio rechazado. MR automático crea copias con degradación controlada y tensores pareados, sin JSON manual de pares. Consulta la guía de carpeta a LoRA en [Español](docs/FOLDER_TO_LORA_ES.md), [English](docs/FOLDER_TO_LORA.md) o [Português](docs/FOLDER_TO_LORA_PTBR.md).

Un test funcional acotado completó la preparación de carpeta con modelos locales, JSON/tensores, entrenamiento SFT iniciado por la UI y exportación. Otro test MR automático preparó tres pares controlados de audio en portugués, codificó ambas ramas, ejecutó una actualización Flow-DPO real y exportó pesos finitos. Dos RTX 3090 se utilizaron en otro test SFT. Verifican ejecución y archivos guardados; no miden mejora audible, precisión de las letras ni rendimiento con datasets arbitrarios.

## Preparar datasets de preferencias

Cada par contiene una grabación **elegida** y otra **rechazada** con la misma caption y letra pretendidas. Conserva comienzos alineados y duraciones iguales. La elegida debe ser mejor según los criterios que quieres enseñar.

Proporciona las palabras cantadas, una descripción musical útil y metadatos correctos. No etiquetes una canción vocal como `[Instrumental]`. Revisa captions/transcripciones automáticas. Para calidad general, combina géneros, idiomas, cantantes, tempos, instrumentos y arreglos. Un dataset limitado puede enseñar un estilo limitado.

### Pares elegidos por personas

Edita [configs/pair_import.example.json](configs/pair_import.example.json) e impórtalo. Las rutas se resuelven respecto al manifiesto. `group_id` identifica la grabación/prompt; `holdout_group` puede identificar al artista u otro grupo de validación. `pair_weight` controla opcionalmente el peso del par.

```powershell
.\jk-step.bat pairs build --input configs/pair_import.example.json --output datasets/human_pairs --options configs/import.example.json
```

Evita que variantes de una grabación o artista aparezcan tanto en entrenamiento como en validación.

### Selección mediante varias recompensas

Crea una plantilla de puntuación manual, agrupa candidatos comparables, completa los scores y aplica MRSD:

```powershell
.\jk-step.bat pairs score --input my_audio --output datasets/scores.json --template
.\jk-step.bat pairs build --input datasets/scores.json --output datasets/mrsd --options configs/mrsd.example.json
```

El ejemplo usa alineación textual, calidad de producción y consistencia semántica, con `lyric_fidelity` protegido. Completa cada score requerido o configura menos ejes. Los scores ausentes no se inventan. La selección requiere una mejora fuerte del eje primario, mejoras en otros ejes y restricciones de los protegidos; cuantiles, niveles mínimos, filtros de valores atípicos y equilibrio de ejes son configurables.

Los evaluadores automáticos opcionales incluyen CLAP musical y Audiobox Production Quality. La recompensa semántica del artículo requiere HuBERT entrenado para música y centroides compatibles; no están incluidos. Aporta recursos compatibles o scores manuales/precalculados. CLAP mide asociación audio/texto, no cada palabra cantada.

```json
{"providers":["clap","audiobox"],"allow_download":true,"device":"cuda:0"}
```

Guarda estas opciones en JSON y usa `pairs score --options`. Dos scores automáticos no completan todos los ejes del ejemplo de tres recompensas.

### Pares acústicos controlados

```powershell
.\jk-step.bat pairs build --input my_audio --output datasets/acoustic_pairs --options configs/degraded.example.json
```

El generador permite lowpass, ruido, clipping y lowpass vocal etiquetados. La degradación solo vocal requiere stems alineados y `vocal_stems_dir`; filtrar la mezcla completa no aísla al cantante.

Son comparaciones acústicas experimentales, no evidencia de mejor composición ni de solución al problema vocal SFT. Las degradaciones preparan ejemplos de entrenamiento; no procesan las canciones generadas.

### Fuentes de datos

Puedes usar tus propias grabaciones con licencia. El catálogo también enlaza:

| Fuente | Contenido y uso |
| --- | --- |
| [JamendoLyrics PT](https://huggingface.co/datasets/Felipehonorato/pt_it_jamendolyrics) | 20 originales portugueses con letras completas revisadas; licencias por canción, sin timestamps. |
| [Anotaciones MulJam PT](https://github.com/weAreMusicAI/alt-datasets-interspeech2025) | Cuatro originales portugueses seleccionados con tiempos por línea; permite importar fragmentos alineados. MTG establece uso académico/de investigación no comercial. |
| [Muse](https://huggingface.co/datasets/bolshyC/Muse) | Canciones existentes de Suno V5 con letra/estilo; requiere selección y preferencias. Su card declara MIT. |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | Canciones reales con stems vocales/instrumentales alineados para comparaciones vocales. El acceso académico debe solicitarse; aporta letras aparte. |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | Canciones completas con tags de género/instrumento/ambiente. La fuente especifica investigación académica no comercial y licencias por canción. |
| [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) | Benchmark de canto con palabras alineadas; reserva evaluación externa para valorar dicción. |

```powershell
.\jk-step.bat sources
# Descarga solo este archivo, no todo el dataset.
.\jk-step.bat sources --download bolshyC/Muse --file en_part01_of_35.tar --output datasets/downloads/muse
```

Este archivo Muse ocupa unos 11 GB. Los archivos no se extraen ni convierten universalmente a captions/letras de forma automática. Sigue su estructura e importa los audios/metadatos deseados. Consulta las [notas de datasets](docs/DATASETS.md).

Para la colección inicial portuguesa acotada (24 canciones existentes, unos 146 MB MP3 más anotaciones):

```powershell
.\.venv\Scripts\python.exe scripts/download_portuguese_datasets.py
.\.venv\Scripts\python.exe scripts/import_jamendolyrics_pt.py --root datasets/downloads/jamendolyrics_pt --output datasets/jamendolyrics_pt --license-filter all
.\.venv\Scripts\python.exe scripts/import_muljam_pt.py --annotations datasets/downloads/musicai_interspeech2025/dali_muljam_interspeech25.csv --metadata datasets/downloads/muljam_pt/audio_metadata.json --output datasets/muljam_pt
```

En Linux, sustituye `.venv/bin/python`. Los importadores preservan letras, hashes y licencias; el importador alineado escribe FLAC o WAV FLOAT32 sin cambiar ganancia o frecuencia de muestreo. No adjuntes una letra completa a un recorte arbitrario. Son muestras de origen, no preferencias MRSD clasificadas naturalmente. Consulta la [guía portuguesa](docs/PORTUGUESE_DATASETS.md) y sus límites.

La [guía de preparación PT/EN](docs/QUALITY_DATASET_PREPARATION.md) explica alineación forzada experimental portuguesa, secciones completas de Muse, combinación de muestras únicas y equilibrio PT después de preprocesar cada par una vez.

## Validar, preprocesar y entrenar

Para preferencias, ejecuta desde el proyecto con los manifiestos devueltos al construir pares:

```powershell
.\jk-step.bat pairs validate --manifest datasets/acoustic_pairs/pairs.json
.\jk-step.bat pairs preprocess --manifest datasets/acoustic_pairs/pairs.json --checkpoint-dir checkpoints --output datasets/tensors
.\jk-step.bat pairs validate --manifest datasets/tensors/pairs.preprocessed.json --check-tensors

.\jk-step.bat config --preset conservative --output configs/my_training.json
.\jk-step.bat train --config configs/my_training.json --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/my_quality_lora
```

El preprocesamiento almacena tensores alineados VAE, texto/letra y condicionamiento ACE. Ambas ramas comparten condicionamiento y recorte. Usa la media posterior VAE por defecto y no normaliza audio. Los fingerprints registran cambios de origen/configuración para reutilizar tensores compatibles.

`context_mode=chosen_semantic` extrae códigos de la **grabación existente elegida** y comparte su plan detokenizado en ambas ramas. No genera canciones Turbo/LM. `context_mode=silence` usa texto/letra sin ese plan. Entre candidatos con melodías diferentes, un plan solo del ganador puede sesgar la comparación; usa silencio si no hay justificación para el plan común. Pasa ajustes en un archivo JSON con `--options`.

`conservative` comienza con rank 32, alpha 64, LR `1e-6`, beta 100, regularización FM 0.1, batch 1, acumulación 8 y dos épocas. Son ajustes iniciales, no óptimos ACE-Step demostrados. `paper_beta` expone beta 2000 del artículo.

```powershell
# Campos disponibles por flags, JSON y UI.
.\jk-step.bat train --config configs/my_training.json --rank 64 --alpha 128 --layers 0-15 --learning-rate 0.000001 --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/rank64
```

Las opciones CLI prevalecen sobre el JSON. `--set KEY=VALUE` acepta valores JSON; claves incorrectas o no admitidas se rechazan.

| Área | Controles |
| --- | --- |
| Adapter | Rank/alpha/dropout, rsLoRA, ajustes por módulo, proyecciones, capas, self/cross attention y MLP. |
| Objetivo | `sft`: flow matching con un objetivo. `flow_dpo`: beta, regularizaciones FM/referencia, smoothing y pesos. Timesteps continuos y dropout CFG se aplican a ambos. |
| Optimización | Batch/acumulación, épocas o pasos máximos, AdamW/AdamW8bit/Adafactor, LR, weight decay/momentos, warmup, scheduler constante/lineal/coseno y clipping. |
| Hardware/datos | Dispositivo, precisión, offload, gradient checkpointing, workers, seeds, recortes alineados, repeticiones, verificación de caché y validación por grupo. |
| Salidas | Frecuencia de evaluación/logs, TensorBoard, frecuencia/retención de checkpoints, reanudación, inicialización con adapter y exportación ComfyUI. |

En Flow-DPO, batch cuenta **pares** con dos ramas y la referencia añade una pasada. En SFT, cuenta **audios individuales**, sin referencia. La acumulación aumenta el batch efectivo sin conservar todas las activaciones GPU. Instala los paquetes de optimizer opcionales cuando los selecciones; el toolkit heredado ofrece otras opciones.

### Entrenamiento multi-GPU experimental

```powershell
.\jk-step.bat train --config configs/my_training.json --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/two_gpus --multi-gpu --gpu-ids 0,1 --distributed-backend auto
```

`batch_size` es por GPU. El batch efectivo es `batch_size × gradient_accumulation × número_de_GPU`. Cada GPU conserva una réplica del decoder; **dos tarjetas de 24 GB no forman una memoria única de 48 GB**. Los fragmentos y el adapter deben caber en cada una.

En Linux, la selección automática usa NCCL. En Windows, usa `local`: sincroniza gradientes LoRA mediante memoria CPU y un administrador multiprocessing, sin Gloo/NCCL. `gloo` está disponible en builds PyTorch compatibles. El rendimiento depende del hardware/interconexión y la comunicación. Consulta [VALIDATION.md](docs/VALIDATION.md); soporte experimental no garantiza aceleración. La UI incluye estas opciones.

## Checkpoints, parada y ComfyUI

Las salidas incluyen `training_config.json`, `metrics.jsonl`, TensorBoard opcional, checkpoints y adapters PEFT. `final/` contiene `adapter_model.safetensors`, `adapter_config.json` y `training_state.pt`; `latest.json` registra las rutas reales.

```powershell
# Continúa el mismo entrenamiento con la configuración guardada.
.\jk-step.bat train --config output/my_quality_lora/training_config.json --resume-from output/my_quality_lora/checkpoint-00000100

# Exporta preservando el scaling efectivo.
.\jk-step.bat export --adapter output/my_quality_lora/final --output output/quality.safetensors --target native
```

La reanudación restaura optimizer, scheduler, scaler, RNG, época y cursor de datos. Cambios incompatibles de base/datos/opciones se rechazan. `init_adapter` inicia otro experimento con pesos compatibles; rank/alpha/dropout/rsLoRA deben coincidir. Usa una salida nueva para un entrenamiento nuevo.

Ctrl+C o Parar en la UI solicitan una parada cooperativa al finalizar una actualización del optimizer y guardan `stopped`. No se interrumpe instantáneamente un kernel GPU activo.

Con `export_comfyui=true`, la finalización escribe `quality_comfyui.safetensors`. Preserva scaling por módulo, rsLoRA y alpha personalizado. Usa `native` para el mapeo ACE específico o `generic` si lo requiere el loader. Aplica el LoRA a la misma familia SFT del entrenamiento y compara intensidades; no se presupone comportamiento equivalente en Turbo.

## Método y límites de evaluación

Con `objective=sft`, el entrenador aprende la velocidad de flow (`noise - target_latents`) del audio individual desde una interpolación ruidosa, con error cuadrático en FP32 y máscaras. Solo entrenan los pesos LoRA del decoder. Los fragmentos vocales no se pueden acortar aleatoriamente conservando letras completas. `beta`, regularización de preferencias, smoothing y pesos de pares no afectan a SFT.

Con `objective=flow_dpo`, el objetivo principal de cada par es:

```text
softplus(beta * ((chosen_error - rejected_error)
               - (reference_chosen_error - reference_rejected_error)))
```

Son errores de velocidad FP32 con máscaras. Ruido, timestep, caption, letra, contexto y dropout pareado se comparten. La referencia desactiva adapters y gradientes; solo entrenan LoRAs del decoder. Se usa consistentemente la convención ACE de datos a ruido.

Esto implementa Flow-DPO y selección por varias recompensas con adaptaciones ACE explícitas. **El reward prompting numérico del artículo no está implementado.** Añadir `quality=10` no lo reproduce. Sus resultados no se transfieren directamente a este modelo de canto ni a cualquier dataset. Consulta el [artículo](https://arxiv.org/abs/2512.10264), [código de sus autores](https://github.com/lonzi/mrflow_dpo) y [notas de entrenamiento](docs/TRAINING.md).

Menor pérdida o mejor margen no demuestran mejor sonido. Evalúa canciones/artistas/idiomas/prompts externos con iguales seeds y condiciones, comparando SFT con/sin LoRA. Valora claridad, artefactos, naturalidad, armonía, ritmo y palabras por separado. La transcripción del canto requiere revisión humana. Congelar el base o filtrar fidelidad de letra no garantiza pronunciación intacta después de entrenar.

## Desarrollo y distribución

```text
jk_step/          Motor SFT/preferencias, flujo de carpeta, CLI, UI y modelos
jk_engine/        Toolkit de dataset/supervisión adaptado de Side-Step
frontend/         Archivos de interfaz del navegador
configs/          Presets y ejemplos de pares
docs/             Método, datasets, validación y guías heredadas
scripts/          Empaquetado y tests de ejecución
tests/            Tests automatizados
UPSTREAM.md       README histórico del proyecto original
```

```powershell
.venv\Scripts\python.exe -m pip install pytest httpx
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe scripts/package_release.py
```

El ZIP excluye entornos, modelos, datasets, salidas, tokens y sesiones. Distribuye fuente/instaladores, no cachés locales ni `.venv`. Puedes crear una wheel con `uv build --wheel`; para el toolkit completo usa repositorio/ZIP e instaladores.

## Créditos y licencia

JK-Step MR-FlowDPO es una distribución modificada de **[Side-Step](https://github.com/koda-dernet/Side-Step), de koda-dernet**, ampliada con entrenamiento ACE por preferencias, creación/evaluación de pares, nueva UI/CLI, preparación de modelos y documentación. La licencia heredada es **[CC BY-NC-SA 4.0](LICENSE)**. Preserva [NOTICE.md](NOTICE.md), licencia y atribución al redistribuir; no concede uso comercial.

Runtime/arquitectura y modelos ACE proceden de [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). El SFT XL puro se descarga de [jeankassio/acestep_v1.5_sft_xl](https://huggingface.co/jeankassio/acestep_v1.5_sft_xl). El objetivo sigue [MR-FlowDPO](https://arxiv.org/abs/2512.10264); es una adaptación ACE, no el entrenador Audiocraft original. Dependencias, pesos y datasets mantienen sus licencias. No se presume respaldo de los proyectos originales.
