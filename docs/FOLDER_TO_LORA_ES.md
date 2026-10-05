# De una carpeta de música a LoRA: MR-FlowDPO automático o SFT

[English](FOLDER_TO_LORA.md) · [Português (Brasil)](FOLDER_TO_LORA_PTBR.md) · **Español** · [README del proyecto](../README_ES.md)

Coloca grabaciones existentes en una carpeta, selecciónala en JK-Step e inicia la preparación. La aplicación produce descripciones, transcripciones del canto, fragmentos de audio/texto y un dataset JSON. Elige **MR-FlowDPO automático** para crear preferencias sintéticas controladas y codificar ambas ramas, o **SFT normal** para codificar un objetivo por grabación. La UI prepara los manifests automáticamente: no requiere canciones Turbo, JSON manual de pares, scores de preferencias ni una clave Genius/API.

Las etiquetas automáticas son estimaciones. Whisper puede omitir, repetir o inventar palabras cantadas; sus tiempos pueden ser imprecisos. Qwen puede identificar mal instrumentos o voces. Preparar datos o reducir la pérdida del entrenamiento no demuestra mejor calidad audible ni una dicción perfecta.

## Usar la interfaz del navegador

Ejecuta `start_ui.bat` en Windows o `bash jk-step.sh gui` en Linux. Elige **English**, **Português** o **Español** en el selector de idioma de la UI; el navegador guarda la elección. Este ajuste es independiente del idioma de las letras/transcripción, por lo que interfaz y canciones pueden usar idiomas distintos.

1. En la pestaña de dataset, selecciona el flujo de carpeta y su objetivo: MR-FlowDPO (predeterminado en la UI) o SFT normal.
2. Elige la carpeta de música o pega su ruta absoluta. Se recorren las subcarpetas. El control de inspección permite obtener un inventario sin cargar modelos.
3. Elige voces, instrumentales o detección automática en el campo de contenido. Deja el idioma en `auto` o selecciona el conocido, por ejemplo `pt`, `en` o `es`.
4. Inicia la preparación. Los modelos faltantes se descargan automáticamente cuando está habilitado. Sigue el progreso en la pestaña de preparación; los modelos analizan audio local y el pipeline construye fragmentos y tensores.
5. Revisa la vista previa y el informe, especialmente el material excluido. Escucha los fragmentos correspondientes al comprobar palabras y límites.
6. Continúa al entrenamiento. Se rellenan automáticamente el manifest preparado y el preset del objetivo elegido. Selecciona una nueva carpeta de salida, ajusta las opciones e inicia el entrenamiento.

La preparación conserva los originales. Escribe fragmentos y sidecars en una carpeta de salida separada. No aplica normalización de volumen ni masterización; el preprocesamiento ACE convierte su copia de trabajo a la frecuencia de muestreo y los canales requeridos por el modelo.

Las descripciones/transcripciones locales están incluidas en la instalación core, también con `-CoreOnly`. La decodificación usa SoundFile y, cuando es necesario, el ejecutable incluido por `imageio-ffmpeg`; este flujo no requiere configurar FFmpeg externo. Algunas funciones opcionales del toolkit heredado tienen otras dependencias de decodificación.

Un test funcional acotado completó entrada de carpeta, anotación local Qwen/Whisper, JSON/tensores, entrenamiento SFT iniciado por la UI y exportación. Otro test MR automático preparó tres pares controlados de audio en portugués, codificó ambas ramas, ejecutó una actualización Flow-DPO y exportó pesos finitos. Dos RTX 3090 se utilizaron en otro test SFT. Son comprobaciones de funcionamiento, no evidencia de mejor música, precisión de palabras ni rendimiento general con datasets.

## Qué hace la preparación

| Etapa | Resultado |
| --- | --- |
| Exploración | Inventario recursivo de audio, sidecars existentes y hashes de fuentes. |
| Letras | Transcripción local Whisper large-v3 con tiempos estimados de frases. Las letras revisadas existentes se conservan cuando pueden asociarse de forma segura. |
| Fragmentos | Agrupa frases completas con tiempos hasta la duración configurada, normalmente 30 segundos, y adjunta solo las palabras de esas frases. Las frases demasiado largas o inutilizables pueden excluirse. |
| Descripciones | Qwen2.5-Omni-7B local escucha cada fragmento preparado y escribe una descripción musical. Se pueden reutilizar captions existentes compatibles. |
| Dataset | JSON de audio/texto, procedencia, separación entrenamiento/validación por grupos e informe de exclusiones. |
| Preferencias, solo MR | Compara el fragmento original con una copia que contiene defectos controlados; cada par comparte letra y descripción. |
| Preprocesamiento | Tensores VAE/texto/ACE: un objetivo SFT o ramas elegido/rechazado con condicionamiento compartido para MR-FlowDPO. |

Los límites de las frases proceden del reconocimiento automático, no de alineación verificada por personas. No se adjunta la letra completa de una canción a un recorte corto arbitrario. El preset SFT conserva los fragmentos preparados con `max_latent_length=0`; el entrenador rechaza un recorte latente más corto de un ejemplo vocal en vez de conservar silenciosamente palabras ajenas al audio.

## MR-FlowDPO automático desde la carpeta

Este modo usa el fragmento existente anotado como **elegido** y una copia con degradación acústica controlada como **rechazado**. No certifica que el original sea excelente: la etiqueta indica que se prefiere a esa versión artificialmente dañada. Inicio, duración, letra y descripción permanecen alineados; los originales no cambian. Se codifican ambas ramas y se entrenan con el objetivo Flow-DPO real, usando el SFT original congelado como referencia.

| `pair_options` | Valor inicial / función |
| --- | --- |
| `degradation` | `mixed`: pasa-bajos, ruido y clipping; o selecciona `lowpass`, `noise`, `clipping`. |
| `variations_per_audio` | 3 para mixed, 1 para un solo defecto elegido. |
| `cutoff_hz` | Frecuencia de corte pasa-bajos de 6000 Hz. |
| `noise_snr_db` | Relación señal/ruido de 24 dB. |
| `clip_threshold` | Umbral de amplitud de 0.15. |
| `context_mode` | `chosen_semantic`, o `silence` para condicionamiento estándar de texto/letra. |

Son preferencias acústicas sintéticas, **no evaluaciones de rewards MRSD del artículo** ni evidencia de calidad musical general. El filtro pasa-bajos afecta a toda la mezcla; no aísla las voces. Evitar estos defectos no equivale a mejorar composición, naturalidad del canto o pronunciación. Revisa las comparaciones y prueba música generada antes de valorar el adapter.

Los pares seleccionados por personas y la selección MRSD por scores siguen disponibles en el flujo separado; este experimento automático no los sustituye. El modo SFT de carpeta no crea grabaciones rechazadas ni scores de preferencias.

### Voces e instrumentales

- `content_mode="auto"`: usa reconocimiento y clasificación de audio conjuntamente. Una transcripción vacía no demuestra que el audio sea instrumental; los ejemplos inciertos o contradictorios se excluyen para revisión.
- `content_mode="vocal"`: prepara palabras cantadas y tiempos. Una transcripción fallida no recibe una etiqueta `[Instrumental]` inventada.
- `content_mode="instrumental"`: el usuario declara explícitamente que la carpeta es instrumental. La letra es `[Instrumental]`; una letra vocal existente contradictoria se rechaza.

Usa una carpeta coherente al seleccionar solo voces o solo instrumentales. La detección automática permite carpetas mixtas. La clasificación todavía puede equivocarse: inspecciona los ejemplos resultantes.

### Letras existentes o corregidas

Los metadatos son opcionales. Se reconocen sidecars como `cancion.caption.txt`, `cancion.lyrics.txt` o `cancion.txt` con campos clave/valor; `cancion` debe coincidir con el nombre del audio sin extensión. Los fragmentos revisados que caben en el límite de preparación pueden usar directamente su letra completa suministrada. Las grabaciones largas necesitan tiempos; si la letra suministrada discrepa de la transcripción, el material se separa para revisión en vez de imponer texto al audio incompatible.

Para corregir con precisión, prepara un fragmento corto con las palabras que realmente se cantan y su sidecar, y vuelve a ejecutar la preparación. La vista previa permite revisar; no certifica ni corrige automáticamente las letras. No edites solo los metadatos de un tensor en caché esperando cambiar su condicionamiento codificado.

## Modelos y primer uso

El pipeline predeterminado usa SFT XL puro, su arquitectura correspondiente, VAE ACE y embedding Qwen3, además de **Qwen2.5-Omni-7B** para descripciones y **Whisper large-v3** para letras. Los modelos ACE usan `checkpoint_dir`; los de anotación usan la caché local Hugging Face. Se reutilizan descargas completas y anotaciones/tensores compatibles. La primera preparación puede tardar bastante en descargar modelos; las siguientes reutilizan los archivos.

Reserva espacio adicional a los aproximadamente 22 GB de modelos ACE: modelos de anotación, entorno, fragmentos y cachés también ocupan disco. Después de descargarse, los modelos funcionan localmente. La preparación libera los modelos de anotación antes del preprocesamiento ACE y usa un dispositivo seleccionado; sincronizar varias GPU es una opción de entrenamiento.

Fuentes oficiales: [model card Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [model card Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) y [modelos ACE-Step](https://huggingface.co/ACE-Step/Ace-Step1.5). La ficha Qwen identifica Apache 2.0. La ficha Hugging Face de Whisper identifica ese repositorio como Apache 2.0; la [licencia del software Whisper original](https://github.com/openai/whisper/blob/main/LICENSE) es MIT. Conserva las licencias descargadas y la atribución [CC BY-NC-SA 4.0](../LICENSE) del proyecto al redistribuir.

## Flujo por línea de comandos

Activa el entorno del proyecto (`.venv\Scripts\Activate.ps1` en Windows o `source .venv/bin/activate` en Linux). También puedes sustituir `python -m jk_step` por `jk-step.bat` o `bash jk-step.sh`.

```powershell
# Solo inventario: no carga ni descarga modelos.
python -m jk_step dataset inspect --audio-dir "E:/Mi musica"

# MR automático: anotaciones, comparaciones controladas y tensores pareados.
python -m jk_step dataset prepare --audio-dir "E:/Mi musica" --output datasets/mi_musica_mr --objective flow_dpo
python -m jk_step train --preset conservative --pairs-manifest "<pairs_manifest devuelto>" --max-latent-length 0 --output-dir output/mi_lora_mr

# Alternativamente, SFT normal: un objetivo por fragmento preparado.
python -m jk_step dataset prepare --audio-dir "E:/Mi musica" --output datasets/mi_musica --objective sft
python -m jk_step config --preset sft_lora --output configs/mi_sft.json
python -m jk_step train --config configs/mi_sft.json --dataset-manifest datasets/mi_musica/supervised_manifest.json --output-dir output/mi_lora_sft
```

La preparación devuelve `objective`, `ready`, `dataset_json`, `tensor_dir` y `report`. **MR usa el `pairs_manifest` devuelto**; **SFT usa `dataset_manifest`**. Sustituye el texto entre signos de menor/mayor por la ruta real; la UI la rellena automáticamente. Especifica el objetivo en la CLI: su valor histórico predeterminado sigue siendo SFT, mientras la UI abre con MR seleccionado. Un resultado `partial` listo tiene ejemplos válidos y algunas exclusiones; revisa el informe. Una preparación fallida/cancelada no está lista.

Para personalizar la preparación, guarda este ejemplo como `configs/mi_preparacion.json`:

```json
{
  "audio_dir": "E:/Mi musica",
  "output_dir": "datasets/mi_musica",
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
python -m jk_step dataset prepare --config configs/mi_preparacion.json
```

`--options` admite otro objeto JSON o un archivo. Por ejemplo, `{"language":"es"}` fija la transcripción en español; `{"objective":"flow_dpo","pair_options":{"degradation":"noise"}}` selecciona comparaciones automáticas de ruido. Usa `objective="sft"` para un único objetivo. `preprocess=false` produce anotaciones/comparaciones sin un dataset de tensores listo: incluye preprocesamiento antes de entrenar. Consulta `--help` para los argumentos CLI.

## Archivos preparados

| Ruta dentro de la salida elegida | Función |
| --- | --- |
| `audio/` | Fragmentos preparados y sidecars correspondientes. |
| `annotation_cache/` | Transcripciones/descripciones reutilizables según fuentes y ajustes. |
| `dataset.json` | Audio, captions, letras de fragmentos, tiempos y procedencia para revisar/preprocesar. |
| `sources.json` | Rutas/hashes originales y registro de conservación de fuentes. |
| `tensors/<fingerprint>/` | Solo SFT: cachés `.pt` compatibles con Side-Step. |
| `supervised_manifest.json` | Solo SFT: rutas/hashes validados de un objetivo y splits por grupos. |
| `preferences/<fingerprint>/` | Solo MR: copias controladas, pares originales y cachés de tensores pareados. |
| `preparation_report.json` | Recuentos, procedencia de modelos, exclusiones y advertencias. |

`dataset.json` por sí solo no es la caché de entrenamiento. En SFT usa `dataset_manifest=supervised_manifest.json`; también se admiten manifests/directorios de tensores Side-Step existentes. En MR usa el `pairs_manifest` preprocesado devuelto, no `pairs.json` sin codificar. Las variantes agrupadas por grabación/artista permanecen juntas en los splits, y las repeticiones se aplican solo al entrenamiento.

## Ajustes editables de entrenamiento

MR automático usa `conservative`: rank 32/alpha 64, LR `1e-6`, beta 100, regularización chosen-FM 0.1, batch 1, acumulación 8 y dos épocas. La UI conserva fragmentos completos (`max_latent_length=0`) y dropout CFG 0. `rank64` es el preset de preferencias de rank 64. Son valores iniciales, no un óptimo de calidad demostrado.

El preset `sft_lora` empieza con:

| Ajuste | Valor inicial |
| --- | --- |
| Objetivo | `sft` |
| Rank / alpha | 32 / 64 |
| Learning rate / optimizer | `1e-5` / AdamW |
| Batch por GPU / acumulación | 1 / 4 |
| Épocas / actualizaciones de warmup | 100 / 50 |
| Dropout CFG | 0.1 |
| Recorte latente extra | 0: conservar cada fragmento preparado |

Son valores iniciales, no una garantía de calidad ni una duración obligatoria. Revisa validación y canciones generadas. `sft_rank64` ofrece rank 64/alpha 128; también puedes editar cualquier ajuste admitido. Siguen disponibles módulos/capas del adapter, dropout, precisión, optimizer/scheduler, pasos, checkpoints y hardware. Los parámetros de preferencias `beta`, referencia/regularización FM, label smoothing y pesos de pares no afectan a SFT.

Ejemplo SFT con dos GPU:

```powershell
python -m jk_step train --config configs/mi_sft.json --dataset-manifest datasets/mi_musica/supervised_manifest.json --output-dir output/mi_sft_2gpu --multi-gpu --gpu-ids 0,1
```

Cada GPU debe alojar el modelo y su batch local; la VRAM no se suma. El batch efectivo es batch × acumulación × número de GPU: 4 con una o 8 con dos para este preset SFT. Fragmentos más largos, rank 64, módulos adicionales o un batch mayor pueden aumentar memoria. Consulta las [notas multi-GPU](../README_ES.md#entrenamiento-multi-gpu-experimental).

## Detener, reanudar y usar el LoRA

El control de parada de la UI o Ctrl+C en CLI solicita cancelación en el límite de una actualización y guarda un checkpoint `stopped` reanudable. Una operación GPU o descarga puede necesitar terminar antes de detenerse.

```powershell
python -m jk_step train --config output/mi_lora_sft/training_config.json --resume-from output/mi_lora_sft/stopped
```

Usa la ruta real de `latest.json`. Reanudar restaura el estado y rechaza cambios incompatibles de datos, objetivo o ajustes. Usa una nueva salida para una ejecución nueva. Por defecto se exporta `quality_comfyui.safetensors` y se conserva el adapter PEFT en `final/`.

Aplica el LoRA exportado a la misma familia SFT usada al entrenar. En el loader LoRA SFT de ComfyUI, conecta su salida de modelo a Generate; conserva el text encoder original para TextEncode, porque los pesos CLIP/text encoder no se entrenaron. Compara inicialmente intensidades de modelo como 0.5 y 1, con intensidad del text encoder 0, manteniendo seed, prompt, letra, CFG, sampler y códigos LM iguales. Evalúa voz, palabras, instrumentos y estructura por separado; reducir la pérdida no demuestra mejor sonido.

Para preferencias humanas o evaluadas por scores, usa el flujo separado con `objective="flow_dpo"` y `pairs_manifest`. Las comparaciones automáticas de carpeta son explícitamente sintéticas; no reclaman esas etiquetas humanas/rewards. Consulta las [instrucciones de datasets de preferencias](../README_ES.md#preparar-datasets-de-preferencias).
