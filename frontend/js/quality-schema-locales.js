/* Schema presentation only. Field keys and option values remain unchanged. */
"use strict";
(() => {
  const locales = { en: {}, pt: {}, es: {} };
  const languages = ["en", "pt", "es"];
  function field(name, en, pt, es, choices) {
    [en, pt, es].forEach(([label, help], index) => {
      const translated = { label, help };
      if (choices) translated.choices = Object.fromEntries(Object.entries(choices).map(([value, labels]) => [value, labels[index]]));
      locales[languages[index]][name] = translated;
    });
  }

  field("objective",
    ["Training objective", "flow_dpo learns preferences; sft learns directly from single audio targets with flow matching. SFT ignores all preference-only coefficients."],
    ["Objetivo do treino", "flow_dpo aprende preferências; sft aprende diretamente com áudios individuais usando flow matching. O SFT ignora todos os coeficientes exclusivos do objetivo de preferências."],
    ["Objetivo del entrenamiento", "flow_dpo aprende preferencias; sft aprende directamente de audios individuales mediante flow matching. SFT ignora todos los coeficientes exclusivos del objetivo de preferencias."],
    { flow_dpo: ["MR-FlowDPO · learn preferences", "MR-FlowDPO · aprender preferências", "MR-FlowDPO · aprender preferencias"], sft: ["LoRA SFT · learn from music", "LoRA SFT · aprender com músicas", "LoRA SFT · aprender de música"] });
  field("auto_download",
    ["Download the default model automatically", "Download the default SFT XL checkpoint when no local model was selected; uses the trainer's own checkpoint_dir."],
    ["Baixar o modelo padrão automaticamente", "Baixa o checkpoint SFT XL padrão quando nenhum modelo local foi selecionado; usa o checkpoint_dir do próprio treinador."],
    ["Descargar el modelo predeterminado automáticamente", "Descarga el checkpoint SFT XL predeterminado cuando no se seleccionó un modelo local; utiliza el checkpoint_dir del propio entrenador."]);
  field("checkpoint_dir",
    ["Checkpoint directory", "Root of the Hugging Face ACE-Step checkpoint directories."],
    ["Pasta dos checkpoints", "Pasta raiz dos diretórios de checkpoints ACE-Step do Hugging Face."],
    ["Directorio de checkpoints", "Directorio raíz de las carpetas de checkpoints ACE-Step de Hugging Face."]);
  field("checkpoint_file",
    ["Custom checkpoint file", "Optional original/ComfyUI .safetensors file; needs model_config_dir with matching architecture."],
    ["Arquivo de checkpoint próprio", "Arquivo .safetensors original ou do ComfyUI, opcional; exige model_config_dir com a arquitetura correspondente."],
    ["Archivo de checkpoint personalizado", "Archivo .safetensors original o de ComfyUI, opcional; requiere model_config_dir con la arquitectura correspondiente."]);
  field("model_config_dir",
    ["Model architecture directory", "Local config.json and Python architecture files for checkpoint_file. No model weights need downloading here."],
    ["Pasta da arquitetura do modelo", "Contém o config.json e os arquivos Python locais da arquitetura para checkpoint_file. Não é necessário baixar pesos do modelo nesta pasta."],
    ["Directorio de la arquitectura del modelo", "Contiene el config.json y los archivos Python locales de la arquitectura para checkpoint_file. No es necesario descargar pesos del modelo en esta carpeta."]);
  field("model_variant",
    ["Model variant", "SFT/base architecture. Preference training of a distilled Turbo is not enabled."],
    ["Variante do modelo", "Arquitetura SFT/base. O treino por preferências de um modelo Turbo destilado não está habilitado."],
    ["Variante del modelo", "Arquitectura SFT/base. El entrenamiento por preferencias de un modelo Turbo destilado no está habilitado."],
    { "xl-sft": ["XL SFT", "XL SFT", "XL SFT"], sft: ["SFT", "SFT", "SFT"], "xl-base": ["XL Base", "XL Base", "XL Base"], base: ["Base", "Base", "Base"] });
  field("pairs_manifest",
    ["Prepared preference dataset", "Prepared preference manifest, with tensor_path for every selected pair."],
    ["Dataset de preferências preparado", "Manifesto de preferências preparado, com tensor_path para cada par selecionado."],
    ["Conjunto de preferencias preparado", "Manifiesto de preferencias preparado, con tensor_path para cada par seleccionado."]);
  field("dataset_manifest",
    ["Prepared SFT dataset", "SFT only: prepared supervised JSON, a Side-Step manifest.json, or a directory of .pt tensors. No chosen/rejected pairs are needed."],
    ["Dataset SFT preparado", "Somente SFT: JSON supervisionado preparado, manifest.json do Side-Step ou pasta de tensores .pt. Não exige pares de áudios preferidos e rejeitados."],
    ["Conjunto SFT preparado", "Solo SFT: JSON supervisado preparado, manifest.json de Side-Step o directorio de tensores .pt. No requiere pares de audios preferidos y rechazados."]);
  field("output_dir",
    ["Training output directory", "Run directory; checkpoints, logs, configuration and final adapter are written here."],
    ["Pasta de saída do treino", "Pasta da execução; os checkpoints, logs, configuração e adaptador final são gravados aqui."],
    ["Directorio de salida del entrenamiento", "Directorio de la ejecución; aquí se guardan los checkpoints, registros, configuración y adaptador final."]);
  field("max_latent_length",
    ["Maximum latent frames", "Flow-DPO crop in latent frames; 0 preserves prepared clips. SFT defaults to 0 and rejects cropping vocals without aligned clip lyrics."],
    ["Limite de frames latentes", "Recorte do Flow-DPO em frames do espaço latente; 0 preserva os trechos preparados. O SFT usa 0 por padrão e rejeita recortes de voz sem letras alinhadas ao trecho."],
    ["Máximo de frames latentes", "Recorte de Flow-DPO en frames del espacio latente; 0 conserva los fragmentos preparados. SFT usa 0 por defecto y rechaza recortes vocales sin letras alineadas con el fragmento."]);
  field("dataset_repeats",
    ["Dataset repetitions per epoch", "Repeat the training data within each epoch."],
    ["Repetições do dataset por época", "Repete os dados de treino dentro de cada época."],
    ["Repeticiones de datos por época", "Repite los datos de entrenamiento dentro de cada época."]);
  field("verify_checksums",
    ["Verify tensor integrity", "Verify cached tensor fingerprints when supported by the dataset."],
    ["Verificar integridade dos tensores", "Verifica as assinaturas dos tensores em cache quando o dataset oferece esse recurso."],
    ["Verificar la integridad de los tensores", "Verifica las huellas de los tensores en caché cuando el conjunto de datos admite esta función."]);
  field("rank",
    ["LoRA rank", "LoRA rank, for example 16, 32, 64 or 128."],
    ["Rank do LoRA", "Rank do LoRA, por exemplo 16, 32, 64 ou 128."],
    ["Rango de LoRA", "Rango de LoRA, por ejemplo 16, 32, 64 o 128."]);
  field("alpha",
    ["LoRA scaling (alpha)", "LoRA scaling alpha; inference exports preserve alpha/rank."],
    ["Escala do LoRA (alpha)", "Fator de escala alpha do LoRA; os arquivos exportados para inferência preservam alpha/rank."],
    ["Escala de LoRA (alpha)", "Factor de escala alpha de LoRA; las exportaciones para inferencia conservan alpha/rank."]);
  field("dropout",
    ["Adapter dropout", "Adapter dropout; Flow-DPO shares masks across branches, SFT uses ordinary dropout. Base-model dropout stays disabled."],
    ["Dropout do adaptador", "Dropout do adaptador; o Flow-DPO compartilha máscaras entre os ramos, e o SFT usa dropout comum. O dropout do modelo base permanece desativado."],
    ["Dropout del adaptador", "Dropout del adaptador; Flow-DPO comparte máscaras entre los dos caminos, y SFT usa dropout convencional. El dropout del modelo base permanece desactivado."]);
  field("target_modules",
    ["LoRA target modules", "Linear decoder module suffixes; comma-separated or a JSON list."],
    ["Módulos alvo do LoRA", "Sufixos dos módulos lineares do decoder; separados por vírgulas ou em uma lista JSON."],
    ["Módulos objetivo de LoRA", "Sufijos de los módulos lineales del decoder; separados por comas o en una lista JSON."]);
  field("attention_type",
    ["Attention types to adapt", "Attention projections to adapt."],
    ["Tipos de atenção a adaptar", "Projeções de atenção que serão adaptadas."],
    ["Tipos de atención que se adaptarán", "Proyecciones de atención que se adaptarán."],
    { both: ["Self-attention and cross-attention", "Autoatenção e atenção cruzada", "Autoatención y atención cruzada"], self: ["Self-attention", "Autoatenção", "Autoatención"], cross: ["Cross-attention", "Atenção cruzada", "Atención cruzada"] });
  field("target_mlp",
    ["Also adapt decoder MLPs", "Also adapt gate_proj/up_proj/down_proj in the decoder MLPs."],
    ["Adaptar também os MLPs do decoder", "Adapta também gate_proj/up_proj/down_proj nos MLPs do decoder."],
    ["Adaptar también los MLP del decoder", "Adapta también gate_proj/up_proj/down_proj en los MLP del decoder."]);
  field("layers",
    ["Decoder layers to adapt", "Decoder layer selection: all, 0,1,2, or 0-7,16-23."],
    ["Camadas do decoder a adaptar", "Seleção das camadas do decoder: all, 0,1,2 ou 0-7,16-23."],
    ["Capas del decoder que se adaptarán", "Selección de capas del decoder: all, 0,1,2 o 0-7,16-23."]);
  field("rank_pattern",
    ["Rank overrides by module", "Optional PEFT per-module rank overrides (JSON object)."],
    ["Rank por módulo", "Valores de rank por módulo do PEFT, opcionais, que substituem o valor geral; use um objeto JSON."],
    ["Rango por módulo", "Valores de rango por módulo de PEFT, opcionales, que sustituyen el valor general; utilice un objeto JSON."]);
  field("alpha_pattern",
    ["Alpha overrides by module", "Optional PEFT per-module alpha overrides (JSON object)."],
    ["Alpha por módulo", "Valores de alpha por módulo do PEFT, opcionais, que substituem o valor geral; use um objeto JSON."],
    ["Alpha por módulo", "Valores de alpha por módulo de PEFT, opcionales, que sustituyen el valor general; utilice un objeto JSON."]);
  field("use_rslora",
    ["Use rank-stabilized LoRA", "Rank-stabilized LoRA; export embeds its effective scaling."],
    ["Usar LoRA com estabilização de rank", "LoRA com estabilização de rank; a exportação incorpora seu fator de escala efetivo."],
    ["Usar LoRA con estabilización de rango", "LoRA con estabilización de rango; la exportación incorpora su factor de escala efectivo."]);
  field("beta",
    ["Preference loss scale (beta)", "Preference-loss scale. 2000 is the published MelodyFlow setup, not a calibrated ACE-Step optimum."],
    ["Escala da loss de preferências (beta)", "Escala da loss de preferências. 2000 é o valor da configuração publicada do MelodyFlow, e não um valor ótimo calibrado para ACE-Step."],
    ["Escala de la pérdida de preferencias (beta)", "Escala de la pérdida de preferencias. 2000 es el valor de la configuración publicada de MelodyFlow, no un valor óptimo calibrado para ACE-Step."]);
  field("fm_regularization",
    ["Flow-matching regularization", "Optional flow-matching error on chosen samples to anchor generation; 0 reproduces the core DPO objective."],
    ["Regularização por flow matching", "Erro de flow matching opcional nas amostras preferidas para ancorar a geração; 0 reproduz o objetivo central do DPO."],
    ["Regularización mediante flow matching", "Error de flow matching opcional en las muestras preferidas para anclar la generación; 0 reproduce el objetivo central de DPO."]);
  field("reference_regularization",
    ["Frozen-reference regularization", "Experimental velocity penalty versus the frozen SFT on both branches; does not guarantee diction preservation."],
    ["Regularização pela referência congelada", "Penalidade experimental sobre a velocidade prevista em relação ao SFT congelado, nos dois ramos; não garante a preservação da dicção."],
    ["Regularización con la referencia congelada", "Penalización experimental de la velocidad predicha respecto al SFT congelado, en ambos caminos; no garantiza la conservación de la dicción."]);
  field("label_smoothing",
    ["Preference label smoothing", "Preference-label smoothing for uncertain pairs (0 to 0.49)."],
    ["Suavização dos rótulos de preferência", "Suavização dos rótulos de preferência para pares incertos (0 a 0.49)."],
    ["Suavizado de etiquetas de preferencia", "Suavizado de las etiquetas de preferencia para pares inciertos (de 0 a 0.49)."]);
  field("pair_weighting",
    ["Use dataset pair weights", "Use pair_weight values in the prepared dataset instead of uniform weights."],
    ["Usar os pesos dos pares do dataset", "Usa os valores de pair_weight do dataset preparado em vez de atribuir pesos uniformes."],
    ["Usar los pesos de los pares del conjunto", "Usa los valores de pair_weight del conjunto preparado en lugar de asignar pesos uniformes."]);
  field("timestep_sampling",
    ["Timestep sampling", "ACE's continuous max-of-two logit-normal sampling or the paper's uniform sampling."],
    ["Amostragem de timesteps", "Amostragem contínua do ACE pelo máximo de dois valores de uma distribuição logit-normal, ou amostragem uniforme do artigo."],
    ["Muestreo de timesteps", "Muestreo continuo de ACE mediante el máximo de dos valores de una distribución logit-normal, o muestreo uniforme del artículo."],
    { logit_normal: ["ACE logit-normal · maximum of two draws", "Logit-normal ACE · máximo de dois valores", "Logit-normal ACE · máximo de dos valores"], uniform: ["Uniform", "Uniforme", "Uniforme"] });
  field("timestep_mu",
    ["Logit-normal location (mu)", "Location of the ACE logit-normal timestep distribution."],
    ["Localização logit-normal (mu)", "Parâmetro de localização da distribuição logit-normal de timesteps do ACE."],
    ["Ubicación logit-normal (mu)", "Parámetro de ubicación de la distribución logit-normal de timesteps de ACE."]);
  field("timestep_sigma",
    ["Logit-normal scale (sigma)", "Scale of the ACE logit-normal timestep distribution."],
    ["Escala logit-normal (sigma)", "Parâmetro de escala da distribuição logit-normal de timesteps do ACE."],
    ["Escala logit-normal (sigma)", "Parámetro de escala de la distribución logit-normal de timesteps de ACE."]);
  field("timestep_min",
    ["Minimum sampled timestep", "Lower endpoint clamp for sampled timesteps."],
    ["Timestep mínimo amostrado", "Limite inferior aplicado aos timesteps amostrados."],
    ["Timestep mínimo muestreado", "Límite inferior aplicado a los timesteps muestreados."]);
  field("timestep_max",
    ["Maximum sampled timestep", "Upper endpoint clamp for sampled timesteps."],
    ["Timestep máximo amostrado", "Limite superior aplicado aos timesteps amostrados."],
    ["Timestep máximo muestreado", "Límite superior aplicado a los timesteps muestreados."]);
  field("cfg_dropout",
    ["Conditioning dropout (CFG)", "Shared condition dropout for both branches. 0 preserves lyrics conditioning on every pair."],
    ["Dropout do condicionamento (CFG)", "Dropout do condicionamento compartilhado pelos dois ramos. 0 preserva o condicionamento pelas letras em todos os pares."],
    ["Dropout del condicionamiento (CFG)", "Dropout del condicionamiento compartido por ambos caminos. 0 conserva el condicionamiento con letras en todos los pares."]);
  field("batch_size",
    ["Microbatch size per GPU", "Examples per microbatch per GPU: one audio branch in SFT, two in Flow-DPO."],
    ["Tamanho do microbatch por GPU", "Exemplos por microbatch em cada GPU: um ramo de áudio no SFT e dois no Flow-DPO."],
    ["Tamaño del microlote por GPU", "Ejemplos por microlote en cada GPU: un camino de audio en SFT y dos en Flow-DPO."]);
  field("gradient_accumulation",
    ["Gradient accumulation", "Microbatches per optimizer update."],
    ["Acumulação de gradientes", "Quantidade de microbatches por atualização do otimizador."],
    ["Acumulación de gradientes", "Cantidad de microlotes por actualización del optimizador."]);
  field("epochs",
    ["Training epochs", "Epoch count when max_steps is 0."],
    ["Épocas de treino", "Número de épocas quando max_steps é 0."],
    ["Épocas de entrenamiento", "Número de épocas cuando max_steps es 0."]);
  field("max_steps",
    ["Maximum optimizer updates", "Positive optimizer-step limit overrides epochs; 0 uses epochs."],
    ["Limite de atualizações do otimizador", "Um limite positivo de passos do otimizador substitui o número de épocas; 0 usa as épocas."],
    ["Límite de actualizaciones del optimizador", "Un límite positivo de pasos del optimizador sustituye el número de épocas; 0 utiliza las épocas."]);
  field("learning_rate",
    ["Learning rate", "Adapter optimizer learning rate."],
    ["Taxa de aprendizado", "Taxa de aprendizado do otimizador do adaptador."],
    ["Tasa de aprendizaje", "Tasa de aprendizaje del optimizador del adaptador."]);
  field("optimizer",
    ["Optimizer", "AdamW, optional CUDA bitsandbytes AdamW8bit, or Adafactor."],
    ["Otimizador", "AdamW, AdamW8bit opcional do bitsandbytes em CUDA ou Adafactor."],
    ["Optimizador", "AdamW, AdamW8bit opcional de bitsandbytes en CUDA o Adafactor."],
    { adamw: ["AdamW", "AdamW", "AdamW"], adamw8bit: ["AdamW · 8-bit (CUDA)", "AdamW · 8 bits (CUDA)", "AdamW · 8 bits (CUDA)"], adafactor: ["Adafactor", "Adafactor", "Adafactor"] });
  field("scheduler",
    ["Learning-rate schedule", "Learning-rate schedule after warmup."],
    ["Programação da taxa de aprendizado", "Programação da taxa de aprendizado depois do aquecimento."],
    ["Programación de la tasa de aprendizaje", "Programación de la tasa de aprendizaje después del calentamiento."],
    { constant: ["Constant", "Constante", "Constante"], linear: ["Linear", "Linear", "Lineal"], cosine: ["Cosine", "Cosseno", "Coseno"] });
  field("warmup_steps",
    ["Learning-rate warmup updates", "Optimizer updates used for learning-rate warmup."],
    ["Atualizações de aquecimento", "Atualizações do otimizador usadas no aquecimento da taxa de aprendizado."],
    ["Actualizaciones de calentamiento", "Actualizaciones del optimizador utilizadas para el calentamiento de la tasa de aprendizaje."]);
  field("min_lr_ratio",
    ["Final learning-rate ratio", "Final learning rate relative to the initial value for linear/cosine."],
    ["Proporção final da taxa de aprendizado", "Taxa de aprendizado final em relação ao valor inicial nas programações linear/cosine."],
    ["Proporción final de la tasa de aprendizaje", "Tasa de aprendizaje final respecto al valor inicial en las programaciones linear/cosine."]);
  field("weight_decay",
    ["Weight decay", "Decoupled weight decay."],
    ["Decaimento dos pesos", "Decaimento de pesos desacoplado."],
    ["Decaimiento de los pesos", "Decaimiento de pesos desacoplado."]);
  field("adam_beta1",
    ["AdamW first-moment decay (beta1)", "First-moment decay for AdamW."],
    ["Decaimento do primeiro momento (beta1)", "Decaimento do primeiro momento no AdamW."],
    ["Decaimiento del primer momento (beta1)", "Decaimiento del primer momento en AdamW."]);
  field("adam_beta2",
    ["AdamW second-moment decay (beta2)", "Second-moment decay for AdamW."],
    ["Decaimento do segundo momento (beta2)", "Decaimento do segundo momento no AdamW."],
    ["Decaimiento del segundo momento (beta2)", "Decaimiento del segundo momento en AdamW."]);
  field("adam_epsilon",
    ["AdamW numerical epsilon", "AdamW denominator epsilon."],
    ["Epsilon numérico do AdamW", "Epsilon no denominador do AdamW."],
    ["Epsilon numérico de AdamW", "Epsilon en el denominador de AdamW."]);
  field("max_grad_norm",
    ["Maximum gradient norm", "Gradient clipping; 0 disables clipping."],
    ["Norma máxima dos gradientes", "Limitação dos gradientes (gradient clipping); 0 desativa a limitação."],
    ["Norma máxima de los gradientes", "Limitación de los gradientes (gradient clipping); 0 desactiva la limitación."]);
  field("precision",
    ["Model computation precision", "Base-model/forward precision; trainable adapter weights and loss reduction remain FP32."],
    ["Precisão de cálculo do modelo", "Precisão do modelo base e do forward; os pesos treináveis do adaptador e a redução da loss permanecem em FP32."],
    ["Precisión de cálculo del modelo", "Precisión del modelo base y del forward; los pesos entrenables del adaptador y la reducción de la pérdida permanecen en FP32."],
    { auto: ["Automatic", "Automática", "Automática"], bf16: ["BF16", "BF16", "BF16"], fp16: ["FP16", "FP16", "FP16"], fp32: ["FP32", "FP32", "FP32"] });
  field("device",
    ["Training device", "Single-device mode: auto, cpu, cuda:0, cuda:1, mps or xpu:0. Multi-GPU mode uses gpu_ids instead."],
    ["Dispositivo de treino", "Modo com um dispositivo: auto, cpu, cuda:0, cuda:1, mps ou xpu:0. O modo com múltiplas GPUs usa gpu_ids."],
    ["Dispositivo de entrenamiento", "Modo con un dispositivo: auto, cpu, cuda:0, cuda:1, mps o xpu:0. El modo con varias GPU utiliza gpu_ids."]);
  field("multi_gpu",
    ["Train on multiple GPUs", "Train synchronized LoRA replicas on multiple CUDA GPUs. Each GPU keeps its own frozen decoder; VRAM is not pooled."],
    ["Treinar com múltiplas GPUs", "Treina réplicas sincronizadas do LoRA em múltiplas GPUs CUDA. Cada GPU mantém seu próprio decoder congelado; a VRAM das placas não é somada."],
    ["Entrenar con varias GPU", "Entrena réplicas sincronizadas de LoRA en varias GPU CUDA. Cada GPU mantiene su propio decoder congelado; la VRAM de las tarjetas no se suma."]);
  field("gpu_ids",
    ["CUDA GPU indices", "CUDA GPU indices for multi-GPU training, e.g. 0,1. Batch size and gradient accumulation apply per GPU."],
    ["Índices das GPUs CUDA", "Índices das GPUs CUDA para treino com múltiplas GPUs, por exemplo 0,1. O tamanho do batch e a acumulação de gradientes se aplicam a cada GPU."],
    ["Índices de las GPU CUDA", "Índices de las GPU CUDA para entrenamiento con varias GPU, por ejemplo 0,1. El tamaño del lote y la acumulación de gradientes se aplican a cada GPU."]);
  field("distributed_backend",
    ["GPU communication backend", "Communication backend: auto uses local CPU IPC on Windows and NCCL on Linux when available. local synchronizes LoRA gradients between processes without requiring Gloo/NCCL."],
    ["Backend de comunicação entre GPUs", "Backend de comunicação: auto usa IPC local pela CPU no Windows e NCCL no Linux quando disponível. local sincroniza gradientes do LoRA entre processos sem exigir Gloo/NCCL."],
    ["Backend de comunicación entre GPU", "Backend de comunicación: auto usa IPC local mediante CPU en Windows y NCCL en Linux cuando está disponible. local sincroniza gradientes de LoRA entre procesos sin requerir Gloo/NCCL."],
    { auto: ["Automatic", "Automático", "Automático"], local: ["Local CPU IPC", "IPC local pela CPU", "IPC local mediante CPU"], gloo: ["Gloo", "Gloo", "Gloo"], nccl: ["NCCL", "NCCL", "NCCL"] });
  field("offload_non_decoder",
    ["Keep conditioning encoders on CPU", "Keep condition encoders/tokenizers on CPU after tensors have been prepared; decoder stays on the selected device."],
    ["Manter codificadores de condicionamento na CPU", "Mantém os codificadores e tokenizadores de condicionamento na CPU depois que os tensores foram preparados; o decoder permanece no dispositivo selecionado."],
    ["Mantener los codificadores de condicionamiento en CPU", "Mantiene los codificadores y tokenizadores de condicionamiento en CPU después de preparar los tensores; el decoder permanece en el dispositivo seleccionado."]);
  field("gradient_checkpointing",
    ["Recompute activations to save memory", "Recompute decoder activations during backward to reduce VRAM."],
    ["Recalcular ativações para economizar memória", "Recalcula as ativações do decoder durante o backward para reduzir o uso de VRAM."],
    ["Recalcular activaciones para ahorrar memoria", "Recalcula las activaciones del decoder durante el backward para reducir el uso de VRAM."]);
  field("seed",
    ["Random seed", "Training, split and shuffle seed."],
    ["Seed aleatória", "Seed usada no treino, na divisão dos dados e no embaralhamento."],
    ["Semilla aleatoria", "Semilla utilizada en el entrenamiento, la división de datos y su mezcla aleatoria."]);
  field("num_workers",
    ["Data-loading worker processes", "DataLoader worker processes; 0 is reliable on Windows."],
    ["Processos de carregamento dos dados", "Processos de trabalho do DataLoader; 0 é uma opção confiável no Windows."],
    ["Procesos de carga de datos", "Procesos de trabajo del DataLoader; 0 es una opción fiable en Windows."]);
  field("pin_memory",
    ["Pin CPU memory for CUDA transfers", "Pinned CPU data transfer when training on CUDA."],
    ["Fixar memória da CPU para transferências CUDA", "Transferência de dados usando memória fixada na CPU durante o treino em CUDA."],
    ["Fijar memoria de CPU para transferencias CUDA", "Transferencia de datos mediante memoria fijada en CPU durante el entrenamiento en CUDA."]);
  field("validation_fraction",
    ["Validation fraction", "Fallback group-level holdout when no explicit validation split exists. Never splits variants of the same recording."],
    ["Fração para validação", "Reserva de validação por grupo quando não existe uma divisão explícita de validação. Nunca separa variantes da mesma gravação entre treino e validação."],
    ["Fracción para validación", "Reserva de validación por grupo cuando no existe una división explícita de validación. Nunca separa variantes de la misma grabación entre entrenamiento y validación."]);
  field("eval_every",
    ["Updates between validation runs", "Optimizer updates between deterministic validation runs; 0 disables periodic evaluation."],
    ["Atualizações entre validações", "Atualizações do otimizador entre execuções determinísticas de validação; 0 desativa a avaliação periódica."],
    ["Actualizaciones entre validaciones", "Actualizaciones del optimizador entre ejecuciones deterministas de validación; 0 desactiva la evaluación periódica."]);
  field("eval_batches",
    ["Maximum validation microbatches", "Maximum validation microbatches per evaluation; 0 evaluates the complete validation set."],
    ["Limite de microbatches de validação", "Quantidade máxima de microbatches de validação por avaliação; 0 avalia todo o conjunto de validação."],
    ["Máximo de microlotes de validación", "Cantidad máxima de microlotes de validación por evaluación; 0 evalúa todo el conjunto de validación."]);
  field("save_every",
    ["Updates between checkpoint saves", "Optimizer updates between resumable checkpoints; 0 saves only at the end."],
    ["Atualizações entre salvamentos", "Atualizações do otimizador entre checkpoints que permitem retomada; 0 salva somente ao final."],
    ["Actualizaciones entre guardados", "Actualizaciones del optimizador entre checkpoints que permiten reanudar; 0 guarda solo al final."]);
  field("save_total_limit",
    ["Periodic checkpoints to retain", "Periodic checkpoints retained; 0 keeps all. final/best/stopped are not deleted."],
    ["Checkpoints periódicos a manter", "Quantidade de checkpoints periódicos mantidos; 0 mantém todos. final/best/stopped não são excluídos."],
    ["Checkpoints periódicos que se conservarán", "Cantidad de checkpoints periódicos que se conservan; 0 conserva todos. final/best/stopped no se eliminan."]);
  field("log_every",
    ["Updates between progress logs", "Optimizer updates between JSONL/progress records."],
    ["Atualizações entre registros de progresso", "Atualizações do otimizador entre registros JSONL/de progresso."],
    ["Actualizaciones entre registros de progreso", "Actualizaciones del optimizador entre registros JSONL/de progreso."]);
  field("tensorboard",
    ["Write TensorBoard metrics", "Write TensorBoard scalars; requires tensorboard when enabled."],
    ["Gravar métricas do TensorBoard", "Grava métricas escalares do TensorBoard; exige tensorboard quando habilitado."],
    ["Guardar métricas de TensorBoard", "Guarda métricas escalares de TensorBoard; requiere tensorboard cuando está habilitado."]);
  field("resume_from",
    ["Checkpoint to resume", "JK-Step checkpoint directory containing training_state.pt; restores optimizer, scheduler, RNG and batch cursor."],
    ["Checkpoint para retomar", "Pasta de checkpoint do JK-Step contendo training_state.pt; restaura o otimizador, scheduler, geradores de números aleatórios e posição no processamento dos batches."],
    ["Checkpoint para reanudar", "Directorio de checkpoint de JK-Step que contiene training_state.pt; restaura el optimizador, scheduler, generadores de números aleatorios y posición en el procesamiento de los lotes."]);
  field("init_adapter",
    ["Initial adapter", "Optional PEFT adapter used to initialize policy. Reference remains the original SFT; mutually exclusive with resume_from."],
    ["Adaptador inicial", "Adaptador PEFT opcional usado para inicializar o modelo treinável. A referência permanece o SFT original; não pode ser usado junto com resume_from."],
    ["Adaptador inicial", "Adaptador PEFT opcional utilizado para inicializar el modelo entrenable. La referencia sigue siendo el SFT original; no puede utilizarse junto con resume_from."]);
  field("export_comfyui",
    ["Export a ComfyUI LoRA", "Export final PEFT adapter as a ComfyUI .safetensors LoRA."],
    ["Exportar LoRA para ComfyUI", "Exporta o adaptador PEFT final como um LoRA .safetensors para ComfyUI."],
    ["Exportar LoRA para ComfyUI", "Exporta el adaptador PEFT final como un LoRA .safetensors para ComfyUI."]);
  field("comfyui_target",
    ["ComfyUI key mapping", "Side-Step ComfyUI key mapping preset."],
    ["Mapeamento de chaves do ComfyUI", "Preset de mapeamento de chaves do ComfyUI usado pelo Side-Step."],
    ["Mapeo de claves de ComfyUI", "Preset de mapeo de claves de ComfyUI utilizado por Side-Step."],
    { native: ["Native", "Nativo", "Nativo"], generic: ["Generic", "Genérico", "Genérico"] });

  window.JKQualitySchemaLocales = locales;
})();
