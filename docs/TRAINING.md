# Treinamento e avaliação

JK-Step usa o erro de velocidade do flow matching como estimativa de preferência. Para um par escolhido/rejeitado, calcula:

`softplus(beta * ((erro_escolhido - erro_rejeitado) - (erro_ref_escolhido - erro_ref_rejeitado)))`.

O ruído, timestep, caption, letra e contexto são compartilhados pelos dois lados. A referência é o checkpoint original com adapters desativados, em modo determinístico e sem gradientes. Só o LoRA do decoder é atualizado. Os erros são calculados em FP32 e ignoram padding; sequências são cortadas antes do forward porque algumas versões ACE não respeitam todas as máscaras.

## O que está adaptado do artigo

Implementado: objetivo Flow-DPO, seleção MRSD por múltiplas recompensas, quantis e balanceamento, condicionamento compartilhado, referência congelada e métricas por preferência. O runtime e timestep seguem ACE-Step SFT (contínuo), com opção uniforme para experimentos.

O artigo treinou outros modelos em música instrumental. A implementação original usou candidatos gerados pelos próprios modelos, beta 2000 e recompensas CLAP, Audiobox PQ e uma representação HuBERT musical privada. Aqui candidatos existentes ou preferências humanas são aceitos, e degradados artificiais são explicitamente uma adaptação. Não é uma reprodução dos resultados do artigo. **Reward prompting numérico do artigo não está implementado**: não acrescente `quality=10` a captions esperando o mesmo efeito. Veja o [artigo](https://arxiv.org/abs/2512.10264) e o [código dos autores](https://github.com/lonzi/mrflow_dpo).

## Dataset para qualidade geral

Use diversidade de gênero, idioma, cantor, andamento, instrumentação e densidade. Evite um artista ou gênero dominar os pares: um LoRA aprende a distribuição que vê, não uma noção abstrata de qualidade. Letras incorretas, baixa qualidade do escolhido ou escolhas baseadas só no brilho podem ensinar comportamentos indesejados.

Pares precisam representar a mesma intenção e ser alinhados: não compare duas músicas sem relação como se fossem um par do mesmo prompt. Para seu problema de voz abafada, uma alternativa controlada é mix original com voz clara vs. a mesma mix com apenas o stem vocal degradado. Isso exige stems separados e alinhados; o gerador não isola automaticamente uma voz por filtro de frequência.

MRSD usa um eixo primário com diferença alta (quantil 0.95 por padrão), secundários com diferenças suficientes (mediana), pisos de qualidade (quantil 0.05), limites superiores opcionais e balanceamento por eixo primário. Um eixo protegido de `lyric_fidelity` pode exigir nenhuma regressão e um piso independente; empate de dicção perfeita é permitido. Esse filtro adicional é uma adaptação para canto, não uma garantia de que o modelo não regredirá.

## Pré-processamento

`chosen_semantic` codifica/detokeniza o áudio escolhido uma vez e usa o mesmo contexto nos dois lados. São códigos extraídos de um áudio existente, sem gerar música com Turbo/LM. `silence` usa o contexto de texto/letra padrão. Comparar esses modos em validação é útil: o primeiro se aproxima do caminho com códigos, mas também muda a distribuição de condicionamento. O VAE usa média do posterior por padrão para não criar diferenças aleatórias entre pares.

Em candidatos independentes com melodias diferentes, códigos extraídos apenas do escolhido podem favorecer esse lado por alinhamento ao próprio plano, confundindo musicalidade com fidelidade aos códigos. Prefira `silence` nesses pares quando não houver um plano semântico comum conhecido. Para original/degradado da mesma gravação, o contexto compartilhado tem uma justificativa mais direta.

O cache registra áudios, letras, metadata, modelo e opções. As duas branches recebem o mesmo crop. Nunca corrija um cache manualmente mantendo o mesmo fingerprint. Normalização é configurável, desativada por padrão; normalizar os dois lados separadamente pode apagar diferenças de produção que o par pretendia representar.

## Experimento inicial

Comece com rank 32 (ou 16), alpha aproximadamente 2 × rank, LR 1e-6, batch 1 e acumulação 8. O preset conservador usa beta 100 e regularização FM 0.1. São pontos iniciais, não valores medidos como ótimos para ACE-Step. Se tiver poucos pares, prefira epochs pequenas, LR baixa e compare checkpoints. O preset `paper_beta` expõe o beta original 2000, com esse nome para não confundir com uma configuração ACE validada.

Monitore loss, erro FM, margem de preferência, accuracy dos pares, grad_norm e validação. Uma loss menor não comprova melhor áudio. Margens saturadas, clipping frequente ou deterioração de erros FM podem indicar beta/LR altos ou pares enviesados. Batch é contado em **pares**; cada par exige dois áudios e uma passagem de referência adicional.

Faça validação por música/artista e um conjunto externo de prompts, idiomas e letras. Gere com as mesmas seeds e condições usando SFT sem LoRA e com LoRA em várias forças; avalie clareza vocal, naturalidade, harmonia, artefatos e palavras cantadas. CLAP não mede dicção literal. Transcrições automáticas de canto também exigem conferência humana. Não use o conjunto de avaliação como treino.

## Checkpoints e exportação

`checkpoint-00000010` contém adapter, optimizer, scheduler, scaler, RNG e cursor do dataset. `final` e `stopped` também podem ser retomados. Use `resume_from` para continuar o mesmo treino; alterações incompatíveis de dataset, modelo, rank ou configuração são rejeitadas. Use `init_adapter` para um novo experimento compatível a partir de um adapter existente.

O LoRA PEFT fica em `adapter_model.safetensors` e `adapter_config.json`. A exportação ComfyUI inclui scaling efetivo por módulo, inclusive rsLoRA/alpha overrides. Escolha `native` para o loader ACE do ComfyUI ou `generic` conforme seu loader. A saída `quality_comfyui.safetensors` é gerada automaticamente se `export_comfyui=true`.

No terminal, Ctrl+C pede parada e salva. Na UI, Parar cria o arquivo de parada cooperativa. A operação conclui a fronteira segura de atualização antes de sair; ela não libera GPU instantaneamente durante um kernel ativo. Um servidor reiniciado não inicia outro job enquanto detectar um job anterior ativo.
