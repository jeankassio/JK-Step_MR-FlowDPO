/* JK-Step MR-FlowDPO: schema-driven configuration and authenticated local jobs. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const token = window.__SIDESTEP_TOKEN__ || new URLSearchParams(location.search).get("token") || "";
  const state = { fields: [], defaults: {}, config: {}, presets: [], job: null, cursor: 0, polling: false, log: [], appliedResults: new Set() };
  const labels = { Model: "Modelo e checkpoint", Data: "Dataset e saída", LoRA: "LoRA · capacidade do ajuste", "MR-FlowDPO": "Objetivo de preferências", Training: "Treinamento", Optimizer: "Otimizador", Hardware: "Dispositivo e memória", "Resume/export": "Retomada e exportação", Logging: "Logs e checkpoints", Evaluation: "Validação" };
  const statuses = { running: "Em execução", stopping: "Salvando / parando", completed: "Concluído", failed: "Falhou", cancelled: "Interrompido", interrupted: "Servidor interrompido", orphaned: "Processo externo ativo" };
  const kinds = { pairs: "Criação de pares", score: "Avaliação de candidatos", preprocess: "Pré-processamento", train: "Treinamento MR-FlowDPO", model_setup: "Download / verificação do SFT XL" };
  const clone = (value) => JSON.parse(JSON.stringify(value));

  function error(message) { $("error-banner").textContent = message; $("error-banner").hidden = false; }
  function clearError() { $("error-banner").hidden = true; }
  async function api(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch("/api/mrflow" + path, { method: body === undefined ? "GET" : "POST", headers: { Authorization: "Bearer " + token, ...(body === undefined ? {} : { "Content-Type": "application/json" }) }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : payload.error || JSON.stringify(payload.detail || payload));
      return payload;
    } finally { clearTimeout(timer); }
  }
  function handle(action) { return async (event) => { if (event) event.preventDefault(); clearError(); try { await action(event); } catch (exception) { error(exception.name === "AbortError" ? "A solicitação demorou demais. Confira a conexão com o servidor." : exception.message); } }; }
  function jsonObject(text, label) { const value = JSON.parse(text); if (!value || Array.isArray(value) || typeof value !== "object") throw new Error(label + " precisa ser um objeto JSON."); return value; }
  function tab(name) {
    document.querySelectorAll("[data-panel]").forEach((button) => { const active = button.dataset.panel === name; button.setAttribute("aria-selected", String(active)); button.tabIndex = active ? 0 : -1; });
    document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== "panel-" + name; });
  }
  document.querySelectorAll("[data-panel]").forEach((button, index, buttons) => {
    button.addEventListener("click", () => tab(button.dataset.panel));
    button.addEventListener("keydown", (event) => { let next = index; if (event.key === "ArrowRight") next = (index + 1) % buttons.length; else if (event.key === "ArrowLeft") next = (index + buttons.length - 1) % buttons.length; else return; event.preventDefault(); tab(buttons[next].dataset.panel); buttons[next].focus(); });
  });
  $("quality-link").href = "/quality?token=" + encodeURIComponent(token);

  function readField(field, input) {
    const kind = field.type;
    if (kind === "bool" || kind === "boolean") return input.checked;
    if (kind === "int" || kind === "integer" || kind === "float" || kind === "number") { const value = Number(input.value); if (!input.value.trim() || !Number.isFinite(value)) throw new Error(field.name + " precisa ser um número."); if ((kind === "int" || kind === "integer") && !Number.isInteger(value)) throw new Error(field.name + " precisa ser inteiro."); return value; }
    if (kind === "list") { const text = input.value.trim(); if (!text) return []; if (text.startsWith("[")) { const value = JSON.parse(text); if (!Array.isArray(value)) throw new Error(field.name + " precisa ser uma lista."); return value; } return text.split(",").map((value) => value.trim()).filter(Boolean); }
    if (kind === "json" || kind === "object") return jsonObject(input.value || "{}", field.name);
    return input.value;
  }
  function collectConfig() {
    const config = clone(state.config);
    state.fields.forEach((field) => { const input = $("field-" + field.name); if (input) config[field.name] = readField(field, input); });
    state.config = config;
    $("config-json").value = JSON.stringify(config, null, 2);
    renderBatchSize(config);
    return config;
  }
  function renderBatchSize(config) {
    const ids = Array.isArray(config.gpu_ids) ? config.gpu_ids : String(config.gpu_ids || "").split(",").filter(Boolean);
    const gpus = config.multi_gpu ? ids.length : 1;
    const batch = Number(config.batch_size), accumulation = Number(config.gradient_accumulation);
    $("effective-batch").textContent = "Lote efetivo: " + (batch * accumulation * gpus) + " pares (" + batch + " por GPU × " + accumulation + " de acumulação × " + gpus + " GPU" + (gpus === 1 ? "" : "s") + ").";
  }
  function renderConfig(config) {
    state.config = clone(config);
    const container = $("schema-fields"); container.replaceChildren();
    const groups = new Map();
    state.fields.forEach((field) => { const key = field.group || "Avançado"; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(field); });
    groups.forEach((fields, group) => {
      const section = document.createElement("section"); section.className = "schema-group";
      const heading = document.createElement("h3"); heading.textContent = labels[group] || group; section.append(heading);
      const grid = document.createElement("div"); grid.className = "field-grid";
      fields.forEach((field) => {
        const label = document.createElement("label"); label.className = "field"; label.htmlFor = "field-" + field.name;
        const isBool = ["bool", "boolean"].includes(field.type); const value = config[field.name] === undefined ? field.default : config[field.name];
        const title = document.createElement("span"); title.textContent = field.label || field.name;
        const help = document.createElement("small"); help.textContent = (field.help || "") + " (" + field.name + ")"; help.id = "help-" + field.name;
        let input;
        if (field.choices) { input = document.createElement("select"); field.choices.forEach((choice) => { const option = document.createElement("option"); option.value = typeof choice === "object" ? choice.value : choice; option.textContent = typeof choice === "object" ? choice.label || choice.value : choice; input.append(option); }); input.value = value; }
        else if (field.type === "list" || field.type === "json" || field.type === "object") { input = document.createElement("textarea"); input.rows = field.type === "list" ? 2 : 4; input.value = JSON.stringify(value ?? (field.type === "list" ? [] : {}), null, 2); input.spellcheck = false; }
        else { input = document.createElement("input"); input.type = isBool ? "checkbox" : ["int", "integer", "float", "number"].includes(field.type) ? "number" : "text"; if (isBool) input.checked = Boolean(value); else input.value = value ?? ""; if (field.min !== undefined) input.min = field.min; if (field.max !== undefined) input.max = field.max; if (input.type === "number") input.step = ["int", "integer"].includes(field.type) ? 1 : "any"; }
        input.id = "field-" + field.name; input.name = field.name; input.setAttribute("aria-describedby", help.id);
        if (isBool) { label.classList.add("boolean"); const copy = document.createElement("span"); copy.className = "field-copy"; copy.append(title, help); label.append(input, copy); } else label.append(title, input, help);
        input.addEventListener("change", handle(() => collectConfig())); grid.append(label);
      });
      section.append(grid); container.append(section);
    });
    $("config-json").value = JSON.stringify(state.config, null, 2);
    $("resume-path").value = state.config.resume_from || "";
    if (!$("preprocess-checkpoint").value) $("preprocess-checkpoint").value = state.config.checkpoint_dir || "";
    $("preprocess-variant").value = state.config.model_variant || "xl-sft";
    renderBatchSize(state.config);
  }

  function normalizePresets(payload) {
    const items = [];
    const builtin = payload.builtin || [];
    if (Array.isArray(builtin)) builtin.forEach((item) => items.push({ name: item.name, config: item.config || item.data || item, description: item.description }));
    else Object.entries(builtin).forEach(([name, item]) => items.push({ name, config: item.config || item.data || item, description: item.description }));
    (payload.saved || []).forEach((item) => items.push({ ...item, name: item.name + " · salvo" }));
    return items;
  }
  async function loadPresets() { state.presets = normalizePresets(await api("/presets")); const select = $("preset-select"); select.replaceChildren(new Option("Configuração atual", "")); state.presets.forEach((preset, index) => select.add(new Option(preset.name, String(index)))); }
  $("apply-preset").addEventListener("click", handle(() => { const index = $("preset-select").value; if (index === "") return; const preset = state.presets[Number(index)]; renderConfig({ ...clone(state.defaults), ...collectConfig(), ...preset.config }); appendLog("Preset aplicado: " + preset.name + (preset.description ? ". " + preset.description : "")); }));
  $("save-preset").addEventListener("click", handle(async () => { const name = window.prompt("Nome do preset:"); if (!name || !name.trim()) return; await api("/presets", { name: name.trim(), config: collectConfig() }); await loadPresets(); }));
  $("reset-config").addEventListener("click", handle(() => renderConfig(state.defaults)));
  $("apply-json").addEventListener("click", handle(() => renderConfig({ ...clone(state.defaults), ...jsonObject($("config-json").value, "Configuração") })));
  $("export-json").addEventListener("click", handle(() => { const blob = new Blob([JSON.stringify(collectConfig(), null, 2)], { type: "application/json" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "jk-step-mrflow-config.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); }));
  $("import-json").addEventListener("change", handle(async (event) => { const file = event.target.files[0]; if (file) renderConfig({ ...clone(state.defaults), ...jsonObject(await file.text(), "Arquivo de configuração") }); event.target.value = ""; }));
  $("apply-resume").addEventListener("click", handle(async () => { const path = $("resume-path").value.trim(); if (!path) throw new Error("Informe um checkpoint para retomar."); const report = await api("/training/resume-config", { adapter_dir: path }); renderConfig(report.config); appendLog("Configuração original do checkpoint carregada para retomada."); }));

  function appendLog(message) { state.log.push(String(message)); if (state.log.length > 1500) state.log.splice(0, state.log.length - 1500); const area = $("job-log"); const bottom = area.scrollTop + area.clientHeight >= area.scrollHeight - 50; area.textContent = state.log.join("\n"); if (bottom) area.scrollTop = area.scrollHeight; }
  function renderJob(job) {
    state.job = job;
    const active = ["running", "stopping", "orphaned"].includes(job.status);
    $("job-status").textContent = statuses[job.status] || job.status; $("job-status").className = "status-chip " + job.status;
    $("job-description").textContent = (kinds[job.kind] || job.kind) + " · " + job.id;
    $("cancel-job").disabled = !active || job.status !== "running";
    document.querySelectorAll("form button[type=submit]").forEach((button) => { button.disabled = active; });
    $("download-model").disabled = active;
    $("score-candidates").disabled = active;
    const progress = job.progress || {}; const total = progress.total_steps || progress.total || progress.max_steps; const current = progress.step ?? progress.current ?? progress.global_step;
    const fraction = progress.percent ?? (total && current !== undefined ? 100 * current / total : null);
    if (fraction === null && active) $("job-progress").removeAttribute("value"); else $("job-progress").value = job.status === "completed" ? 100 : Math.min(100, Math.max(0, fraction || 0));
    const metrics = $("job-metrics"); metrics.replaceChildren();
    const values = { ...progress, ...(progress.metrics || {}) };
    ["step", "global_step", "epoch", "loss", "dpo_loss", "preference_accuracy", "learning_rate", "preference_margin", "validation_loss", "validation_preference_accuracy"].forEach((key) => { const value = values[key]; if (value === undefined || value === null) return; const card = document.createElement("div"); card.className = "metric"; const label = document.createElement("span"); label.textContent = key.replaceAll("_", " "); card.append(label, document.createTextNode(typeof value === "number" ? Number(value.toPrecision(5)).toString() : String(value))); metrics.append(card); });
    if (job.error) error(job.error);
    if (job.result && job.result.manifest) { const manifest = job.result.manifest; if (job.kind === "score") { $("pairs-score-file").value = manifest; $("score-output").value = manifest; } else if (job.kind === "pairs") { $("pairs-manifest").value = manifest; $("preprocess-manifest").value = manifest; } else if (job.kind === "preprocess") { state.config.pairs_manifest = manifest; const input = $("field-pairs_manifest"); if (input) input.value = manifest; $("config-json").value = JSON.stringify(state.config, null, 2); } }
    if (job.result && job.kind === "model_setup" && !state.appliedResults.has(job.id)) {
      const downloaded = job.result; const config = collectConfig();
      ["checkpoint_dir", "checkpoint_file", "model_config_dir", "model_variant"].forEach((key) => { if (downloaded[key]) config[key] = downloaded[key]; });
      renderConfig(config);
      if (downloaded.checkpoint_dir) { $("setup-checkpoint").value = downloaded.checkpoint_dir; $("preprocess-checkpoint").value = downloaded.checkpoint_dir; }
      if (downloaded.checkpoint_file) $("preprocess-checkpoint-file").value = downloaded.checkpoint_file;
      if (downloaded.model_config_dir) $("preprocess-model-config").value = downloaded.model_config_dir;
      if (downloaded.model_variant) $("preprocess-variant").value = downloaded.model_variant;
      state.appliedResults.add(job.id);
    }
    if (job.result && job.kind === "train" && job.result.adapter_dir) $("resume-path").value = job.result.adapter_dir;
  }
  async function selectJob(job) { state.cursor = 0; state.log = []; $("job-log").textContent = ""; renderJob(job); await pollJob(); }
  async function refreshJobs() {
    const payload = await api("/jobs"); const history = $("job-history"); history.replaceChildren();
    payload.jobs.forEach((job) => { const button = document.createElement("button"); button.textContent = (kinds[job.kind] || job.kind) + " · " + (statuses[job.status] || job.status) + " · " + new Date(job.started_at * 1000).toLocaleString("pt-BR"); button.addEventListener("click", handle(() => selectJob(job))); history.append(button); });
    if (payload.active && (!state.job || state.job.id !== payload.active.id)) await selectJob(payload.active);
  }
  async function pollJob() {
    if (!state.job || state.polling) return;
    state.polling = true;
    try { const payload = await api("/jobs/" + state.job.id + "?after=" + state.cursor); payload.events.forEach((event) => appendLog(event.message || (event.type === "status" ? "Estado: " + (statuses[event.status] || event.status) : JSON.stringify(event)))); state.cursor = payload.cursor; renderJob(payload.job); } finally { state.polling = false; }
  }
  async function start(kind, config) { const job = await api("/jobs", { kind, config }); await selectJob(job); await refreshJobs(); }
  $("cancel-job").addEventListener("click", handle(async () => { if (!state.job) return; renderJob(await api("/jobs/" + state.job.id + "/stop", {})); }));
  $("refresh-jobs").addEventListener("click", handle(refreshJobs));
  $("clear-log").addEventListener("click", () => { state.log = []; $("job-log").textContent = ""; });
  $("download-model").addEventListener("click", handle(() => start("model_setup", { checkpoint_dir: $("setup-checkpoint").value.trim() })));
  $("score-candidates").addEventListener("click", handle(() => { const options = jsonObject($("score-options").value, "Opções dos avaliadores"); options.device = $("score-device").value.trim(); options.allow_download = $("score-download").checked; return start("score", { input_manifest_or_audio_dir: $("pairs-source").value.trim(), output_file: $("score-output").value.trim(), template: $("score-mode").value === "manual", options }); }));
  $("read-scores").addEventListener("click", handle(async () => { const report = await api("/scores/read", { path: $("score-output").value.trim() }); $("scores-json").value = JSON.stringify(report.data, null, 2); $("score-output").value = report.path; }));
  $("save-scores").addEventListener("click", handle(async () => { const report = await api("/scores/save", { path: $("score-output").value.trim(), data: jsonObject($("scores-json").value, "Avaliações") }); $("pairs-score-file").value = report.path; appendLog("Avaliações salvas: " + report.samples + " candidatos."); }));

  function pairMode() { const mode = $("pairs-mode").value; $("scored-fields").hidden = mode !== "scored"; $("degraded-fields").hidden = mode !== "degraded"; }
  $("pairs-mode").addEventListener("change", pairMode);
  $("pairs-form").addEventListener("submit", handle(async () => { const options = jsonObject($("pairs-options").value, "Opções dos pares"); options.mode = $("pairs-mode").value; options.validation_fraction = Number($("pairs-validation").value); if (options.mode === "scored") options.score_file = $("pairs-score-file").value.trim(); if (options.mode === "degraded") { options.degradation = $("pairs-degradation").value; options.cutoff_hz = Number($("pairs-cutoff").value); if (options.degradation === "vocal_lowpass" && !$("pairs-stems").value.trim()) throw new Error("Informe a pasta de stems para degradar apenas a voz."); options.vocal_stems_dir = $("pairs-stems").value.trim(); } await start("pairs", { input_manifest_or_audio_dir: $("pairs-source").value.trim(), output_dir: $("pairs-output").value.trim(), options }); }));
  $("preprocess-form").addEventListener("submit", handle(() => { const options = jsonObject($("preprocess-options").value, "Opções de pré-processamento"); if ($("preprocess-checkpoint-file").value.trim()) options.checkpoint_file = $("preprocess-checkpoint-file").value.trim(); if ($("preprocess-model-config").value.trim()) options.model_config_dir = $("preprocess-model-config").value.trim(); return start("preprocess", { manifest: $("preprocess-manifest").value.trim(), checkpoint_dir: $("preprocess-checkpoint").value.trim(), model_variant: $("preprocess-variant").value, output_dir: $("preprocess-output").value.trim(), options }); }));
  $("train-form").addEventListener("submit", handle(() => start("train", collectConfig())));
  $("validate-pairs").addEventListener("click", handle(async () => { const report = await api("/pairs/validate", { manifest: $("pairs-manifest").value.trim() }); appendLog(JSON.stringify(report, null, 2)); $("dataset-summary").textContent = report.valid ? "Manifesto válido · " + report.pairs + " pares · " + report.groups + " grupos" : "Manifesto inválido: " + (report.errors || []).join("; "); if (!report.valid) error((report.errors || ["Manifesto inválido"]).join("; ")); }));
  $("preview-pairs").addEventListener("click", handle(async () => {
    const payload = await api("/pairs/preview", { manifest: $("pairs-manifest").value.trim(), limit: 5 }); $("dataset-summary").textContent = payload.count + " pares · exibindo " + payload.pairs.length;
    const preview = $("dataset-preview"); preview.replaceChildren();
    payload.pairs.forEach((pair, index) => { const card = document.createElement("article"); card.className = "pair-card"; const title = document.createElement("h4"); title.textContent = "Par " + (index + 1) + " · " + (pair.split || "sem split"); card.append(title); const caption = document.createElement("p"); caption.textContent = pair.caption || "Sem caption"; card.append(caption); if (pair.lyrics) { const lyrics = document.createElement("p"); lyrics.textContent = "Letra: " + pair.lyrics.slice(0, 180); card.append(lyrics); } const audios = document.createElement("div"); audios.className = "pair-audios"; [["chosen", "Preferido"], ["rejected", "Rejeitado"]].forEach(([key, label]) => { const path = pair[key] || pair[key + "_audio"] || pair[key + "_path"]; if (!path) return; const box = document.createElement("label"); box.textContent = label; const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none"; audio.src = "/api/mrflow/pairs/audio?" + new URLSearchParams({ manifest: payload.path, index, branch: key, token }); box.append(audio); const filename = document.createElement("p"); filename.textContent = path; box.append(filename); audios.append(box); }); card.append(audios); preview.append(card); });
    $("preprocess-manifest").value = payload.path;
  }));
  async function boot() {
    try { const schema = await api("/schema"); state.defaults = schema.defaults; state.fields = Array.isArray(schema.fields) ? schema.fields : Object.entries(schema.fields).map(([name, field]) => ({ name, ...field })); renderConfig(state.defaults); await loadPresets(); await refreshJobs(); $("connection-status").textContent = "Servidor local conectado"; $("connection-status").classList.add("connected"); } catch (exception) { $("connection-status").textContent = "Falha na conexão"; error(exception.message); }
    setInterval(() => { if (state.job && ["running", "stopping"].includes(state.job.status)) pollJob().catch((exception) => error(exception.message)); }, 1500);
  }
  boot();
})();
