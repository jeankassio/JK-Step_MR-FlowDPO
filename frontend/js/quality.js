/* JK-Step: folder preparation, SFT/Flow-DPO configuration and authenticated jobs. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const i18n = window.QualityI18n;
  const t = (key, params) => i18n.t(key, params);
  i18n.init();
  const token = window.__SIDESTEP_TOKEN__ || new URLSearchParams(location.search).get("token") || "";
  const state = { fields: [], defaults: {}, config: {}, presets: [], job: null, cursor: 0, polling: false, log: [], appliedResults: new Set(), prepared: null, workflowMode: "folder", inspection: 0, inspected: null, browsing: 0, browserEmpty: false, browserReturnFocus: null, reportIdentity: null, reportLoading: null, reports: new Map(), error: null, connection: "connecting", pairSummary: null };
  const preferenceFields = new Set(["beta", "fm_regularization", "reference_regularization", "label_smoothing", "pair_weighting", "pairs_manifest"]);
  const stageOrder = ["scan", "lyrics", "caption", "models", "pairs", "preprocess"];
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const statusLabel = (status) => t("status." + status);
  const kindLabel = (kind) => t("kind." + kind);
  const fieldLabel = (name) => i18n.field(name, "label", name);
  function errorParams(params = {}) { return { ...params, ...(params.fieldKey ? { field: t(params.fieldKey) } : {}), ...(params.fieldName ? { field: fieldLabel(params.fieldName) } : {}) }; }
  function localizedError(key, params = {}) { const exception = new Error(t(key, errorParams(params))); exception.translationKey = key; exception.translationParams = params; return exception; }

  function error(message) { state.error = message; $("error-banner").textContent = message?.translationKey ? t(message.translationKey, errorParams(message.translationParams)) : message?.message || String(message); $("error-banner").hidden = false; }
  function clearError() { state.error = null; $("error-banner").hidden = true; }
  async function request(url, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(url, { method: body === undefined ? "GET" : "POST", headers: { Authorization: "Bearer " + token, "Accept-Language": i18n.language, ...(body === undefined ? {} : { "Content-Type": "application/json" }) }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : payload.error || JSON.stringify(payload.detail || payload));
      return payload;
    } finally { clearTimeout(timer); }
  }
  const api = (path, body) => request("/api/mrflow" + path, body);
  function handle(action) { return async (event) => { if (event) event.preventDefault(); clearError(); try { await action(event); } catch (exception) { error(exception.name === "AbortError" ? localizedError("error.timeout") : exception); } }; }
  function jsonObject(text, labelKey, schemaField = false) { const params = schemaField ? { fieldName: labelKey } : { fieldKey: labelKey }; let value; try { value = JSON.parse(text); } catch (_) { throw localizedError("error.json", params); } if (!value || Array.isArray(value) || typeof value !== "object") throw localizedError("error.object", params); return value; }
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
    if (kind === "int" || kind === "integer" || kind === "float" || kind === "number") { const value = Number(input.value); if (!input.value.trim() || !Number.isFinite(value)) throw localizedError("error.number", { fieldName: field.name }); if ((kind === "int" || kind === "integer") && !Number.isInteger(value)) throw localizedError("error.integer", { fieldName: field.name }); return value; }
    if (kind === "list") { const text = input.value.trim(); if (!text) return []; if (text.startsWith("[")) { let value; try { value = JSON.parse(text); } catch (_) { throw localizedError("error.json", { field: fieldLabel(field.name) }); } if (!Array.isArray(value)) throw localizedError("error.list", { field: fieldLabel(field.name) }); return value; } return text.split(",").map((value) => value.trim()).filter(Boolean); }
    if (kind === "json" || kind === "object") return jsonObject(input.value || "{}", field.name, true);
    return input.value;
  }
  function collectConfig() {
    const config = clone(state.config);
    state.fields.forEach((field) => { const input = $("field-" + field.name); if (input && !input.closest("[data-config-field]")?.hidden) config[field.name] = readField(field, input); });
    state.config = config;
    $("config-json").value = JSON.stringify(config, null, 2);
    renderBatchSize(config);
    updateObjectiveUi(config);
    return config;
  }
  function renderBatchSize(config) {
    const ids = Array.isArray(config.gpu_ids) ? config.gpu_ids : String(config.gpu_ids || "").split(",").filter(Boolean);
    const gpus = config.multi_gpu ? ids.length : 1;
    const batch = Number(config.batch_size), accumulation = Number(config.gradient_accumulation);
    $("effective-batch").textContent = t("batch", { count: batch * accumulation * gpus, unit: t(config.objective === "sft" ? "samples_unit" : "pairs_unit"), batch, accumulation, gpus });
  }
  function updateObjectiveUi(config) {
    const sft = config.objective === "sft";
    $("dataset-mode").value = state.workflowMode;
    document.querySelectorAll("[data-workflow]").forEach((element) => { element.hidden = element.dataset.workflow !== state.workflowMode; });
    document.querySelectorAll("[data-config-field]").forEach((element) => { const name = element.dataset.configField; element.hidden = preferenceFields.has(name) ? sft : name === "dataset_manifest" ? !sft : false; });
    document.querySelectorAll("[data-config-group]").forEach((section) => { section.hidden = [...section.querySelectorAll("[data-config-field]")].every((field) => field.hidden); });
    $("train-title").textContent = t(sft ? "train_title_sft" : "train_title_flow");
    $("train-description").textContent = t(sft ? "train_description_sft" : "train_description_flow");
    $("start-training").textContent = t(sft ? "train_button_sft" : "train_button_flow");
    $("workflow-notice").textContent = t(state.workflowMode === "folder" ? "notice_folder" : "notice_pairs");
  }
  function renderConfig(config) {
    state.config = clone(config);
    const container = $("schema-fields"); container.replaceChildren();
    const groups = new Map();
    state.fields.forEach((field) => { const key = field.group || "Advanced"; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(field); });
    groups.forEach((fields, group) => {
      const section = document.createElement("section"); section.className = "schema-group"; section.dataset.configGroup = group;
      const heading = document.createElement("h3"); heading.dataset.schemaGroup = group; heading.textContent = t("group." + group); section.append(heading);
      const grid = document.createElement("div"); grid.className = "field-grid";
      fields.forEach((field) => {
        const label = document.createElement("label"); label.className = "field"; label.htmlFor = "field-" + field.name; label.dataset.configField = field.name;
        const isBool = ["bool", "boolean"].includes(field.type); const value = config[field.name] === undefined ? field.default : config[field.name];
        const title = document.createElement("span"); title.dataset.schemaTitle = field.name; title.textContent = fieldLabel(field.name);
        const help = document.createElement("small"); help.dataset.schemaHelp = field.name; help.textContent = i18n.field(field.name, "help", field.help || "") + " (" + field.name + ")"; help.id = "help-" + field.name;
        let input;
        if (field.choices) { input = document.createElement("select"); field.choices.forEach((choice) => { const option = document.createElement("option"); option.value = typeof choice === "object" ? choice.value : choice; option.dataset.schemaChoice = field.name; option.textContent = i18n.field(field.name, "choices", {})[option.value] || (typeof choice === "object" ? choice.label || choice.value : choice); input.append(option); }); input.value = value; }
        else if (field.type === "list" || field.type === "json" || field.type === "object") { input = document.createElement("textarea"); input.rows = field.type === "list" ? 2 : 4; input.value = JSON.stringify(value ?? (field.type === "list" ? [] : {}), null, 2); input.spellcheck = false; }
        else { input = document.createElement("input"); input.type = isBool ? "checkbox" : ["int", "integer", "float", "number"].includes(field.type) ? "number" : "text"; if (isBool) input.checked = Boolean(value); else input.value = value ?? ""; if (field.min !== undefined) input.min = field.min; if (field.max !== undefined) input.max = field.max; if (input.type === "number") input.step = ["int", "integer"].includes(field.type) ? 1 : "any"; }
        input.id = "field-" + field.name; input.name = field.name; input.setAttribute("aria-describedby", help.id);
        if (isBool) { label.classList.add("boolean"); const copy = document.createElement("span"); copy.className = "field-copy"; copy.append(title, help); label.append(input, copy); } else label.append(title, input, help);
        input.addEventListener("change", handle(() => { const current = collectConfig(); if (field.name === "objective" && current.objective === "sft") renderConfig({ ...current, max_latent_length: 0 }); })); grid.append(label);
      });
      section.append(grid); container.append(section);
    });
    $("config-json").value = JSON.stringify(state.config, null, 2);
    $("resume-path").value = state.config.resume_from || "";
    if (!$("preprocess-checkpoint").value) $("preprocess-checkpoint").value = state.config.checkpoint_dir || "";
    $("preprocess-variant").value = state.config.model_variant || "xl-sft";
    renderBatchSize(state.config);
    updateObjectiveUi(state.config);
  }

  function finiteInput(id, labelKey, min, max, integer = false) {
    const raw = $(id).value.trim(); const value = Number(raw);
    if (!raw || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw localizedError(integer ? "error.integer_range" : "error.range", { fieldKey: labelKey, min, max });
    return value;
  }
  function prepareConfig() {
    const config = {
      objective: $("prepare-objective").value, audio_dir: $("folder-input").value.trim(), checkpoint_dir: $("prepare-checkpoint").value.trim(), model_variant: $("prepare-variant").value,
      device: $("prepare-device").value, precision: $("prepare-precision").value, caption_model: $("prepare-caption-model").value.trim(), caption_tier: $("prepare-caption-tier").value,
      lyrics_model: $("prepare-lyrics-model").value.trim(), language: $("prepare-language").value.trim() || "auto", content_mode: $("prepare-content-mode").value,
      clip_seconds: finiteInput("prepare-clip-seconds", "maximum_duration", 10, 120), min_clip_seconds: finiteInput("prepare-min-seconds", "minimum_duration", 1, 120),
      custom_tag: $("prepare-custom-tag").value.trim(), normalize: "none", allow_download: $("prepare-download").checked, reuse: $("prepare-reuse").checked, preprocess: true,
      validation_fraction: finiteInput("prepare-validation", "validation_fraction", 0, 0.9), seed: finiteInput("prepare-seed", "seed", 0, 2147483647, true),
    };
    if (!config.audio_dir) throw localizedError("error.folder");
    if (!config.checkpoint_dir || !config.caption_model || !config.lyrics_model) throw localizedError("error.models");
    if (config.min_clip_seconds > config.clip_seconds) throw localizedError("error.durations");
    if (config.objective === "flow_dpo") config.pair_options = { degradation: $("prepare-degradation").value, variations_per_audio: finiteInput("prepare-variations", "variations", 1, 100, true), cutoff_hz: finiteInput("prepare-cutoff", "cutoff", 100, 20000), noise_snr_db: finiteInput("prepare-noise-snr", "noise_snr", 0, 80), clip_threshold: finiteInput("prepare-clipping", "clipping_threshold", 0.00001, 1) };
    if ($("prepare-output").value.trim()) config.output_dir = $("prepare-output").value.trim();
    if ($("prepare-name").value.trim()) config.dataset_name = $("prepare-name").value.trim();
    return config;
  }
  async function inspectFolder() {
    const audioDir = $("folder-input").value.trim(); const sequence = ++state.inspection;
    if (!audioDir) { state.inspected = null; $("folder-summary").textContent = t("folder_empty"); return; }
    $("folder-summary").textContent = t("folder_reading");
    const report = await api("/datasets/inspect", { audio_dir: audioDir });
    if (sequence !== state.inspection || audioDir !== $("folder-input").value.trim()) return;
    state.inspected = report; renderFolderInspection(report);
  }
  function renderFolderInspection(report) {
    const count = Number(report.count ?? report.supported_files?.length ?? report.supported_files ?? 0);
    const metadata = typeof report.existing_metadata === "number" ? report.existing_metadata : report.existing_metadata?.any ?? report.existing_metadata?.count;
    const coverage = report.existing_metadata && typeof report.existing_metadata === "object" ? t("folder_metadata", { captions: report.existing_metadata.caption || 0, lyrics: report.existing_metadata.lyrics || 0 }) : Number.isFinite(metadata) ? t("folder_metadata_count", { count: metadata }) : "";
    $("folder-summary").textContent = t("folder_count", { count }) + coverage + t(count === 0 ? "folder_no_audio" : "folder_ready");
  }
  function normalizePath(path) { return String(path || "/").replace(/\\/g, "/").replace(/\/+$/, "") || "/"; }
  function parentPath(path) {
    const normalized = normalizePath(path);
    if (normalized === "/" || /^[A-Za-z]:$/.test(normalized)) return "/";
    const parent = normalized.slice(0, normalized.lastIndexOf("/")) || "/";
    return /^[A-Za-z]:$/.test(parent) ? parent + "/" : parent;
  }
  async function browseFolder(path) {
    const sequence = ++state.browsing; const normalized = normalizePath(path); const target = /^[A-Za-z]:$/.test(normalized) ? normalized + "/" : normalized;
    $("folder-browser-path").value = target; $("folder-browser-error").hidden = true;
    $("folder-browser-select").disabled = true; $("folder-browser-list").replaceChildren();
    try {
      const payload = await request("/api/browse", { path: target, dirs_only: true });
      if (sequence !== state.browsing) return;
      if (payload.error) throw new Error(payload.error);
      $("folder-browser-path").value = payload.path || target;
      const directories = (payload.entries || []).filter((entry) => entry.is_dir);
      directories.forEach((entry) => { const button = document.createElement("button"); button.className = "folder-entry"; button.type = "button"; button.textContent = "▸ " + entry.name; button.addEventListener("click", () => browseFolder(entry.path)); $("folder-browser-list").append(button); });
      state.browserEmpty = !directories.length;
      if (!directories.length) { const empty = document.createElement("p"); empty.className = "muted"; empty.dataset.i18n = "browser_empty"; empty.textContent = t("browser_empty"); $("folder-browser-list").append(empty); }
      $("folder-browser-select").disabled = normalizePath(payload.path || target) === "/";
    } catch (exception) { if (sequence === state.browsing) { $("folder-browser-error").textContent = exception.message; $("folder-browser-error").hidden = false; } }
  }
  function closeFolderBrowser() { ++state.browsing; $("folder-browser").close(); state.browserReturnFocus?.focus(); }
  $("choose-folder").addEventListener("click", handle(async () => { state.browserReturnFocus = document.activeElement; $("folder-browser").showModal(); await browseFolder($("folder-input").value.trim() || "/"); }));
  $("folder-browser-close").addEventListener("click", closeFolderBrowser);
  $("folder-browser-cancel").addEventListener("click", closeFolderBrowser);
  $("folder-browser").addEventListener("cancel", () => { ++state.browsing; });
  $("folder-browser-go").addEventListener("click", () => browseFolder($("folder-browser-path").value));
  $("folder-browser-up").addEventListener("click", () => browseFolder(parentPath($("folder-browser-path").value)));
  $("folder-browser-path").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); browseFolder($("folder-browser-path").value); } });
  $("folder-browser-select").addEventListener("click", handle(async () => { $("folder-input").value = $("folder-browser-path").value; closeFolderBrowser(); await inspectFolder(); }));
  $("folder-input").addEventListener("change", handle(inspectFolder));
  $("inspect-folder").addEventListener("click", handle(inspectFolder));
  $("dataset-mode").addEventListener("change", handle(() => { const mode = $("dataset-mode").value; const config = collectConfig(); state.workflowMode = mode; config.objective = mode === "folder" ? $("prepare-objective").value : "flow_dpo"; config.max_latent_length = 0; renderConfig(config); }));
  function preparationObjective() { const flow = $("prepare-objective").value === "flow_dpo"; $("prepare-pair-options").hidden = !flow; $("prepare-pair-options").querySelectorAll("input").forEach((input) => { input.disabled = !flow; }); }
  $("prepare-objective").addEventListener("change", handle(() => { preparationObjective(); renderConfig({ ...collectConfig(), objective: $("prepare-objective").value, max_latent_length: 0 }); }));
  $("prepare-degradation").addEventListener("change", () => { $("prepare-variations").value = $("prepare-degradation").value === "mixed" ? "3" : "1"; });
  $("back-to-folder").addEventListener("click", () => tab("pairs"));
  $("prepare-form").addEventListener("submit", handle(async () => { const config = prepareConfig(); state.prepared = null; $("prepared-result").hidden = true; $("folder-preview").replaceChildren(); await start("prepare_dataset", config); tab("preprocess"); }));

  function readyDataset(job) {
    const result = job.result;
    return job.kind === "prepare_dataset" && job.status === "completed" && result?.ready === true && ["ready", "partial"].includes(result.status) && Boolean(result.objective === "flow_dpo" ? result.pairs_manifest : result.dataset_manifest) && Boolean(result.tensor_dir);
  }
  function sampleCount(value) { const count = Array.isArray(value) ? value.length : Number(value || 0); return Number.isFinite(count) ? count : 0; }
  function applyPreparedDataset(job) {
    if (!readyDataset(job)) return;
    const result = job.result; const objective = result.objective || "sft"; const flow = objective === "flow_dpo"; const preset = state.presets.find((item) => item.name === (flow ? "conservative" : "sft_lora"));
    if (!state.appliedResults.has(job.id)) {
      const name = job.config?.dataset_name || result.dataset_name || normalizePath(job.config?.audio_dir || "dataset").split("/").pop() || "dataset";
      const safeName = name.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/[. ]+$/, "") || "dataset";
      const config = { ...clone(state.defaults), ...state.config, ...(preset?.config || {}), objective, dataset_manifest: flow ? "" : result.dataset_manifest, pairs_manifest: flow ? result.pairs_manifest : "", checkpoint_dir: result.checkpoint_dir || job.config?.checkpoint_dir || "checkpoints", model_variant: result.model_variant || job.config?.model_variant || "xl-sft", max_latent_length: 0, output_dir: "output/" + safeName, resume_from: "", init_adapter: "", ...(flow ? { cfg_dropout: 0 } : {}) };
      // A folder result uses the prepared checkpoint, never a stale custom-weight override.
      config.checkpoint_file = ""; config.model_config_dir = "";
      state.workflowMode = "folder"; renderConfig(config); state.appliedResults.add(job.id);
      if (preset) $("preset-select").value = String(state.presets.indexOf(preset));
      if (job.config?.audio_dir) $("folder-input").value = job.config.audio_dir;
      if (job.config?.dataset_name) $("prepare-name").value = job.config.dataset_name;
      $("prepare-checkpoint").value = config.checkpoint_dir; $("prepare-variant").value = config.model_variant;
      $("prepare-objective").value = objective; preparationObjective();
      appendLog(t("prepared_log", { objective: flow ? "MR-FlowDPO" : "LoRA SFT", manifest: flow ? result.pairs_manifest : result.dataset_manifest }));
    }
    state.prepared = job;
    $("prepared-dataset-json").value = result.dataset_json || ""; $("prepared-manifest").value = flow ? result.pairs_manifest : result.dataset_manifest; $("prepared-tensors").value = result.tensor_dir;
    $("prepared-summary").textContent = t("prepared_summary", { samples: sampleCount(flow ? result.pairs : result.samples), unit: t(flow ? "pairs_unit" : "samples_unit"), prepared: sampleCount(result.preprocessed), excluded: sampleCount(result.quarantined) ? t("excluded_summary", { count: sampleCount(result.quarantined) }) : "" });
    $("prepared-result").hidden = false;
    $("use-prepared-dataset").disabled = false;
  }
  function renderReport(payload) {
    const exclusions = Array.isArray(payload.quarantine) ? payload.quarantine : [];
    $("preparation-exclusions").replaceChildren();
    exclusions.forEach((entry) => {
      const item = document.createElement("li"); const title = document.createElement("strong"); title.textContent = entry.id || entry.source_audio_path || t("exclusion_default");
      const reason = document.createElement("p"); reason.textContent = entry.reason || t("reason_unknown"); item.append(title, reason); $("preparation-exclusions").append(item);
    });
    $("preparation-report-status").textContent = exclusions.length ? t(payload.truncated ? "report_truncated" : "report_count", { count: exclusions.length, total: sampleCount(payload.quarantined) }) : t("report_no_exclusions");
  }
  function renderPreparationOutcome(job) {
    const result = job.result; const terminal = ["completed", "failed", "cancelled", "interrupted"].includes(job.status);
    const show = terminal && Boolean(result);
    $("preparation-outcome").hidden = !show;
    $("show-preparation-report").disabled = !show || !result?.report;
    if (!show) { $("preparation-report-path").value = ""; return; }
    $("preparation-counts").textContent = t("report_counts", { samples: sampleCount(result.samples), prepared: sampleCount(result.preprocessed), excluded: sampleCount(result.quarantined) });
    const identity = job.id + ":" + (result.report || "");
    $("preparation-report-path").value = result.report || "";
    $("show-preparation-report").disabled = !result.report || state.reportLoading === identity;
    if (state.reportIdentity !== identity) {
      state.reportIdentity = identity; $("preparation-issues").open = false; $("preparation-exclusions").replaceChildren();
      $("preparation-report-status").textContent = t(result.report ? "report_instruction" : "report_missing");
      if (state.reports.has(identity)) renderReport(state.reports.get(identity));
    }
  }
  $("show-preparation-report").addEventListener("click", handle(async () => {
    const job = state.job;
    if (job?.kind !== "prepare_dataset" || !job.result?.report || !["completed", "failed", "cancelled", "interrupted"].includes(job.status)) throw localizedError("error.report");
    const identity = job.id + ":" + job.result.report; state.reportLoading = identity; $("show-preparation-report").disabled = true;
    $("preparation-report-status").textContent = t("report_loading"); $("preparation-issues").open = true;
    try {
      const payload = await api("/datasets/report", { report: job.result.report }); state.reports.set(identity, payload);
      if (state.reportIdentity === identity && state.job?.id === job.id) renderReport(payload);
    } catch (exception) { if (state.reportIdentity === identity) $("preparation-report-status").textContent = t("report_error", { message: exception.message }); }
    finally { if (state.reportLoading === identity) state.reportLoading = null; if (state.reportIdentity === identity) $("show-preparation-report").disabled = false; }
  }));
  function renderPreparation(job) {
    if (job.kind !== "prepare_dataset") { $("preparation-outcome").hidden = true; return; }
    const progress = job.progress || {}; const rawStage = String(progress.stage || progress.stage_name || progress.phase || "scan").toLowerCase();
    const stage = stageOrder.find((name) => rawStage.includes(name)) || (rawStage.includes("transcri") || rawStage.includes("whisper") || rawStage.includes("align") ? "lyrics" : rawStage.includes("tensor") || ["ready", "result"].includes(rawStage) ? "preprocess" : rawStage.includes("manifest") || rawStage.includes("model") ? "models" : "scan");
    const current = stageOrder.indexOf(stage); const ready = readyDataset(job);
    document.querySelectorAll("[data-stage]").forEach((element) => { const index = stageOrder.indexOf(element.dataset.stage); element.hidden = element.dataset.stage === "pairs" && (job.result?.objective || job.config?.objective) !== "flow_dpo"; element.classList.toggle("done", ready || index < current); element.classList.toggle("active", !ready && index === current && job.status === "running"); });
    $("preparation-summary").textContent = ready ? t("preparation_complete") : job.status === "completed" ? t("preparation_pending") : ["failed", "cancelled", "interrupted"].includes(job.status) ? t("preparation_stopped") : t("stage_running." + stage) + (progress.current !== undefined && progress.total ? " · " + progress.current + " / " + progress.total : "…");
    renderPreparationOutcome(job);
    if (!ready) { state.prepared = null; $("prepared-result").hidden = true; $("use-prepared-dataset").disabled = true; ["prepared-dataset-json", "prepared-manifest", "prepared-tensors"].forEach((id) => { $(id).value = ""; }); }
    applyPreparedDataset(job);
  }
  async function previewDataset() {
    if (state.prepared?.result?.objective === "flow_dpo") { const payload = await api("/pairs/preview", { manifest: state.prepared.result.pairs_manifest, limit: 5 }); renderPairCards(payload, "folder-preview"); return; }
    const manifest = state.prepared?.result?.dataset_json || state.prepared?.result?.dataset_manifest || $("prepared-dataset-json").value;
    if (!manifest) throw localizedError("error.preview");
    const payload = await api("/datasets/preview", { manifest, limit: 5 }); const preview = $("folder-preview"); preview.replaceChildren();
    (payload.samples || []).forEach((sample, index) => {
      const card = document.createElement("article"); card.className = "pair-card";
      const heading = document.createElement("h4"); heading.dataset.sampleIndex = String(index + 1); heading.dataset.sampleKind = sample.is_instrumental ? "instrumental" : "voice"; heading.textContent = t("sample_heading", { index: index + 1, kind: t(heading.dataset.sampleKind) }); card.append(heading);
      const caption = document.createElement("p"); if (!sample.caption) caption.dataset.i18n = "no_caption"; caption.textContent = sample.caption || t("no_caption"); card.append(caption);
      if (sample.lyrics) { const details = document.createElement("details"); const summary = document.createElement("summary"); summary.dataset.i18n = "view_lyrics"; summary.textContent = t("view_lyrics"); const lyrics = document.createElement("p"); lyrics.className = "sample-lyrics"; lyrics.textContent = sample.lyrics; details.append(summary, lyrics); card.append(details); }
      if (sample.audio_path) { const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none"; audio.src = "/api/mrflow/datasets/audio?" + new URLSearchParams({ manifest: payload.path || manifest, index, token }); card.append(audio); }
      preview.append(card);
    });
    if (!(payload.samples || []).length) { const empty = document.createElement("p"); empty.dataset.i18n = "no_preview"; empty.textContent = t("no_preview"); preview.append(empty); }
  }
  $("preview-dataset").addEventListener("click", handle(previewDataset));
  $("use-prepared-dataset").addEventListener("click", handle(() => { if (!state.prepared || !readyDataset(state.prepared)) throw localizedError("error.not_ready"); const result = state.prepared.result; const flow = result.objective === "flow_dpo"; renderConfig({ ...collectConfig(), objective: flow ? "flow_dpo" : "sft", dataset_manifest: flow ? "" : result.dataset_manifest, pairs_manifest: flow ? result.pairs_manifest : "", checkpoint_dir: result.checkpoint_dir || state.prepared.config?.checkpoint_dir || "checkpoints", model_variant: result.model_variant || state.prepared.config?.model_variant || "xl-sft", checkpoint_file: "", model_config_dir: "", max_latent_length: 0, ...(flow ? { cfg_dropout: 0 } : {}) }); tab("train"); }));

  function normalizePresets(payload) {
    const items = [];
    const builtin = payload.builtin || [];
    if (Array.isArray(builtin)) builtin.forEach((item) => items.push({ name: item.name, config: item.config || item.data || item, description: item.description }));
    else Object.entries(builtin).forEach(([name, item]) => items.push({ name, config: item.config || item.data || item, description: item.description }));
    (payload.saved || []).forEach((item) => items.push({ ...item, saved: true }));
    return items;
  }
  function presetLabel(preset) { const key = "preset." + preset.name; const label = preset.saved || t(key) === key ? preset.name : t(key); return label + (preset.saved ? " · " + t("preset_saved") : ""); }
  async function loadPresets() { state.presets = normalizePresets(await api("/presets")); const select = $("preset-select"); select.replaceChildren(new Option(t("current_config"), "")); state.presets.forEach((preset, index) => select.add(new Option(presetLabel(preset), String(index)))); }
  $("apply-preset").addEventListener("click", handle(() => { const index = $("preset-select").value; if (index === "") return; const preset = state.presets[Number(index)]; renderConfig({ ...clone(state.defaults), ...collectConfig(), ...preset.config }); appendLog(t("preset_applied", { name: presetLabel(preset) })); }));
  $("save-preset").addEventListener("click", handle(async () => { const name = window.prompt(t("preset_name")); if (!name || !name.trim()) return; await api("/presets", { name: name.trim(), config: collectConfig() }); await loadPresets(); }));
  $("reset-config").addEventListener("click", handle(() => renderConfig(state.defaults)));
  $("apply-json").addEventListener("click", handle(() => renderConfig({ ...clone(state.defaults), ...jsonObject($("config-json").value, "configuration") })));
  $("export-json").addEventListener("click", handle(() => { const blob = new Blob([JSON.stringify(collectConfig(), null, 2)], { type: "application/json" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "jk-step-mrflow-config.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); }));
  $("import-json").addEventListener("change", handle(async (event) => { const file = event.target.files[0]; if (file) renderConfig({ ...clone(state.defaults), ...jsonObject(await file.text(), "configuration_file") }); event.target.value = ""; }));
  $("apply-resume").addEventListener("click", handle(async () => { const path = $("resume-path").value.trim(); if (!path) throw localizedError("error.resume"); const report = await api("/training/resume-config", { adapter_dir: path }); renderConfig(report.config); appendLog(t("resume_log")); }));

  function appendLog(message) { state.log.push(String(message)); if (state.log.length > 1500) state.log.splice(0, state.log.length - 1500); const area = $("job-log"); const bottom = area.scrollTop + area.clientHeight >= area.scrollHeight - 50; area.textContent = state.log.join("\n"); if (bottom) area.scrollTop = area.scrollHeight; }
  function renderJob(job) {
    state.job = job;
    const active = ["running", "stopping", "orphaned"].includes(job.status);
    $("job-status").textContent = statusLabel(job.status); $("job-status").className = "status-chip " + job.status;
    $("job-description").textContent = (job.kind === "train" ? t(job.config?.objective === "sft" ? "train_job_sft" : "train_job_flow") : kindLabel(job.kind)) + " · " + job.id;
    $("cancel-job").disabled = !active || job.status !== "running";
    document.querySelectorAll("form button[type=submit]").forEach((button) => { button.disabled = active; });
    $("download-model").disabled = active;
    $("score-candidates").disabled = active;
    $("use-prepared-dataset").disabled = active;
    const progress = job.progress || {}; const total = progress.total_steps || progress.total || progress.max_steps; const current = progress.step ?? progress.current ?? progress.global_step;
    const fraction = progress.percent ?? (total && current !== undefined ? 100 * current / total : null);
    if (fraction === null && active) $("job-progress").removeAttribute("value"); else $("job-progress").value = job.status === "completed" ? 100 : Math.min(100, Math.max(0, fraction || 0));
    const metrics = $("job-metrics"); metrics.replaceChildren();
    const values = { ...progress, ...(progress.metrics || {}) };
    ["step", "global_step", "epoch", "loss", "dpo_loss", "preference_accuracy", "learning_rate", "preference_margin", "validation_loss", "validation_preference_accuracy"].forEach((key) => { const value = values[key]; if (value === undefined || value === null) return; const card = document.createElement("div"); card.className = "metric"; const label = document.createElement("span"); label.textContent = t("metric." + key); card.append(label, document.createTextNode(typeof value === "number" ? Number(value.toPrecision(5)).toString() : String(value))); metrics.append(card); });
    if (job.error) error(job.error);
    if (job.status === "completed" && job.result && job.result.manifest) { const manifest = job.result.manifest; if (job.kind === "score") { $("pairs-score-file").value = manifest; $("score-output").value = manifest; } else if (job.kind === "pairs") { $("pairs-manifest").value = manifest; $("preprocess-manifest").value = manifest; } else if (job.kind === "preprocess") { state.config.pairs_manifest = manifest; const input = $("field-pairs_manifest"); if (input) input.value = manifest; $("config-json").value = JSON.stringify(state.config, null, 2); } }
    if (job.status === "completed" && job.result && job.kind === "model_setup" && !state.appliedResults.has(job.id)) {
      const downloaded = job.result; const config = collectConfig();
      ["checkpoint_dir", "checkpoint_file", "model_config_dir", "model_variant"].forEach((key) => { if (downloaded[key]) config[key] = downloaded[key]; });
      renderConfig(config);
      if (downloaded.checkpoint_dir) { $("setup-checkpoint").value = downloaded.checkpoint_dir; $("preprocess-checkpoint").value = downloaded.checkpoint_dir; }
      if (downloaded.checkpoint_file) $("preprocess-checkpoint-file").value = downloaded.checkpoint_file;
      if (downloaded.model_config_dir) $("preprocess-model-config").value = downloaded.model_config_dir;
      if (downloaded.model_variant) $("preprocess-variant").value = downloaded.model_variant;
      if (downloaded.checkpoint_dir) $("prepare-checkpoint").value = downloaded.checkpoint_dir;
      if (downloaded.model_variant) $("prepare-variant").value = downloaded.model_variant;
      state.appliedResults.add(job.id);
    }
    if (job.status === "completed" && job.result && job.kind === "train" && job.result.adapter_dir) $("resume-path").value = job.result.adapter_dir;
    renderPreparation(job);
  }
  async function selectJob(job) { state.cursor = 0; state.log = []; $("job-log").textContent = ""; renderJob(job); await pollJob(); }
  function historyLabel(job) { return kindLabel(job.kind || job.jobKind) + " · " + statusLabel(job.status || job.jobStatus) + " · " + new Date(Number(job.started_at || job.jobStarted) * 1000).toLocaleString(i18n.dateLocale()); }
  async function refreshJobs() {
    const payload = await api("/jobs"); const history = $("job-history"); history.replaceChildren();
    payload.jobs.forEach((job) => { const button = document.createElement("button"); button.dataset.jobKind = job.kind; button.dataset.jobStatus = job.status; button.dataset.jobStarted = String(job.started_at); button.textContent = historyLabel(job); button.addEventListener("click", handle(() => selectJob(job))); history.append(button); });
    if (payload.active && (!state.job || state.job.id !== payload.active.id)) await selectJob(payload.active);
    else if (!state.job) { const prepared = payload.jobs.find(readyDataset); if (prepared) await selectJob(prepared); }
  }
  async function pollJob() {
    if (!state.job || state.polling) return;
    const jobId = state.job.id;
    state.polling = true;
    try { const payload = await api("/jobs/" + jobId + "?after=" + state.cursor); if (state.job?.id !== jobId) return; payload.events.forEach((event) => appendLog(event.message || (event.type === "status" ? t("status_log", { status: statusLabel(event.status) }) : JSON.stringify(event)))); state.cursor = payload.cursor; renderJob(payload.job); } finally { state.polling = false; }
  }
  async function start(kind, config) { const job = await api("/jobs", { kind, config }); await selectJob(job); await refreshJobs(); }
  $("cancel-job").addEventListener("click", handle(async () => { if (!state.job) return; renderJob(await api("/jobs/" + state.job.id + "/stop", {})); }));
  $("refresh-jobs").addEventListener("click", handle(refreshJobs));
  $("clear-log").addEventListener("click", () => { state.log = []; $("job-log").textContent = ""; });
  $("download-model").addEventListener("click", handle(() => start("model_setup", { checkpoint_dir: $("setup-checkpoint").value.trim() })));
  $("score-candidates").addEventListener("click", handle(() => { const options = jsonObject($("score-options").value, "scorer_options"); options.device = $("score-device").value.trim(); options.allow_download = $("score-download").checked; return start("score", { input_manifest_or_audio_dir: $("pairs-source").value.trim(), output_file: $("score-output").value.trim(), template: $("score-mode").value === "manual", options }); }));
  $("read-scores").addEventListener("click", handle(async () => { const report = await api("/scores/read", { path: $("score-output").value.trim() }); $("scores-json").value = JSON.stringify(report.data, null, 2); $("score-output").value = report.path; }));
  $("save-scores").addEventListener("click", handle(async () => { const report = await api("/scores/save", { path: $("score-output").value.trim(), data: jsonObject($("scores-json").value, "ratings_editor") }); $("pairs-score-file").value = report.path; appendLog(t("ratings_saved", { count: report.samples })); }));

  function pairMode() { const mode = $("pairs-mode").value; $("scored-fields").hidden = mode !== "scored"; $("degraded-fields").hidden = mode !== "degraded"; }
  $("pairs-mode").addEventListener("change", pairMode);
  $("pairs-form").addEventListener("submit", handle(async () => { const options = jsonObject($("pairs-options").value, "pairs_options"); options.mode = $("pairs-mode").value; options.validation_fraction = Number($("pairs-validation").value); if (options.mode === "scored") options.score_file = $("pairs-score-file").value.trim(); if (options.mode === "degraded") { options.degradation = $("pairs-degradation").value; options.cutoff_hz = Number($("pairs-cutoff").value); if (options.degradation === "vocal_lowpass" && !$("pairs-stems").value.trim()) throw localizedError("error.stems"); options.vocal_stems_dir = $("pairs-stems").value.trim(); } await start("pairs", { input_manifest_or_audio_dir: $("pairs-source").value.trim(), output_dir: $("pairs-output").value.trim(), options }); }));
  $("preprocess-form").addEventListener("submit", handle(() => { const options = jsonObject($("preprocess-options").value, "preprocess_options"); if ($("preprocess-checkpoint-file").value.trim()) options.checkpoint_file = $("preprocess-checkpoint-file").value.trim(); if ($("preprocess-model-config").value.trim()) options.model_config_dir = $("preprocess-model-config").value.trim(); return start("preprocess", { manifest: $("preprocess-manifest").value.trim(), checkpoint_dir: $("preprocess-checkpoint").value.trim(), model_variant: $("preprocess-variant").value, output_dir: $("preprocess-output").value.trim(), options }); }));
  $("train-form").addEventListener("submit", handle(() => start("train", collectConfig())));
  function renderPairSummary() { if (!state.pairSummary) return; $("dataset-summary").textContent = t(state.pairSummary.key, state.pairSummary.params); }
  $("validate-pairs").addEventListener("click", handle(async () => { const report = await api("/pairs/validate", { manifest: $("pairs-manifest").value.trim() }); appendLog(JSON.stringify(report, null, 2)); state.pairSummary = report.valid ? { key: "manifest_valid", params: { pairs: report.pairs, groups: report.groups } } : { key: "manifest_invalid", params: { errors: (report.errors || []).join("; ") } }; renderPairSummary(); if (!report.valid) error(localizedError(state.pairSummary.key, state.pairSummary.params)); }));
  function renderPairCards(payload, targetId) {
    const preview = $(targetId); preview.replaceChildren();
    (payload.pairs || []).forEach((pair, index) => {
      const card = document.createElement("article"); card.className = "pair-card";
      const title = document.createElement("h4"); title.dataset.pairIndex = String(index + 1); title.dataset.pairSplit = pair.split || ""; title.textContent = t("pair_heading", { index: index + 1, split: pair.split || t("no_split") }); card.append(title);
      const caption = document.createElement("p"); if (!pair.caption) caption.dataset.i18n = "no_caption"; caption.textContent = pair.caption || t("no_caption"); card.append(caption);
      if (pair.lyrics) { const details = document.createElement("details"); const summary = document.createElement("summary"); summary.dataset.i18n = "view_lyrics"; summary.textContent = t("view_lyrics"); const lyrics = document.createElement("p"); lyrics.className = "sample-lyrics"; lyrics.textContent = pair.lyrics; details.append(summary, lyrics); card.append(details); }
      const audios = document.createElement("div"); audios.className = "pair-audios";
      [["chosen", "preferred"], ["rejected", "rejected"]].forEach(([key, labelKey]) => {
        const path = pair[key] || pair[key + "_audio"] || pair[key + "_path"]; if (!path) return;
        const box = document.createElement("label"); const label = document.createElement("span"); label.dataset.i18n = labelKey; label.textContent = t(labelKey); box.append(label);
        const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none"; audio.src = "/api/mrflow/pairs/audio?" + new URLSearchParams({ manifest: payload.path, index, branch: key, token }); box.append(audio);
        const filename = document.createElement("p"); filename.textContent = path; box.append(filename); audios.append(box);
      });
      card.append(audios); preview.append(card);
    });
    if (!(payload.pairs || []).length) { const empty = document.createElement("p"); empty.dataset.i18n = "no_preview"; empty.textContent = t("no_preview"); preview.append(empty); }
  }
  $("preview-pairs").addEventListener("click", handle(async () => {
    const payload = await api("/pairs/preview", { manifest: $("pairs-manifest").value.trim(), limit: 5 }); state.pairSummary = { key: "pairs_preview", params: { count: payload.count, shown: payload.pairs.length } }; renderPairSummary(); renderPairCards(payload, "dataset-preview");
    $("preprocess-manifest").value = payload.path;
  }));
  function refreshLocale() {
    document.querySelectorAll("[data-schema-title]").forEach((element) => { element.textContent = fieldLabel(element.dataset.schemaTitle); });
    document.querySelectorAll("[data-schema-help]").forEach((element) => { const name = element.dataset.schemaHelp; element.textContent = i18n.field(name, "help", state.fields.find((field) => field.name === name)?.help || "") + " (" + name + ")"; });
    document.querySelectorAll("[data-schema-choice]").forEach((element) => { element.textContent = i18n.field(element.dataset.schemaChoice, "choices", {})[element.value] || element.value; });
    document.querySelectorAll("[data-schema-group]").forEach((element) => { element.textContent = t("group." + element.dataset.schemaGroup); });
    [...$("preset-select").children].forEach((option) => { option.textContent = option.value === "" ? t("current_config") : presetLabel(state.presets[Number(option.value)]); });
    document.querySelectorAll("[data-job-kind]").forEach((element) => { element.textContent = historyLabel(element.dataset); });
    document.querySelectorAll("[data-sample-index]").forEach((element) => { element.textContent = t("sample_heading", { index: element.dataset.sampleIndex, kind: t(element.dataset.sampleKind) }); });
    document.querySelectorAll("[data-pair-index]").forEach((element) => { element.textContent = t("pair_heading", { index: element.dataset.pairIndex, split: element.dataset.pairSplit || t("no_split") }); });
    $("connection-status").textContent = t(state.connection);
    updateObjectiveUi(state.config); renderBatchSize(state.config); renderPairSummary();
    if (state.inspected) renderFolderInspection(state.inspected);
    const currentError = state.error;
    if (state.job) renderJob(state.job);
    if (state.reportIdentity && state.reports.has(state.reportIdentity)) renderReport(state.reports.get(state.reportIdentity));
    else if (state.job?.kind === "prepare_dataset" && !$("preparation-outcome").hidden) $("preparation-report-status").textContent = t(state.reportLoading ? "report_loading" : state.job.result?.report ? "report_instruction" : "report_missing");
    if (currentError) error(currentError);
  }
  window.addEventListener("jk-step-language-changed", refreshLocale);
  async function boot() {
    preparationObjective(); pairMode();
    try { const schema = await api("/schema"); state.defaults = schema.defaults; state.fields = Array.isArray(schema.fields) ? schema.fields : Object.entries(schema.fields).map(([name, field]) => ({ name, ...field })); renderConfig({ ...state.defaults, objective: "flow_dpo", max_latent_length: 0, cfg_dropout: 0 }); await loadPresets(); const preset = state.presets.find((item) => item.name === "conservative"); if (preset) { renderConfig({ ...state.defaults, ...preset.config, objective: "flow_dpo", max_latent_length: 0, cfg_dropout: 0 }); $("preset-select").value = String(state.presets.indexOf(preset)); } await refreshJobs(); state.connection = "connected"; $("connection-status").textContent = t(state.connection); $("connection-status").classList.add("connected"); } catch (exception) { state.connection = "connection_failed"; $("connection-status").textContent = t(state.connection); error(exception); }
    setInterval(() => { if (state.job && ["running", "stopping", "orphaned"].includes(state.job.status)) pollJob().catch((exception) => error(exception)); }, 1500);
  }
  boot();
})();
