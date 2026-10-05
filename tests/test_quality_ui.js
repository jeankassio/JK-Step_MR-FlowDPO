/* Run with: node --test tests/test_quality_ui.js. No browser or server required. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const token = "quality-ui-test-token";
const copy = (value) => JSON.parse(JSON.stringify(value));

class Element {
  constructor(tagName, document) {
    this.tagName = tagName.toUpperCase();
    this.nodeType = this.tagName === "#TEXT" ? 3 : this.tagName === "DOCUMENT" ? 9 : 1;
    this.ownerDocument = document;
    this.children = [];
    this.parentElement = null;
    this.attributes = {};
    this.dataset = {};
    this.listeners = new Map();
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this.required = false;
    this.open = false;
    this.scrollTop = 0;
    this.clientHeight = 100;
    this.scrollHeight = 100;
    this._text = "";
    this._value = undefined;
    this.classList = {
      contains: (name) => this.className.split(/\s+/).includes(name),
      add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(" "); },
      remove: (...names) => { this.className = this.className.split(/\s+/).filter((name) => !names.includes(name)).join(" "); },
      toggle: (name, force) => {
        const enabled = force === undefined ? !this.classList.contains(name) : Boolean(force);
        this.classList[enabled ? "add" : "remove"](name);
        return enabled;
      },
    };
  }
  get id() { return this.attributes.id || ""; }
  set id(value) { this.attributes.id = String(value); }
  get className() { return this.attributes.class || ""; }
  set className(value) { this.attributes.class = String(value); }
  get childNodes() { return this.children; }
  get firstChild() { return this.children[0] || null; }
  get lang() { return this.attributes.lang || ""; }
  set lang(value) { this.attributes.lang = String(value); }
  get options() { return this.children; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.replaceChildren(); }
  get value() {
    if (this._value !== undefined) return this._value;
    if (this.tagName === "TEXTAREA") return this.textContent;
    if (this.tagName === "SELECT") return (this.children.find((child) => child.attributes.selected !== undefined) || this.children[0] || {}).value || "";
    return this.attributes.value || "";
  }
  set value(value) {
    const text = String(value);
    this._value = this.tagName === "SELECT" && !this.children.some((option) => option.value === text) ? "" : text;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
    if (["hidden", "checked", "disabled", "required", "open"].includes(name)) this[name] = true;
  }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) {
    delete this.attributes[name];
    if (["hidden", "checked", "disabled", "required", "open"].includes(name)) this[name] = false;
    if (name === "value") this._value = undefined;
  }
  append(...children) {
    for (const child of children) {
      const node = typeof child === "string" ? this.ownerDocument.createTextNode(child) : child;
      if (node.parentElement) node.remove();
      node.parentElement = this;
      this.children.push(node);
    }
  }
  appendChild(child) { this.append(child); return child; }
  add(child) { this.append(child); }
  replaceChildren(...children) { this.children.forEach((child) => { child.parentElement = null; }); this.children = []; this.append(...children); }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter((child) => child !== this); this.parentElement = null; }
  addEventListener(type, listener) { const listeners = this.listeners.get(type) || []; listeners.push(listener); this.listeners.set(type, listeners); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener)); }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    event.currentTarget = this;
    for (const listener of this.listeners.get(event.type) || []) listener.call(this, event);
    return !event.defaultPrevented;
  }
  async emit(type, extra = {}) {
    const event = { type, target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}, ...extra };
    for (const listener of this.listeners.get(type) || []) await listener.call(this, event);
  }
  click() { return this.emit("click"); }
  focus() { this.ownerDocument.activeElement = this; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  closest(selector) { let node = this; while (node) { if (matches(node, selector)) return node; node = node.parentElement; } return null; }
  querySelectorAll(selector) { return descendants(this).filter((node) => matches(node, selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function descendants(element) { return element.children.flatMap((child) => [child, ...descendants(child)]); }
function matchesSimple(element, selector) {
  const attributes = [...selector.matchAll(/\[([^\]=]+)(?:=["']?([^\]"']*)["']?)?\]/g)];
  const plain = selector.replace(/\[[^\]]*\]/g, "");
  const tag = plain.match(/^[a-z][a-z\d-]*/i);
  if (tag && element.tagName !== tag[0].toUpperCase()) return false;
  for (const [, id] of plain.matchAll(/#([\w-]+)/g)) if (element.id !== id) return false;
  for (const [, name] of plain.matchAll(/\.([\w-]+)/g)) if (!element.classList.contains(name)) return false;
  for (const [, name, value] of attributes) {
    const actual = name.startsWith("data-") ? element.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] : element.getAttribute(name);
    if (actual === undefined || actual === null || (value !== undefined && actual !== value)) return false;
  }
  return true;
}
function matches(element, selector) {
  return selector.split(",").some((part) => {
    const chain = part.trim().split(/\s+(?![^\[]*\])/);
    if (!matchesSimple(element, chain.pop())) return false;
    let ancestor = element.parentElement;
    while (chain.length) {
      const next = chain.pop();
      while (ancestor && !matchesSimple(ancestor, next)) ancestor = ancestor.parentElement;
      if (!ancestor) return false;
      ancestor = ancestor.parentElement;
    }
    return true;
  });
}

class Document extends Element {
  constructor(html) {
    super("document", null);
    this.ownerDocument = this;
    const stack = [this];
    const voidTags = new Set(["AREA", "BASE", "BR", "COL", "EMBED", "HR", "IMG", "INPUT", "LINK", "META", "PARAM", "SOURCE", "TRACK", "WBR"]);
    for (const match of html.replace(/<!--[\s\S]*?-->/g, "").matchAll(/<\/?([a-z][\w-]*)\b([^>]*)>|([^<]+)/gi)) {
      const [, tag, attributes, text] = match;
      if (text) { stack.at(-1).append(this.createTextNode(text)); continue; }
      if (match[0].startsWith("</")) { while (stack.length > 1 && stack.pop().tagName !== tag.toUpperCase()) {} continue; }
      const element = this.createElement(tag);
      for (const attribute of attributes.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) element.setAttribute(attribute[1], attribute[2] ?? attribute[3] ?? attribute[4] ?? "");
      stack.at(-1).append(element);
      if (!voidTags.has(element.tagName) && !match[0].endsWith("/>")) stack.push(element);
    }
    this.body = this.querySelector("body");
    this.documentElement = this.querySelector("html");
    this.readyState = "complete";
  }
  createElement(tag) { return new Element(tag, this); }
  createTextNode(text) { const node = new Element("#text", this); node._text = String(text); return node; }
  getElementById(id) { return descendants(this).find((node) => node.id === id) || null; }
}

const fixtureDefaults = {
  objective: "flow_dpo", checkpoint_dir: "checkpoints", model_variant: "xl-sft",
  checkpoint_file: "old/custom.safetensors", model_config_dir: "old/model-config",
  pairs_manifest: "original/pairs.json", dataset_manifest: "", output_dir: "output/original",
  max_latent_length: 750, learning_rate: 0.00001, rank: 8, batch_size: 1,
  gradient_accumulation: 4, multi_gpu: false, gpu_ids: ["0", "1"], fm_regularization: 0,
  timestep_sampling: "logit_normal", timestep_mu: -0.4, timestep_sigma: 1,
  timestep_min: 0.0001, timestep_max: 0.9999, cfg_dropout: 0.1,
};
function actualSchemaFields() {
  // Reading the lightweight declaration keeps the VM fixture current without Python/ML dependencies.
  const source = fs.readFileSync(path.join(root, "jk_step/config.py"), "utf8");
  const declarations = [...source.matchAll(/_field\("([a-z][a-z\d_]*)", ([\s\S]*?), "(string|bool|int|float|list|json)", "([^"]+)", "((?:\\.|[^"\\])*)"/g)];
  assert.ok(declarations.length > 60, "The harness must load the real field metadata");
  return declarations.map((match, index) => {
    const [, name, rawDefault, type, group, help] = match;
    const defaultValue = JSON.parse(rawDefault.replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false"));
    const extras = source.slice(match.index + match[0].length, declarations[index + 1]?.index || source.indexOf("def schema_fields"));
    const choices = extras.match(/choices=\[([^\]]*)\]/);
    return {
      name, default: defaultValue, type, group, help, label: name.replaceAll("_", " "),
      ...(choices ? { choices: JSON.parse("[" + choices[1] + "]") } : {}),
    };
  });
}
const realFields = actualSchemaFields();
const defaults = { ...Object.fromEntries(realFields.map((field) => [field.name, field.default])), ...fixtureDefaults };
const fields = realFields.map((field) => ({ ...field, default: defaults[field.name] }));
const sftPreset = { objective: "sft", learning_rate: 0.0001, rank: 64, max_latent_length: 0 };
const flowPreset = { objective: "flow_dpo", learning_rate: 0.000001, rank: 32, gradient_accumulation: 8, fm_regularization: 0.1 };

function preparedJob(id = "prepared", overrides = {}) {
  return {
    id, kind: "prepare_dataset", status: "completed", started_at: 20, progress: {},
    config: { audio_dir: "E:/Music/" + (overrides.dataset_name || "demo"), dataset_name: overrides.dataset_name || "demo", checkpoint_dir: "checkpoints", model_variant: "sft" },
    result: { status: "ready", ready: true, objective: "sft", dataset_manifest: "E:/prepared/demo/dataset.json", tensor_dir: "E:/prepared/demo/tensors", dataset_name: "demo", checkpoint_dir: "checkpoints", model_variant: "sft", ...overrides },
  };
}

function preparedFlowJob(id = "prepared-flow", overrides = {}) {
  return preparedJob(id, {
    objective: "flow_dpo", dataset_manifest: "", pairs_manifest: "E:/prepared/demo/preferences/preprocessed/pairs.json",
    dataset_json: "E:/prepared/demo/dataset.json", tensor_dir: "E:/prepared/demo/preferences/tensors", ...overrides,
  });
}

async function harness(initialJobs = [], reports = {}, options = {}) {
  const document = new Document(fs.readFileSync(path.join(root, "frontend/quality.html"), "utf8"));
  const requests = [];
  const jobs = copy(initialJobs);
  const timers = [];
  const intervals = [];
  const element = (id) => { const node = document.getElementById(id); assert.ok(node, "Expected actual HTML/rendered element #" + id); return node; };
  const fetch = async (url, fetchOptions = {}) => {
    const body = fetchOptions.body === undefined ? undefined : JSON.parse(fetchOptions.body);
    const request = { url: String(url), method: fetchOptions.method || "GET", headers: fetchOptions.headers || {}, body };
    requests.push(request);
    let payload;
    if (url === "/api/mrflow/schema") payload = { defaults, fields };
    else if (url === "/api/mrflow/presets") payload = { builtin: [{ name: "sft_lora", config: sftPreset }, { name: "conservative", config: flowPreset }], saved: [] };
    else if (url === "/api/mrflow/jobs" && request.method === "GET") payload = { jobs, active: jobs.find((job) => ["running", "stopping", "orphaned"].includes(job.status)) || null };
    else if (url === "/api/mrflow/jobs" && request.method === "POST") {
      payload = { id: "started-" + requests.length, kind: body.kind, config: body.config, status: "running", started_at: 100, progress: {}, result: null };
      jobs.unshift(payload);
    } else if (String(url).startsWith("/api/mrflow/jobs/")) {
      const id = String(url).split("/")[4].split("?")[0];
      const job = jobs.find((item) => item.id === id);
      assert.ok(job, "Unexpected job fetch: " + url);
      if (String(url).endsWith("/stop") && request.method === "POST") { job.status = "stopping"; payload = job; }
      else payload = { job, events: [], cursor: 0 };
    } else if (url === "/api/mrflow/datasets/report") {
      assert.ok(reports[body.report], "Unexpected preparation report: " + body.report);
      payload = reports[body.report];
    } else if (url === "/api/browse") payload = { path: body.path || "/", entries: [{ name: "Songs", path: "E:/Music/Songs", is_dir: true }] };
    else if (url === "/api/mrflow/datasets/inspect") payload = { audio_dir: body.audio_dir, files: 2, count: 2, audio_count: 2, total: 2, duration: 60, total_duration: 60, total_duration_seconds: 60, estimated_clips: 2, samples: [], warnings: [] };
    else if (url === "/api/mrflow/datasets/preview") payload = { path: body.manifest, samples: options.previewSamples || [] };
    else throw new Error("Unexpected fetch: " + request.method + " " + url);
    return { ok: true, status: 200, json: async () => copy(payload) };
  };
  const storage = new Map();
  const storedLanguage = Object.hasOwn(options, "storedLanguage") ? options.storedLanguage : "pt";
  if (storedLanguage !== undefined && storedLanguage !== null) storage.set("jk_step_language", String(storedLanguage));
  const localStorage = { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, String(value)), removeItem: (key) => storage.delete(key) };
  const navigator = { language: options.navigatorLanguage || "en-US", languages: options.navigatorLanguages || [options.navigatorLanguage || "en-US"] };
  const windowEvents = new Map();
  const window = {
    __SIDESTEP_TOKEN__: token, prompt: () => null, confirm: () => true, localStorage, navigator,
    addEventListener(type, listener) { const listeners = windowEvents.get(type) || []; listeners.push(listener); windowEvents.set(type, listeners); },
    removeEventListener(type, listener) { windowEvents.set(type, (windowEvents.get(type) || []).filter((item) => item !== listener)); },
    dispatchEvent(event) { for (const listener of windowEvents.get(event.type) || []) listener.call(window, event); return !event.defaultPrevented; },
  };
  class CustomEvent {
    constructor(type, init = {}) { this.type = type; this.detail = init.detail; this.defaultPrevented = false; }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() {}
  }
  window.CustomEvent = CustomEvent;
  function Option(text, value) { const option = document.createElement("option"); option.textContent = text; option.value = value; return option; }
  const context = vm.createContext({
    window, document, location: { search: "" }, fetch, Option, console, URL, URLSearchParams,
    AbortController, Blob, navigator, localStorage, CustomEvent,
    setTimeout: (callback) => { timers.push(callback); return timers.length; }, clearTimeout() {},
    setInterval: (callback) => { intervals.push(callback); return intervals.length; }, clearInterval() {},
  });
  for (const filename of ["quality-schema-locales.js", "quality-i18n.js", "quality.js"]) {
    vm.runInContext(fs.readFileSync(path.join(root, "frontend/js", filename), "utf8"), context, { filename });
  }
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(element("error-banner").hidden, true, "Boot failed: " + element("error-banner").textContent);
  assert.equal(element("connection-status").textContent, window.QualityI18n.t("connected"));
  return {
    document, element, requests, jobs, intervals, window, storage,
    language: () => window.QualityI18n.language,
    async changeLanguage(locale) { element("quality-language").value = locale; await element("quality-language").emit("change"); await new Promise((resolve) => setImmediate(resolve)); },
    config: () => JSON.parse(element("config-json").value),
    set(id, value) { const node = element(id); if (typeof value === "boolean") node.checked = value; else node.value = value; return node; },
    async select(job) {
      if (!jobs.some((item) => item.id === job.id)) jobs.unshift(copy(job));
      await element("refresh-jobs").emit("click");
      const date = new Date(job.started_at * 1000);
      const button = element("job-history").children.find((item) => item.tagName === "BUTTON" && ["pt-BR", "en-US", "es-ES", "en", "pt", "es"].some((locale) => item.textContent.includes(date.toLocaleString(locale))));
      assert.ok(button, "Expected actual history button for " + job.id);
      await button.emit("click");
    },
  };
}

test("folder preparation sends explicit settings without requiring an output path", async () => {
  const ui = await harness();
  ui.set("dataset-mode", "folder");
  await ui.element("dataset-mode").emit("change");
  ui.set("prepare-objective", "sft");
  await ui.element("prepare-objective").emit("change");
  const settings = {
    "folder-input": "E:/Music", "prepare-output": "", "prepare-name": "", "prepare-checkpoint": "E:/Models",
    "prepare-variant": "sft", "prepare-device": "cuda:1", "prepare-precision": "bf16",
    "prepare-caption-model": "test/caption-model", "prepare-caption-tier": "8-10gb",
    "prepare-lyrics-model": "test/lyrics-model", "prepare-language": "pt", "prepare-content-mode": "vocal",
    "prepare-clip-seconds": "45", "prepare-min-seconds": "12", "prepare-custom-tag": "demo",
    "prepare-validation": "0.2", "prepare-seed": "23", "prepare-download": false, "prepare-reuse": true,
  };
  Object.entries(settings).forEach(([id, value]) => ui.set(id, value));
  assert.equal(ui.element("prepare-output").required, false);
  await ui.element("prepare-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.url === "/api/mrflow/jobs");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.headers.Authorization, "Bearer " + token);
  assert.equal(request.body.kind, "prepare_dataset");
  assert.deepEqual(request.body.config, {
    audio_dir: "E:/Music", checkpoint_dir: "E:/Models", model_variant: "sft", device: "cuda:1", precision: "bf16",
    caption_model: "test/caption-model", caption_tier: "8-10gb", lyrics_model: "test/lyrics-model", language: "pt", content_mode: "vocal",
    clip_seconds: 45, min_clip_seconds: 12, custom_tag: "demo", validation_fraction: 0.2, seed: 23,
    normalize: "none", allow_download: false, reuse: true, preprocess: true,
    objective: "sft",
  });
});

test("preparation defaults use Omni-7B and automatic content detection", async () => {
  const ui = await harness();
  assert.equal(ui.element("prepare-objective").value, "flow_dpo");
  assert.equal(ui.element("prepare-caption-model").value, "Qwen/Qwen2.5-Omni-7B");
  assert.equal(ui.element("prepare-content-mode").value, "auto");
  ui.set("folder-input", "E:/Music");
  await ui.element("prepare-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.body?.kind === "prepare_dataset");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.body.config.caption_model, "Qwen/Qwen2.5-Omni-7B");
  assert.equal(request.body.config.content_mode, "auto");
  assert.equal(request.body.config.objective, "flow_dpo");
  assert.equal(request.body.config.pair_options.degradation, "mixed");
  assert.equal(request.body.config.pair_options.variations_per_audio, 3);
});

test("automatic preference preparation submits controlled degradation parameters", async () => {
  const ui = await harness();
  ui.set("folder-input", "E:/Music");
  ui.set("prepare-degradation", "noise");
  await ui.element("prepare-degradation").emit("change");
  assert.equal(ui.element("prepare-variations").value, "1");
  for (const [id, value] of Object.entries({
    "prepare-variations": "4", "prepare-cutoff": "7500", "prepare-noise-snr": "32", "prepare-clipping": "0.25",
  })) ui.set(id, value);
  await ui.element("prepare-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.body?.kind === "prepare_dataset");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.body.config.objective, "flow_dpo");
  assert.deepEqual(request.body.config.pair_options, {
    degradation: "noise", variations_per_audio: 4, cutoff_hz: 7500, noise_snr_db: 32, clip_threshold: 0.25,
  });
});

test("SFT preparation hides preference settings and never submits pair options", async () => {
  const ui = await harness();
  ui.set("folder-input", "E:/Music");
  ui.set("prepare-objective", "sft");
  await ui.element("prepare-objective").emit("change");
  assert.equal(ui.element("prepare-pair-options").hidden, true);
  ui.set("prepare-cutoff", "invalid-unused-pair-setting");
  await ui.element("prepare-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.body?.kind === "prepare_dataset");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.body.config.objective, "sft");
  assert.equal(Object.hasOwn(request.body.config, "pair_options"), false);
});

test("invalid automatic preference parameters never start a job", async (t) => {
  for (const [name, id, value] of [
    ["invalid cutoff", "prepare-cutoff", "Infinity"],
    ["fractional variations", "prepare-variations", "1.5"],
    ["out-of-range noise", "prepare-noise-snr", "81"],
    ["out-of-range clipping", "prepare-clipping", "0"],
  ]) await t.test(name, async () => {
    const ui = await harness();
    ui.set("folder-input", "E:/Music");
    ui.set(id, value);
    await ui.element("prepare-form").emit("submit");
    assert.equal(ui.requests.some((item) => item.method === "POST" && item.body?.kind === "prepare_dataset"), false);
    assert.equal(ui.element("error-banner").hidden, false);
  });
});

test("download guidance distinguishes ACE text encoding from annotation models", async () => {
  const ui = await harness();
  const setup = ui.element("model-setup-title").closest("section").textContent;
  assert.match(setup, /VAE/i);
  assert.match(setup, /encoder de texto/i);
  assert.match(setup, /1[,.]6\s*GB/i);
  const hint = ui.element("prepare-download-info");
  assert.equal(hint.hidden, false);
  assert.equal(hint.closest("details"), null);
  assert.match(hint.textContent, /Omni[\s-]?7B|Qwen2\.5-Omni-7B/i);
  assert.match(hint.textContent, /22[,.]4\s*GB/i);
  assert.match(hint.textContent, /Whisper/i);
  assert.match(hint.textContent, /3[,.]1\s*GB/i);
  assert.match(hint.textContent, /primeir[ao]/i);
});

test("invalid preparation numbers show an error without starting a job", async (t) => {
  const cases = [
    ["malformed duration", { "prepare-clip-seconds": "thirty" }],
    ["infinite duration", { "prepare-clip-seconds": "Infinity" }],
    ["minimum exceeds maximum", { "prepare-clip-seconds": "20", "prepare-min-seconds": "21" }],
  ];
  for (const [name, settings] of cases) await t.test(name, async () => {
    const ui = await harness();
    ui.set("folder-input", "E:/Music");
    Object.entries(settings).forEach(([id, value]) => ui.set(id, value));
    await ui.element("prepare-form").emit("submit");
    assert.equal(ui.requests.some((item) => item.method === "POST" && item.url === "/api/mrflow/jobs"), false);
    assert.equal(ui.element("error-banner").hidden, false);
    assert.ok(ui.element("error-banner").textContent.length > 0);
  });
});

test("incomplete or unsuccessful preparation never fills training inputs", async (t) => {
  const cases = [
    ["running", "running", {}], ["stopping", "stopping", {}], ["orphaned", "orphaned", {}],
    ["failed", "failed", {}], ["cancelled", "cancelled", {}],
    ["not ready", "completed", { ready: false }], ["cancelled result", "completed", { status: "cancelled" }],
    ["missing manifest", "completed", { dataset_manifest: "" }], ["missing tensors", "completed", { tensor_dir: "" }],
  ];
  for (const [name, status, result] of cases) await t.test(name, async () => {
    const ui = await harness();
    const before = ui.config();
    const job = { ...preparedJob(name.replaceAll(" ", "-"), result), status };
    await ui.select(job);
    assert.deepEqual(ui.config(), before);
    assert.equal(ui.element("field-dataset_manifest").value, before.dataset_manifest);
  });
});

test("safe stop keeps folder preparation from filling or starting training", async (t) => {
  for (const job of [preparedJob("cancel-sft"), preparedFlowJob("cancel-flow")]) await t.test(job.result.objective, async () => {
    job.status = "running";
    const ui = await harness([job]);
    const before = ui.config();
    assert.equal(ui.element("start-training").disabled, true);
    assert.equal(ui.element("cancel-job").disabled, false);
    await ui.element("cancel-job").emit("click");
    const stop = ui.requests.find((request) => request.method === "POST" && request.url.endsWith("/" + job.id + "/stop"));
    assert.ok(stop, ui.element("error-banner").textContent);
    assert.equal(stop.headers.Authorization, "Bearer " + token);
    assert.equal(ui.element("cancel-job").disabled, true);
    assert.equal(ui.element("prepared-result").hidden, true);
    assert.deepEqual(ui.config(), before);
    const persisted = ui.jobs.find((item) => item.id === job.id);
    persisted.status = "cancelled";
    await ui.select(persisted);
    assert.equal(ui.element("prepared-result").hidden, true);
    assert.deepEqual(ui.config(), before);
    assert.equal(ui.requests.some((request) => request.method === "POST" && request.body?.kind === "train"), false);
  });
});

test("automatic preference pair creation appears as its own preparation stage", async () => {
  const job = { ...preparedFlowJob("building-pairs"), status: "running", progress: { stage: "pairs", current: 4, total: 12 } };
  const ui = await harness([job]);
  const stage = ui.document.querySelector('[data-stage="pairs"]');
  assert.ok(stage, "The actual preparation steps must include preference pair creation");
  assert.equal(stage.classList.contains("active"), true);
  assert.equal(stage.classList.contains("done"), false);
  assert.equal(ui.document.querySelector('[data-stage="scan"]').classList.contains("done"), true);
  assert.equal(ui.document.querySelector('[data-stage="preprocess"]').classList.contains("active"), false);
  for (const locale of ["en", "pt", "es"]) {
    await ui.changeLanguage(locale);
    assert.equal(stage.classList.contains("active"), true);
    const stageLabel = ui.window.QualityI18n.t("stage.pairs");
    assert.notEqual(stageLabel, "stage.pairs");
    assert.equal(stage.textContent, stageLabel);
    assert.ok(ui.element("preparation-summary").textContent.includes(ui.window.QualityI18n.t("stage_running.pairs")));
    assert.ok(ui.element("preparation-summary").textContent.includes("4 / 12"));
  }
  assert.equal(ui.element("prepared-result").hidden, true);
  assert.equal(ui.requests.some((request) => request.method === "POST" && request.body?.kind === "train"), false);
});

test("ready preparation applies SFT preset once and retains subsequent edits", async () => {
  const ui = await harness();
  const job = preparedJob();
  await ui.select(job);
  const config = ui.config();
  assert.equal(config.objective, "sft");
  assert.equal(config.dataset_manifest, job.result.dataset_manifest);
  assert.equal(config.pairs_manifest, "");
  assert.equal(config.max_latent_length, 0);
  assert.equal(config.output_dir, "output/demo");
  assert.equal(config.checkpoint_file, "");
  assert.equal(config.model_config_dir, "");
  assert.equal(config.rank, sftPreset.rank);
  assert.equal(config.learning_rate, sftPreset.learning_rate);
  ui.set("field-learning_rate", "0.0000123");
  await ui.element("field-learning_rate").emit("change");
  await ui.element("refresh-jobs").emit("click");
  await ui.select(job);
  assert.equal(ui.config().learning_rate, 0.0000123);
  assert.equal(ui.element("field-learning_rate").value, "0.0000123");
});

test("failed and partial preparation show counts and report paths without applying SFT", async (t) => {
  for (const status of ["failed", "partial"]) await t.test(status, async () => {
    const ui = await harness();
    const before = ui.config();
    const report = "E:/prepared/" + status + "/preparation_report.json";
    const job = {
      ...preparedJob(status, { status, ready: false, samples: 17, preprocessed: 6, quarantined: 11, report, dataset_manifest: "", tensor_dir: "" }),
      status: status === "failed" ? "failed" : "completed",
    };
    await ui.select(job);
    assert.equal(ui.element("preparation-outcome").hidden, false);
    const counts = ui.element("preparation-counts").textContent;
    assert.match(counts, /17 amostras anotadas/);
    assert.match(counts, /6 prontas para treino/);
    assert.match(counts, /11 exclusões/);
    assert.equal(ui.element("preparation-report-path").value, report);
    assert.notEqual(ui.element("preparation-report-path").getAttribute("readonly"), null);
    assert.equal(ui.element("show-preparation-report").disabled, false);
    assert.equal(ui.element("prepared-result").hidden, true);
    assert.deepEqual(ui.config(), before);
    assert.equal(ui.requests.some((request) => request.method === "POST" && request.body?.kind === "train"), false);
  });
});

test("running preparation does not offer a terminal report", async () => {
  const ui = await harness();
  const job = {
    ...preparedJob("running-report", { ready: false, status: "partial", report: "E:/prepared/in-progress/preparation_report.json" }),
    status: "running",
  };
  await ui.select(job);
  assert.equal(ui.element("show-preparation-report").disabled, true);
  assert.equal(ui.requests.some((request) => request.url === "/api/mrflow/datasets/report"), false);
});

test("selecting failed preparation clears the previous ready card and its action", async () => {
  const ready = preparedJob();
  const ui = await harness([ready]);
  assert.equal(ui.element("prepared-result").hidden, false);
  const before = ui.config();
  const failed = {
    ...preparedJob("failed-after-ready", { status: "failed", ready: false, samples: 4, preprocessed: 0, quarantined: 4, dataset_manifest: "", tensor_dir: "" }),
    status: "failed", started_at: 30,
  };
  await ui.select(failed);
  assert.equal(ui.element("prepared-result").hidden, true);
  assert.equal(ui.element("use-prepared-dataset").disabled, true);
  for (const id of ["prepared-dataset-json", "prepared-manifest", "prepared-tensors"]) assert.equal(ui.element(id).value, "");
  assert.deepEqual(ui.config(), before);
  await ui.element("use-prepared-dataset").emit("click");
  assert.equal(ui.element("error-banner").hidden, false);
  assert.match(ui.element("error-banner").textContent, /ainda não está pronto/i);
  assert.equal(ui.requests.some((request) => request.method === "POST" && request.body?.kind === "train"), false);
});

test("report action requests the selected report and renders exclusion reasons as text", async () => {
  const reportPath = "E:/prepared/review/preparation_report.json";
  const reason = '<img src=x onerror="alert(1)"> no aligned lyric segment';
  const job = {
    ...preparedJob("review-report", { status: "partial", ready: false, samples: 17, preprocessed: 6, quarantined: 11, report: reportPath, dataset_manifest: "", tensor_dir: "" }),
    status: "completed",
  };
  const ui = await harness([], {
    [reportPath]: {
      path: reportPath, status: "partial", ready: false, samples: 17, preprocessed: 6, quarantined: 11,
      quarantine: [{ id: "song-1", source_audio_path: "E:/Music/song-1.wav", reason }, { id: "song-2", source_audio_path: "E:/Music/song-2.wav", reason: "caption returned empty" }],
      truncated: true, warnings: ["Showing the first exclusions only"],
    },
  });
  await ui.select(job);
  assert.equal(ui.requests.some((request) => request.url === "/api/mrflow/datasets/report"), false);
  await ui.element("show-preparation-report").emit("click");
  const request = ui.requests.find((item) => item.url === "/api/mrflow/datasets/report");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.method, "POST");
  assert.equal(request.headers.Authorization, "Bearer " + token);
  assert.deepEqual(request.body, { report: reportPath });
  const exclusions = ui.element("preparation-exclusions");
  assert.ok(exclusions.textContent.includes(reason));
  assert.ok(exclusions.textContent.includes("caption returned empty"));
  assert.equal(exclusions.querySelector("img"), null);
  assert.ok(ui.element("preparation-report-status").textContent.length > 0);
  assert.equal(ui.requests.some((item) => item.method === "POST" && item.body?.kind === "train"), false);
});

test("reload restores the latest completed ready or partial preparation", async () => {
  const latest = { ...preparedJob("latest", { status: "partial", dataset_name: "latest", dataset_manifest: "E:/prepared/latest/dataset.json" }), started_at: 30 };
  const failed = { ...preparedJob("failed"), status: "failed", started_at: 50 };
  const cancelled = { ...preparedJob("cancelled"), status: "cancelled", started_at: 40 };
  const ui = await harness([failed, cancelled, latest, { ...preparedJob("older"), started_at: 10 }]);
  assert.equal(ui.config().objective, "sft");
  assert.equal(ui.config().dataset_manifest, latest.result.dataset_manifest);
  assert.equal(ui.config().output_dir, "output/latest");
  assert.ok(ui.element("job-description").textContent.includes(latest.id));
});

test("reload keeps an active job selected instead of restoring an older dataset", async () => {
  const active = { id: "active-training", kind: "train", status: "running", started_at: 60, progress: {}, result: null };
  const ui = await harness([active, preparedJob()]);
  assert.equal(ui.config().dataset_manifest, "");
  assert.ok(ui.element("job-description").textContent.includes(active.id));
});

test("training submit uses supervised objective and prepared manifest", async () => {
  const job = preparedJob();
  const ui = await harness([job]);
  await ui.element("train-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.body?.kind === "train");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.url, "/api/mrflow/jobs");
  assert.equal(request.body.config.objective, "sft");
  assert.equal(request.body.config.dataset_manifest, job.result.dataset_manifest);
  assert.equal(request.body.config.pairs_manifest, "");
  assert.equal(request.body.config.max_latent_length, 0);
});

test("folder workflow defaults to automatic MR-FlowDPO independently of the data source", async () => {
  const ui = await harness();
  assert.equal(ui.element("dataset-mode").value, "folder");
  assert.equal(ui.element("prepare-objective").value, "flow_dpo");
  assert.equal(ui.config().objective, "flow_dpo");
  for (const mode of ["pairs", "folder"]) {
    ui.set("dataset-mode", mode);
    await ui.element("dataset-mode").emit("change");
    assert.equal(ui.config().objective, "flow_dpo");
    assert.equal(ui.element("prepare-objective").value, "flow_dpo");
  }
  ui.set("prepare-objective", "sft");
  await ui.element("prepare-objective").emit("change");
  for (const mode of ["pairs", "folder"]) {
    ui.set("dataset-mode", mode);
    await ui.element("dataset-mode").emit("change");
    assert.equal(ui.element("prepare-objective").value, "sft", "Manual pair mode must preserve the selected folder objective");
    assert.equal(ui.config().objective, mode === "pairs" ? "flow_dpo" : "sft", "Returning to folder mode restores its selected objective");
  }
});

test("ready folder preference preparation fills MR-FlowDPO and clears supervised data", async () => {
  const ui = await harness([preparedJob("previous-sft")]);
  const job = { ...preparedFlowJob(), started_at: 40 };
  await ui.select(job);
  const config = ui.config();
  assert.equal(config.objective, "flow_dpo");
  assert.equal(config.pairs_manifest, job.result.pairs_manifest);
  assert.equal(config.dataset_manifest, "");
  assert.equal(config.checkpoint_dir, job.result.checkpoint_dir);
  assert.equal(config.model_variant, job.result.model_variant);
  assert.equal(config.checkpoint_file, "");
  assert.equal(config.model_config_dir, "");
  assert.equal(config.rank, flowPreset.rank);
  assert.equal(config.learning_rate, flowPreset.learning_rate);
  assert.equal(ui.element("prepared-manifest").value, job.result.pairs_manifest);
  assert.equal(ui.element("prepared-tensors").value, job.result.tensor_dir);
  await ui.element("use-prepared-dataset").emit("click");
  assert.equal(ui.config().objective, "flow_dpo", "Using a ready preference dataset must preserve its objective");
  assert.equal(ui.config().pairs_manifest, job.result.pairs_manifest);
  assert.equal(ui.config().dataset_manifest, "");
  assert.equal(ui.element("panel-train").hidden, false);
  await ui.element("train-form").emit("submit");
  const request = ui.requests.find((item) => item.method === "POST" && item.body?.kind === "train");
  assert.ok(request, ui.element("error-banner").textContent);
  assert.equal(request.body.config.objective, "flow_dpo");
  assert.equal(request.body.config.pairs_manifest, job.result.pairs_manifest);
  assert.equal(request.body.config.dataset_manifest, "");
});

test("a preference dataset needs both its pair manifest and tensors before autofill", async (t) => {
  for (const [name, result] of [
    ["missing pair manifest", { pairs_manifest: "" }],
    ["missing pair tensors", { tensor_dir: "" }],
    ["supervised manifest cannot replace pairs", { pairs_manifest: "", dataset_manifest: "E:/prepared/demo/supervised.json" }],
  ]) await t.test(name, async () => {
    const ui = await harness();
    const before = ui.config();
    await ui.select(preparedFlowJob(name.replaceAll(" ", "-"), result));
    assert.deepEqual(ui.config(), before);
    assert.equal(ui.element("prepared-result").hidden, true);
    assert.equal(ui.element("use-prepared-dataset").disabled, true);
  });
});

test("reload restores a ready MR-FlowDPO folder preparation and retains edits", async () => {
  const job = preparedFlowJob();
  const ui = await harness([job]);
  assert.equal(ui.config().objective, "flow_dpo");
  assert.equal(ui.config().pairs_manifest, job.result.pairs_manifest);
  ui.set("field-learning_rate", "0.0000027");
  await ui.element("field-learning_rate").emit("change");
  await ui.select(job);
  assert.equal(ui.config().learning_rate, 0.0000027);
  assert.equal(ui.element("field-learning_rate").value, "0.0000027");
});

test("older completed preparations without an objective still restore supervised training", async () => {
  const job = preparedJob("legacy-supervised");
  delete job.result.objective;
  const ui = await harness([job]);
  assert.equal(ui.config().objective, "sft");
  assert.equal(ui.config().dataset_manifest, job.result.dataset_manifest);
  assert.equal(ui.config().pairs_manifest, "");
  assert.equal(ui.element("prepared-result").hidden, false);
});

test("workflow and objective switches reveal only relevant controls", async () => {
  const ui = await harness();
  for (const mode of ["pairs", "folder"]) {
    ui.set("dataset-mode", mode);
    await ui.element("dataset-mode").emit("change");
    const containers = ui.document.querySelectorAll("[data-workflow]");
    assert.ok(containers.some((node) => node.dataset.workflow === "folder"));
    assert.ok(containers.some((node) => node.dataset.workflow === "pairs"));
    containers.forEach((node) => assert.equal(node.hidden, node.dataset.workflow !== mode));
  }
  ui.set("field-cfg_dropout", "0.23");
  await ui.element("field-cfg_dropout").emit("change");
  ui.set("field-timestep_sampling", "uniform");
  await ui.element("field-timestep_sampling").emit("change");
  const beforeObjectiveSwitch = ui.config();
  for (const objective of ["sft", "flow_dpo"]) {
    ui.set("field-objective", objective);
    await ui.element("field-objective").emit("change");
    assert.equal(ui.element("field-fm_regularization").closest("label").hidden, objective === "sft");
    assert.equal(ui.element("field-fm_regularization").closest("section").hidden, false);
    assert.equal(ui.element("field-pairs_manifest").closest("label").hidden, objective === "sft");
    assert.equal(ui.element("field-dataset_manifest").closest("label").hidden, objective !== "sft");
    for (const name of ["timestep_sampling", "timestep_mu", "timestep_sigma", "timestep_min", "timestep_max", "cfg_dropout"]) {
      assert.equal(ui.element("field-" + name).closest("label").hidden, false);
      assert.equal(ui.config()[name], beforeObjectiveSwitch[name]);
    }
  }
});

test("folder picker reuses authenticated directories-only browsing and inspection", async () => {
  const ui = await harness();
  ui.set("folder-input", "E:/Music");
  await ui.element("choose-folder").emit("click");
  const browse = ui.requests.find((item) => item.url === "/api/browse");
  assert.ok(browse, ui.element("error-banner").textContent);
  assert.equal(browse.method, "POST");
  assert.equal(browse.headers.Authorization, "Bearer " + token);
  assert.equal(browse.body.dirs_only, true);
  assert.equal(ui.element("folder-browser").open, true);
  await ui.element("folder-browser-select").emit("click");
  assert.equal(ui.element("folder-browser").open, false);
  assert.equal(ui.element("folder-input").value, "E:/Music");
  const inspect = ui.requests.find((item) => item.url === "/api/mrflow/datasets/inspect");
  assert.ok(inspect, ui.element("error-banner").textContent);
  assert.equal(inspect.headers.Authorization, "Bearer " + token);
  assert.equal(inspect.body.audio_dir, "E:/Music");
});

test("quality scripts load schema translations before interface and runtime", () => {
  const html = fs.readFileSync(path.join(root, "frontend/quality.html"), "utf8");
  const scripts = ["/js/quality-schema-locales.js", "/js/quality-i18n.js", "/js/quality.js"].map((filename) => html.indexOf(filename));
  assert.ok(scripts.every((index) => index >= 0), "The actual HTML must include all three quality scripts");
  assert.ok(scripts[0] < scripts[1] && scripts[1] < scripts[2], "Schema catalog must load before i18n and runtime");
});

test("every marked interface string has an explicit EN/PT/ES translation", async () => {
  const ui = await harness();
  const markup = new Document(fs.readFileSync(path.join(root, "frontend/quality.html"), "utf8"));
  const keys = new Set();
  markup.querySelectorAll("[data-i18n],[data-i18n-placeholder],[data-i18n-label]").forEach((node) => {
    for (const name of ["i18n", "i18nPlaceholder", "i18nLabel"]) if (node.dataset[name]) keys.add(node.dataset[name]);
  });
  assert.ok(keys.size > 100, "Inspect the actual HTML translation markers");
  for (const locale of ["en", "pt", "es"]) {
    for (const key of keys) {
      const translated = ui.window.QualityI18n.catalogs[locale][key];
      assert.equal(typeof translated, "string", "Missing interface translation " + locale + ":" + key);
      assert.ok(translated.trim().length > 0, "Empty interface translation " + locale + ":" + key);
    }
  }
});

test("static prose is marked for translation except explicit technical labels", () => {
  const markup = new Document(fs.readFileSync(path.join(root, "frontend/quality.html"), "utf8"));
  const technicalLabels = new Set([
    "JK", "JK-Step", "Dataset · LoRA SFT · MR-FlowDPO", "JK-Step · LoRA SFT · MR-FlowDPO",
    "English", "Português", "Español", "01", "02", "03", "JSON", "Log",
    "XL SFT", "SFT XL", "SFT", "XL Base", "Base XL", "Base", "CUDA 0", "CUDA 1", "GPU 0", "GPU 1", "CPU", "BF16", "FP16", "FP32",
    "8–10 GB", "16 GB",
  ]);
  const unmarked = descendants(markup).filter((node) => node.nodeType === 3 && node.closest("html") && !node.closest("[data-i18n]") && node.textContent.trim());
  const missed = unmarked.filter((node) => {
    const text = node.textContent.trim();
    return !technicalLabels.has(text) && !(text === "{}" && node.closest("textarea"));
  }).map((node) => node.textContent.trim());
  assert.deepEqual(missed, [], "User-facing static prose needs a translation marker; only explicit technical labels may stay unmarked");
});

test("numeric defaults satisfy native HTML ranges and step constraints", () => {
  const markup = new Document(fs.readFileSync(path.join(root, "frontend/quality.html"), "utf8"));
  const inputs = markup.querySelectorAll('input[type="number"]').filter((input) => input.getAttribute("value") !== null);
  assert.ok(inputs.length >= 10, "Validate numeric defaults in the actual HTML");
  for (const input of inputs) {
    const value = Number(input.getAttribute("value"));
    const minimum = input.getAttribute("min") === null ? null : Number(input.getAttribute("min"));
    const maximum = input.getAttribute("max") === null ? null : Number(input.getAttribute("max"));
    const step = input.getAttribute("step");
    assert.ok(Number.isFinite(value), "Finite default: " + input.id);
    if (minimum !== null) assert.ok(value >= minimum, "Default below native minimum: " + input.id);
    if (maximum !== null) assert.ok(value <= maximum, "Default above native maximum: " + input.id);
    if (step !== "any") {
      // HTML uses min, then the value attribute, then zero as the step base.
      const increment = step === null ? 1 : Number(step);
      const steps = (value - (minimum ?? value)) / increment;
      assert.ok(increment > 0 && Number.isFinite(increment), "Valid native step: " + input.id);
      assert.ok(Math.abs(steps - Math.round(steps)) < 1e-7, "Default has native stepMismatch: " + input.id);
    }
  }
});

test("language selector persists the same EN/PT/ES preference used by the workspace", async () => {
  const ui = await harness([], {}, { storedLanguage: "en", navigatorLanguage: "pt-BR" });
  assert.equal(ui.language(), "en", "Stored preference takes priority over the navigator");
  const selector = ui.element("quality-language");
  assert.deepEqual(selector.children.map((option) => option.value), ["en", "pt", "es"]);
  const headlines = new Set();
  for (const locale of ["pt", "es", "en"]) {
    await ui.changeLanguage(locale);
    assert.equal(ui.language(), locale);
    assert.equal(ui.storage.get("jk_step_language"), locale);
    assert.equal(selector.value, locale);
    assert.equal(ui.document.documentElement.lang.split("-")[0], locale);
    headlines.add(ui.document.querySelector("[data-i18n=headline]").textContent);
  }
  assert.equal(headlines.size, 3, "The headline must translate visibly in all three languages");
});

test("language defaults map navigator locales and fall back to English", async (t) => {
  for (const [navigatorLanguage, expected] of [["pt-BR", "pt"], ["es-MX", "es"], ["en-GB", "en"], ["de-DE", "en"]]) {
    await t.test(navigatorLanguage, async () => {
      const ui = await harness([], {}, { storedLanguage: null, navigatorLanguage });
      assert.equal(ui.language(), expected);
      assert.equal(ui.element("quality-language").value, expected);
    });
  }
  await t.test("unknown saved preference falls back to navigator", async () => {
    const ui = await harness([], {}, { storedLanguage: "fr", navigatorLanguage: "es-AR" });
    assert.equal(ui.language(), "es");
    assert.equal(ui.element("quality-language").value, "es");
  });
});

test("workspace storage events update the quality language without changing inputs", async () => {
  const ui = await harness([], {}, { storedLanguage: "en" });
  const input = ui.set("folder-input", "E:/Jean/Songs");
  ui.storage.set("jk_step_language", "es");
  ui.window.dispatchEvent({ type: "storage", key: "jk_step_language", newValue: "es" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(ui.language(), "es");
  assert.equal(ui.element("quality-language").value, "es");
  assert.equal(ui.element("folder-input"), input);
  assert.equal(input.value, "E:/Jean/Songs");
  ui.window.dispatchEvent({ type: "storage", key: "jk_step_language", newValue: "fr" });
  assert.equal(ui.language(), "es");
  ui.window.QualityI18n.setLanguage("unknown");
  assert.equal(ui.language(), "es");
  assert.equal(ui.storage.get("jk_step_language"), "es");
});

test("switching language preserves field instances, values, focus and configuration", async () => {
  const ui = await harness();
  ui.set("folder-input", "E:/Música de Jean/Canções");
  ui.set("prepare-custom-tag", "voz Jean; don't translate");
  ui.set("prepare-caption-model", "Jean/Caption-model");
  ui.set("field-checkpoint_dir", "E:/Models/Jean");
  ui.set("field-learning_rate", "0.000017");
  await ui.element("field-learning_rate").emit("change");
  const inputs = ui.document.querySelectorAll("input,textarea,select").filter((node) => node.id !== "quality-language");
  const values = inputs.map((node) => ({ node, id: node.id, value: node.value, checked: node.checked }));
  const config = ui.config();
  const focus = ui.element("field-learning_rate");
  focus.focus();
  const labels = new Set();
  const helps = new Set();
  for (const locale of ["en", "es", "pt"]) {
    await ui.changeLanguage(locale);
    for (const saved of values) {
      if (saved.id) assert.equal(ui.element(saved.id), saved.node, "Language switch recreated #" + saved.id);
      assert.equal(saved.node.value, saved.value, "Language switch changed #" + saved.id);
      assert.equal(saved.node.checked, saved.checked, "Language switch changed checkbox #" + saved.id);
    }
    assert.equal(ui.document.activeElement, focus);
    assert.deepEqual(ui.config(), config);
    labels.add(ui.element("field-learning_rate").closest("label").querySelector("span").textContent);
    helps.add(ui.element("help-learning_rate").textContent);
  }
  assert.equal(labels.size, 3, "Rendered schema labels must be translated");
  assert.equal(helps.size, 3, "Rendered schema explanations must be translated");
});

test("schema translation catalog covers every real training field in EN/PT/ES", async () => {
  const source = fs.readFileSync(path.join(root, "jk_step/config.py"), "utf8");
  const names = [...source.matchAll(/_field\("([a-z][a-z_\d]*)"/g)].map((match) => match[1]);
  assert.ok(names.length > 60, "Coverage must inspect the real schema, not the reduced harness fixture");
  const ui = await harness();
  for (const locale of ["en", "pt", "es"]) {
    await ui.changeLanguage(locale);
    for (const name of names) {
      for (const part of ["label", "help"]) {
        const fallback = "MISSING_TRANSLATION:" + locale + ":" + name + ":" + part;
        const catalogValue = ui.window.JKQualitySchemaLocales[locale]?.[name]?.[part];
        assert.equal(typeof catalogValue, "string", "Missing explicit schema catalog entry " + locale + ":" + name + ":" + part);
        const translated = ui.window.QualityI18n.field(name, part, fallback);
        assert.equal(typeof translated, "string", locale + ":" + name + ":" + part);
        assert.ok(translated.trim().length > 0, locale + ":" + name + ":" + part);
        assert.notEqual(translated, fallback, "Missing schema translation " + locale + ":" + name + ":" + part);
        const rendered = part === "label" ? ui.document.querySelector('[data-schema-title="' + name + '"]') : ui.element("help-" + name);
        assert.ok(rendered, "Expected actual rendered schema " + part + ":" + name);
        assert.equal(rendered.textContent, translated + (part === "help" ? " (" + name + ")" : ""), "Rendered schema metadata: " + locale + ":" + name + ":" + part);
      }
    }
    ui.document.querySelectorAll("[data-config-group]").forEach((group) => {
      const key = "group." + group.dataset.configGroup;
      const translated = ui.window.QualityI18n.t(key);
      assert.notEqual(translated, key, "Group heading needs a translation: " + locale + ":" + key);
      assert.equal(group.querySelector("h3").textContent, translated);
    });
    for (const field of realFields.filter((item) => item.choices)) {
      const translated = ui.window.QualityI18n.field(field.name, "choices", null);
      assert.ok(translated && typeof translated === "object", "Missing translated choices: " + locale + ":" + field.name);
      assert.deepEqual(Object.keys(translated).sort(), [...field.choices].sort(), "Every real choice needs presentation metadata: " + locale + ":" + field.name);
      const select = ui.element("field-" + field.name);
      assert.deepEqual(select.children.map((option) => option.value), field.choices);
      select.children.forEach((option) => assert.equal(option.textContent, translated[option.value], "Rendered choice: " + locale + ":" + field.name + ":" + option.value));
    }
  }
});

test("numeric validation errors use the current language and update an existing banner", async () => {
  const ui = await harness([], {}, { storedLanguage: "en" });
  ui.set("folder-input", "E:/Music");
  ui.set("prepare-clip-seconds", "thirty");
  await ui.element("prepare-form").emit("submit");
  assert.equal(ui.element("error-banner").hidden, false);
  const errors = [ui.element("error-banner").textContent];
  assert.match(errors[0], /number/i);
  for (const [locale, word] of [["pt", /número/i], ["es", /número/i]]) {
    await ui.changeLanguage(locale);
    assert.equal(ui.element("error-banner").hidden, false);
    assert.match(ui.element("error-banner").textContent, word);
    errors.push(ui.element("error-banner").textContent);
    await ui.element("prepare-form").emit("submit");
    assert.equal(ui.element("error-banner").textContent, errors.at(-1));
  }
  assert.equal(new Set(errors).size, 3);
  assert.equal(ui.requests.some((item) => item.method === "POST" && item.body?.kind === "prepare_dataset"), false);
});

test("language switching preserves music captions, lyrics and backend exclusion reasons", async () => {
  const caption = "Portuguese singer in a soft pop arrangement; piano, acoustic guitar.";
  const lyrics = "[Verse]\nMeu coração canta em português\n[Chorus]\nNão traduza esta canção";
  const reason = "No aligned lyric segment: 'Meu coração' at 19.3s";
  const report = "E:/prepared/review/raw-report.json";
  const ui = await harness([preparedJob()], {
    [report]: { path: report, status: "partial", ready: false, samples: 1, preprocessed: 0, quarantined: 1, quarantine: [{ id: "song-1", source_audio_path: "E:/Music/song-1.wav", reason }] },
  }, { previewSamples: [{ caption, lyrics, is_instrumental: false }] });
  await ui.element("preview-dataset").emit("click");
  const preview = ui.element("folder-preview");
  assert.ok(preview.textContent.includes(caption));
  assert.equal(preview.querySelector(".sample-lyrics").textContent, lyrics);
  for (const locale of ["en", "pt", "es"]) {
    await ui.changeLanguage(locale);
    assert.ok(preview.textContent.includes(caption));
    assert.equal(preview.querySelector(".sample-lyrics").textContent, lyrics);
  }
  await ui.select({ ...preparedJob("raw-report", { status: "partial", ready: false, report, dataset_manifest: "", tensor_dir: "" }), status: "completed", started_at: 40 });
  await ui.element("show-preparation-report").emit("click");
  for (const locale of ["en", "pt", "es"]) {
    await ui.changeLanguage(locale);
    assert.ok(ui.element("preparation-exclusions").textContent.includes(reason));
  }
});
