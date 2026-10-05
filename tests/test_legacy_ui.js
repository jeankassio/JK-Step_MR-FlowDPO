/* Node-only regressions for the real legacy localization adapter.
 * Run: node tests/test_legacy_ui.js. No browser, network or GPU required.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Small DOM fixture supplies the browser APIs used by the adapter, not its logic.
class Element {
  constructor(tag = 'div', id = '', attrs = {}) {
    this.nodeType = 1; this.tagName = tag.toUpperCase(); this.id = id;
    this.parentElement = null; this.childNodes = []; this.attributes = {...attrs};
    this.value = ''; this.listeners = {};
  }
  append(child) { child.parentElement = this; this.childNodes.push(child); return child; }
  get textContent() { return this.childNodes.map(child => child.textContent).join(''); }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  getAttribute(name) { return this.attributes[name]; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  closest(selectors) {
    for (let current = this; current; current = current.parentElement) {
      for (const selector of selectors.split(',')) {
        if (selector === '[data-no-i18n]' && current.hasAttribute('data-no-i18n')) return current;
        if (selector.startsWith('#') && current.id === selector.slice(1)) return current;
        if (selector.startsWith('.') && (current.attributes.class || '').split(' ').includes(selector.slice(1))) return current;
        if (selector.toUpperCase() === current.tagName) return current;
      }
    }
    return null;
  }
}
function text(parent, value) {
  return parent.append({nodeType: 3, nodeValue: value, parentElement: parent,
    get textContent() { return this.nodeValue; }});
}
const body = new Element('body'), ids = new Map();
function field(tag, id, value, attrs = {}) {
  const element = body.append(new Element(tag, id, attrs));
  if (id) ids.set(id, element);
  if (value !== undefined) text(element, value);
  return element;
}
const start = field('button', 'start', 'Start Training', {'aria-label': 'Start Training'});
const picker = field('select', 'legacy-language');
const status = field('span', 'ez-vram-status', '[ok] fits');
const preview = field('div', 'caption', 'Training', {'data-no-i18n': ''});
const filename = field('span', 'filename', 'Save', {'data-no-i18n': '', title: 'Training'});
const lyrics = field('textarea', 'lyrics', 'Ready\nTraining'); lyrics.value = 'Ready\nTraining';
const logs = field('div', 'monitor-log', 'Ready', {class: 'log-viewer'});
const apiOutput = field('pre', 'api-output', 'Start Training');
const datasetSelect = field('select', 'full-dataset-dir');
const dataset = datasetSelect.append(new Element('option')); dataset.value = 'datasets/Training'; text(dataset, 'Save');
const presetStatus = field('div', 'preset-status', "Preset 'Training' saved");
const runName = field('input', 'run-name'); runName.value = 'Training';
const storage = new Map([['jk_step_language', 'pt']]);
const document = {
  body, title: 'JK-Step MR-FlowDPO · SFT e Datasets', readyState: 'complete', documentElement: {},
  getElementById: id => ids.get(id), dispatchEvent() {}, addEventListener() {},
  createTreeWalker(root) {
    const nodes = [];
    function visit(element) { for (const child of element.childNodes || []) { nodes.push(child); visit(child); } }
    visit(root); let index = 0;
    return {nextNode: () => nodes[index++] || null};
  }
};
const context = vm.createContext({document, navigator: {language: 'es-ES'},
  localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
  Node: {ELEMENT_NODE: 1, TEXT_NODE: 3}, NodeFilter: {SHOW_ELEMENT: 1, SHOW_TEXT: 4},
  MutationObserver: class { observe() {} }, CustomEvent: class {},
  setTimeout, clearTimeout, console, $: id => ids.get(id)});
context.window = context; context.addEventListener = () => {};
const jsRoot = path.resolve(__dirname, '../frontend/js');
for (const file of ['legacy-i18n-messages.js', 'legacy-i18n-dynamic.js',
                    'legacy-i18n-tutorial.js', 'legacy-i18n.js', 'vram.js']) {
  vm.runInContext(fs.readFileSync(path.join(jsRoot, file), 'utf8'), context, {filename: file});
}
for (const [language, label, vramLabel] of [
  ['pt', 'Iniciar treino', '[ok] cabe na memória'],
  ['es', 'Iniciar entrenamiento', '[ok] cabe en la memoria'],
  ['en', 'Start Training', '[ok] fits']
]) {
  context.JKLegacyI18n.setLanguage(language);
  assert.equal(start.textContent, label);
  assert.equal(start.getAttribute('aria-label'), label);
  assert.equal(status.textContent, vramLabel);
  assert.equal(vm.runInContext('VRAM.getVerdict()', context), 'green');
  assert.equal(context.jkSourceText(start), 'Start Training');
  assert.equal(context.jkSourceText(status), '[ok] fits');
  assert.equal(picker.value, language);
  assert.equal(storage.get('jk_step_language'), language);
  assert.equal(preview.textContent, 'Training');
  assert.equal(filename.textContent, 'Save');
  assert.equal(filename.getAttribute('title'), 'Training');
  assert.equal(lyrics.value, 'Ready\nTraining');
  assert.equal(lyrics.textContent, 'Ready\nTraining');
  assert.equal(logs.textContent, 'Ready');
  assert.equal(apiOutput.textContent, 'Start Training');
  assert.equal(dataset.textContent, 'Save');
  assert.equal(dataset.value, 'datasets/Training');
  assert.equal(runName.value, 'Training');
  assert.ok(presetStatus.textContent.includes("'Training'"));
}
for (const [message, verdict] of [['[!] tight fit', 'yellow'], ['[!] WILL OOM', 'red']]) {
  status.childNodes[0].nodeValue = message;
  context.JKLegacyI18n.setLanguage('es');
  assert.equal(vm.runInContext('VRAM.getVerdict()', context), verdict);
}
assert.equal(context.JKLegacyI18n.translate('Uncatalogued user path  with  spaces'), 'Uncatalogued user path  with  spaces');
assert.equal(context.JKLegacyI18n.translate('Select a dataset', 'es'), 'Selecciona un dataset');
console.log('PASS: legacy EN/PT/ES, persistence, immutable user data and VRAM source readers.');
