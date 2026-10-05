/* English / Portuguese / Spanish for the original toolkit interface.
 * Display text has an immutable source, so UI readers and model data stay stable.
 */
(() => {
  'use strict';
  const KEY = 'jk_step_language';
  const languages = ['en', 'pt', 'es'];
  const catalogs = Object.assign({}, window.JK_LEGACY_MESSAGES || {}, window.JK_LEGACY_DYNAMIC_MESSAGES || {}, window.JK_LEGACY_TUTORIAL_MESSAGES || {}, window.JK_LEGACY_THEME_MESSAGES || {});
  const pageTitle = document.title;
  const originals = new WeakMap();
  const attributes = new WeakMap();
  const reverse = new Map();
  Object.entries(catalogs).forEach(([source, row]) => languages.forEach(lang => {
    if (row[lang] && !reverse.has(row[lang])) reverse.set(row[lang], source);
  }));
  const prefixes = Object.keys(catalogs).filter(key => /[:~(]$/.test(key)).sort((a, b) => b.length - a.length);
  const skip = '[data-no-i18n],script,style,code,pre,textarea,.log-viewer,.log-entry,#console-line,#cli-export-content,#file-browser-breadcrumb,#sidecar-filename,#resume-run-name,#ap-track-name,#ap-track-artist,#ap-title,#ap-artist,#compare-legend-a,#compare-legend-b';
  let language;
  try { language = localStorage.getItem(KEY); } catch (_) {}
  if (!languages.includes(language)) language = String(navigator.language || 'en').split('-')[0].toLowerCase();
  if (!languages.includes(language)) language = 'en';
  const normalize = value => String(value).replace(/\s+/g, ' ').trim();
  const canonical = value => reverse.get(normalize(value)) || normalize(value);
  const templates = window.JK_LEGACY_TEMPLATES || [];
  function translate(value, selected = language) {
    if (String(value).includes('\n')) return String(value).split('\n').map(line => translate(line, selected)).join('\n');
    const source = canonical(value);
    if (catalogs[source]) return catalogs[source][selected] || catalogs[source].en || source;
    for (const [pattern, pt, es] of templates) {
      if (pattern.test(source)) return selected === 'en' ? source : source.replace(pattern, selected === 'pt' ? pt : es);
    }
    for (const prefix of prefixes) {
      if (source.startsWith(prefix + ' ') || source.startsWith(prefix) && /[~(]$/.test(prefix)) {
        return (catalogs[prefix][selected] || prefix) + source.slice(prefix.length);
      }
    }
    return String(value);
  }
  function sourceText(element) {
    if (!element) return '';
    if (element.nodeType === Node.TEXT_NODE) {
      const item = originals.get(element);
      return item && element.nodeValue === item.rendered ? item.source : element.nodeValue;
    }
    return Array.from(element.childNodes || []).map(sourceText).join('');
  }
  function text(node) {
    if (!node.parentElement || node.parentElement.closest(skip)) return;
    const option = node.parentElement.closest('option');
    if (option?.value && /dataset|resume-checkpoint|compare-run|preset|theme/.test(option.closest('select')?.id || '')) {
      option.setAttribute('data-no-i18n', '');
      return;
    }
    const value = node.nodeValue;
    if (!value || !/[A-Za-zÀ-ÿ]/.test(value)) return;
    const previous = originals.get(node);
    const source = previous && value === previous.rendered ? previous.source : value;
    const trimmed = normalize(source);
    const rendered = translate(trimmed);
    if (rendered === trimmed && !previous) return;
    const whitespace = source.match(/^(\s*)[\s\S]*?(\s*)$/);
    const next = (whitespace ? whitespace[1] : '') + rendered + (whitespace ? whitespace[2] : '');
    originals.set(node, {source, rendered: next});
    if (value !== next) node.nodeValue = next;
  }
  function attrs(element) {
    if (!element || element.nodeType !== Node.ELEMENT_NODE || element.closest('[data-no-i18n],script,style,.log-viewer,.log-entry')) return;
    const stored = attributes.get(element) || {};
    ['title','placeholder','aria-label','data-hint'].forEach(name => {
      if (!element.hasAttribute(name)) return;
      const current = element.getAttribute(name);
      const old = stored[name];
      const source = old && current === old.rendered ? old.source : current;
      const next = translate(source);
      stored[name] = {source, rendered: next};
      if (next !== current) element.setAttribute(name, next);
    });
    attributes.set(element, stored);
  }
  function walk(root) {
    if (root.nodeType === Node.TEXT_NODE) { text(root); return; }
    if (root.nodeType === Node.ELEMENT_NODE) attrs(root);
    const iterator = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node;
    while ((node = iterator.nextNode())) {
      if (node.nodeType === Node.TEXT_NODE) text(node); else attrs(node);
    }
  }
  function setLanguage(value, persist = true) {
    if (!languages.includes(value)) return;
    language = value;
    if (persist) { try { localStorage.setItem(KEY, language); } catch (_) {} }
    document.documentElement.lang = language === 'pt' ? 'pt-BR' : language;
    const picker = document.getElementById('legacy-language');
    if (picker) picker.value = language;
    walk(document.body);
    document.title = translate(pageTitle);
    document.dispatchEvent(new CustomEvent('jk-step:language', {detail: language}));
  }
  window.JKLegacyI18n = {translate, sourceText, setLanguage, get language() { return language; }};
  window.jkSourceText = sourceText;
  function initialize() {
    setLanguage(language);
    document.getElementById('legacy-language')?.addEventListener('change', event => setLanguage(event.target.value));
    new MutationObserver(mutations => {
      mutations.forEach(change => {
        if (change.type === 'characterData') text(change.target);
        else if (change.type === 'attributes') attrs(change.target);
        else change.addedNodes.forEach(walk);
      });
    }).observe(document.body, {subtree: true,childList: true,characterData: true,attributes: true,attributeFilter:['title','placeholder','aria-label','data-hint']});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else initialize();
  window.addEventListener('storage', event => { if (event.key === KEY) setLanguage(event.newValue, false); });
})();
