// Small PDF viewer on top of PDF.js: one page at a time, text layer for
// selection, and highlighting of quotes by fuzzy text matching.
import * as pdfjsLib from './vendor/pdfjs/pdf.min.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdfjs/pdf.worker.min.mjs', import.meta.url).toString();

// Letters and digits only, lower case. Makes matching robust against line
// breaks, hyphenation, double spaces and typographic quotes.
export function norm(text) {
  return (text || '').toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}]+/gu, '');
}

export async function createViewer({ root, url, onPageChange, onSelection }) {
  const scroll = root.querySelector('[data-viewer-scroll]');
  const pageInput = root.querySelector('[data-page-input]');
  const pageCount = root.querySelector('[data-page-count]');
  const status = root.querySelector('[data-viewer-status]');

  const pdf = await pdfjsLib.getDocument({ url }).promise;
  const textCache = new Map();   // page -> {items, joined, map}
  let current = 1;
  let zoom = 1;
  let highlights = [];
  let renderToken = 0;
  pageCount.textContent = pdf.numPages;

  async function pageText(n) {
    if (textCache.has(n)) return textCache.get(n);
    const page = await pdf.getPage(n);
    const content = await page.getTextContent();
    // joined normalised text plus, for each character, the item it came from
    let joined = '';
    const map = [];
    content.items.forEach((it, idx) => {
      const t = norm(it.str);
      joined += t;
      for (let i = 0; i < t.length; i++) map.push(idx);
    });
    const entry = { content, joined, map };
    textCache.set(n, entry);
    return entry;
  }

  // Find the item indices on page n that a quote covers. Tries the whole
  // quote, then its start and end (quotes are sometimes trimmed or merged).
  async function locate(n, quote) {
    const q = norm(quote);
    if (q.length < 8) return null;
    const { joined, map } = await pageText(n);
    const tries = [q];
    if (q.length > 80) tries.push(q.slice(0, 60), q.slice(-60));
    for (const t of tries) {
      const at = joined.indexOf(t);
      if (at >= 0) {
        const items = new Set();
        for (let i = at; i < at + t.length && i < map.length; i++) items.add(map[i]);
        return { items, partial: t !== q };
      }
    }
    return null;
  }

  // Which page holds the quote? Checks the stated page first, then nearby
  // pages, then the whole document.
  async function findQuote(quote, hint) {
    const order = [];
    const seen = new Set();
    const add = (p) => { if (p >= 1 && p <= pdf.numPages && !seen.has(p)) { seen.add(p); order.push(p); } };
    if (hint) { add(hint); add(hint - 1); add(hint + 1); }
    for (let p = 1; p <= pdf.numPages; p++) add(p);
    for (const p of order) {
      const hit = await locate(p, quote);
      if (hit) return { found: true, page: p, partial: hit.partial };
    }
    return { found: false, page: null };
  }

  async function render() {
    const token = ++renderToken;
    const page = await pdf.getPage(current);
    const base = page.getViewport({ scale: 1 });
    const width = Math.max(320, scroll.clientWidth - 24);
    const scale = (width / base.width) * zoom;
    const viewport = page.getViewport({ scale });
    const ratio = window.devicePixelRatio || 1;

    const wrap = document.createElement('div');
    wrap.className = 'pdf-page';
    wrap.style.width = `${Math.floor(viewport.width)}px`;
    wrap.style.height = `${Math.floor(viewport.height)}px`;
    wrap.style.setProperty('--scale-factor', viewport.scale);
    const canvas = document.createElement('canvas');
    canvas.width = Math.floor(viewport.width * ratio);
    canvas.height = Math.floor(viewport.height * ratio);
    canvas.style.width = `${Math.floor(viewport.width)}px`;
    canvas.style.height = `${Math.floor(viewport.height)}px`;
    const layer = document.createElement('div');
    layer.className = 'textLayer';
    wrap.append(canvas, layer);

    await page.render({
      canvasContext: canvas.getContext('2d'),
      viewport,
      transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : null,
    }).promise;
    if (token !== renderToken) return;

    const { content } = await pageText(current);
    const textLayer = new pdfjsLib.TextLayer({ textContentSource: content, container: layer, viewport });
    await textLayer.render();
    if (token !== renderToken) return;

    scroll.replaceChildren(wrap);
    pageInput.value = current;

    // Apply highlights. Selected one last so it wins, then scroll to it.
    const divs = textLayer.textDivs;
    let firstSelected = null;
    let missingSelected = false;
    const ordered = [...highlights].sort((a, b) => (a.selected ? 1 : 0) - (b.selected ? 1 : 0));
    for (const h of ordered) {
      const hit = await locate(current, h.text);
      if (!hit) { if (h.selected) missingSelected = true; continue; }
      hit.items.forEach((i) => {
        const d = divs[i];
        if (!d) return;
        d.classList.add('hl', h.domain || 'targets');
        if (h.selected) { d.classList.add('selected'); if (!firstSelected) firstSelected = d; }
      });
    }
    if (firstSelected) {
      const top = firstSelected.getBoundingClientRect().top - wrap.getBoundingClientRect().top;
      scroll.scrollTop = Math.max(0, top - 80);
    } else {
      scroll.scrollTop = 0;
    }
    if (status) {
      status.textContent = missingSelected ? 'The selected quote was not found word for word on this page.' : '';
    }
    if (onPageChange) onPageChange(current);
  }

  async function goTo(n, hl) {
    current = Math.min(Math.max(1, Number(n) || 1), pdf.numPages);
    if (hl) highlights = hl;
    await render();
  }

  root.querySelector('[data-prev]')?.addEventListener('click', () => goTo(current - 1));
  root.querySelector('[data-next]')?.addEventListener('click', () => goTo(current + 1));
  root.querySelector('[data-zoom-in]')?.addEventListener('click', () => { zoom = Math.min(3, zoom + 0.25); render(); });
  root.querySelector('[data-zoom-out]')?.addEventListener('click', () => { zoom = Math.max(0.5, zoom - 0.25); render(); });
  pageInput?.addEventListener('change', () => goTo(pageInput.value));

  if (onSelection) {
    scroll.addEventListener('mouseup', () => {
      const sel = window.getSelection();
      const text = sel ? sel.toString().replace(/\s+/g, ' ').trim() : '';
      onSelection(text, current);
    });
  }

  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 200); });

  return {
    numPages: pdf.numPages,
    get page() { return current; },
    goTo,
    findQuote,
    setHighlights(hl) { highlights = hl; return render(); },
  };
}
