// Results page: filters for the passage list, mitigation drill-down, PDF preview.
import { createViewer } from './pdfviewer.mjs';

const CFG = window.RESULTS;
const rows = CFG.rowData || [];
let viewer = null;
let domain = 'all';

// mitigation categories: open/close instrument lists
document.querySelectorAll('[data-toggle]').forEach((b) => b.addEventListener('click', () => {
  const box = document.getElementById(b.dataset.toggle);
  const open = box.classList.toggle('hidden') === false;
  b.textContent = open ? '▾' : '▸';
  b.setAttribute('aria-expanded', String(open));
}));

// passage list filters
const cards = [...document.querySelectorAll('#cards .card')];
const search = document.getElementById('search');
function applyFilter() {
  const q = (search?.value || '').trim().toLowerCase();
  cards.forEach((c) => {
    const okDom = domain === 'all' || c.dataset.dom === domain;
    const okText = !q || c.textContent.toLowerCase().includes(q);
    c.classList.toggle('hidden', !(okDom && okText));
  });
}
document.querySelectorAll('#domain-chips [data-dom]').forEach((b) => b.addEventListener('click', () => {
  domain = b.dataset.dom;
  document.querySelectorAll('#domain-chips [data-dom]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  applyFilter();
}));
search?.addEventListener('input', applyFilter);

function show(page, selectedId) {
  if (!viewer || !page) return;
  const hl = rows.filter((r) => r.page === page && r.text).map((r) => ({ text: r.text, domain: r.domain, selected: r.id === selectedId }));
  viewer.goTo(page, hl);
}

cards.forEach((c) => {
  const open = (e) => {
    if (e.target.closest('a')) return;
    cards.forEach((x) => x.classList.toggle('active', x === c));
    const r = rows.find((x) => x.id === c.dataset.id);
    show(r ? r.page : Number(c.dataset.page), c.dataset.id);
  };
  c.addEventListener('click', open);
  c.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(e); });
});

// page links in the summary and the page strip
document.querySelectorAll('[data-goto]').forEach((a) => a.addEventListener('click', () => show(Number(a.dataset.goto))));

if (CFG.hasPdf) {
  createViewer({ root: document.getElementById('viewer'), url: CFG.pdfUrl })
    .then((v) => {
      viewer = v;
      const bad = new Set(cards.filter((c) => c.querySelector('[style*="var(--bad)"]')).map((c) => c.dataset.id));
      const first = rows.find((r) => r.page && !bad.has(r.id)) || rows.find((r) => r.page);
      if (first) { show(first.page, first.id); cards.find((c) => c.dataset.id === first.id)?.classList.add('active'); }
      else v.goTo(1, []);
    })
    .catch((e) => { document.getElementById('viewer').innerHTML = `<div class="banner bad">The PDF could not be shown: ${String(e.message).replace(/</g, '&lt;')}</div>`; });
}

// run statistics: click to copy
document.querySelectorAll('[data-copy]').forEach((b) => b.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(b.dataset.copy);
    b.classList.add('copied');
    const old = b.textContent;
    if (b.classList.contains('btn')) b.textContent = 'Copied';
    setTimeout(() => { b.classList.remove('copied'); b.textContent = old; }, 1200);
  } catch (e) { /* clipboard blocked: the value stays selectable */ }
}));
