// Validation page: edit rows next to the PDF, save as you go.
import { createViewer } from './pdfviewer.mjs';

const CFG = window.REVIEW;
const DOMAINS = ['targets', 'mitigation', 'adaptation', 'benefits'];
const DOMAIN_LABEL = { targets: 'Targets', mitigation: 'Mitigation', adaptation: 'Adaptation', benefits: 'Benefits' };
const TAG_LABEL = { targets: 'Target', mitigation: 'Mitigation', adaptation: 'Adaptation', benefits: 'Benefit' };
const REQUIRED = {
  targets: ['target_area', 'conditionality', 'target_year'],
  mitigation: ['category', 'instrument'],
  adaptation: ['adaptation_category', 'adaptation_measure'],
  benefits: ['benefit'],
};
const FIELD_LABEL = {
  target_area: 'Target area', target_scope: 'Target scope', ghg_target: 'GHG target?', target_type: 'Target type',
  conditionality: 'Conditionality', target_year: 'Target year', category: 'Category', purpose: 'Purpose',
  instrument: 'Instrument', asi: 'A-S-I', adaptation_category: 'Category', adaptation_measure: 'Measure',
  benefit: 'Type of benefit',
};
const LOCAL_KEY = `review:${CFG.taskId}`;

let state = null;
let tax = null;
let viewer = null;
const ui = { domain: 'targets', filter: 'all', selected: null, selectionText: '', selectionPage: null };

// ---------------------------------------------------------------- helpers
const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const uid = () => Math.random().toString(16).slice(2, 14);
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

function toast(msg, action) {
  document.querySelectorAll('.toast').forEach((t) => t.remove());
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = esc(msg);
  if (action) {
    const b = document.createElement('button');
    b.textContent = action.label;
    b.className = 'btn small';
    b.style.marginLeft = '12px';
    b.onclick = () => { action.run(); el.remove(); };
    el.append(b);
  }
  document.body.append(el);
  setTimeout(() => el.remove(), action ? 8000 : 3500);
}

function newRow(domain, extra = {}) {
  return {
    id: uid(), domain, verbatim: '', translation: '', page: null, source: 'manual', status: 'pending',
    deleted: false, edited: false, note: '', pipeline_verified: null, pdf_found: null, pdf_page: null,
    target_area: '', target_scope: '', ghg_target: '', target_type: '', conditionality: '', target_year: '',
    category: '', purpose: '', instrument: '', asi: '', adaptation_category: '', adaptation_measure: '', benefit: '',
    ...extra,
  };
}

const active = () => state.rows.filter((r) => !r.deleted);
const missingOf = (r) => {
  const m = (REQUIRED[r.domain] || []).filter((f) => !String(r[f] ?? '').trim());
  if (!String(r.verbatim || '').trim() || !r.page) m.push('verbatim/page');
  return m;
};

// Same logic as derive_flags() in review.py. Keep both in sync.
function deriveFlags(rows) {
  const tMit = rows.filter((r) => r.domain === 'targets' && r.target_area === 'Transport sector mitigation target');
  const tAda = rows.filter((r) => r.domain === 'targets' && r.target_area === 'Transport sector adaptation target');
  const hasMit = rows.some((r) => r.domain === 'mitigation');
  const hasAda = rows.some((r) => r.domain === 'adaptation');
  const hasBen = rows.some((r) => r.domain === 'benefits');
  const yn = (b) => (b ? 'yes' : 'no');
  const kinds = new Set(tMit.map((r) => r.ghg_target));
  let summary = 'No transport target';
  if (tMit.length) summary = kinds.has('GHG') && kinds.has('Non-GHG') ? 'GHG and non-GHG target' : kinds.has('GHG') ? 'GHG target' : 'non-GHG target';
  const ghg = tMit.filter((r) => r.ghg_target === 'GHG');
  const conds = new Set(ghg.map((r) => r.conditionality));
  let ghgType = 'No transport target';
  if (tMit.length && !ghg.length) ghgType = 'No GHG transport target';
  else if (ghg.length) {
    if (conds.size === 1 && conds.has('Unconditional')) ghgType = 'GHG unconditional only';
    else if (conds.size === 1 && conds.has('Conditional')) ghgType = 'GHG conditional only';
    else if (conds.has('Unconditional') && conds.has('Conditional')) ghgType = 'GHG unconditional and conditional';
    else ghgType = 'Conditionality unclear';
  }
  const target = tMit.length > 0 || tAda.length > 0;
  const measures = hasMit || hasAda;
  return {
    transport_content: yn(target || measures || hasBen), transport_target: yn(target), transport_measures: yn(measures),
    summary_transport_target: summary, contains_transport_mitigation_target: yn(tMit.length > 0),
    ghg_transport_target_type: ghgType, contains_transport_adaptation_target: yn(tAda.length > 0),
    contains_transport_mitigation_measures: yn(hasMit), contains_transport_adaptation_measures: yn(hasAda),
    contains_benefits: yn(hasBen), contains_just_transition: '',
  };
}

// ---------------------------------------------------------------- saving
const saveLabel = $('#savestate');
function markSaved(text) { saveLabel.textContent = text; }

function changed({ rerenderList = true } = {}) {
  state.local_saved = Date.now();
  try { localStorage.setItem(LOCAL_KEY, JSON.stringify(state)); } catch (e) { /* storage full or blocked */ }
  markSaved('Unsaved changes, saving…');
  saveToServer();
  renderTabs();
  renderFilters();
  renderFlags();
  if (rerenderList) renderRows();
}

const saveToServer = debounce(async () => {
  try {
    const res = await fetch(CFG.stateUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.saved_at = data.saved_at;
    markSaved(`Saved on the server at ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}. Also kept in this browser.`);
  } catch (e) {
    markSaved('Could not reach the server. Changes are kept in this browser; use "Save draft (JSON)" to be safe.');
  }
}, 1500);

// ---------------------------------------------------------------- taxonomy
function opts(domain, dim, filter) {
  const list = (tax.domains[domain] || {})[dim] || [];
  return list.filter((v) => !filter || filter(v));
}
const codeOf = (domain, dim, label) => (opts(domain, dim).find((v) => v.label === label) || {}).code;

function optionsFor(row, field) {
  switch (field) {
    case 'target_area': return opts('Targets', 'Target area').map((v) => v.label);
    case 'target_scope': {
      const c = codeOf('Targets', 'Target area', row.target_area);
      return opts('Targets', 'Target scope', (v) => !c || !v.parent_1 || v.parent_1.split(/,\s*/).includes(c)).map((v) => v.label);
    }
    case 'ghg_target': return opts('Targets', 'GHG target?').map((v) => v.label);
    case 'target_type': {
      const c = codeOf('Targets', 'Target area', row.target_area);
      return opts('Targets', 'Target type', (v) => !c || !v.parent_1 || v.parent_1.split(/,\s*/).includes(c)).map((v) => v.label);
    }
    case 'conditionality': return opts('Targets', 'Conditionality').map((v) => v.label);
    case 'category': return opts('Mitigation', 'Type of measure: category').map((v) => v.label);
    case 'purpose': {
      const c = codeOf('Mitigation', 'Type of measure: category', row.category);
      return opts('Mitigation', 'Type of measure: purpose', (v) => !c || v.parent_1 === c).map((v) => v.label);
    }
    case 'instrument': {
      const c = codeOf('Mitigation', 'Type of measure: category', row.category);
      const p = codeOf('Mitigation', 'Type of measure: purpose', row.purpose);
      return opts('Mitigation', 'Type of measure: instruments', (v) => (!c || v.parent_1 === c || !v.parent_1) && (!p || v.parent_2 === p || !v.parent_2)).map((v) => v.label);
    }
    case 'asi': return (tax.cross_cutting_dimensions.asi.values || []).map((v) => v.label);
    case 'adaptation_category': return opts('Adaptation', 'Category').map((v) => v.label);
    case 'adaptation_measure': {
      const c = codeOf('Adaptation', 'Category', row.adaptation_category);
      return opts('Adaptation', 'Measure', (v) => !c || !v.parent_1 || v.parent_1 === c).map((v) => v.label);
    }
    case 'benefit': return opts('Benefits', 'Type of benefit').map((v) => v.label);
    default: return [];
  }
}

const FIELDS = {
  targets: ['target_area', 'target_scope', 'ghg_target', 'target_type', 'conditionality', 'target_year'],
  mitigation: ['category', 'purpose', 'instrument', 'asi'],
  adaptation: ['adaptation_category', 'adaptation_measure'],
  benefits: ['benefit'],
};
const DEPENDENT = { target_area: ['target_scope', 'target_type'], category: ['purpose', 'instrument'], purpose: ['instrument'], adaptation_category: ['adaptation_measure'] };

// ---------------------------------------------------------------- document panel
const DOC_FIELDS = [
  ['document_id', 'Document ID', 'Assigned in the database'], ['country_code', 'Country code', 'e.g. KEN'],
  ['country', 'Country', ''], ['type_of_document', 'Type of document', ''], ['document_name', 'Document name', ''],
  ['last_submission', 'Last submission', ''], ['generation', 'Generation', 'e.g. 3'], ['version', 'Version', 'e.g. 0'],
  ['date', 'Date', 'YYYY-MM-DD'], ['status', 'Status', ''], ['language', 'Language', 'e.g. English'],
];
const DOC_CHOICES = {
  type_of_document: ['NDC', 'LTS', 'National policy document', 'Other'],
  last_submission: ['First Generation NDC', 'Second Generation NDC', 'Third Generation NDC', 'LTS', 'National policy document'],
  status: ['Active', 'Archived', 'Covered by EU', 'Covered by EU archived'],
};
const FLAG_FIELDS = [
  ['transport_content', 'Transport content', ['yes', 'no']],
  ['transport_target', 'Transport target (mitigation or/and adaptation)', ['yes', 'no']],
  ['transport_measures', 'Transport measures (mitigation or/and adaptation)', ['yes', 'no']],
  ['summary_transport_target', 'Summary transport target', ['No transport target', 'non-GHG target', 'GHG target', 'GHG and non-GHG target']],
  ['contains_transport_mitigation_target', 'Contains transport mitigation target', ['yes', 'no']],
  ['ghg_transport_target_type', 'GHG transport target type', ['No transport target', 'No GHG transport target', 'GHG unconditional only', 'GHG conditional only', 'GHG unconditional and conditional', 'Conditionality unclear']],
  ['contains_transport_adaptation_target', 'Contains transport adaptation target', ['yes', 'no']],
  ['contains_transport_mitigation_measures', 'Contains transport mitigation measures', ['yes', 'no']],
  ['contains_transport_adaptation_measures', 'Contains transport adaptation measures', ['yes', 'no']],
  ['contains_benefits', 'Contains benefits', ['yes', 'no']],
  ['contains_just_transition', 'Contains reference to just transition', ['yes', 'no']],
];

function renderDoc() {
  const d = state.document;
  const box = $('#doc-fields');
  const input = ([k, label, ph]) => {
    const choices = DOC_CHOICES[k];
    const val = d[k] ?? '';
    const control = choices
      ? `<select data-doc="${k}"><option value=""></option>${[...new Set([...choices, val].filter(Boolean))].map((c) => `<option ${c === val ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`
      : `<input data-doc="${k}" value="${esc(val)}" placeholder="${esc(ph)}">`;
    return `<label class="field mono-label">${esc(label)}${control}</label>`;
  };
  box.innerHTML = `
    <div class="grid g6">${DOC_FIELDS.map(input).join('')}
      <label class="field mono-label">URL<input data-doc="url" value="${esc(d.url)}" placeholder="https://unfccc.int/…"></label></div>
    <label class="field mono-label">Note<textarea data-doc="note" rows="2" placeholder="e.g. which earlier submission this document replaces">${esc(d.note)}</textarea></label>
    <div style="display: flex; flex-direction: column; gap: 8px; padding-top: 12px; border-top: 1px solid #EAF2F5">
      <div class="spread small"><b style="font-size: 13px">Transport content</b><span class="muted">Filled in from the rows. Change a value only if the rows are incomplete.</span></div>
      <div class="grid g4" id="flag-fields"></div>
    </div>
    <p class="small muted" style="margin: 0">Version number in the export: <b id="vn"></b></p>`;
  box.querySelectorAll('[data-doc]').forEach((el) => {
    el.addEventListener('input', () => { d[el.dataset.doc] = el.value; updateVersion(); changed({ rerenderList: false }); });
  });
  renderFlags();
  updateVersion();
}

function updateVersion() {
  const d = state.document;
  const vn = d.generation ? `${d.type_of_document || 'NDC'} ${d.generation}.${d.version || '0'}` : '(set generation)';
  const el = $('#vn');
  if (el) el.textContent = vn;
}

function renderFlags() {
  const box = $('#flag-fields');
  if (!box) return;
  const derived = deriveFlags(active());
  const over = state.document.flags || (state.document.flags = {});
  box.innerHTML = FLAG_FIELDS.map(([k, label, choices]) => {
    const auto = derived[k];
    const autoLabel = k === 'contains_just_transition' ? 'Not checked' : `Automatic: ${auto}`;
    const val = over[k] || '';
    const manualWarn = k === 'contains_just_transition' || (k === 'contains_benefits' && !val);
    return `<label class="field">${esc(label)}
      <select data-flag="${k}" style="${manualWarn ? 'border-color: var(--flag); background: #FFF9EF' : ''}">
        <option value="">${esc(autoLabel)}</option>
        ${choices.map((c) => `<option ${c === val ? 'selected' : ''}>${esc(c)}</option>`).join('')}
      </select>
      ${k === 'contains_benefits' ? '<span class="hint" style="color:#7A4A00">Benefits are not extracted yet, check manually</span>' : ''}
      ${k === 'contains_just_transition' ? '<span class="hint" style="color:#7A4A00">Manual</span>' : ''}</label>`;
  }).join('');
  box.querySelectorAll('[data-flag]').forEach((el) => el.addEventListener('change', () => {
    over[el.dataset.flag] = el.value;
    changed({ rerenderList: false });
  }));
}

// ---------------------------------------------------------------- tabs and filters
function inTab(r) { return !r.deleted && r.domain === ui.domain; }

function passes(r) {
  const f = ui.filter;
  if (f === 'all') return true;
  if (['pending', 'confirmed', 'flagged'].includes(f)) return r.status === f;
  if (f === 'missing') return missingOf(r).length > 0;
  if (f === 'notfound') return r.pdf_found === false;
  if (f.startsWith('source:')) return (r.source || '') === f.slice(7);
  return true;
}

function renderTabs() {
  $('#tabs').innerHTML = DOMAINS.map((d) => {
    const n = state.rows.filter((r) => !r.deleted && r.domain === d).length;
    return `<button class="tab" role="tab" aria-selected="${d === ui.domain}" data-tab="${d}">${DOMAIN_LABEL[d]} ${n}</button>`;
  }).join('');
  $('#tabs').querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => {
    ui.domain = b.dataset.tab; ui.selected = null; renderTabs(); renderFilters(); renderRows();
  }));
}

function renderFilters() {
  const rows = state.rows.filter(inTab);
  const count = (fn) => rows.filter(fn).length;
  const sources = [...new Set(rows.map((r) => r.source || ''))].filter(Boolean);
  const chips = [
    ['all', `All ${rows.length}`],
    ['pending', `Pending ${count((r) => r.status === 'pending')}`],
    ['confirmed', `Confirmed ${count((r) => r.status === 'confirmed')}`],
    ['flagged', `Flagged ${count((r) => r.status === 'flagged')}`],
    ['missing', `Missing field ${count((r) => missingOf(r).length)}`, true],
    ['notfound', `Not found in PDF ${count((r) => r.pdf_found === false)}`, true],
    ...(sources.length > 1 ? sources.map((s) => [`source:${s}`, `${s} ${count((r) => r.source === s)}`]) : []),
  ];
  $('#filters').innerHTML = chips.map(([k, label, warn]) =>
    `<button class="chip ${warn ? 'warn' : ''}" aria-pressed="${ui.filter === k}" data-filter="${esc(k)}">${esc(label)}</button>`).join('');
  $('#filters').querySelectorAll('[data-filter]').forEach((b) => b.addEventListener('click', () => {
    ui.filter = b.dataset.filter; renderFilters(); renderRows();
  }));
}

// ---------------------------------------------------------------- rows
function labelLine(r) {
  return FIELDS[r.domain].map((f) => r[f]).filter(Boolean).join(', ') || 'Not classified yet';
}

function checkText(r) {
  if (r.pdf_found === true) return r.pdf_page && r.page && r.pdf_page !== r.page
    ? `<span class="notfound">Found on p. ${r.pdf_page}, row says p. ${r.page}</span>` : `<span class="found">Found on p. ${r.pdf_page || r.page}</span>`;
  if (r.pdf_found === false) return '<span class="notfound">Not found in the PDF</span>';
  if (r.pipeline_verified === false) return '<span class="notfound">Pipeline could not verify</span>';
  return '';
}

function statusPill(r) {
  const label = { pending: 'Pending', confirmed: 'Confirmed', flagged: 'Flagged' }[r.status];
  return `<span class="tag st-${r.status}">${label}</span>`;
}

function renderRows() {
  const list = state.rows.filter(inTab).filter(passes)
    .sort((a, b) => (a.page || 9999) - (b.page || 9999));
  $('#list-info').textContent = `${list.length} rows shown, sorted by page`;
  const box = $('#rows');
  if (!list.length) {
    box.innerHTML = `<div class="panel muted">No rows here. Add one, or select text in the PDF to start a row from it.</div>`;
    return;
  }
  box.innerHTML = list.map((r) => (r.id === ui.selected ? editorHtml(r) : cardHtml(r))).join('');
  box.querySelectorAll('[data-open]').forEach((el) => el.addEventListener('click', () => select(el.dataset.open)));
  const ed = box.querySelector('.editor');
  if (ed) bindEditor(ed, state.rows.find((r) => r.id === ui.selected));
}

function cardHtml(r) {
  const miss = missingOf(r);
  return `<button type="button" class="rowcard ${miss.length ? 'missing' : ''}" data-open="${r.id}">
    <span class="meta"><span class="tag ${r.domain}">${TAG_LABEL[r.domain]}</span>${statusPill(r)}
      <span class="tag">${esc(r.source || 'n/a')}</span>${checkText(r)}
      <span class="mono" style="margin-left:auto">p. ${esc(r.page ?? '?')}</span></span>
    <span class="quote">“${esc(r.verbatim || '(no quote yet)')}”</span>
    <span class="labels">${esc(labelLine(r))}${miss.length ? ` <b style="color:#7A2E12">Missing: ${esc(miss.map((m) => FIELD_LABEL[m] || m).join(', '))}</b>` : ''}</span>
  </button>`;
}

function selectHtml(r, f, miss) {
  const options = optionsFor(r, f);
  const val = r[f] || '';
  const extra = val && !options.includes(val) ? `<option selected>${esc(val)}</option>` : '';
  const isMissing = miss.includes(f);
  return `<label class="field ${isMissing ? 'missing' : ''}">${FIELD_LABEL[f]}${REQUIRED[r.domain].includes(f) ? ' (required)' : ''}
    <select data-f="${f}"><option value=""></option>${extra}${options.map((o) => `<option ${o === val ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>
    ${extra ? '<span class="hint" style="color:#7A4A00">Value is not in the taxonomy</span>' : ''}</label>`;
}

function editorHtml(r) {
  const miss = missingOf(r);
  const fields = FIELDS[r.domain].map((f) => (f === 'target_year'
    ? `<label class="field ${miss.includes(f) ? 'missing' : ''}">Target year (required)<input data-f="target_year" value="${esc(r.target_year)}" placeholder="e.g. 2035"></label>`
    : selectHtml(r, f, miss))).join('');
  return `<div class="editor ${r.status}" data-editor="${r.id}">
    <div class="meta row-flex small"><span class="tag ${r.domain}">${TAG_LABEL[r.domain]}</span>${statusPill(r)}
      <span class="tag">${esc(r.source || 'n/a')}</span>${checkText(r)}
      <label style="margin-left:auto; display:flex; gap:6px; align-items:center">p.<input data-f="page" type="number" min="1" value="${esc(r.page ?? '')}" style="width:64px;height:30px;border:1px solid var(--line-strong);border-radius:6px;padding:0 6px"></label></div>
    <div class="grid g2">
      <label class="field">Verbatim quote (exported)<textarea data-f="verbatim" rows="3">${esc(r.verbatim)}</textarea></label>
      <label class="field">English translation (for checking only)<textarea data-f="translation" rows="3" placeholder="Not provided yet">${esc(r.translation)}</textarea></label>
    </div>
    <div class="grid g3">${fields}</div>
    ${miss.length ? `<div class="small" style="color:#7A2E12;font-weight:600">Needed for export: ${esc(miss.map((m) => FIELD_LABEL[m] || m).join(', '))}</div>` : ''}
    <label class="field">Note<input data-f="note" value="${esc(r.note)}" placeholder="Why you changed or flagged this row"></label>
    <div class="row-flex">
      <button type="button" class="btn primary" data-act="confirm">Confirm and next</button>
      <button type="button" class="btn" data-act="flag">${r.status === 'flagged' ? 'Unflag' : 'Flag'}</button>
      <button type="button" class="btn" data-act="duplicate">Duplicate row</button>
      <label class="field" style="flex-direction:row;align-items:center;gap:6px;font-size:13px">Move to
        <select data-act-move style="height:40px;width:auto">${DOMAINS.map((d) => `<option value="${d}" ${d === r.domain ? 'selected' : ''}>${DOMAIN_LABEL[d]}</option>`).join('')}</select></label>
      <button type="button" class="btn" data-act="close">Close</button>
      <button type="button" class="btn danger" data-act="delete" style="margin-left:auto">Delete</button>
    </div>
  </div>`;
}

const recheck = debounce((r) => checkRow(r).then(() => { renderFilters(); }), 800);

function bindEditor(ed, r) {
  ed.querySelectorAll('[data-f]').forEach((el) => {
    const f = el.dataset.f;
    const handler = () => {
      r[f] = f === 'page' ? (parseInt(el.value, 10) || null) : el.value;
      if (DEPENDENT[f]) {
        // clear dependent fields that no longer fit
        DEPENDENT[f].forEach((dep) => { if (r[dep] && !optionsFor(r, dep).includes(r[dep])) r[dep] = ''; });
      }
      r.edited = true;
      if (f === 'verbatim' || f === 'page') { r.pdf_found = null; recheck(r); }
      const structural = el.tagName === 'SELECT' || f === 'page';
      changed({ rerenderList: structural });
      if (structural && f === 'page') showRowInPdf(r);
    };
    el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', handler);
  });
  ed.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => act(b.dataset.act, r)));
  ed.querySelector('[data-act-move]').addEventListener('change', (e) => {
    r.domain = e.target.value; r.edited = true; ui.domain = r.domain; changed(); showRowInPdf(r);
  });
}

function act(what, r) {
  if (what === 'confirm') {
    r.status = 'confirmed';
    const list = state.rows.filter(inTab).filter(passes).sort((a, b) => (a.page || 9999) - (b.page || 9999));
    const idx = list.findIndex((x) => x.id === r.id);
    const next = list.slice(idx + 1).find((x) => x.status === 'pending');
    ui.selected = next ? next.id : null;
    changed();
    if (next) showRowInPdf(next);
  } else if (what === 'flag') {
    r.status = r.status === 'flagged' ? 'pending' : 'flagged'; changed();
  } else if (what === 'duplicate') {
    const copy = { ...r, id: uid(), status: 'pending', source: r.source, edited: true };
    state.rows.splice(state.rows.indexOf(r) + 1, 0, copy);
    ui.selected = copy.id; changed();
  } else if (what === 'delete') {
    r.deleted = true; ui.selected = null; changed();
    toast('Row deleted.', { label: 'Undo', run: () => { r.deleted = false; ui.selected = r.id; changed(); } });
  } else if (what === 'close') {
    ui.selected = null; renderRows();
  }
}

function select(id) {
  ui.selected = id;
  renderRows();
  const r = state.rows.find((x) => x.id === id);
  if (r) showRowInPdf(r);
  document.querySelector(`[data-editor="${id}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function showRowInPdf(r) {
  if (!viewer) return;
  const page = r.pdf_page || r.page || viewer.page;
  const hl = active().filter((x) => (x.pdf_page || x.page) === page && x.verbatim)
    .map((x) => ({ text: x.verbatim, domain: x.domain, selected: x.id === r.id }));
  viewer.goTo(page, hl);
}

// ---------------------------------------------------------------- quote check
async function checkRow(r) {
  if (!viewer || !r.verbatim) return;
  const res = await viewer.findQuote(r.verbatim, r.page);
  r.pdf_found = res.found;
  r.pdf_page = res.page;
  const card = document.querySelector(`[data-open="${r.id}"]`);
  if (card) card.outerHTML = cardHtml(r);
  document.querySelector(`[data-open="${r.id}"]`)?.addEventListener('click', () => select(r.id));
}

async function checkAll() {
  const todo = active().filter((r) => r.verbatim && r.pdf_found == null);
  for (const r of todo) { await checkRow(r); }
  if (todo.length) {
    renderFilters();
    const nf = active().filter((r) => r.pdf_found === false).length;
    if (nf) toast(`${nf} quotes were not found word for word in the PDF. Use the "Not found in PDF" filter to check them.`);
    try { localStorage.setItem(LOCAL_KEY, JSON.stringify(state)); } catch (e) { /* ignore */ }
    saveToServer();
  }
}

// ---------------------------------------------------------------- import / export
const ALIASES = {
  verbatim: ['verbatim', 'quote', 'content', 'text', 'verbatim_en'], page: ['page', 'page_number', 'pagenumber', 'p'],
  domain: ['domain', 'type', 'sheet', 'kind'], translation: ['translation', 'translation_en'],
  target_year: ['target_year', 'year'], ghg_target: ['ghg_target', 'ghg'], adaptation_measure: ['adaptation_measure', 'measure'],
  benefit: ['benefit', 'type_of_benefit'],
};
const DOMAIN_ALIAS = { target: 'targets', targets: 'targets', mitigation: 'mitigation', mitigation_measure: 'mitigation', measure: 'mitigation', adaptation: 'adaptation', adaptation_measure: 'adaptation', benefit: 'benefits', benefits: 'benefits' };

function normaliseRow(raw, fallbackDomain) {
  const k = {};
  Object.entries(raw || {}).forEach(([key, v]) => { k[key.trim().toLowerCase().replace(/[\s-]+/g, '_')] = v; });
  const r = newRow(fallbackDomain || 'targets', { source: '' });
  Object.keys(r).forEach((f) => {
    if (f === 'id') return;
    for (const a of ALIASES[f] || [f]) { if (k[a] != null) { r[f] = k[a]; break; } }
  });
  r.domain = DOMAIN_ALIAS[String(r.domain).toLowerCase()] || fallbackDomain || 'targets';
  r.page = parseInt(r.page, 10) || null;
  if (k.status == null) r.status = k.confirmed === true || k.confirmed === 'true' ? 'confirmed' : (k.flagged === true || k.flagged === 'true') ? 'flagged' : 'pending';
  if (!['pending', 'confirmed', 'flagged'].includes(r.status)) r.status = 'pending';
  r.pdf_found = null; r.pdf_page = null;
  Object.keys(r).forEach((f) => { if (typeof r[f] === 'number' && f !== 'page') r[f] = String(r[f]); });
  return r;
}

function parseImport(data) {
  let rows = [];
  if (Array.isArray(data)) rows = data.map((x) => normaliseRow(x));
  else if (Array.isArray(data.rows)) rows = data.rows.map((x) => normaliseRow(x));
  else DOMAINS.forEach((d) => (data[d] || []).forEach((x) => rows.push(normaliseRow(x, d))));
  const doc = (!Array.isArray(data) && (data.document || data.metadata)) || null;
  return { rows, doc };
}

function openImport() {
  $('#imp-error').classList.add('hidden');
  $('#imp-text').value = '';
  $('#imp-file').value = '';
  $('#import-dialog').showModal();
}

async function runImport() {
  const err = $('#imp-error');
  try {
    let text = $('#imp-text').value.trim();
    const file = $('#imp-file').files[0];
    if (file) text = await file.text();
    if (!text) throw new Error('Paste JSON or choose a file.');
    const { rows, doc } = parseImport(JSON.parse(text));
    if (!rows.length) throw new Error('No rows found. Expected "rows": [...] or "targets"/"mitigation"/"adaptation"/"benefits" lists.');
    const source = $('#imp-source').value.trim();
    rows.forEach((r) => { if (!r.source || r.source === 'pipeline') r.source = source || 'import'; });
    const mode = document.querySelector('input[name="imp-mode"]:checked').value;
    if (mode === 'replace') {
      state.rows = rows;
      if (doc) state.document = { ...state.document, ...doc, flags: { ...(state.document.flags || {}), ...(doc.flags || {}) } };
      renderDoc();
    } else {
      state.rows.push(...rows);
    }
    $('#import-dialog').close();
    ui.selected = null;
    changed();
    toast(`${rows.length} rows imported${mode === 'add' ? ` as source "${rows[0].source}"` : ''}.`);
    checkAll();
  } catch (e) {
    err.textContent = `Could not import: ${e.message}`;
    err.classList.remove('hidden');
  }
}

function downloadDraft() {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  const stem = (state.filename || 'review').replace(/\.pdf$/i, '');
  a.href = URL.createObjectURL(blob);
  a.download = `${stem}_review_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function finish() {
  const miss = active().filter((r) => missingOf(r).length).length;
  const pending = active().filter((r) => r.status === 'pending').length;
  const parts = [];
  if (miss) parts.push(`${miss} rows still miss a required field`);
  if (pending) parts.push(`${pending} rows are still pending`);
  if (parts.length && !window.confirm(`${parts.join(' and ')}. They will be exported as they are. Finish anyway?`)) return;
  const btn = $('#btn-finish');
  btn.disabled = true;
  btn.textContent = 'Rebuilding Excel, PDF and summary…';
  try {
    const res = await fetch(CFG.finalizeUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    localStorage.removeItem(LOCAL_KEY);
    if (data.warnings && data.warnings.length) {
      $('#banner').innerHTML = `<div class="banner warn">Results rebuilt with warnings: ${data.warnings.map(esc).join(' ')} <a href="${esc(data.redirect)}">Open results</a></div>`;
      btn.textContent = 'Finish and rebuild results';
      btn.disabled = false;
    } else {
      window.location.href = data.redirect;
    }
  } catch (e) {
    $('#banner').innerHTML = `<div class="banner bad">Could not rebuild the results: ${esc(e.message)}. Your changes are saved in this browser.</div>`;
    btn.textContent = 'Finish and rebuild results';
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- start
async function init() {
  const [srv, taxonomy] = await Promise.all([
    fetch(CFG.stateUrl).then((r) => r.json()),
    fetch(CFG.taxonomyUrl).then((r) => r.json()),
  ]);
  tax = taxonomy;
  state = srv;
  let local = null;
  try { local = JSON.parse(localStorage.getItem(LOCAL_KEY) || 'null'); } catch (e) { local = null; }
  const serverTime = Date.parse(srv.saved_at || srv.created || 0) || 0;
  if (local && local.local_saved && local.local_saved > serverTime + 2000) {
    state = local;
    toast('Restored changes that were only saved in this browser.');
    saveToServer();
  }
  state.rows = (state.rows || []).map((r) => ({ ...newRow(r.domain), ...r }));
  state.document = state.document || {};
  state.document.flags = state.document.flags || {};
  const firstWithRows = DOMAINS.find((d) => state.rows.some((r) => !r.deleted && r.domain === d));
  ui.domain = firstWithRows || 'targets';
  // ?row=<id> from "Fix in validation" on the results page, ?filter=missing from the banner
  const params = new URLSearchParams(window.location.search);
  const wanted = state.rows.find((r) => r.id === params.get('row'));
  if (wanted) { ui.domain = wanted.domain; ui.selected = wanted.id; }
  if (params.get('filter')) {
    ui.filter = params.get('filter');
    if (ui.filter === 'missing') {
      const d = DOMAINS.find((x) => active().some((r) => r.domain === x && missingOf(r).length));
      if (d) ui.domain = d;
    }
  }

  if (state.validated) {
    $('#banner').innerHTML = `<div class="banner ok">This review was finished on ${esc((state.validated_at || '').slice(0, 10))}. You can keep editing; press "Finish and rebuild results" again to update the downloads.</div>`;
  }
  markSaved(state.saved_at ? `Last saved ${new Date(state.saved_at).toLocaleString()}` : 'Not reviewed yet. Changes are saved automatically.');
  renderDoc();
  renderTabs();
  renderFilters();
  renderRows();

  $('#docpanel').addEventListener('toggle', () => { $('#doc-toggle-hint').textContent = $('#docpanel').open ? 'Hide' : 'Show'; });
  $('#btn-import').addEventListener('click', openImport);
  $('#imp-go').addEventListener('click', runImport);
  $('#btn-download').addEventListener('click', downloadDraft);
  $('#btn-finish').addEventListener('click', finish);
  $('#btn-add').addEventListener('click', () => {
    const r = newRow(ui.domain, { page: viewer ? viewer.page : null, edited: true });
    state.rows.push(r); ui.filter = 'all'; ui.selected = r.id; changed();
  });

  if (CFG.hasPdf) {
    const selBtn = $('#btn-from-selection');
    try {
      viewer = await createViewer({
        root: $('#viewer'),
        url: CFG.pdfUrl,
        onSelection: (text, page) => {
          ui.selectionText = text; ui.selectionPage = page;
          selBtn.classList.toggle('hidden', text.length < 10);
        },
      });
      selBtn.addEventListener('click', () => {
        const r = newRow(ui.domain, { verbatim: ui.selectionText, page: ui.selectionPage, pdf_found: true, pdf_page: ui.selectionPage, edited: true });
        state.rows.push(r); ui.filter = 'all'; ui.selected = r.id; selBtn.classList.add('hidden');
        window.getSelection()?.removeAllRanges();
        changed();
      });
      const first = state.rows.find((r) => r.id === ui.selected) || state.rows.find(inTab);
      if (first) showRowInPdf(first); else viewer.goTo(1, []);
      checkAll();
    } catch (e) {
      $('#viewer').innerHTML = `<div class="banner bad">The PDF could not be shown: ${esc(e.message)}</div>`;
    }
  }
}

init().catch((e) => {
  $('#banner').innerHTML = `<div class="banner bad">The validation page could not load: ${esc(e.message)}</div>`;
});
