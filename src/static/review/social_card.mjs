// Draws the social media card on a canvas and writes the post texts.
const CFG = window.CARD;
const canvas = document.getElementById('card');
const ctx = canvas.getContext('2d');
const C = { bg: '#153743', line: '#2F5564', light: '#CDE7EE', accent: '#83C8D1', green: '#89AE10', blue: '#06B6F6', faint: '#A9CBD5' };
const BLUESKY_LIMIT = 300;
const val = (id) => document.getElementById(id).value.trim();

function points() {
  return [...document.querySelectorAll('#points .point')]
    .map((row) => {
      const [check, text, page] = row.querySelectorAll('input');
      return { on: check.checked, text: text.value.trim(), page: page.value.trim() };
    })
    .filter((p) => p.on && p.text)
    .slice(0, 3);
}

// Word wrap; returns the lines that fit in maxLines (last one with an ellipsis if cut).
function wrap(text, maxWidth, maxLines) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = '';
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (ctx.measureText(test).width > maxWidth && line) { lines.push(line); line = w; } else line = test;
  }
  if (line) lines.push(line);
  if (lines.length > maxLines) {
    const cut = lines.slice(0, maxLines);
    let last = cut[maxLines - 1];
    while (ctx.measureText(`${last}…`).width > maxWidth && last.includes(' ')) last = last.slice(0, last.lastIndexOf(' '));
    cut[maxLines - 1] = `${last}…`;
    return cut;
  }
  return lines;
}

function draw() {
  const square = document.querySelector('input[name="fmt"]:checked').value === 'square';
  const W = square ? 1080 : 1200;
  const H = square ? 1080 : 675;
  canvas.width = W; canvas.height = H;
  const pad = 60;
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  ctx.textBaseline = 'alphabetic';

  // top line
  ctx.font = '500 15px "IBM Plex Mono", monospace';
  ctx.fillStyle = C.accent;
  ctx.fillText('NDC TRANSPORT TRACKER: NEW SUBMISSION', pad, 72);
  const stamp = CFG.validated ? 'Reviewed by an analyst' : 'Preliminary: AI-extracted, not reviewed';
  ctx.font = '600 15px Inter, sans-serif';
  const sw = ctx.measureText(stamp).width + 28;
  ctx.fillStyle = CFG.validated ? C.green : '#F2C46B';
  roundRect(W - pad - sw, 48, sw, 34, 17); ctx.fill();
  ctx.fillStyle = '#0C2730';
  ctx.fillText(stamp, W - pad - sw + 14, 71);

  // heading
  ctx.fillStyle = '#fff';
  ctx.font = '700 64px Inter, sans-serif';
  ctx.fillText(val('f-country') || 'Country', pad, 160);
  ctx.font = '400 22px Inter, sans-serif';
  ctx.fillStyle = C.light;
  ctx.fillText(val('f-sub'), pad, 198);

  // layout: points left, chart right (landscape) or below (square)
  const showAsi = document.getElementById('f-asi').checked && (CFG.asi.Avoid + CFG.asi.Shift + CFG.asi.Improve) > 0;
  const top = 250;
  const footY = H - 40;
  const pointsW = showAsi && !square ? 620 : W - 2 * pad;
  let y = top;
  ctx.font = '400 24px Inter, sans-serif';
  for (const p of points()) {
    const lines = wrap(p.text, pointsW - 60, 2);
    ctx.fillStyle = C.green;
    ctx.fillRect(pad, y - 16, 10, 10);
    ctx.fillStyle = '#fff';
    ctx.font = '400 24px Inter, sans-serif';
    lines.forEach((l, i) => ctx.fillText(l, pad + 28, y + i * 32));
    if (p.page) {
      const lastW = ctx.measureText(lines[lines.length - 1]).width;
      ctx.font = '400 16px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.accent;
      ctx.fillText(p.page, pad + 28 + lastW + 10, y + (lines.length - 1) * 32);
    }
    y += lines.length * 32 + 22;
  }

  if (showAsi) {
    const x0 = square ? pad : pad + pointsW + 50;
    const y0 = square ? Math.max(y + 20, 620) : top - 20;
    if (!square) { ctx.fillStyle = C.line; ctx.fillRect(x0 - 30, top - 40, 1, 300); }
    ctx.fillStyle = C.light;
    ctx.font = '400 16px Inter, sans-serif';
    ctx.fillText('Transport mitigation passages by approach', x0, y0);
    const max = Math.max(CFG.asi.Avoid, CFG.asi.Shift, CFG.asi.Improve, 1);
    const barW = (square ? W - 2 * pad : W - pad - x0) - 140;
    [['Avoid', C.accent], ['Shift', C.blue], ['Improve', C.green]].forEach(([k, col], i) => {
      const by = y0 + 36 + i * 44;
      ctx.fillStyle = '#fff'; ctx.font = '400 19px Inter, sans-serif';
      ctx.fillText(k, x0, by + 14);
      ctx.fillStyle = '#1F4756'; roundRect(x0 + 90, by, barW, 18, 3); ctx.fill();
      ctx.fillStyle = col; roundRect(x0 + 90, by, Math.max(4, barW * CFG.asi[k] / max), 18, 3); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '600 19px Inter, sans-serif';
      ctx.fillText(String(CFG.asi[k]), x0 + 100 + barW, by + 15);
    });
  }

  // footer
  ctx.fillStyle = C.line; ctx.fillRect(pad, footY - 34, W - 2 * pad, 1);
  ctx.font = '400 14px Inter, sans-serif'; ctx.fillStyle = C.faint;
  const pages = [...new Set(points().map((p) => p.page).filter(Boolean))].join(', ');
  ctx.fillText(`Source: ${val('f-sub')}${pages ? `, ${pages}` : ''}. Extracted with the Transport Policy Miner.`, pad, footY);
  ctx.font = '600 14px Inter, sans-serif'; ctx.fillStyle = '#fff';
  const brand = `GIZ, SLOCAT  |  ${CFG.link}`;
  ctx.fillText(brand, W - pad - ctx.measureText(brand).width, footY);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

let userEdited = { 't-linkedin': false, 't-bluesky': false, 't-alt': false };
function texts() {
  const country = val('f-country') || 'This country';
  const doc = val('f-sub');
  const pts = points();
  const pre = CFG.validated ? '' : 'Preliminary, AI-extracted: ';
  const linkedin = `${pre}${country} has submitted ${doc}. On transport:\n\n${pts.map((p) => `- ${p.text}${p.page ? ` (${p.page})` : ''}`).join('\n')}\n\nFull analysis in the NDC Transport Tracker (GIZ, SLOCAT): ${CFG.link}`;
  let bluesky = `${pre}${country}, ${doc}: ${pts.map((p) => p.text.replace(/\.$/, '')).join('; ')}. More: ${CFG.link}`;
  if (bluesky.length > BLUESKY_LIMIT) bluesky = `${pre}${country}, ${doc}: ${pts.slice(0, 2).map((p) => p.text.replace(/\.$/, '')).join('; ')}. More: ${CFG.link}`;
  const alt = `Card from the NDC Transport Tracker on ${country}, ${doc}. ${CFG.validated ? 'Reviewed by an analyst.' : 'Preliminary, AI-extracted and not reviewed.'} Key points: ${pts.map((p) => p.text).join('; ')}.`;
  if (!userEdited['t-linkedin']) document.getElementById('t-linkedin').value = linkedin;
  if (!userEdited['t-bluesky']) document.getElementById('t-bluesky').value = bluesky;
  if (!userEdited['t-alt']) document.getElementById('t-alt').value = alt;
  count();
}

function count() {
  const n = [...new Intl.Segmenter().segment(document.getElementById('t-bluesky').value)].length;
  const el = document.getElementById('bsky-count');
  el.textContent = `${n} / ${BLUESKY_LIMIT} characters`;
  el.classList.toggle('over', n > BLUESKY_LIMIT);
}

function update() { draw(); texts(); }

document.querySelectorAll('input').forEach((el) => el.addEventListener('input', update));
document.querySelectorAll('input[type=radio], input[type=checkbox]').forEach((el) => el.addEventListener('change', update));
Object.keys(userEdited).forEach((id) => document.getElementById(id).addEventListener('input', () => { userEdited[id] = true; count(); }));
document.querySelectorAll('[data-copy]').forEach((b) => b.addEventListener('click', async () => {
  await navigator.clipboard.writeText(document.getElementById(b.dataset.copy).value);
  const old = b.textContent; b.textContent = 'Copied'; setTimeout(() => { b.textContent = old; }, 1500);
}));
document.getElementById('b-png').addEventListener('click', () => {
  const a = document.createElement('a');
  a.download = `${(val('f-country') || 'ndc').replace(/\W+/g, '_')}_tracker_card.png`;
  a.href = canvas.toDataURL('image/png');
  a.click();
});

document.fonts.ready.then(update);
update();
