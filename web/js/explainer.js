// "How it works" tab
import { drawLine } from './charts.js';

const fmt = (v) => { const r = Math.round(v * 1000) / 1000; return (Object.is(r, -0) ? 0 : r).toLocaleString('en-US', { maximumFractionDigits: 3 }); };

export function initExplainer() {
  // edge filter
  const row = document.getElementById('edgeRow'), filt = document.getElementById('edgeFilter'), out = document.getElementById('edgeOut');
  const cell = (v, max) => { const t = Math.max(0, Math.min(1, Math.abs(v) / (max || 1))); const bg = v < 0 ? `rgba(255,93,115,${0.15 + 0.7 * t})` : `rgba(54,214,195,${0.12 + 0.75 * t})`; return `<div class="cell" style="background:${bg}">${fmt(v)}</div>`; };
  const bright = (v) => `<div class="cell" style="background:rgb(${20 + v * 28},${20 + v * 28},${24 + v * 28});color:${v > 4 ? '#111' : '#fff'}">${fmt(v)}</div>`;
  function edge() {
    const r = row.value.trim().split(/[\s,]+/).map(Number).filter((v) => !isNaN(v));
    const f = filt.value.trim().split(/[\s,]+/).map(Number).filter((v) => !isNaN(v));
    const o = [];
    for (let i = 0; i + f.length <= r.length; i++) o.push(f.reduce((s, w, k) => s + w * r[i + k], 0));
    const m = Math.max(1, ...o.map(Math.abs));
    out.innerHTML = `<div class="cellrow"><span class="lbl">input</span>${r.map(bright).join('')}</div>
      <div class="cellrow"><span class="lbl">filter</span>${f.map((v) => cell(v, 1)).join('')}</div>
      <div class="cellrow"><span class="lbl">output</span>${o.map((v) => cell(v, m)).join('')}</div>`;
  }
  row.addEventListener('input', edge); filt.addEventListener('input', edge); edge();

  // toy gradient descent: w=-1, lr=0.001, x=-20, target=-15
  const pic = [[8, 8, 1], [8, 8, 1], [8, 8, 2]];
  document.getElementById('grid3').innerHTML = pic.flat().map((v) => `<div style="background:${v > 4 ? '#6b7180' : '#2f7a3a'};color:#fff">${v}</div>`).join('');
  const x = pic.reduce((s, r) => s + r[2], 0) - pic.reduce((s, r) => s + r[0], 0); // 4 - 24 = -20
  const target = -15;
  const wIn = document.getElementById('gdW'), lrIn = document.getElementById('gdLr'), tbody = document.getElementById('gdTable');
  let ws = [];
  function render(highlight) {
    tbody.innerHTML = ws.map((w, i) => {
      const guess = w * x, miss = guess - target, slope = 2 * miss * x;
      return `<tr class="${highlight && i === ws.length - 1 ? 'new' : ''}"><td>${i === 0 ? 'start' : i}</td><td>${fmt(w)}</td><td>${guess > 0 ? '+' : ''}${fmt(guess)}</td><td>${fmt(miss)}</td><td>${fmt(miss * miss)}</td><td>${fmt(slope)}</td></tr>`;
    }).join('');
    drawLine(document.getElementById('gdChart'), ws.map((w) => w * x), { target, label: 'target -15' });
  }
  function reset() { ws = [Number(wIn.value)]; render(false); }
  document.getElementById('gdStep').addEventListener('click', () => {
    const w = ws[ws.length - 1], lr = Number(lrIn.value);
    const slope = 2 * (w * x - target) * x;
    if (ws.length < 12) ws.push(w - lr * slope);
    render(true);
  });
  document.getElementById('gdReset').addEventListener('click', reset);
  wIn.addEventListener('change', reset); lrIn.addEventListener('change', reset);
  reset();

  // autonomy calculator
  const t = document.getElementById('calcT'), s = document.getElementById('calcS'), o = document.getElementById('calcOut');
  const calc = () => { o.textContent = `${fmt(Math.max(0, (1 - (Number(t.value) * 6) / Math.max(1e-6, Number(s.value))) * 100))}%`; };
  t.addEventListener('input', calc); s.addEventListener('input', calc); calc();
  return { stepTo(n) { reset(); for (let i = 0; i < n; i++) document.getElementById('gdStep').click(); } };
}

// (-1, +1) filter on the live camera image
export function edgeImage(img, ctx) {
  const W = 200, H = 66, id = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 3, j = (y * W + Math.min(W - 1, x + 1)) * 3;
    const a = (img[i] + img[i + 1] + img[i + 2]) / 3, b = (img[j] + img[j + 1] + img[j + 2]) / 3;
    const v = Math.min(255, Math.abs(b - a) * 3), o = (y * W + x) * 4;
    id.data[o] = v * 0.4; id.data[o + 1] = v; id.data[o + 2] = v * 0.9; id.data[o + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}
