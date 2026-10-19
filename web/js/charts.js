// simple canvas charts, didn't want a chart lib for this
function setup(canvas, hRatio) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round((canvas.clientWidth || canvas.width) * dpr), h = Math.round(w * hRatio);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

export function drawLossChart(canvas, batchLosses, epochs) {
  const { ctx, w, h, dpr } = setup(canvas, 0.5);
  const padL = 44 * dpr, padR = 10 * dpr, padT = 22 * dpr, padB = 24 * dpr;
  ctx.font = `${11 * dpr}px system-ui, sans-serif`;
  const all = [...batchLosses.map((b) => b.ema), ...epochs.flatMap((e) => [e.loss, e.valLoss])].filter((v) => v > 0 && isFinite(v));
  if (!all.length) { ctx.fillStyle = '#8b97bd'; ctx.fillText('loss shows up here during training', padL, h / 2); return; }
  let lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
  if (hi - lo < 0.5) { const m = (hi + lo) / 2; lo = m - 0.25; hi = m + 0.25; }
  lo = Math.floor(lo * 2) / 2; hi = Math.ceil(hi * 2) / 2;
  const total = Math.max(1, batchLosses.length ? batchLosses[batchLosses.length - 1].x : 1, epochs.length);
  const X = (x) => padL + (x / total) * (w - padL - padR);
  const Y = (v) => padT + (1 - (Math.log10(v) - lo) / (hi - lo)) * (h - padT - padB);
  ctx.strokeStyle = '#263256'; ctx.fillStyle = '#8b97bd'; ctx.lineWidth = 1;
  for (let e = lo; e <= hi + 1e-9; e += 0.5) { const y = Y(10 ** e); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.fillText((10 ** e).toExponential(0), 4 * dpr, y + 4 * dpr); }
  for (let e = 1; e <= Math.ceil(total); e++) { const x = X(e); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke(); ctx.fillText(String(e), x - 3 * dpr, h - 8 * dpr); }
  // smoothed per-batch loss
  ctx.strokeStyle = 'rgba(90,169,255,.55)'; ctx.lineWidth = 1.2 * dpr; ctx.beginPath();
  batchLosses.forEach((b, i) => (i ? ctx.lineTo(X(b.x), Y(b.ema)) : ctx.moveTo(X(b.x), Y(b.ema)))); ctx.stroke();
  const line = (key, color) => {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2 * dpr; ctx.beginPath();
    epochs.forEach((e, i) => (i ? ctx.lineTo(X(e.epoch), Y(e[key])) : ctx.moveTo(X(e.epoch), Y(e[key])))); ctx.stroke();
    epochs.forEach((e) => { ctx.beginPath(); ctx.arc(X(e.epoch), Y(e[key]), 3.5 * dpr, 0, 7); ctx.fill(); });
  };
  line('loss', '#5aa9ff'); line('valLoss', '#ffb547');
  ctx.font = `${11.5 * dpr}px system-ui, sans-serif`;
  ctx.fillStyle = '#5aa9ff'; ctx.fillText('train loss', padL, 14 * dpr);
  ctx.fillStyle = '#ffb547'; ctx.fillText('val loss', padL + 90 * dpr, 14 * dpr);
  ctx.fillStyle = '#8b97bd'; ctx.fillText('epoch', w - padR - 50 * dpr, 14 * dpr);
}

export function drawHistogram(canvas, hist) {
  const { ctx, w, h, dpr } = setup(canvas, 0.22);
  const max = Math.max(1, ...hist), bw = w / hist.length;
  hist.forEach((v, i) => {
    const bh = (v / max) * (h - 18 * dpr);
    const c = i === Math.floor(hist.length / 2) ? '#ffb547' : '#36d6c3';
    ctx.fillStyle = c; ctx.globalAlpha = 0.85; ctx.fillRect(i * bw + 1, h - 14 * dpr - bh, bw - 2, bh);
  });
  ctx.globalAlpha = 1; ctx.fillStyle = '#8b97bd'; ctx.font = `${10 * dpr}px system-ui`;
  ctx.fillText('-1 (left)', 2, h - 2 * dpr); ctx.fillText('0', w / 2 - 3 * dpr, h - 2 * dpr); ctx.fillText('+1 (right)', w - 52 * dpr, h - 2 * dpr);
}

export function drawLine(canvas, values, { target, label } = {}) {
  const { ctx, w, h, dpr } = setup(canvas, 0.33);
  const pad = 26 * dpr; const all = [...values, target ?? values[0]];
  const lo = Math.min(...all) - 0.1, hi = Math.max(...all) + 0.1;
  const X = (i) => pad + (i / Math.max(1, values.length - 1 || 1)) * (w - 2 * pad), Y = (v) => pad / 2 + (1 - (v - lo) / (hi - lo)) * (h - pad * 1.5);
  if (target !== undefined) { ctx.setLineDash([5, 5]); ctx.strokeStyle = '#ffb547'; ctx.beginPath(); ctx.moveTo(pad, Y(target)); ctx.lineTo(w - pad, Y(target)); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#ffb547'; ctx.font = `${10.5 * dpr}px system-ui`; ctx.fillText(label || '', w - pad - 120 * dpr, Y(target) - 5 * dpr); }
  ctx.strokeStyle = '#36d6c3'; ctx.fillStyle = '#36d6c3'; ctx.lineWidth = 2 * dpr; ctx.beginPath();
  values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
  values.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 4 * dpr, 0, 7); ctx.fill(); ctx.fillText(`${i}`, X(i) - 3 * dpr, h - 4 * dpr); });
}
