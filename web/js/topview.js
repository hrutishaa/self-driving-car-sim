// top-down map
export class TopView {
  constructor(canvas, track) {
    this.c = canvas; this.ctx = canvas.getContext('2d'); this.track = track;
    this.base = document.createElement('canvas');
    this.resize();
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(this.c.clientWidth * dpr) || 600;
    const h = Math.round(w * 0.7);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
    const b = this.track.bounds, pad = 14 + 8 * dpr;
    this.scale = Math.min((w - 2 * pad) / (b.maxX - b.minX), (h - 2 * pad) / (b.maxY - b.minY));
    this.offX = (w - (b.maxX - b.minX) * this.scale) / 2 - b.minX * this.scale;
    this.offY = (h - (b.maxY - b.minY) * this.scale) / 2 - b.minY * this.scale;
    this.drawBase();
  }
  X(x) { return x * this.scale + this.offX; }
  Y(y) { return y * this.scale + this.offY; }
  path(ctx) {
    const t = this.track; ctx.beginPath();
    for (let i = 0; i <= t.n; i++) { const k = i % t.n; i ? ctx.lineTo(this.X(t.x[k]), this.Y(t.y[k])) : ctx.moveTo(this.X(t.x[k]), this.Y(t.y[k])); }
    ctx.closePath();
  }
  drawBase() {
    const b = this.base; b.width = this.c.width; b.height = this.c.height;
    const ctx = b.getContext('2d'), t = this.track, s = this.scale;
    const g = ctx.createLinearGradient(0, 0, b.width, b.height); g.addColorStop(0, '#173322'); g.addColorStop(1, '#0f2619');
    ctx.fillStyle = g; ctx.fillRect(0, 0, b.width, b.height);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    this.path(ctx); ctx.strokeStyle = '#c93a3f'; ctx.lineWidth = (t.halfWidth + 0.9) * 2 * s; ctx.stroke();
    ctx.setLineDash([1.5 * s, 1.5 * s]); ctx.strokeStyle = '#f0f0f0'; ctx.stroke(); ctx.setLineDash([]);
    this.path(ctx); ctx.strokeStyle = '#4d5363'; ctx.lineWidth = t.halfWidth * 2 * s; ctx.stroke();
    this.path(ctx); ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = Math.max(1, 0.3 * s);
    ctx.save(); ctx.lineWidth = t.halfWidth * 2 * s - 0.7 * s; ctx.strokeStyle = '#eee'; ctx.stroke();
    ctx.lineWidth = t.halfWidth * 2 * s - 1.2 * s; ctx.strokeStyle = '#4d5363'; ctx.stroke(); ctx.restore();
    this.path(ctx); ctx.setLineDash([3 * s, 3 * s]); ctx.strokeStyle = 'rgba(236,200,70,.8)'; ctx.lineWidth = Math.max(1, 0.25 * s); ctx.stroke(); ctx.setLineDash([]);
    // start / finish line
    const p = t.pointAt(0), nx = -Math.sin(p.heading), ny = Math.cos(p.heading);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = Math.max(2, 1.2 * s); ctx.setLineDash([0.8 * s, 0.8 * s]);
    ctx.beginPath(); ctx.moveTo(this.X(p.x - nx * t.halfWidth), this.Y(p.y - ny * t.halfWidth)); ctx.lineTo(this.X(p.x + nx * t.halfWidth), this.Y(p.y + ny * t.halfWidth)); ctx.stroke(); ctx.setLineDash([]);
  }
  draw(sim, trails, takeovers) {
    const ctx = this.ctx, s = this.scale;
    ctx.drawImage(this.base, 0, 0);
    const colors = { manual: '#5aa9ff', expert: '#ffb547', auto: '#36d6c3' };
    ctx.lineWidth = Math.max(1.5, 0.45 * s); ctx.lineCap = 'round';
    for (const seg of trails) {
      if (seg.pts.length < 2) continue;
      ctx.strokeStyle = colors[seg.mode] || '#fff'; ctx.globalAlpha = 0.9; ctx.beginPath();
      seg.pts.forEach(([x, y], i) => (i ? ctx.lineTo(this.X(x), this.Y(y)) : ctx.moveTo(this.X(x), this.Y(y)))); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (const [x, y] of takeovers) { ctx.fillStyle = '#ff5d73'; ctx.beginPath(); ctx.arc(this.X(x), this.Y(y), Math.max(4, 1.6 * s), 0, 7); ctx.fill(); }
    const c = sim.car;
    ctx.save(); ctx.translate(this.X(c.x), this.Y(c.y)); ctx.rotate(c.h);
    const L = Math.max(10, 4.4 * s), W = Math.max(5.5, 1.9 * s);
    ctx.shadowColor = 'rgba(0,0,0,.6)'; ctx.shadowBlur = 6;
    ctx.fillStyle = '#e8333f'; roundRect(ctx, -L / 2, -W / 2, L, W, W * 0.3); ctx.fill();
    ctx.shadowBlur = 0; ctx.fillStyle = '#ffd9dc'; ctx.fillRect(L * 0.1, -W * 0.35, L * 0.18, W * 0.7);
    ctx.restore();
  }
}
function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
