// Track: Catmull-Rom spline through some hand-placed points, resampled every 0.5 m,
// plus a distance field around it. Units are meters.

export const ROAD_HALF_WIDTH = 4.0; // 8 m wide road
export const CURB_WIDTH = 0.9;

const CONTROL_POINTS = [
  [0, 0], [45, -4], [90, 0], [122, 18], [134, 50], [124, 80], [100, 94],
  [76, 94], [56, 104], [42, 124], [18, 138], [-16, 136], [-40, 114],
  [-48, 84], [-64, 62], [-74, 32], [-58, 8], [-28, -2],
].map(([x, y]) => [x * 1.3, y * 1.3]);

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return [
    0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
    0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
  ];
}

export class Track {
  constructor({ controlPoints = CONTROL_POINTS, spacing = 0.5, cell = 0.25 } = {}) {
    this.halfWidth = ROAD_HALF_WIDTH;
    const dense = [];
    const n = controlPoints.length;
    for (let i = 0; i < n; i++) {
      const p0 = controlPoints[(i - 1 + n) % n], p1 = controlPoints[i];
      const p2 = controlPoints[(i + 1) % n], p3 = controlPoints[(i + 2) % n];
      for (let k = 0; k < 200; k++) dense.push(catmull(p0, p1, p2, p3, k / 200));
    }
    // resample at even spacing
    const pts = [dense[0]];
    let acc = 0;
    for (let i = 1; i <= dense.length; i++) {
      const a = dense[i - 1], b = dense[i % dense.length];
      let seg = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let start = 0;
      while (acc + (seg - start) >= spacing) {
        const need = spacing - acc;
        start += need;
        const t = start / seg;
        pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        acc = 0;
      }
      acc += seg - start;
    }
    if (Math.hypot(pts[pts.length - 1][0] - pts[0][0], pts[pts.length - 1][1] - pts[0][1]) < spacing * 0.5) pts.pop();
    this.n = pts.length;
    this.x = new Float64Array(this.n); this.y = new Float64Array(this.n);
    this.s = new Float64Array(this.n + 1);
    for (let i = 0; i < this.n; i++) { this.x[i] = pts[i][0]; this.y[i] = pts[i][1]; }
    for (let i = 0; i < this.n; i++) {
      const j = (i + 1) % this.n;
      this.s[i + 1] = this.s[i] + Math.hypot(this.x[j] - this.x[i], this.y[j] - this.y[i]);
    }
    this.length = this.s[this.n];
    this.tx = new Float64Array(this.n); this.ty = new Float64Array(this.n);
    this.heading = new Float64Array(this.n); this.kappa = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const a = (i - 1 + this.n) % this.n, b = (i + 1) % this.n;
      const dx = this.x[b] - this.x[a], dy = this.y[b] - this.y[a];
      const l = Math.hypot(dx, dy);
      this.tx[i] = dx / l; this.ty[i] = dy / l;
      this.heading[i] = Math.atan2(dy, dx);
    }
    // curvature, positive = turning right (y is down)
    for (let i = 0; i < this.n; i++) {
      const a = (i - 2 + this.n) % this.n, b = (i + 2) % this.n;
      let dh = this.heading[b] - this.heading[a];
      dh = Math.atan2(Math.sin(dh), Math.cos(dh));
      this.kappa[i] = dh / (4 * spacing);
    }
    this.spacing = spacing;
    this._buildField(cell);
  }

  _buildField(cell) {
    const margin = 30;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < this.n; i++) {
      minX = Math.min(minX, this.x[i]); maxX = Math.max(maxX, this.x[i]);
      minY = Math.min(minY, this.y[i]); maxY = Math.max(maxY, this.y[i]);
    }
    this.bounds = { minX, minY, maxX, maxY };
    this.ox = minX - margin; this.oy = minY - margin;
    this.cell = cell;
    this.gw = Math.ceil((maxX - minX + 2 * margin) / cell);
    this.gh = Math.ceil((maxY - minY + 2 * margin) / cell);
    const N = this.gw * this.gh;
    const FAR = 30;
    this.FAR = FAR;
    this.dist = new Float32Array(N).fill(FAR);
    this.along = new Float32Array(N);
    this.idx = new Int32Array(N).fill(-1);
    const reach = 16; // metres around the centreline where we compute exact distances
    const r = Math.ceil(reach / cell);
    for (let i = 0; i < this.n; i++) {
      const j = (i + 1) % this.n;
      const ax = this.x[i], ay = this.y[i], bx = this.x[j], by = this.y[j];
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const cx = Math.floor(((ax + bx) / 2 - this.ox) / cell), cy = Math.floor(((ay + by) / 2 - this.oy) / cell);
      for (let gy = Math.max(0, cy - r); gy <= Math.min(this.gh - 1, cy + r); gy++) {
        const wy = this.oy + (gy + 0.5) * cell;
        for (let gx = Math.max(0, cx - r); gx <= Math.min(this.gw - 1, cx + r); gx++) {
          const wx = this.ox + (gx + 0.5) * cell;
          let t = ((wx - ax) * dx + (wy - ay) * dy) / l2;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const px = ax + dx * t - wx, py = ay + dy * t - wy;
          const d = Math.sqrt(px * px + py * py);
          const k = gy * this.gw + gx;
          if (d < Math.abs(this.dist[k])) {
            // signed so bilinear sampling works across the center line
            const sign = dx * (-py) - dy * (-px) >= 0 ? 1 : -1;
            this.dist[k] = d * sign; this.idx[k] = i; this.along[k] = this.s[i] + t * Math.sqrt(l2);
          }
        }
      }
    }
  }

  sampleDist(wx, wy) { return Math.abs(this.sampleSigned(wx, wy)); }
  sampleSigned(wx, wy) {
    const fx = (wx - this.ox) / this.cell - 0.5, fy = (wy - this.oy) / this.cell - 0.5;
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    if (x0 < 0 || y0 < 0 || x0 >= this.gw - 1 || y0 >= this.gh - 1) return this.FAR;
    const ax = fx - x0, ay = fy - y0, k = y0 * this.gw + x0, d = this.dist;
    const d00 = d[k], d01 = d[k + 1], d10 = d[k + this.gw], d11 = d[k + this.gw + 1];
    // far from the road the field is capped, don't interpolate there
    if (Math.abs(d00) > 10 || Math.abs(d01) > 10 || Math.abs(d10) > 10 || Math.abs(d11) > 10)
      return Math.min(Math.abs(d00), Math.abs(d01), Math.abs(d10), Math.abs(d11));
    return (d00 * (1 - ax) + d01 * ax) * (1 - ay) + (d10 * (1 - ax) + d11 * ax) * ay;
  }
  // distance along the track, falls back to nearest cell at the start/finish seam
  sampleAlong(wx, wy) {
    const fx = (wx - this.ox) / this.cell - 0.5, fy = (wy - this.oy) / this.cell - 0.5;
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    if (x0 < 0 || y0 < 0 || x0 >= this.gw - 1 || y0 >= this.gh - 1) return 0;
    const ax = fx - x0, ay = fy - y0, k = y0 * this.gw + x0, a = this.along;
    const a00 = a[k], a01 = a[k + 1], a10 = a[k + this.gw], a11 = a[k + this.gw + 1];
    if (Math.abs(a00 - a11) > 5 || Math.abs(a01 - a10) > 5 || Math.abs(a00 - a01) > 5) return a[ax < 0.5 ? (ay < 0.5 ? k : k + this.gw) : (ay < 0.5 ? k + 1 : k + this.gw + 1)];
    return (a00 * (1 - ax) + a01 * ax) * (1 - ay) + (a10 * (1 - ax) + a11 * ax) * ay;
  }

  // project onto the center line. d is signed, + = right of the car's direction
  project(wx, wy, hint = -1) {
    let i0 = hint;
    if (i0 < 0) {
      const gx = Math.floor((wx - this.ox) / this.cell), gy = Math.floor((wy - this.oy) / this.cell);
      if (gx >= 0 && gy >= 0 && gx < this.gw && gy < this.gh) i0 = this.idx[gy * this.gw + gx];
    }
    let range = 12;
    if (i0 < 0) { i0 = 0; range = this.n; }
    let best = { d2: Infinity };
    for (let o = -range; o <= range; o++) {
      const i = ((i0 + o) % this.n + this.n) % this.n;
      const j = (i + 1) % this.n;
      const ax = this.x[i], ay = this.y[i], dx = this.x[j] - ax, dy = this.y[j] - ay;
      const l2 = dx * dx + dy * dy;
      let t = ((wx - ax) * dx + (wy - ay) * dy) / l2;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = wx - (ax + dx * t), py = wy - (ay + dy * t);
      const d2 = px * px + py * py;
      if (d2 < best.d2) best = { d2, i, t, px, py, dx, dy, l: Math.sqrt(l2) };
    }
    const cross = best.dx * best.py - best.dy * best.px;
    const d = Math.sqrt(best.d2) * (cross >= 0 ? 1 : -1);
    return { i: best.i, s: this.s[best.i] + best.t * best.l, d, heading: Math.atan2(best.dy, best.dx) };
  }

  pointAt(s) {
    s = ((s % this.length) + this.length) % this.length;
    let lo = 0, hi = this.n;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.s[m] <= s) lo = m; else hi = m; }
    const j = (lo + 1) % this.n;
    const t = (s - this.s[lo]) / (this.s[lo + 1] - this.s[lo]);
    return { x: this.x[lo] + (this.x[j] - this.x[lo]) * t, y: this.y[lo] + (this.y[j] - this.y[lo]) * t, heading: this.heading[lo], i: lo };
  }
  curvatureAt(s) { return this.kappa[this.pointAt(s).i]; }
}
