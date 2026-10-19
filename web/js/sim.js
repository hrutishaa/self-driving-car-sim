// Car, camera renderer, preprocessing, expert driver, speed controller, lap tracking.
// No DOM in here so node can use it too.

export const CAM_W = 320, CAM_H = 160;          // raw "camera" image (same size as Udacity's sim)
export const CROP_TOP = 60, CROP_BOTTOM = 25;    // sky / hood rows removed (same as naokishibuya)
export const IMG_H = 66, IMG_W = 200, IMG_C = 3; // network input (Nvidia, Bojarski et al. 2016)
export const SIDE_CAM_OFFSET = 1.0;              // metres left/right of centre camera
export const SIDE_STEER_CORRECTION = 0.2;        // steering nudge for side cameras
export const MAX_STEER_DEG = 25;
export const WHEELBASE = 2.6;
export const CAR_HALF_WIDTH = 0.9;
export const TAKEOVER_PENALTY_S = 6;             // Nvidia autonomy formula

const MAX_STEER = MAX_STEER_DEG * Math.PI / 180;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// kinematic bicycle model, with a crude grip limit
export class Car {
  constructor() { this.reset(0, 0, 0); }
  reset(x, y, heading, v = 0) { this.x = x; this.y = y; this.h = heading; this.v = v; this.steer = 0; this.throttle = 0; }
  step(dt, steerCmd, throttleCmd, offRoad = false) {
    const rate = 3.5 * dt; // steering actuator rate limit (units / s)
    this.steer += clamp(clamp(steerCmd, -1, 1) - this.steer, -rate, rate);
    this.throttle = clamp(throttleCmd, -1, 1);
    const accel = this.throttle >= 0 ? this.throttle * 5.5 : this.throttle * 11;
    const drag = 0.25 + 0.012 * this.v * this.v + (offRoad ? 2.5 : 0);
    this.v = Math.max(0, this.v + (accel - (this.v > 0 ? drag : 0)) * dt);
    let k = Math.tan(this.steer * MAX_STEER) / WHEELBASE; // path curvature
    const kGrip = 8.5 / Math.max(1, this.v * this.v);      // lateral grip ~0.87 g
    k = clamp(k, -kGrip, kGrip);
    this.h = wrap(this.h + this.v * k * dt);
    this.x += Math.cos(this.h) * this.v * dt;
    this.y += Math.sin(this.h) * this.v * dt;
  }
}

// Dashcam: project each pixel onto the ground and color it from the distance field.
// Writes RGBA into out.
function hash2(ix, iy) {
  let n = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) & 1023) / 1023;
}
const HFOV = 80 * Math.PI / 180, CAM_HEIGHT = 1.45;

export function renderCamera(track, cx, cy, heading, out, W = CAM_W, H = CAM_H, rowStart = 0, rowEnd = H) {
  const f = (W / 2) / Math.tan(HFOV / 2);
  const horizon = H * (CROP_TOP / CAM_H) - 0.5 * (H / CAM_H) * 4; // horizon a few rows above the crop line
  const fx = Math.cos(heading), fy = Math.sin(heading);
  const rx = -fy, ry = fx; // right vector (y-down world)
  const hw = track.halfWidth;
  const hoodTop = H * 0.865;
  for (let y = rowStart; y < rowEnd; y++) {
    let o = y * W * 4;
    const dy = y + 0.5 - horizon;
    if (dy <= 0.6) {
      // sky and hills
      const t = clamp((y) / horizon, 0, 1);
      const sr = 92 + 95 * t, sg = 148 + 70 * t, sb = 214 + 22 * t;
      for (let x = 0; x < W; x++, o += 4) {
        const az = heading + Math.atan((x + 0.5 - W / 2) / f);
        const hill = (6 + 4 * Math.sin(az * 3 + 1.3) + 3 * Math.sin(az * 7 + 0.2) + 1.6 * Math.sin(az * 17)) * (H / CAM_H);
        const tree = (1.8 + 1.4 * Math.sin(az * 61) * Math.sin(az * 23 + 2)) * (H / CAM_H);
        if (horizon - y < tree) { out[o] = 52; out[o + 1] = 96; out[o + 2] = 58; }
        else if (horizon - y < hill) { out[o] = 120 + 0.25 * sr; out[o + 1] = 140 + 0.2 * sg; out[o + 2] = 168 + 0.15 * sb; }
        else { out[o] = sr; out[o + 1] = sg; out[o + 2] = sb; }
        out[o + 3] = 255;
      }
      continue;
    }
    const z = CAM_HEIGHT * f / dy;               // forward distance of this row on the ground
    const fog = clamp(Math.pow(z / 140, 1.25), 0, 1);
    const lx0 = (0.5 - W / 2) / f * z, ldx = z / f; // lateral offset at column 0 and per column
    let wx = cx + fx * z + rx * lx0, wy = cy + fy * z + ry * lx0;
    const sx = rx * ldx, sy = ry * ldx;
    const fine = z < 25; // texture detail only up close (reduces shimmer)
    for (let x = 0; x < W; x++, o += 4, wx += sx, wy += sy) {
      if (y >= hoodTop + Math.pow((x - W / 2) / (W / 2), 2) * H * 0.07) {
        // hood (cropped before the network sees it)
        const g = 1 - (y - hoodTop) / (H - hoodTop);
        out[o] = 150 + 60 * g; out[o + 1] = 28 + 20 * g; out[o + 2] = 36 + 20 * g; out[o + 3] = 255; continue;
      }
      const d = track.sampleDist(wx, wy);
      let r, g, b;
      if (d < hw) {
        const n = fine ? hash2(Math.floor(wx * 3), Math.floor(wy * 3)) * 9 : 4.5;
        r = g = 84 + n; b = 90 + n;
        if (d > hw - 0.5 && d < hw - 0.22) { r = g = b = 236; }
        else if (d < 0.13) {
          const s = track.sampleAlong(wx, wy);
          if (s % 6 < 3) { r = 236; g = 200; b = 70; }
        }
      } else if (d < hw + 0.9) {
        const s = track.sampleAlong(wx, wy);
        if (Math.floor(s / 1.5) & 1) { r = 206; g = 42; b = 46; } else { r = g = b = 240; }
      } else {
        const coarse = hash2(Math.floor(wx / 4), Math.floor(wy / 4));
        const n = fine ? hash2(Math.floor(wx * 2), Math.floor(wy * 2)) * 18 : 9;
        const edge = d < hw + 2.2 ? 0.82 : 1; // darker strip just beyond the curb
        r = (58 + coarse * 18 + n) * edge; g = (122 + coarse * 26 + n) * edge; b = (52 + coarse * 10) * edge;
      }
      out[o] = r + (196 - r) * fog; out[o + 1] = g + (214 - g) * fog; out[o + 2] = b + (232 - b) * fog; out[o + 3] = 255;
    }
  }
  return out;
}

// crop + resize to 66x200 RGB. used for recording and driving, must stay identical
export function preprocess(rgba, W = CAM_W, H = CAM_H, out = new Uint8Array(IMG_H * IMG_W * IMG_C)) {
  const top = Math.round(CROP_TOP * H / CAM_H), bottom = H - Math.round(CROP_BOTTOM * H / CAM_H);
  const ch = bottom - top;
  const syScale = ch / IMG_H, sxScale = W / IMG_W;
  let o = 0;
  for (let y = 0; y < IMG_H; y++) {
    const fy = top + (y + 0.5) * syScale - 0.5;
    const y0 = clamp(Math.floor(fy), top, bottom - 1), y1 = Math.min(y0 + 1, bottom - 1), ay = fy - y0;
    for (let x = 0; x < IMG_W; x++) {
      const fxx = (x + 0.5) * sxScale - 0.5;
      const x0 = clamp(Math.floor(fxx), 0, W - 1), x1 = Math.min(x0 + 1, W - 1), ax = clamp(fxx - x0, 0, 1);
      const i00 = (y0 * W + x0) * 4, i01 = (y0 * W + x1) * 4, i10 = (y1 * W + x0) * 4, i11 = (y1 * W + x1) * 4;
      for (let c = 0; c < 3; c++) {
        const top_ = rgba[i00 + c] * (1 - ax) + rgba[i01 + c] * ax;
        const bot_ = rgba[i10 + c] * (1 - ax) + rgba[i11 + c] * ax;
        out[o++] = top_ * (1 - ay) + bot_ * ay + 0.5;
      }
    }
  }
  return out;
}

// pure pursuit on the center line, slows down for curves
export class Expert {
  constructor(track, { vmax = 15, aLat = 4.2 } = {}) { this.track = track; this.vmax = vmax; this.aLat = aLat; }
  steering(car, proj) {
    const Ld = 5 + 0.42 * car.v;
    const p = this.track.pointAt(proj.s + Ld);
    const alpha = wrap(Math.atan2(p.y - car.y, p.x - car.x) - car.h);
    const delta = Math.atan(2 * WHEELBASE * Math.sin(alpha) / Ld);
    return clamp(delta / MAX_STEER, -1, 1);
  }
  targetSpeed(proj) {
    let kmax = 0;
    for (let a = 0; a <= 45; a += 3) kmax = Math.max(kmax, Math.abs(this.track.curvatureAt(proj.s + a)));
    return Math.min(this.vmax, Math.sqrt(this.aLat / Math.max(kmax, 1e-4)));
  }
  throttle(car, proj) { return clamp(0.6 * (this.targetSpeed(proj) - car.v), -1, 1); }
}

// random steering kicks while recording expert laps, gives us recovery data
export class Disturbance {
  constructor(seed = 1, strength = 1) { this.rng = mulberry32(seed); this.n = 0; this.kick = 0; this.kickT = 0; this.next = 2; this.t = 0; this.strength = strength; }
  step(dt) {
    this.t += dt;
    const g = gauss(this.rng);
    this.n += -this.n * dt / 0.6 + 0.16 * Math.sqrt(dt) * g;
    if (this.t > this.next) { this.kick = (this.rng() < 0.5 ? -1 : 1) * (0.12 + 0.2 * this.rng()); this.kickT = 0.35 + 0.6 * this.rng(); this.next = this.t + 2.5 + 4 * this.rng(); }
    let k = 0;
    if (this.kickT > 0) { this.kickT -= dt; k = this.kick; }
    return (this.n + k) * this.strength;
  }
}

// PI speed controller for autonomous mode
export class SpeedController {
  constructor({ kp = 0.35, ki = 0.06, maxSpeed = 13, minSpeed = 7 } = {}) { Object.assign(this, { kp, ki, maxSpeed, minSpeed }); this.integral = 0; }
  target(steer) { return clamp(this.maxSpeed - 22 * Math.abs(steer), this.minSpeed, this.maxSpeed); }
  update(speed, steer, dt) {
    const err = this.target(steer) - speed;
    this.integral = clamp(this.integral + err * dt, -20, 20);
    return clamp(this.kp * err + this.ki * this.integral, -1, 1);
  }
  reset() { this.integral = 0; }
}

export function mulberry32(a) {
  return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }


export class Simulator {
  constructor(track) {
    this.track = track;
    this.car = new Car();
    this.expert = new Expert(track);
    this._rgba = new Uint8ClampedArray(CAM_W * CAM_H * 4);
    this.reset();
  }
  reset(s = 0) {
    const p = this.track.pointAt(s);
    this.car.reset(p.x, p.y, p.heading, 0);
    this.proj = this.track.project(p.x, p.y);
    this.time = 0; this.progress = 0; this.laps = 0; this.lapStart = 0; this.lapTimes = [];
    this.takeovers = 0; this.lapTakeovers = 0; this.lastLapTakeovers = []; this.offRoadEvents = 0;
    this.maxAbsD = 0; this.lapMaxAbsD = 0; this.lapMaxAbsDs = [];
    this.trail = [];
  }
  get offRoad() { return Math.abs(this.proj.d) > this.track.halfWidth; }
  get wheelsOff() { return Math.abs(this.proj.d) + CAR_HALF_WIDTH > this.track.halfWidth; }

  step(dt, steer, throttle) {
    const onGrass = Math.abs(this.proj.d) > this.track.halfWidth + 0.9;
    this.car.step(dt, steer, throttle, onGrass);
    const prevS = this.proj.s;
    this.proj = this.track.project(this.car.x, this.car.y, this.proj.i);
    let ds = this.proj.s - prevS;
    if (ds < -this.track.length / 2) ds += this.track.length;
    if (ds > this.track.length / 2) ds -= this.track.length;
    this.progress += ds;
    this.time += dt;
    const ad = Math.abs(this.proj.d);
    this.maxAbsD = Math.max(this.maxAbsD, ad); this.lapMaxAbsD = Math.max(this.lapMaxAbsD, ad);
    let lapDone = false;
    if (this.progress >= (this.laps + 1) * this.track.length) {
      this.laps++; lapDone = true;
      this.lapTimes.push(this.time - this.lapStart); this.lapStart = this.time;
      this.lastLapTakeovers.push(this.lapTakeovers); this.lapTakeovers = 0;
      this.lapMaxAbsDs.push(this.lapMaxAbsD); this.lapMaxAbsD = 0;
    }
    if (this.trail.length === 0 || Math.hypot(this.trail[this.trail.length - 1][0] - this.car.x, this.trail[this.trail.length - 1][1] - this.car.y) > 1.5) {
      this.trail.push([this.car.x, this.car.y]); if (this.trail.length > 600) this.trail.shift();
    }
    return lapDone;
  }

  // reset to the center line, like a safety driver taking over
  takeover() {
    this.takeovers++; this.lapTakeovers++;
    const p = this.track.pointAt(this.proj.s);
    this.car.reset(p.x, p.y, p.heading, Math.min(this.car.v, 6));
    this.proj = this.track.project(p.x, p.y, this.proj.i);
  }

  // (1 - takeovers * 6 / seconds) * 100, from the Nvidia paper
  autonomy(takeovers = this.takeovers, seconds = this.time) {
    if (seconds <= 0) return 100;
    return Math.max(0, (1 - (takeovers * TAKEOVER_PENALTY_S) / seconds) * 100);
  }

  // lateral in meters, + is right. only renders the rows that survive the crop
  capture(lateral = 0, out) {
    const c = this.car, rx = -Math.sin(c.h), ry = Math.cos(c.h);
    const rowStart = CROP_TOP, rowEnd = CAM_H - CROP_BOTTOM; // only the rows that survive the crop
    renderCamera(this.track, c.x + rx * lateral, c.y + ry * lateral, c.h, this._rgba, CAM_W, CAM_H, rowStart, rowEnd);
    return preprocess(this._rgba, CAM_W, CAM_H, out);
  }
  captureAll() {
    return { center: this.capture(0), left: this.capture(-SIDE_CAM_OFFSET), right: this.capture(SIDE_CAM_OFFSET) };
  }
}
