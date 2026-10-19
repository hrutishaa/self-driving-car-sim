// Recorded frames + train/val split + augmentation (side cams, flips).
// Images are stored already cropped/resized (66x200x3 uint8).
import { IMG_H, IMG_W, IMG_C, SIDE_STEER_CORRECTION, mulberry32 } from './sim.js';

export const IMG_SIZE = IMG_H * IMG_W * IMG_C;

export class Dataset {
  constructor() { this.frames = []; }
  get size() { return this.frames.length; }
  add(frame) { this.frames.push(frame); } // {center, left, right: Uint8Array(66*200*3), steering, speed, t}
  clear() { this.frames = []; }

  // split by frame so the val set never shares a moment with training
  split(valFraction = 0.2, seed = 42, { sideCameras = true } = {}) {
    const rng = mulberry32(seed);
    const idx = this.frames.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nVal = Math.max(1, Math.round(idx.length * valFraction));
    const val = [], train = [];
    idx.forEach((fi, k) => {
      const f = this.frames[fi];
      if (k < nVal) { val.push({ img: f.center, y: f.steering }); return; }
      train.push({ img: f.center, y: f.steering });
      if (sideCameras && f.left && f.right) {
        // left cam looks like the car drifted left, so steer a bit more right
        train.push({ img: f.left, y: clampSteer(f.steering + SIDE_STEER_CORRECTION) });
        train.push({ img: f.right, y: clampSteer(f.steering - SIDE_STEER_CORRECTION) });
      }
    });
    return { train, val };
  }

  steeringHistogram(bins = 21) {
    const h = new Array(bins).fill(0);
    for (const f of this.frames) h[Math.min(bins - 1, Math.max(0, Math.floor((f.steering + 1) / 2 * bins)))]++;
    return h;
  }

  // same column order as Udacity's driving_log.csv
  toCSV() {
    const rows = this.frames.map((f, i) => {
      const n = String(i).padStart(5, '0');
      return `IMG/center_${n}.png,IMG/left_${n}.png,IMG/right_${n}.png,${f.steering.toFixed(5)},${(f.throttle ?? 0).toFixed(4)},0,${(f.speed * 2.237).toFixed(3)}`;
    });
    return rows.join('\n') + '\n';
  }
}

const clampSteer = (v) => Math.max(-1, Math.min(1, v));

export function writeImage(dst, offset, img, flip) {
  if (!flip) { for (let i = 0; i < IMG_SIZE; i++) dst[offset + i] = img[i]; return; }
  const row = IMG_W * IMG_C;
  for (let y = 0; y < IMG_H; y++) {
    const r = y * row;
    for (let x = 0; x < IMG_W; x++) {
      const s = r + (IMG_W - 1 - x) * IMG_C, d = offset + r + x * IMG_C;
      dst[d] = img[s]; dst[d + 1] = img[s + 1]; dst[d + 2] = img[s + 2];
    }
  }
}

// flip half the batch left/right and negate steering
export function makeBatch(samples, indices, rng, augment = true) {
  const n = indices.length;
  const x = new Float32Array(n * IMG_SIZE), y = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const s = samples[indices[k]];
    const flip = augment && rng() < 0.5;
    writeImage(x, k * IMG_SIZE, s.img, flip);
    y[k] = flip ? -s.y : s.y;
  }
  return { x, y, n };
}
