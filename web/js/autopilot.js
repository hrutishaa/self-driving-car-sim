// Network steers, PI controller does throttle. Used by the app and by the node eval.
import { SpeedController } from './sim.js';

export const PHYSICS_DT = 1 / 60;
export const INFER_EVERY = 4; // 15 network decisions per simulated second

export class Autopilot {
  constructor(sim, predict, opts = {}) {
    this.sim = sim; this.predict = predict; this.speed = new SpeedController(opts.speed);
    this.steer = 0; this.k = 0; this.lastImage = null; this.autoTakeover = opts.autoTakeover ?? true;
  }
  reset() { this.steer = 0; this.k = 0; this.speed.reset(); }
  step(dt = PHYSICS_DT) {
    if (this.k++ % INFER_EVERY === 0) {
      this.lastImage = this.sim.capture(0, this.lastImage || undefined);
      this.steer = Math.max(-1, Math.min(1, this.predict(this.lastImage)));
    }
    const throttle = this.speed.update(this.sim.car.v, this.steer, dt);
    const lapDone = this.sim.step(dt, this.steer, throttle);
    let tookOver = false;
    if (this.autoTakeover && this.sim.offRoad) { this.sim.takeover(); this.speed.reset(); tookOver = true; }
    return { lapDone, tookOver };
  }
}
