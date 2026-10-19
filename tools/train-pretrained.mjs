// Records laps with the scripted driver, trains the model with tfjs-node and saves it
// to web/model/ so the demo has something to drive with.
//   node tools/train-pretrained.mjs [--laps 8] [--epochs 10]
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as tf from '@tensorflow/tfjs-node';
import { Track } from '../web/js/track.js';
import { Simulator, Disturbance } from '../web/js/sim.js';
import { Dataset } from '../web/js/data.js';
import { buildNvidiaModel, trainModel, predictSteering } from '../web/js/model.js';
import { Autopilot, PHYSICS_DT } from '../web/js/autopilot.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? Number(process.argv[i + 1]) : d; };
const LAPS = arg('laps', 6), EPOCHS = arg('epochs', 8), EVAL_LAPS = arg('eval', 3);

export function recordExpertLaps(sim, dataset, { laps = 1, seed = 1, strength = 1, vmax = 14, recordEvery = 6 } = {}) {
  const dist = new Disturbance(seed, strength);
  sim.expert.vmax = vmax;
  sim.reset((seed * 97) % sim.track.length);
  let k = 0;
  while (sim.laps < laps) {
    const label = sim.expert.steering(sim.car, sim.proj);
    const throttle = sim.expert.throttle(sim.car, sim.proj);
    if (k++ % recordEvery === 0) {
      const cams = sim.captureAll();
      dataset.add({ ...cams, steering: label, throttle, speed: sim.car.v, t: sim.time });
    }
    sim.step(PHYSICS_DT, label + dist.step(PHYSICS_DT), throttle); // label stays clean, the car gets the noise
    if (sim.offRoad) sim.takeover();
  }
}

async function main() {
  const track = new Track();
  const sim = new Simulator(track);
  const ds = new Dataset();
  const t0 = Date.now();
  const strengths = [0, 0.8, 1.2, 1.6, 2.0, 1.0, 1.4, 1.8];
  for (let l = 0; l < LAPS; l++) {
    recordExpertLaps(sim, ds, { laps: 1, seed: 11 + l, strength: strengths[l % strengths.length], vmax: 12 + (l % 4) });
    console.log(`lap ${l + 1}/${LAPS}: ${ds.size} frames`);
  }
  console.log(`data generated in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  const split = ds.split(0.2, 42);
  console.log(`train samples ${split.train.length} (centre+left+right, flipped on the fly), val ${split.val.length}`);

  const model = buildNvidiaModel(tf);
  model.summary();
  const history = await trainModel(tf, model, split, {
    epochs: EPOCHS, batchSize: 64,
    onEpoch: (r) => console.log(`epoch ${r.epoch}: loss ${r.loss.toFixed(5)} val ${r.valLoss.toFixed(5)}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`),
  });

  // drive a few laps with it
  const evalSim = new Simulator(track);
  const ap = new Autopilot(evalSim, (img) => predictSteering(tf, model, img));
  while (evalSim.laps < EVAL_LAPS && evalSim.time < 400) ap.step();
  const evalRes = {
    laps: evalSim.laps, lapTimes: evalSim.lapTimes.map((t) => +t.toFixed(2)), takeovers: evalSim.takeovers,
    maxAbsOffset: +evalSim.maxAbsD.toFixed(3), autonomy: +evalSim.autonomy().toFixed(1),
  };
  console.log('closed-loop eval', evalRes);
  // and one more lap with random steering kicks added
  const rSim = new Simulator(track); const kicks = new Disturbance(999, 1.0);
  const rAp = new Autopilot(rSim, (img) => predictSteering(tf, model, img) + kicks.step(PHYSICS_DT * 4));
  while (rSim.laps < 1 && rSim.time < 200) rAp.step();
  evalRes.withDisturbance = { laps: rSim.laps, takeovers: rSim.takeovers, maxAbsOffset: +rSim.maxAbsD.toFixed(3) };
  console.log('with disturbance', evalRes.withDisturbance);

  const out = path.join(root, 'web', 'model');
  fs.mkdirSync(out, { recursive: true });
  await model.save('file://' + out);
  fs.writeFileSync(path.join(out, 'meta.json'), JSON.stringify({
    created: new Date().toISOString(), architecture: 'nvidia', params: model.countParams(),
    frames: ds.size, trainSamples: split.train.length, valSamples: split.val.length,
    history, eval: evalRes,
  }, null, 2));
  console.log('saved to', out, 'total', ((Date.now() - t0) / 1000).toFixed(0) + 's');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
