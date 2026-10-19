// UI glue: simulator loop, recording, training and autonomous mode.
import { Track } from './track.js';
import { Simulator, Disturbance, renderCamera, preprocess, CAM_W, CAM_H, CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H } from './sim.js';
import { Dataset } from './data.js';
import { buildNvidiaModel, compile, trainModel, predictSteering, describeModel } from './model.js';
import { Autopilot, PHYSICS_DT } from './autopilot.js';
import { TopView } from './topview.js';
import { drawLossChart, drawHistogram } from './charts.js';
import { initExplainer, edgeImage } from './explainer.js';

const $ = (id) => document.getElementById(id);
const tf = window.tf;
const params = new URLSearchParams(location.search);

// state
const track = new Track();
const sim = new Simulator(track);
const dataset = new Dataset();
const state = {
  mode: 'manual', recording: false, training: false, stopTraining: false,
  step: 0, humanSteer: 0, expertTargetLaps: 0, disturbance: null, expertLapIdx: 0,
  pretrained: null, userModel: null, prediction: null, overrideHeld: false,
  trails: [{ mode: 'manual', pts: [] }], takeoverPts: [], lapRows: [], errors: [],
  batchLosses: [], epochs: [], lastFrameTs: 0, acc: 0, frame: 0,
};
const keys = { left: false, right: false, up: false, down: false };
const MANUAL_RECORD_EVERY = 6;  // 10 Hz
const EXPERT_RECORD_EVERY = 12; // 5 Hz (keeps memory modest on phones)
const MAX_FRAMES = 4000;
let autopilot = null;

// canvases
const dash = $('dashcam'), dctx = dash.getContext('2d');
let dashImg = null;
function sizeDash() {
  const mobile = window.matchMedia('(max-width: 760px)').matches;
  const w = mobile ? 400 : 576, h = mobile ? 300 : 288;
  if (dash.width !== w || dash.height !== h || !dashImg) { dash.width = w; dash.height = h; dashImg = dctx.createImageData(w, h); }
}
const rawc = $('rawcam'), rctx = rawc.getContext('2d'), rawImg = rctx.createImageData(CAM_W, CAM_H);
const netc = $('netin'), nctx = netc.getContext('2d'), netImg = nctx.createImageData(IMG_W, IMG_H);
const edgeCtx = $('edgeCam').getContext('2d');
const topview = new TopView($('topview'), track);
$('trackInfo').textContent = `${Math.round(track.length)} m, 8 m wide`;

// helpers
function toast(msg, ms = 1800) {
  const t = $('toast'); t.textContent = msg; t.classList.remove('hidden');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('hidden'), ms);
}
function activeModel() { return $('modelChoice').value === 'user' && state.userModel ? state.userModel : state.pretrained; }
function newTrail(mode) { state.trails.push({ mode, pts: [] }); let total = 0; for (const t of state.trails) total += t.pts.length; while (total > 2500 && state.trails.length > 1) total -= state.trails.shift().pts.length; }
function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode; newTrail(mode);
  const pill = $('modePill'); pill.className = 'pill ' + mode;
  pill.textContent = { manual: 'MANUAL', expert: 'EXPERT REC', auto: 'AUTONOMOUS' }[mode];
  $('btnManual').classList.toggle('active', mode === 'manual');
  $('btnExpert').textContent = mode === 'expert' ? 'Stop expert' : 'Auto-record expert laps';
  $('btnAuto').textContent = mode === 'auto' ? 'Restart' : 'Start';
}
function setRecording(on) {
  state.recording = on; $('recPill').classList.toggle('hidden', !on && state.mode !== 'expert');
  $('btnRecord').classList.toggle('on', on); $('btnRecord').textContent = on ? 'Stop recording' : 'Record';
}
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
  state.tab = name;
}
function recordFrame(label, throttle) {
  if (dataset.size >= MAX_FRAMES) { if (state.recording) { setRecording(false); toast(`Hit the recording limit (${MAX_FRAMES} frames)`); } return; }
  const cams = sim.captureAll();
  dataset.add({ ...cams, steering: label, throttle, speed: sim.car.v, t: sim.time });
}

// simulation step
function stepOnce() {
  state.step++;
  const dt = PHYSICS_DT;
  if (state.mode === 'manual') {
    const target = ((keys.right ? 1 : 0) - (keys.left ? 1 : 0)) * 0.55;
    const rate = (target === 0 ? 3 : 2.2) * dt;
    state.humanSteer += Math.max(-rate, Math.min(rate, target - state.humanSteer));
    const throttle = keys.up ? 1 : keys.down ? -1 : 0;
    if (state.recording && sim.car.v > 0.5 && state.step % MANUAL_RECORD_EVERY === 0) recordFrame(sim.car.steer, throttle);
    const lap = sim.step(dt, state.humanSteer, throttle);
    if (lap) toast(`Lap ${sim.laps}: ${sim.lapTimes.at(-1).toFixed(1)} s`);
  } else if (state.mode === 'expert') {
    const label = sim.expert.steering(sim.car, sim.proj), throttle = sim.expert.throttle(sim.car, sim.proj);
    if (state.step % EXPERT_RECORD_EVERY === 0) recordFrame(label, throttle);
    const lap = sim.step(dt, label + state.disturbance.step(dt), throttle);
    if (sim.offRoad) sim.takeover();
    if (lap) {
      state.expertLapIdx++;
      state.disturbance = new Disturbance(100 + state.expertLapIdx + dataset.size, [0.8, 1.4, 1.8, 1.1][state.expertLapIdx % 4]);
      sim.expert.vmax = 12 + (state.expertLapIdx % 4);
      if (sim.laps >= state.expertTargetLaps || dataset.size >= MAX_FRAMES) { stopExpert(); toast(`Recorded ${sim.laps} expert lap(s), ${dataset.size} frames total`, 2600); }
    }
  } else if (state.mode === 'auto') {
    const human = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
    if (human !== 0) {
      if (!state.overrideHeld) { sim.takeovers++; sim.lapTakeovers++; state.takeoverPts.push([sim.car.x, sim.car.y]); toast('Manual takeover'); }
      state.overrideHeld = true;
      state.humanSteer = human * 0.55;
      const thr = autopilot.speed.update(sim.car.v, state.humanSteer, dt);
      if (sim.step(dt, state.humanSteer, thr)) onAutoLap();
      if (sim.offRoad) { state.takeoverPts.push([sim.car.x, sim.car.y]); sim.takeover(); }
    } else {
      state.overrideHeld = false;
      const at = [sim.car.x, sim.car.y];
      const r = autopilot.step(dt);
      state.prediction = autopilot.steer;
      if (r.tookOver) { state.takeoverPts.push(at); newTrail('auto'); toast('Left the road, takeover'); }
      if (r.lapDone) onAutoLap();
    }
  }
  const tr = state.trails[state.trails.length - 1].pts, last = tr[tr.length - 1];
  if (!last || Math.hypot(last[0] - sim.car.x, last[1] - sim.car.y) > 1.2) tr.push([sim.car.x, sim.car.y]);
}

function onAutoLap() {
  const i = sim.laps - 1, t = sim.lapTimes[i], to = sim.lastLapTakeovers[i];
  state.lapRows.push({ lap: sim.laps, time: t, takeovers: to, autonomy: sim.autonomy(to, t), maxOffset: sim.lapMaxAbsDs[i] });
  renderLapTable();
  toast(`Lap ${sim.laps}: ${t.toFixed(1)} s, ${to} takeover${to === 1 ? '' : 's'}`, 2200);
}
function renderLapTable() {
  $('lapTable').innerHTML = state.lapRows.slice().reverse().map((r) => `<tr><td>${r.lap}</td><td>${r.time.toFixed(1)} s</td><td>${r.takeovers}</td><td>${r.autonomy.toFixed(0)}%</td><td>${r.maxOffset.toFixed(2)} m</td></tr>`).join('');
}

// modes
function startManual() { if (state.mode === 'expert') stopExpert(); setMode('manual'); state.humanSteer = 0; }
function startExpert() {
  if (state.mode === 'expert') { stopExpert(); return; }
  setRecording(false);
  setMode('expert'); $('recPill').classList.remove('hidden');
  sim.reset(dataset.size % 2 ? track.length * 0.5 : 0);
  state.expertTargetLaps = Number($('expertLaps').value); state.expertLapIdx = 0;
  state.disturbance = new Disturbance(7 + dataset.size, 0.8); sim.expert.vmax = 12;
  switchTab('drive');
}
function stopExpert() { setMode('manual'); $('recPill').classList.toggle('hidden', !state.recording); sim.car.v = 0; state.humanSteer = 0; }
function startAuto() {
  const m = activeModel();
  if (!m) { toast('Model is still loading'); return false; }
  if (state.training) { toast('Wait for training to finish'); return false; }
  setRecording(false);
  if (state.mode === 'expert') stopExpert();
  sim.reset(0); state.lapRows = []; state.takeoverPts = []; renderLapTable();
  autopilot = new Autopilot(sim, (img) => predictSteering(tf, activeModel(), img), { speed: { maxSpeed: Number($('maxSpeed').value) } });
  state.mode = null; setMode('auto');
  switchTab('auto');
  return true;
}
function stopAuto() { if (state.mode === 'auto') { setMode('manual'); state.humanSteer = 0; } }

// training
async function train() {
  if (state.training) return;
  if (dataset.size < 30) { toast('Record some driving first'); switchTab('drive'); return; }
  stopAuto(); if (state.mode === 'expert') stopExpert();
  state.training = true; state.stopTraining = false;
  $('btnTrain').disabled = true; $('btnStop').disabled = false; $('btnAuto').disabled = true;
  const epochs = Math.max(1, Math.min(50, Number($('epochs').value) || 5));
  const batchSize = Math.max(8, Math.min(256, Number($('batch').value) || 64));
  const lr = Number($('lr').value) || 1e-3;
  const val = Number($('valSplit').value) / 100;
  const lite = $('arch').value === 'lite';
  try {
    let model;
    if ($('initFrom').value === 'pretrained' && !lite && state.pretrained) {
      model = await tf.loadLayersModel('model/model.json'); compile(tf, model, lr);
    } else model = buildNvidiaModel(tf, { lite, learningRate: lr });
    renderArch(model);
    const split = dataset.split(val, 42, { sideCameras: $('useSide').checked });
    state.batchLosses = []; state.epochs = [];
    let ema = null; const t0 = performance.now();
    $('trainStatus').textContent = `Training on ${split.train.length} images, ${split.val.length} held out for validation`;
    await trainModel(tf, model, split, {
      epochs, batchSize, flip: $('useFlip').checked, shouldStop: () => state.stopTraining,
      onBatch: ({ epoch, batch, nBatches, loss }) => {
        ema = ema === null ? loss : ema * 0.9 + loss * 0.1;
        state.batchLosses.push({ x: epoch + (batch + 1) / nBatches, ema });
        const done = (epoch + (batch + 1) / nBatches) / epochs;
        $('trainBar').style.width = `${(done * 100).toFixed(1)}%`;
        if (batch % 3 === 0) { drawLossChart($('lossChart'), state.batchLosses, state.epochs); $('trainStatus').textContent = `epoch ${epoch + 1}/${epochs}, batch ${batch + 1}/${nBatches}, loss ${loss.toFixed(4)} (${((performance.now() - t0) / 1000).toFixed(0)} s)`; }
      },
      onEpoch: (r) => { state.epochs.push(r); drawLossChart($('lossChart'), state.batchLosses, state.epochs); },
    });
    const last = state.epochs.at(-1);
    state.userModel?.dispose?.();
    state.userModel = model;
    const opt = $('modelChoice').querySelector('option[value=user]'); opt.disabled = false; $('modelChoice').value = 'user';
    $('btnSaveModel').disabled = false;
    $('trainStatus').textContent = last ? `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s. train ${last.loss.toFixed(4)}, val ${last.valLoss.toFixed(4)}` : 'Stopped before the first epoch finished.';
    $('modelBadge').textContent = 'model: yours'; $('modelBadge').classList.add('ok');
    toast('Training done');
  } catch (e) {
    console.error(e); state.errors.push(String(e)); $('trainStatus').textContent = 'Training failed: ' + e.message;
  } finally {
    state.training = false; $('btnTrain').disabled = false; $('btnStop').disabled = true; $('btnAuto').disabled = false;
  }
}
function renderArch(model) {
  $('archTable').querySelector('tbody').innerHTML = describeModel(model).map((l) => `<tr><td>${l.name} <span class="muted">${l.type}</span></td><td>${l.shape}</td><td>${l.params.toLocaleString()}</td></tr>`).join('') +
    `<tr><td><b>total</b></td><td></td><td><b>${model.countParams().toLocaleString()}</b></td></tr>`;
}

// rendering
function drawRGB(ctx, imgData, rgb) {
  const d = imgData.data; for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { d[j] = rgb[i]; d[j + 1] = rgb[i + 1]; d[j + 2] = rgb[i + 2]; d[j + 3] = 255; }
  ctx.putImageData(imgData, 0, 0);
}
function render() {
  state.frame++;
  sizeDash();
  const c = sim.car;
  renderCamera(track, c.x, c.y, c.h, dashImg.data, dash.width, dash.height);
  dctx.putImageData(dashImg, 0, 0);

  if (state.frame % 3 === 0) {
    renderCamera(track, c.x, c.y, c.h, rawImg.data, CAM_W, CAM_H);
    rctx.putImageData(rawImg, 0, 0);
    rctx.fillStyle = 'rgba(70,140,255,.38)'; rctx.fillRect(0, 0, CAM_W, CROP_TOP);
    rctx.fillStyle = 'rgba(255,70,100,.38)'; rctx.fillRect(0, CAM_H - CROP_BOTTOM, CAM_W, CROP_BOTTOM);
    rctx.strokeStyle = '#fff'; rctx.setLineDash([6, 4]); rctx.strokeRect(1, CROP_TOP, CAM_W - 2, CAM_H - CROP_TOP - CROP_BOTTOM); rctx.setLineDash([]);
    const img = state.mode === 'auto' && autopilot?.lastImage ? autopilot.lastImage : preprocess(rawImg.data);
    drawRGB(nctx, netImg, img);
    if (state.tab === 'learn') edgeImage(img, edgeCtx);
    // show what the model would steer even when it's not driving
    if (state.mode !== 'auto' && !state.training && activeModel() && state.frame % 12 === 0) state.prediction = predictSteering(tf, activeModel(), img);
    const ex = sim.expert.steering(c, sim.proj);
    $('predSteer').textContent = state.prediction == null ? '–' : state.prediction.toFixed(3);
    $('expertSteer').textContent = ex.toFixed(3);
    $('laneOffset').textContent = `${sim.proj.d >= 0 ? '+' : ''}${sim.proj.d.toFixed(2)} m`;
  }
  if (state.frame % 2 === 0) topview.draw(sim, state.trails, state.takeoverPts);

  $('hudSpeed').textContent = Math.round(c.v * 3.6);
  $('hudLap').textContent = sim.laps;
  $('hudTime').textContent = (sim.time - sim.lapStart).toFixed(1);
  $('steerNeedle').style.left = `${50 + c.steer * 50}%`;
  $('steerText').textContent = (c.steer >= 0 ? '+' : '') + c.steer.toFixed(2);

  if (state.frame % 10 === 0) {
    $('statFrames').textContent = dataset.size.toLocaleString();
    $('statImages').textContent = (dataset.size * 3).toLocaleString();
    $('statMem').textContent = `${((dataset.size * 3 * IMG_W * IMG_H * 3) / 1e6).toFixed(0)} MB`;
    drawHistogram($('hist'), dataset.steeringHistogram());
    if (state.mode === 'auto') {
      const a = sim.autonomy();
      $('autonomyPct').textContent = `${a.toFixed(0)}%`; document.querySelector('.gauge').style.setProperty('--p', a.toFixed(1));
      $('fTake').textContent = sim.takeovers; $('fTime').textContent = sim.time.toFixed(0);
      $('aLaps').textContent = sim.laps; $('aTake').textContent = sim.takeovers;
      $('aBest').textContent = sim.lapTimes.length ? Math.min(...sim.lapTimes).toFixed(1) : '–';
    }
  }
}

function loop(ts) {
  const real = Math.min(0.1, (ts - (state.lastFrameTs || ts)) / 1000);
  state.lastFrameTs = ts;
  const mul = state.mode === 'expert' ? Number($('expertSpeed').value) : state.mode === 'auto' ? Number($('autoSpeed').value) : 1;
  state.acc += real * mul;
  let n = 0;
  const maxSteps = 6 * mul;
  try {
    while (state.acc >= PHYSICS_DT && n < maxSteps) { stepOnce(); state.acc -= PHYSICS_DT; n++; }
    if (n >= maxSteps) state.acc = 0; // device too slow for this speed: run slower instead of skipping
    render();
  } catch (e) { console.error(e); state.errors.push(String(e)); }
  requestAnimationFrame(loop);
}

// input
const KEYMAP = { ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right', ArrowUp: 'up', KeyW: 'up', ArrowDown: 'down', KeyS: 'down' };
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName) && !KEYMAP[e.code]) return;
  if (KEYMAP[e.code]) { keys[KEYMAP[e.code]] = true; e.preventDefault(); if (document.activeElement?.blur) document.activeElement.blur(); }
  if (e.code === 'KeyR' && !e.repeat) { if (state.mode !== 'manual') startManual(); setRecording(!state.recording); }
  if (e.code === 'Space') { e.preventDefault(); if (state.mode === 'manual') { sim.takeover(); sim.takeovers--; sim.lapTakeovers--; sim.car.v = 0; } }
});
window.addEventListener('keyup', (e) => { if (KEYMAP[e.code]) keys[KEYMAP[e.code]] = false; });
window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
document.querySelectorAll('#touch button').forEach((b) => {
  const k = b.dataset.key;
  const on = (e) => { e.preventDefault(); keys[k] = true; b.classList.add('down'); b.setPointerCapture?.(e.pointerId); };
  const off = (e) => { e.preventDefault(); keys[k] = false; b.classList.remove('down'); };
  b.addEventListener('pointerdown', on); b.addEventListener('pointerup', off); b.addEventListener('pointercancel', off); b.addEventListener('pointerleave', off);
  b.addEventListener('contextmenu', (e) => e.preventDefault());
});

// buttons
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
$('btnManual').addEventListener('click', () => { startManual(); stopAuto(); toast('Manual driving'); });
$('btnRecord').addEventListener('click', () => { if (state.mode !== 'manual') { stopAuto(); startManual(); } setRecording(!state.recording); });
$('btnExpert').addEventListener('click', () => { stopAuto(); startExpert(); });
$('btnClear').addEventListener('click', () => { if (dataset.size && confirm(`Delete ${dataset.size} recorded frames?`)) { dataset.clear(); toast('Data cleared'); } });
$('btnCSV').addEventListener('click', () => {
  if (!dataset.size) { toast('No data yet'); return; }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([dataset.toCSV()], { type: 'text/csv' })); a.download = 'driving_log.csv'; a.click();
});
$('btnTrain').addEventListener('click', train);
$('btnStop').addEventListener('click', () => { state.stopTraining = true; });
$('btnSaveModel').addEventListener('click', () => state.userModel?.save('downloads://self-driving-car-model'));
$('btnAuto').addEventListener('click', startAuto);
$('btnAutoStop').addEventListener('click', stopAuto);
$('heroAuto').addEventListener('click', () => { if (startAuto() && window.innerWidth < 1100) $('dashWrap').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
$('valSplit').addEventListener('input', () => { $('valSplitText').textContent = $('valSplit').value + '%'; });
$('maxSpeed').addEventListener('input', () => { $('maxSpeedText').textContent = `${Math.round($('maxSpeed').value * 3.6)} km/h`; if (autopilot) autopilot.speed.maxSpeed = Number($('maxSpeed').value); });
$('arch').addEventListener('change', () => { if ($('arch').value === 'lite') $('initFrom').value = 'scratch'; });
window.addEventListener('resize', () => topview.resize());
window.addEventListener('error', (e) => { state.errors.push(String(e.message)); });
window.addEventListener('unhandledrejection', (e) => { state.errors.push(String(e.reason)); });

// boot
async function boot() {
  initExplainer();
  drawLossChart($('lossChart'), [], []);
  requestAnimationFrame(loop);
  try {
    await tf.ready();
    $('backendBadge').textContent = `TF.js ${tf.version.tfjs} (${tf.getBackend()})`; $('backendBadge').classList.add('ok');
    state.pretrained = await tf.loadLayersModel('model/model.json');
    tf.tidy(() => state.pretrained.predict(tf.zeros([1, IMG_H, IMG_W, 3]))); // warm-up
    renderArch(state.pretrained);
    $('modelBadge').textContent = 'model: pretrained'; $('modelBadge').classList.add('ok');
    if (params.get('auto') === '1') { if (params.get('speed')) $('autoSpeed').value = params.get('speed'); startAuto(); }
  } catch (e) {
    console.error(e); state.errors.push(String(e)); $('modelBadge').textContent = 'model: failed to load';
  }
}
boot();

// handy for poking around in devtools (and used by tools/verify-browser.mjs)
window.app = { sim, dataset, state, startAuto, stopAuto, startExpert, train, switchTab, setRecording, keys, track };
