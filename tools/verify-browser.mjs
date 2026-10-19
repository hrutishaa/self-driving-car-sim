// Headless Chrome test of the web app, also takes the screenshots in docs/.
//   node tools/verify-browser.mjs [--chrome /usr/bin/google-chrome]
import path from 'path';
import { fileURLToPath } from 'url';
import { chromium } from 'playwright-core';
import { serve } from './serve.mjs';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const docs = path.join(root, 'docs');
const ci = process.argv.indexOf('--chrome');
const executablePath = ci > 0 ? process.argv[ci + 1] : (process.env.CHROME_PATH || '/usr/bin/google-chrome');
const PORT = 8765;
const results = { date: new Date().toISOString(), checks: {} };
const ok = (name, pass, info = {}) => { results.checks[name] = { pass, ...info }; console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}`, JSON.stringify(info)); };

const server = await serve(PORT);
const browser = await chromium.launch({ executablePath, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const consoleErrors = [];
try {
  // desktop
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(e.message));
  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForFunction(() => window.app?.state.pretrained, null, { timeout: 90000 });
  results.backend = await page.textContent('#backendBadge');
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(docs, 'screenshot-home.png') });

  // pretrained model, one full lap
  await page.click('.tab[data-tab=auto]');
  await page.selectOption('#autoSpeed', '4');
  await page.click('#heroAuto');
  const t0 = Date.now();
  await page.waitForFunction(() => window.app.sim.laps >= 0 && window.app.sim.progress > 300, null, { timeout: 300000 });
  await page.screenshot({ path: path.join(docs, 'screenshot-autonomous.png') });
  await page.waitForFunction(() => window.app.sim.laps >= 1, null, { timeout: 600000, polling: 500 });
  const lap = await page.evaluate(() => {
    const s = window.app.sim;
    return { laps: s.laps, lapTime: +s.lapTimes[0].toFixed(2), takeovers: s.lastLapTakeovers[0], maxAbsOffset: +s.lapMaxAbsDs[0].toFixed(3), halfWidth: s.track.halfWidth, autonomy: +s.autonomy(s.lastLapTakeovers[0], s.lapTimes[0]).toFixed(1) };
  });
  lap.realSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  lap.allWheelsOnAsphalt = lap.maxAbsOffset + 0.9 < lap.halfWidth;
  ok('pretrained model drives a full lap without leaving the road', lap.laps >= 1 && lap.takeovers === 0 && lap.allWheelsOnAsphalt, lap);
  await page.screenshot({ path: path.join(docs, 'screenshot-lap-complete.png') });
  await page.click('#btnAutoStop');

  // manual driving + recording
  await page.click('.tab[data-tab=drive]');
  await page.click('#btnRecord');
  await page.keyboard.down('ArrowUp');
  await page.waitForTimeout(1500);
  await page.keyboard.down('ArrowRight'); await page.waitForTimeout(500); await page.keyboard.up('ArrowRight');
  await page.keyboard.down('ArrowLeft'); await page.waitForTimeout(500); await page.keyboard.up('ArrowLeft');
  await page.waitForTimeout(1000);
  await page.keyboard.up('ArrowUp');
  await page.click('#btnRecord');
  const manualFrames = await page.evaluate(() => window.app.dataset.size);
  ok('manual recording', manualFrames > 10, { frames: manualFrames });

  // expert laps
  await page.selectOption('#expertLaps', '1');
  await page.selectOption('#expertSpeed', '8');
  await page.click('#btnExpert');
  await page.waitForFunction(() => window.app.state.mode === 'expert' && window.app.sim.progress > 250, null, { timeout: 300000 });
  await page.screenshot({ path: path.join(docs, 'screenshot-recording.png') });
  await page.waitForFunction(() => window.app.state.mode === 'manual', null, { timeout: 600000, polling: 500 });
  const frames = await page.evaluate(() => window.app.dataset.size);
  ok('expert lap recording', frames > manualFrames + 200, { frames });

  // training
  await page.click('.tab[data-tab=train]');
  await page.fill('#epochs', '2');
  const tt = Date.now();
  await page.click('#btnTrain');
  await page.waitForFunction(() => window.app.state.training === true, null, { timeout: 20000 });
  await page.waitForFunction(() => window.app.state.epochs.length >= 1, null, { timeout: 900000, polling: 1000 });
  await page.screenshot({ path: path.join(docs, 'screenshot-training.png') });
  await page.waitForFunction(() => window.app.state.training === false, null, { timeout: 1800000, polling: 1000 });
  const tr = await page.evaluate(() => ({ epochs: window.app.state.epochs, errors: window.app.state.errors, status: document.getElementById('trainStatus').textContent }));
  tr.realSeconds = +((Date.now() - tt) / 1000).toFixed(1);
  ok('training in the browser', tr.epochs.length === 2 && tr.errors.length === 0 && tr.epochs.every((e) => isFinite(e.loss) && isFinite(e.valLoss)), tr);
  await page.evaluate(() => document.querySelector('details.arch').open = true);
  await page.screenshot({ path: path.join(docs, 'screenshot-training-done.png') });

  // see how the model we just trained does
  await page.click('.tab[data-tab=auto]');
  await page.selectOption('#autoSpeed', '4');
  await page.click('#btnAuto');
  await page.waitForFunction(() => window.app.sim.laps >= 1 || window.app.sim.time > 150, null, { timeout: 900000, polling: 1000 });
  const userLap = await page.evaluate(() => ({ model: document.getElementById('modelChoice').value, laps: window.app.sim.laps, takeovers: window.app.sim.takeovers, simSeconds: +window.app.sim.time.toFixed(1), autonomy: +window.app.sim.autonomy().toFixed(1) }));
  results.userTrainedModelDrive = userLap; console.log('info  browser-trained model:', JSON.stringify(userLap));
  await page.click('#btnAutoStop');

  // how it works tab
  await page.click('.tab[data-tab=learn]');
  for (let i = 0; i < 3; i++) await page.click('#gdStep');
  const gd = await page.$$eval('#gdTable tr', (rows) => rows.map((r) => [...r.children].map((c) => c.textContent)));
  const ws = gd.map((r) => r[1]);
  ok('gradient descent toy (w: -1, 0.4, 0.68, 0.736)', JSON.stringify(ws) === JSON.stringify(['-1', '0.4', '0.68', '0.736']) && gd[0][5] === '-1,400', { rows: gd });
  const edge = await page.$$eval('#edgeOut .cellrow:last-child .cell', (c) => c.map((x) => x.textContent));
  ok('edge filter (1 1 8 8 8 -> 0 7 0 0)', edge.join(' ') === '0 7 0 0', { output: edge });
  const calc = await page.textContent('#calcOut');
  ok('autonomy calc (2 takeovers, 120 s -> 90%)', calc === '90%', { value: calc });
  await page.evaluate(() => { window.scrollTo(0, 0); const pnl = document.querySelector('.panel'); pnl.scrollTop = document.getElementById('grid3').offsetTop - 260; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(docs, 'screenshot-explainer.png') });
  const pageErrors = await page.evaluate(() => window.app.state.errors);
  await page.close();

  // mobile
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const m = await ctx.newPage();
  m.on('pageerror', (e) => consoleErrors.push('mobile: ' + e.message));
  const failed = [];
  m.on('response', (r) => { if (r.status() >= 400) failed.push(r.url()); });
  // load from a sub-path like GitHub Pages does
  await m.goto(`http://localhost:${PORT}/self-driving-car-sim/`);
  await m.waitForFunction(() => window.app?.state.pretrained, null, { timeout: 90000 });
  ok('loads under /self-driving-car-sim/ with no 404s', failed.length === 0, { failed });
  const touchVisible = await m.isVisible('#touch');
  const box = await m.locator('#touch button[data-key=up]').boundingBox();
  await m.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
  await m.dispatchEvent('#touch button[data-key=up]', 'pointerdown', { pointerId: 1 });
  await m.waitForTimeout(1500);
  await m.dispatchEvent('#touch button[data-key=up]', 'pointerup', { pointerId: 1 });
  const v = await m.evaluate(() => window.app.sim.car.v);
  ok('mobile touch controls', touchVisible && v > 1, { touchVisible, speed: +v.toFixed(2) });
  await m.screenshot({ path: path.join(docs, 'screenshot-mobile.png') });
  await m.click('#heroAuto');
  await m.waitForTimeout(4000);
  await m.screenshot({ path: path.join(docs, 'screenshot-mobile-autonomous.png') });
  await ctx.close();

  ok('no page errors', consoleErrors.length === 0 && pageErrors.length === 0, { consoleErrors, pageErrors });
} catch (e) {
  ok('verification script completed', false, { error: String(e) });
} finally {
  await browser.close(); server.close();
  results.allPassed = Object.values(results.checks).every((c) => c.pass);
  console.log(results.allPassed ? 'all passed' : 'FAILED');
  process.exit(results.allPassed ? 0 : 1);
}
