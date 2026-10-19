// Nvidia model (Bojarski et al. 2016). tf is passed in so this works with the CDN build
// in the browser and with tfjs-node for the pretraining script.
import { IMG_H, IMG_W, IMG_C, mulberry32 } from './sim.js';
import { IMG_SIZE, makeBatch } from './data.js';

// lite = half the filters/units (~64k weights), for slow phones
export function buildNvidiaModel(tf, { lite = false, dropout = 0.5, learningRate = 1e-3 } = {}) {
  const k = lite ? 0.5 : 1;
  const c = (n) => Math.max(4, Math.round(n * k));
  const model = tf.sequential({ name: lite ? 'nvidia_lite' : 'nvidia' });
  // normalize inside the model so there's nothing to forget at drive time
  model.add(tf.layers.rescaling({ inputShape: [IMG_H, IMG_W, IMG_C], scale: 1 / 127.5, offset: -1, name: 'normalize' }));
  model.add(tf.layers.conv2d({ filters: c(24), kernelSize: 5, strides: 2, activation: 'elu', name: 'conv1' }));
  model.add(tf.layers.conv2d({ filters: c(36), kernelSize: 5, strides: 2, activation: 'elu', name: 'conv2' }));
  model.add(tf.layers.conv2d({ filters: c(48), kernelSize: 5, strides: 2, activation: 'elu', name: 'conv3' }));
  model.add(tf.layers.conv2d({ filters: c(64), kernelSize: 3, activation: 'elu', name: 'conv4' }));
  model.add(tf.layers.conv2d({ filters: c(64), kernelSize: 3, activation: 'elu', name: 'conv5' }));
  model.add(tf.layers.dropout({ rate: dropout, name: 'dropout' }));
  model.add(tf.layers.flatten({ name: 'flatten' }));
  model.add(tf.layers.dense({ units: c(100), activation: 'elu', name: 'fc1' }));
  model.add(tf.layers.dense({ units: c(50), activation: 'elu', name: 'fc2' }));
  model.add(tf.layers.dense({ units: c(10), activation: 'elu', name: 'fc3' }));
  model.add(tf.layers.dense({ units: 1, name: 'steering' }));
  compile(tf, model, learningRate);
  return model;
}

export function compile(tf, model, learningRate = 1e-3) {
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });
}

export function describeModel(model) {
  return model.layers.map((l) => ({ name: l.name, type: l.getClassName(), shape: JSON.stringify(l.outputShape.slice(1)), params: l.countParams() }));
}

export function predictSteering(tf, model, img) {
  return tf.tidy(() => {
    const x = tf.tensor4d(Float32Array.from(img), [1, IMG_H, IMG_W, IMG_C]);
    return model.predict(x).dataSync()[0];
  });
}

export async function evaluate(tf, model, samples, batchSize = 128) {
  let se = 0;
  for (let i = 0; i < samples.length; i += batchSize) {
    const idx = []; for (let j = i; j < Math.min(samples.length, i + batchSize); j++) idx.push(j);
    const b = makeBatch(samples, idx, null, false);
    const pred = tf.tidy(() => model.predict(tf.tensor4d(b.x, [b.n, IMG_H, IMG_W, IMG_C])));
    const p = await pred.data(); pred.dispose();
    for (let j = 0; j < b.n; j++) se += (p[j] - b.y[j]) ** 2;
    await tf.nextFrame?.();
  }
  return se / Math.max(1, samples.length);
}

export async function trainModel(tf, model, { train, val }, {
  epochs = 5, batchSize = 64, seed = 7, flip = true, onBatch, onEpoch, shouldStop, yieldEvery = 1,
} = {}) {
  const rng = mulberry32(seed);
  const history = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = train.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let sum = 0, count = 0;
    const nBatches = Math.ceil(order.length / batchSize);
    for (let b = 0; b < nBatches; b++) {
      if (shouldStop?.()) return history;
      const idx = order.slice(b * batchSize, (b + 1) * batchSize);
      const batch = makeBatch(train, idx, rng, flip);
      const xs = tf.tensor4d(batch.x, [batch.n, IMG_H, IMG_W, IMG_C]);
      const ys = tf.tensor2d(batch.y, [batch.n, 1]);
      const loss = await model.trainOnBatch(xs, ys);
      xs.dispose(); ys.dispose();
      sum += loss * batch.n; count += batch.n;
      onBatch?.({ epoch, batch: b, nBatches, loss });
      if (b % yieldEvery === 0 && tf.nextFrame) await tf.nextFrame();
    }
    const valLoss = await evaluate(tf, model, val);
    const rec = { epoch: epoch + 1, loss: sum / count, valLoss };
    history.push(rec);
    onEpoch?.(rec);
  }
  return history;
}

export { IMG_SIZE };
