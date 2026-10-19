"""Train the Nvidia model on data recorded with the Udacity simulator.

    python model.py -d data/                      # folder with driving_log.csv + IMG/
    python model.py --synthetic -n 2              # smoke test without a simulator

Saves the best epoch (by val loss) to model.keras.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras  # noqa: E402
from keras import layers  # noqa: E402

from utils import INPUT_SHAPE, augment, load_image, preprocess  # noqa: E402

COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']


def load_log(data_dir: str) -> pd.DataFrame:
    # newer sim versions write a header row, older ones don't
    path = os.path.join(data_dir, 'driving_log.csv')
    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True)
    if str(df.iloc[0]['steering']).strip().lower() == 'steering':
        df = df.iloc[1:]
    df['steering'] = df['steering'].astype(float)
    return df.reset_index(drop=True)


def build_model(dropout: float = 0.5, learning_rate: float = 1e-3) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Rescaling(1 / 127.5, offset=-1.0, name='normalize'),
        layers.Conv2D(24, 5, strides=2, activation='elu', name='conv1'),
        layers.Conv2D(36, 5, strides=2, activation='elu', name='conv2'),
        layers.Conv2D(48, 5, strides=2, activation='elu', name='conv3'),
        layers.Conv2D(64, 3, activation='elu', name='conv4'),
        layers.Conv2D(64, 3, activation='elu', name='conv5'),
        layers.Dropout(dropout, name='dropout'),
        layers.Flatten(),
        layers.Dense(100, activation='elu', name='fc1'),
        layers.Dense(50, activation='elu', name='fc2'),
        layers.Dense(10, activation='elu', name='fc3'),
        layers.Dense(1, name='steering'),
    ], name='nvidia')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')
    return model


class DrivingSequence(keras.utils.PyDataset):
    # augmentation only for the training set

    def __init__(self, data_dir, df, batch_size=40, training=True, extra_aug=False, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.data_dir, self.df, self.batch_size = data_dir, df.reset_index(drop=True), batch_size
        self.training, self.extra_aug = training, extra_aug
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.order)

    def __getitem__(self, idx):
        rows = self.df.iloc[self.order[idx * self.batch_size:(idx + 1) * self.batch_size]]
        x = np.empty((len(rows), *INPUT_SHAPE), dtype=np.float32)
        y = np.empty((len(rows),), dtype=np.float32)
        for i, r in enumerate(rows.itertuples(index=False)):
            if self.training:
                img, steer = augment(self.data_dir, r.center, r.left, r.right, r.steering, self.rng, self.extra_aug)
            else:  # validation: centre camera, no augmentation
                img, steer = load_image(self.data_dir, r.center), r.steering
            x[i], y[i] = preprocess(img), steer
        return x, y


def make_synthetic_dataset(out_dir: str, n: int = 240, seed: int = 0) -> str:
    """Fake recording for testing without the simulator."""
    import cv2
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, 'IMG')
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(n):
        steer = float(np.clip(rng.normal(0, 0.3), -1, 1))
        names = []
        for cam, shift in (('center', 0), ('left', -40), ('right', 40)):
            img = np.zeros((160, 320, 3), np.uint8)
            img[:60] = (200, 160, 110)            # sky (BGR)
            img[60:] = (60, 130, 60)              # grass
            top_x = int(160 + steer * 120 + shift)
            pts = np.array([[60 + shift, 160], [260 + shift, 160], [top_x + 25, 60], [top_x - 25, 60]], np.int32)
            cv2.fillPoly(img, [pts], (90, 90, 90))
            img[135:] = (40, 30, 160)             # hood
            name = f'IMG/{cam}_{i:05d}.jpg'
            cv2.imwrite(os.path.join(out_dir, name), img)
            names.append(name)
        rows.append([*names, steer, 0.3, 0, 20.0])
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'driving_log.csv'), header=False, index=False)
    return out_dir


def main():
    p = argparse.ArgumentParser(description='Behavioral cloning: train the Nvidia model')
    p.add_argument('-d', '--data-dir', default='data')
    p.add_argument('-t', '--test-size', type=float, default=0.2, help='validation fraction (0.1-0.2)')
    p.add_argument('-k', '--dropout', type=float, default=0.5)
    p.add_argument('-n', '--epochs', type=int, default=10)
    p.add_argument('-b', '--batch-size', type=int, default=40)
    p.add_argument('-l', '--learning-rate', type=float, default=1e-3)
    p.add_argument('-o', '--output', default='model.keras')
    p.add_argument('--extra-aug', action='store_true', help='also random shifts + brightness')
    p.add_argument('--synthetic', action='store_true', help='generate a fake dataset in --data-dir first')
    args = p.parse_args()

    if args.synthetic:
        make_synthetic_dataset(args.data_dir)
    df = load_log(args.data_dir)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_val = max(1, int(round(len(df) * args.test_size)))
    val_df, train_df = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
    print(f'{len(train_df)} training rows, {len(val_df)} validation rows')

    model = build_model(args.dropout, args.learning_rate)
    model.summary()
    train_seq = DrivingSequence(args.data_dir, train_df, args.batch_size, training=True, extra_aug=args.extra_aug)
    val_seq = DrivingSequence(args.data_dir, val_df, args.batch_size, training=False)
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.output, monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ]
    hist = model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=2)
    for e, (l, v) in enumerate(zip(hist.history['loss'], hist.history['val_loss']), 1):
        print(f'epoch {e}: loss {l:.4f}  val_loss {v:.4f}')
    print(f'saved best model to {args.output}')


if __name__ == '__main__':
    main()
