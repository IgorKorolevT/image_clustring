import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from image_preparation import clean_images

DATA_DIR = Path('cats_and_dogs_filtered')
CHECKPOINT_DIR = Path('checkpoints')
LOG_DIR = Path('logs')
MODEL_DIR = Path('model')
IMAGE_SIZE = (500, 500)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

for directory in (CHECKPOINT_DIR, LOG_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for split in ('train', 'validation'):
    for class_name in ('cats', 'dogs'):
        clean_images(DATA_DIR / split / class_name)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / 'train', image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='binary', seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / 'validation', image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='binary', shuffle=False,
)

class_names = train_ds.class_names
if class_names != val_ds.class_names:
    raise ValueError('Класи train і validation не збігаються')

with (MODEL_DIR / 'class_names.json').open('w', encoding='utf-8') as file:
    json.dump(class_names, file, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    layers.Rescaling(1.0 / 255),
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', name='embedding'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', metrics=['accuracy'],
)
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / 'latest.keras'),
        save_best_only=False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / 'best.keras'),
        monitor='val_accuracy', mode='max', save_best_only=True,
    ),
    tf.keras.callbacks.CSVLogger(str(LOG_DIR / 'training.csv')),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True,
    ),
    tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR / 'tensorboard')),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks,
)
model.save(MODEL_DIR / 'model.keras')

with (LOG_DIR / 'history.json').open('w', encoding='utf-8') as file:
    json.dump(
        {key: [float(v) for v in values] for key, values in history.history.items()},
        file, ensure_ascii=False, indent=2,
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for key, title, ax in [('accuracy', 'Accuracy', axes[0]), ('loss', 'Loss', axes[1])]:
    ax.plot(history.history[key], label='Train')
    ax.plot(history.history['val_' + key], label='Validation')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
fig.tight_layout()
fig.savefig(LOG_DIR / 'training.png')
plt.close(fig)
print('Модель збережено:', MODEL_DIR / 'model.keras')
