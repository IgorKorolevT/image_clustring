import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

IMAGE_DIR = Path('images')
OUTPUT_DIR = Path('output')
MODEL_DIR = Path('model')
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_DIR / 'model.keras', compile=False)
with (MODEL_DIR / 'class_names.json').open(encoding='utf-8') as file:
    class_names = json.load(file)

if set(class_names) != {'cats', 'dogs'} or len(class_names) != 2:
    raise ValueError('Очікуються класи cats та dogs')

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer('embedding').output,
)
image_size = tuple(model.input_shape[1:3])
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
image_paths = sorted(
    path for path in IMAGE_DIR.rglob('*')
    if path.is_file() and path.suffix.lower() in extensions
)
if not image_paths:
    raise ValueError('У папці images немає зображень')


def load_image(path):
    with Image.open(path) as img:
        return np.asarray(img.convert('RGB').resize(image_size), dtype=np.float32)


predictions = []
embeddings = []
for start in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[start:start + BATCH_SIZE]
    batch = np.stack([load_image(path) for path in batch_paths])
    predictions.append(model.predict(batch, verbose=0).reshape(-1))
    embeddings.append(feature_model.predict(batch, verbose=0))

predictions = np.concatenate(predictions)
embeddings = normalize(np.concatenate(embeddings), norm='l2')
class_indices = (predictions >= 0.5).astype(int)

# При повторному запуску не залишаємо результати попередньої обробки.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for class_index, class_name in enumerate(class_names):
    output_name = 'cat' if class_name == 'cats' else 'dog'
    class_dir = OUTPUT_DIR / output_name
    class_dir.mkdir(parents=True, exist_ok=True)

    indices = np.flatnonzero(class_indices == class_index)
    if len(indices) >= 2:
        labels = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit_predict(
            embeddings[indices]
        )
    else:
        labels = np.full(len(indices), -1, dtype=int)

    for index, label in zip(indices, labels):
        folder = class_dir / ('noise' if label == -1 else f'cluster_{label}')
        folder.mkdir(parents=True, exist_ok=True)
        source = image_paths[index]
        destination = folder / source.name
        if destination.exists():
            destination = folder / f'{index}_{source.name}'
        shutil.copy2(source, destination)

    clusters = set(labels) - {-1}
    print(f'{output_name}: {len(indices)} зображень, '
          f'{len(clusters)} кластерів, '
          f'{np.sum(labels == -1)} шумових')

print(f'Усього оброблено: {len(image_paths)} зображень')
