import numpy as np
from struct import unpack
import gzip
import pickle

with open('best_models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

test_images_path = 'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    _, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

with gzip.open(test_labels_path, 'rb') as f:
    _, num_labels = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs.astype(np.float32) / 255.0
test_imgs = test_imgs.reshape(-1, 1, rows, cols)

logits = model(test_imgs)

import mynn as nn
acc = nn.metric.accuracy(logits, test_labs)
print(f"CNN Test Accuracy: {acc:.4f}")
