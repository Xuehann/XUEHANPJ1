import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt

model = nn.models.Model_MLP([784, 512, 256, 128, 10], 'ReLU')
model.load_model('best_models/best_model.pkl')

def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels

test_imgs, test_labs = load_mnist(
    'dataset/MNIST/t10k-images-idx3-ubyte.gz',
    'dataset/MNIST/t10k-labels-idx1-ubyte.gz'
)

first_layer_weights = model.layers[0].params['W']
third_layer_weights = model.layers[2].params['W'] 
num_neurons = 30 
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
axes = axes.ravel()

for i in range(num_neurons):
    weight_img = first_layer_weights[:, i].reshape(28, 28)
    axes[i].imshow(weight_img, cmap='bwr')
    axes[i].axis('off')
    axes[i].set_title(f'N{i}')

plt.suptitle("First Layer MLP Weights")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(third_layer_weights, cmap='bwr', aspect='auto')
plt.colorbar()
plt.title("Third Layer Weight Matrix (256 → 128)")
plt.xlabel("Output Neurons")
plt.ylabel("Input Neurons")
plt.tight_layout()
plt.show()
