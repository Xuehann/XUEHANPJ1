import numpy as np
import matplotlib.pyplot as plt
import gzip
from struct import unpack
from mynn.models import Model_CNN

def load_one_image(path, index=0):
    with gzip.open(path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images[index].astype(np.float32) / 255.0

activation_map_list = []

for index in range(5):
    img = load_one_image("dataset/MNIST/t10k-images-idx3-ubyte.gz", index)
    X = img[np.newaxis, ...]

    model = Model_CNN()
    model.load_model("modelCNN.pkl") 

    activation_maps = None
    for layer in model.layers:
        if layer.__class__.__name__ == "Linear" and X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = layer(X)
        if activation_maps is None and layer.__class__.__name__ == "Conv2D":
            activation_maps = X[0]
    if activation_maps is None:
        raise RuntimeError("First Conv2D output not found.")
    activation_map_list.append(activation_maps)

num_images = len(activation_map_list)
num_filters = activation_map_list[0].shape[0]
cols = 8
rows = num_images

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.reshape(rows, cols)

for row in range(rows):
    for col in range(cols):
        if col < num_filters:
            axes[row, col].imshow(activation_map_list[row][col], cmap='gray')
            axes[row, col].set_title(f"Img {row} - F{col}", fontsize=8)
        axes[row, col].axis('off')

plt.suptitle("First Conv Layer Activations for 5 Images", fontsize=16)
plt.tight_layout()
plt.show()
