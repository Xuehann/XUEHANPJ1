import numpy as np
import scipy.ndimage
import gzip
from struct import unpack
import os

def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels

def augment_image(img, rotation_range=18, zoom_range=0.2):
    img = img[0]  

    angle = np.random.uniform(-rotation_range, rotation_range)
    rotated = scipy.ndimage.rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0)

    zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    zoomed = scipy.ndimage.zoom(rotated, zoom_factor, order=1)

    if zoomed.shape[0] > 28:
        crop = (zoomed.shape[0] - 28) // 2
        final = zoomed[crop:crop+28, crop:crop+28]
    else:
        pad = 28 - zoomed.shape[0]
        final = np.pad(zoomed, ((pad//2, pad - pad//2), (pad//2, pad - pad//2)), mode='constant', constant_values=0)

    return final.reshape(1, 28, 28).astype(np.float32)

def main():
    train_images_path = "dataset/MNIST/train-images-idx3-ubyte.gz"
    train_labels_path = "dataset/MNIST/train-labels-idx1-ubyte.gz"
    output_path = "dataset/MNIST/combined_train.npz"

    train_imgs, train_labs = load_mnist(train_images_path, train_labels_path)

    augmented_imgs = np.zeros_like(train_imgs)
    for i in range(len(train_imgs)):
        augmented_imgs[i] = augment_image(train_imgs[i])
    combined_imgs = np.concatenate([train_imgs, augmented_imgs], axis=0)
    combined_labs = np.concatenate([train_labs, train_labs], axis=0)

    print(f"Saving combined dataset to {output_path}")
    np.savez_compressed(output_path, images=combined_imgs, labels=combined_labs)

if __name__ == "__main__":
    main()
