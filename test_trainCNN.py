
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os
import mynn as nn

SEED = 309
TRAIN_IMAGES_PATH = 'dataset/MNIST/train-images-idx3-ubyte.gz'
TRAIN_LABELS_PATH = 'dataset/MNIST/train-labels-idx1-ubyte.gz'
VALIDATION_SPLIT = 10000
INIT_LR = 0.001 
WEIGHT_DECAY = 0
EPOCHS = 5
BATCH_SIZE = 64
SAVE_DIR = 'best_models'
np.random.seed(SEED)

def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels


train_imgs, train_labs = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
valid_imgs, valid_labs = train_imgs[-VALIDATION_SPLIT:], train_labs[-VALIDATION_SPLIT:]
train_imgs, train_labs = train_imgs[:-VALIDATION_SPLIT], train_labs[:-VALIDATION_SPLIT]

print("[INFO] Train shape:", train_imgs.shape)

model = nn.models.Model_CNN(weight_decay_lambda=WEIGHT_DECAY)

loss_fn = nn.op.MultiCrossEntropyLoss(model)
optimizer = nn.optimizer.MomentGD(INIT_LR, model, mu=0.9)
scheduler = nn.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
metric = nn.metric.accuracy

runner = nn.runner.RunnerM(model, optimizer, metric, loss_fn, batch_size=BATCH_SIZE, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=EPOCHS,
             log_iters=200,
             save_dir=SAVE_DIR)

plt.figure()
plt.plot(runner.train_scores, label="Train Accuracy")
plt.plot(runner.dev_scores, label="Validation Accuracy")
plt.xlabel("Evaluation Steps")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy over Time")
plt.legend()
plt.grid(True)
plt.show()
