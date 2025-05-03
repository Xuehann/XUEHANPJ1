import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from draw_tools.plot import plot
import mynn as nn
import os

SEED = 309
VALIDATION_SPLIT = 10000
INIT_LR = 0.005
WEIGHT_DECAY = 1e-4  
EPOCHS = 5
SAVE_DIR = 'best_models'
np.random.seed(SEED)

'''
TRAIN_IMAGES_PATH = 'dataset/MNIST/train-images-idx3-ubyte.gz'
TRAIN_LABELS_PATH = 'dataset/MNIST/train-labels-idx1-ubyte.gz'

def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

train_imgs, train_labs = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
'''

data = np.load('dataset/MNIST/combined_train.npz')
train_imgs = data['images']  
train_labs = data['labels']
train_imgs = train_imgs.reshape(-1, 28 * 28) 
#Use augmented datasets, comment out normalization 
#train_imgs = train_imgs / 255.0        

idx = np.random.permutation(len(train_imgs))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs, train_labs = train_imgs[idx], train_labs[idx]
valid_imgs = train_imgs[:VALIDATION_SPLIT]
valid_labs = train_labs[:VALIDATION_SPLIT]
train_imgs = train_imgs[VALIDATION_SPLIT:]
train_labs = train_labs[VALIDATION_SPLIT:]

model = nn.models.Model_MLP([784, 512, 256, 128, 10], 'ReLU', WEIGHT_DECAY)
# model = nn.models.Model_MLP([784, 128, 10], weight_decay_lambda=1e-4)
# model = nn.models.Model_CNN(weight_decay_lambda=WEIGHT_DECAY)

optimizer = nn.optimizer.MomentGD(INIT_LR, model, mu=0.9)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
             num_epochs=EPOCHS, log_iters=100, save_dir=SAVE_DIR)

_, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.reshape(-1)
_.set_tight_layout(True)
plot(runner, axes)
plt.show()
