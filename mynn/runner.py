import numpy as np
import os
from tqdm import tqdm

class RunnerM():

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.best_score = -np.inf

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 10)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        early_stop = kwargs.get("early_stop", None)

        os.makedirs(save_dir, exist_ok=True)
        no_improve = 0

        X_train, y_train = train_set
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / self.batch_size))

        for epoch in range(num_epochs):
            idx = np.random.permutation(num_samples)
            X_epoch, y_epoch = X_train[idx], y_train[idx]

            epoch_loss = 0
            epoch_score = 0

            with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for iteration in pbar:
                    start = iteration * self.batch_size
                    end = min((iteration + 1) * self.batch_size, num_samples)
                    X_batch = X_epoch[start:end]
                    y_batch = y_epoch[start:end]

                    logits = self.model(X_batch)
                    loss = self.loss_fn(logits, y_batch)
                    score = self.metric(logits, y_batch)

                    #DEBUG: Logit distribution check
                    if epoch == 0 and iteration == 0:
                        print("[DEBUG] Logits sample:", logits[0])
                        print("[DEBUG] Logits max diff:", np.max(logits[0]) - np.min(logits[0]))

                    self.loss_fn.backward()

                    #print("[DEBUG] Grad mean after backward:", np.mean(self.loss_fn.grads))
                    for layer in self.model.layers:
                        if hasattr(layer, 'grads') and 'W' in layer.grads:
                            #print("[DEBUG] Conv2D dW mean:", np.mean(layer.grads['W']))
                            break 

                    self.optimizer.step()
                    self.model.clear_gradients()

                    self.train_loss.append(loss)
                    self.train_scores.append(score)
                    epoch_loss += loss * (end - start)
                    epoch_score += score * (end - start)

                    if (iteration + 1) % log_iters == 0:
                        dev_score, dev_loss = self.evaluate(dev_set)
                        self.dev_scores.append(dev_score)
                        self.dev_loss.append(dev_loss)

                        pbar.set_postfix({
                            'train_loss': f"{loss:.4f}",
                            'train_acc': f"{score:.4f}",
                            'val_loss': f"{dev_loss:.4f}",
                            'val_acc': f"{dev_score:.4f}"
                        })

            if self.scheduler is not None:
                self.scheduler.step()
                print(f"[Epoch {epoch+1}] LR: {self.optimizer.lr:.5f}")

            dev_score, dev_loss = self.evaluate(dev_set)
            print(f"[Epoch {epoch+1}] Final Val Acc: {dev_score:.4f}, Final Val Loss: {dev_loss:.4f}")

            if dev_score > self.best_score:
                self.best_score = dev_score
                save_path = os.path.join(save_dir, 'best_model.pkl')
                self.save_model(save_path)
                no_improve = 0
            else:
                no_improve += 1

            if early_stop and no_improve >= early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)

    def load_model(self, load_path):
        self.model.load_model(load_path)
