import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.optimizable = True

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad):
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        return grad @ self.W.T

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grads):
        return grads * (self.input > 0)

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = np.zeros((N, C, out_h, out_w))
        self.argmax = np.zeros_like(out, dtype=int)
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * s, i * s + k
                w_start, w_end = j * s, j * s + k
                patch = X[:, :, h_start:h_end, w_start:w_end]
                patch_flat = patch.reshape(N, C, -1)
                self.argmax[:, :, i, j] = np.argmax(patch_flat, axis=-1)
                out[:, :, i, j] = np.max(patch_flat, axis=-1)
        return out

    def backward(self, grad_output):
        if grad_output.ndim == 2:
            N, flat = grad_output.shape
            C = self.input.shape[1]
            out_h = (self.input.shape[2] - self.kernel_size) // self.stride + 1
            out_w = (self.input.shape[3] - self.kernel_size) // self.stride + 1
            grad_output = grad_output.reshape(N, C, out_h, out_w)

        N, C, H, W = self.input.shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        grad_input = np.zeros_like(self.input)

        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * s, i * s + k
                w_start, w_end = j * s, j * s + k

                patch = self.input[:, :, h_start:h_end, w_start:w_end].reshape(N, C, -1)
                max_indices = self.argmax[:, :, i, j]
                grad_patch = np.zeros_like(patch)

                for n in range(N):
                    for c in range(C):
                        idx = max_indices[n, c]
                        grad_patch[n, c, idx] = grad_output[n, c, i, j]

                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_patch.reshape(N, C, k, k)

        return grad_input



class MultiCrossEntropyLoss:
    def __init__(self, model=None, max_classes=10):
        self.model = model
        self.max_classes = max_classes
        self.input = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, logits, labels):
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        self.input = logits
        self.labels = labels
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        N = logits.shape[0]
        correct_logprobs = -np.log(self.probs[np.arange(N), labels] + 1e-9)
        return np.sum(correct_logprobs) / N

    def backward(self):
        N = self.input.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        grad /= N
        self.grads = grad
        self.model.backward(self.grads)


def im2col(X, kernel_size):
    N, C, H, W = X.shape
    k = kernel_size
    out_h = H - k + 1
    out_w = W - k + 1
    cols = np.zeros((N, C, k, k, out_h, out_w))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = X[:, :, i:i+out_h, j:j+out_w]
    return cols.reshape(N, C * k * k, out_h * out_w)

def col2im(cols, X_shape, kernel_size):
    N, C, H, W = X_shape
    k = kernel_size
    out_h = H - k + 1
    out_w = W - k + 1
    cols_reshaped = cols.reshape(N, C, k, k, out_h, out_w)
    X_grad = np.zeros((N, C, H, W))
    for i in range(k):
        for j in range(k):
            X_grad[:, :, i:i+out_h, j:j+out_w] += cols_reshaped[:, :, i, j, :, :]
    return X_grad

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, weight_decay=False, weight_decay_lambda=1e-4):
        super().__init__()
        print(f"Conv2D layer init: in={in_channels}, out={out_channels}, kernel={kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels,))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}

        self.input = None
        self.X_col = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        self.X_col = im2col(X, self.kernel_size)  
        W_col = self.W.reshape(self.out_channels, -1)  

        out = np.matmul(self.X_col.transpose(0, 2, 1), W_col.T).transpose(0, 2, 1) 
        out += self.b[None, :, None]

        H_out = (X.shape[2] - self.kernel_size) // self.stride + 1
        W_out = (X.shape[3] - self.kernel_size) // self.stride + 1
        return out.reshape(X.shape[0], self.out_channels, H_out, W_out)

    def backward(self, dY):
        N, C_out, out_h, out_w = dY.shape
        dY_reshaped = dY.reshape(N, C_out, -1)
        X_col = self.X_col 
        W_col = self.W.reshape(C_out, -1)

        dW_col = np.zeros_like(W_col)
        for n in range(N):
            dW_col += dY_reshaped[n] @ X_col[n].T
        dW = (dW_col / N).reshape(self.W.shape)
        db = np.sum(dY_reshaped, axis=(0, 2)) / N

        dX_col = np.zeros_like(X_col)
        for n in range(N):
            dX_col[n] = W_col.T @ dY_reshaped[n]
        dX = col2im(dX_col, self.input.shape, self.kernel_size)

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
