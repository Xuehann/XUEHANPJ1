import numpy as np
from mynn.op import Linear, ReLU, Conv2D, MaxPool2D
import pickle

class Model_MLP:
    def __init__(self, layer_sizes=[784, 600, 10], activation='ReLU', weight_decay_lambda=None):
        self.layers = []
        self.activation = activation
        num_layers = len(layer_sizes) - 1

        if weight_decay_lambda is None:
            weight_decay_lambda = [0.0] * num_layers
        elif isinstance(weight_decay_lambda, (int, float)):
            weight_decay_lambda = [weight_decay_lambda] * num_layers
        elif isinstance(weight_decay_lambda, list):
            if len(weight_decay_lambda) != num_layers:
                raise ValueError("Length mismatch in weight_decay_lambda")

        for i in range(num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            use_activation = (i < num_layers - 1)
            init_fn = lambda size: np.random.randn(*size) * np.sqrt(2 / size[0])
            self.layers.append(Linear(in_dim, out_dim,
                                      initialize_method=init_fn,
                                      weight_decay=True,
                                      weight_decay_lambda=weight_decay_lambda[i]))
            if use_activation and activation == 'ReLU':
                self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def clear_gradients(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    def save_model(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, path):
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)

class Model_CNN:
    def __init__(self, weight_decay_lambda=1e-4):
        self.layers = []
        self.layers.append(Conv2D(in_channels=1,  out_channels=32, kernel_size=3,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(in_channels=32, out_channels=32, kernel_size=3,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))  

        self.layers.append(Conv2D(in_channels=32, out_channels=64, kernel_size=3,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(in_channels=64, out_channels=64, kernel_size=3,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2)) 

        dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
        x = dummy
        for layer in self.layers:
            x = layer(x)
        self.flatten_shape = x.size
        self.last_conv_shape = None

        self.layers.append(Linear(self.flatten_shape, 128,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        self.layers.append(Linear(128, 10,
                                  weight_decay=True, weight_decay_lambda=weight_decay_lambda))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        from mynn.op import Linear as _Lin
        for layer in self.layers:
            if isinstance(layer, _Lin) and X.ndim > 2:
                self.last_conv_shape = X.shape
                X = X.reshape(X.shape[0], -1)
            X = layer(X)
        return X

    def backward(self, grad):
        from mynn.op import Linear as _Lin
        for layer in reversed(self.layers):
            if isinstance(layer, _Lin) and self.last_conv_shape and grad.ndim == 2:
                if grad.shape[1] == np.prod(self.last_conv_shape[1:]):
                    grad = grad.reshape(self.last_conv_shape)
            grad = layer.backward(grad)
        return grad

    def clear_gradients(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    def save_model(self, path):
        for layer in self.layers:
            if hasattr(layer, 'input'):
                layer.input = None
            if hasattr(layer, 'X_col'):
                layer.X_col = None
            if hasattr(layer, 'grads'):
                layer.grads = {'W': None, 'b': None}
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
