from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """Abstract base optimizer class"""
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.lr = init_lr 
        self.model = model

    @abstractmethod
    def step(self):
        """Perform one optimization step"""
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.lr * layer.grads[key]


class MomentGD(Optimizer):
    
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}

        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocities[i] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params:
                    self.velocities[i][key] = (
                        self.mu * self.velocities[i][key]
                        - self.lr * layer.grads[key]
                    )
                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)
                    layer.params[key] += self.velocities[i][key]
