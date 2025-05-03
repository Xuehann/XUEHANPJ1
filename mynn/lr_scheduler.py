from abc import ABC, abstractmethod

class Scheduler(ABC):
    """Base class for learning rate schedulers"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        """Update the learning rate"""
        pass

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.lr 
        

class StepLR(Scheduler):
    """Decays learning rate by gamma every step_size steps"""
    def __init__(self, optimizer, step_size=30, gamma=0.1):
        """
        Args:
            optimizer: Optimizer to adjust
            step_size: Interval of epochs between decays
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        """Decay learning rate if at step interval"""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            old_lr = self.optimizer.lr
            self.optimizer.lr *= self.gamma
            print(f"[StepLR] Step {self.step_count}: LR decayed from {old_lr:.5f} to {self.optimizer.lr:.5f}")
