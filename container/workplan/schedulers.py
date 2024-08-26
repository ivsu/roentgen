import math
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import AdamW
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from accelerate import Accelerator
from plotly import graph_objects as go


class WarmupAndDecayScheduler(LRScheduler):
    """
    Изменяет шаг обучения с линейным прогревом и экспоненциальным затуханием
    """
    def __init__(self, optimizer, warmup_steps: int, decay_steps: int, decay_rate: float,
                 target_lr: float, initial_lr: float = 1e-5, final_lr: float = 1e-5,
                 change_on_every_batch: bool = False, steps_per_epoch: int = None,
                 device=None):

        self.device = device
        self.steps_per_epoch = self.as_tensor(steps_per_epoch)
        self.change_on_every_batch = change_on_every_batch

        self.name = "WarmupAndDecayScheduler"
        self.initial_lr = self.as_tensor(initial_lr)
        self.warmup_steps = self.as_tensor(warmup_steps, dtype=torch.int)
        self.target_lr = self.as_tensor(target_lr)

        self.decay_steps = self.as_tensor(decay_steps, dtype=torch.int)

        self.final_lr = self.as_tensor(final_lr)
        # self.decay_factor = self.as_tensor(-math.log(decay_rate) / decay_steps)
        self.decay_rate = decay_rate

        # начальное значение шага
        self.update_steps = self.as_tensor(0, dtype=torch.int)
        # устанавливаем начальное значение оптимизатору
        self.set_lr(optimizer, initial_lr)

        super().__init__(optimizer)

    def as_tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    @classmethod
    def set_lr(cls, optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    # линейная функция прогрева
    def _warmup_function(self, step):
        completed_fraction = step / self.warmup_steps
        total_step_delta = self.target_lr - self.initial_lr
        return total_step_delta * completed_fraction + self.initial_lr

    # экспоненциальная функция затухания
    def _decay_function(self, step):

        x = step / self.decay_steps
        # print(f'step: {step}, x: {x}')

        # Рассчитаем коэффициенты a и b экспоненциальной функции вида y = a * exp(-x / tau) + b
        a = (self.target_lr - self.final_lr) / (1. - math.exp(-1 / self.decay_rate))
        b = self.final_lr - a * math.exp(-1 / self.decay_rate)

        # Построим график экспоненциальной функции
        return a * math.exp(-x / self.decay_rate) + b

    def step(self, epoch=None):

        if self.update_steps <= self.warmup_steps:
            lr = self._warmup_function(self.update_steps)
        elif self.update_steps <= self.warmup_steps + self.decay_steps:
            lr = self._decay_function(self.update_steps - self.warmup_steps)
        else:
            lr = self.final_lr

        if self.change_on_every_batch or \
                not self.change_on_every_batch and self.update_steps % self.steps_per_epoch == 0:
            self.set_lr(self.optimizer, lr)

        self.update_steps = self.update_steps.add(1)

        super().step(epoch)


def print_lr(lrs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[step for step in range(len(lrs))],
        y=lrs,
        mode='lines',
        name='Learning rates',
    ))
    fig.show()


if __name__ == '__main__':

    epochs = 40
    num_batches = 2
    warmup_epochs = 5
    # decay_epochs = epochs - warmup_epochs
    decay_epochs = 25

    config = TimeSeriesTransformerConfig(prediction_length=5)
    model = TimeSeriesTransformerForPrediction(config)
    optimizer = AdamW(model.parameters())

    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)

    scheduler = WarmupAndDecayScheduler(
        optimizer,
        warmup_steps=num_batches * warmup_epochs,
        decay_steps=num_batches * decay_epochs,
        decay_rate=0.3,
        target_lr=1e-3,
        initial_lr=5e-5,
        final_lr=5e-5,
        steps_per_epoch=num_batches,
        change_on_every_batch=True,
        device=device
    )

    lrs = []
    step = 0
    for epoch in range(epochs):
        for batch in range(num_batches):
            test_lr = scheduler.get_last_lr()[0]
            lrs.append(test_lr)
            print(f'step: {step}, LR: {test_lr:.6f}')
            scheduler.step()
            step += 1

    print_lr(lrs)