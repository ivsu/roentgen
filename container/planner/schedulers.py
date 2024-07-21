import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmAndDecayScheduler(LRScheduler):
    """
    Изменяет шаг обучения с линейным прогревом и экспоненциальным затуханием
    """
    def __init__(self, optimizer, warmup_steps, decay_steps, decay_rate,
                 target_learning_rate, initial_learning_rate=1e-5,
                 device=None):

        self.dtype = torch.float32
        self.device = device

        # стартовый шаг обучения - начало прогрева
        self.initial_learning_rate = self.as_tensor(initial_learning_rate)
        self.decay_steps = self.as_tensor(decay_steps)
        self.decay_rate = self.as_tensor(decay_rate)

        self.name = "WarmAndDecayScheduler"
        self.warmup_steps = self.as_tensor(warmup_steps)
        self.warmup_target = self.as_tensor(target_learning_rate)

        # начальное значение шага
        # self.current_step = self.as_tensor(0)

        self.learning_rate = self.initial_learning_rate

        super().__init__(optimizer)

        # print(f'self.optimizer.param_groups: {self.optimizer.param_groups}')

    def as_tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def get_lr(self):
        return [self.learning_rate]

    # линейная функция прогрева
    def _warmup_function(self, step):
        completed_fraction = step / self.warmup_steps
        total_step_delta = self.warmup_target - self.initial_learning_rate
        return total_step_delta * completed_fraction + self.initial_learning_rate

    # экспоненциальная функция затухания
    def _decay_function(self, step):
        p = step / self.decay_steps
        return torch.multiply(
            self.warmup_target, torch.pow(self.decay_rate, p)
        )

    def step(self, epoch=None):

        # print(f'scheduler._step_count {self._step_count}, optimizer._step_count {self.optimizer._step_count}, ')

        # self.current_step = self.current_step.add(1)

        global_step_recomp = torch.minimum(
            self.as_tensor(self._step_count),
            self.decay_steps + self.warmup_steps
        )

        self.learning_rate = torch.where(
            global_step_recomp < self.warmup_steps,
            self._warmup_function(global_step_recomp),
            self._decay_function(global_step_recomp - self.warmup_steps)
        )
        # print(f'step: {self.current_step.item()}, LR: {self.learning_rate}')
        super().step(epoch)
        # return self.learning_rate
