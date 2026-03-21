from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(self, model: nn.Module, device: torch.device, num_timesteps: int):
        super().__init__(model, device)

        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps

    def __interpolate_target(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        t = t.reshape(t.shape[0], 1, 1, 1)
        return (1 - t) * x_0 + t * x_1

    def __compute_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor):
        return x_1 - x_0

    def compute_loss(self, x_0: torch.Tensor, **kwargs):
        batch_size = x_0.shape[0]
        x_1 = torch.randn_like(x_0, device=self.device)
        t = (
            torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            / self.num_timesteps
        )
        x_t = self.__interpolate_target(x_0, x_1, t)
        target_velocity = self.__compute_velocity(x_0, x_1)
        predicted_velocity = self.model(x_t, t)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, batch_size: int, image_shape: Tuple[int, int, int], num_steps: int = None, **kwargs):
        self.model.eval()
        num_steps = num_steps or self.num_timesteps
        tensor_shape = (batch_size, *image_shape)
        x_1 = torch.randn(tensor_shape, device=self.device)
        timesteps = torch.linspace(0, 1, num_steps + 1)
        step_size = 1.0 / num_steps
        x_t = x_1
        for timestep in reversed(timesteps[:-1]):
            t = timestep * torch.ones(size=(batch_size,), device=self.device, dtype=torch.float32)
            x_t = x_t - self.model(x_t, t) * step_size
        self.model.train()
        return x_t

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=fm_config["num_timesteps"],
        )
