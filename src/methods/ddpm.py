"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod
from ..models.unet import UNet


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
        parametrization: Literal["epsilon", "x0"] = 'epsilon',
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        # TODO: Implement your own init

        # =========================================================================
        # You can add, delete or modify as many functions as you would like
        # =========================================================================

        # Pro tips: If you have a lot of pseudo parameters that you will specify for each
        # model run but will be fixed once you specified them (say in your config),
        # then you can use super().register_buffer(...) for these parameters

        # Pro tips 2: If you need a specific broadcasting for your tensors,
        # it's a good idea to write a general helper function for that

        valid_parametrizations = ["epsilon", "x0"]
        if parametrization not in valid_parametrizations:
            raise ValueError(
                f'Invalid value passed for parametrization in DDPM, received {self.parametrization} but can only take ["epsilon", "x0"]'
            )
        self.parametrization = parametrization

        beta_t = torch.linspace(start=beta_start, end=beta_end, steps=num_timesteps, device=device)
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        alpha_bar_t_minus_one = torch.cat([torch.tensor([1.0], device=device), alpha_bar_t[:-1]])
        beta_tilde_t = beta_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)
        sqrt_beta_tilde_t = torch.sqrt(beta_tilde_t)

        # Precomputed coefficients for reverse process mean
        sqrt_alpha_t = torch.sqrt(alpha_t)
        # epsilon parametrization: mu = eps_coeff_xt * x_t + eps_coeff_pred * eps_theta
        eps_coeff_xt = 1.0 / sqrt_alpha_t
        eps_coeff_pred = -beta_t / (sqrt_alpha_t * sqrt_one_minus_alpha_bar_t)
        # x0 parametrization: mu = x0_coeff_xt * x_t + x0_coeff_pred * x0_hat
        x0_coeff_xt = sqrt_alpha_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)
        x0_coeff_pred = beta_t * torch.sqrt(alpha_bar_t_minus_one) / (1 - alpha_bar_t)

        # Noise schedule fundamentals (shared by DDPM and DDIM)
        super().register_buffer("beta_t", beta_t)
        super().register_buffer("alpha_t", alpha_t)
        super().register_buffer("alpha_bar_t", alpha_bar_t)
        super().register_buffer("alpha_bar_t_minus_one", alpha_bar_t_minus_one)
        super().register_buffer("sqrt_alpha_t", sqrt_alpha_t)
        super().register_buffer("sqrt_alpha_bar_t", sqrt_alpha_bar_t)
        super().register_buffer("sqrt_one_minus_alpha_bar_t", sqrt_one_minus_alpha_bar_t)
        # Reverse process buffers
        super().register_buffer("sqrt_beta_tilde_t", sqrt_beta_tilde_t)
        super().register_buffer("eps_coeff_xt", eps_coeff_xt)
        super().register_buffer("eps_coeff_pred", eps_coeff_pred)
        super().register_buffer("x0_coeff_xt", x0_coeff_xt)
        super().register_buffer("x0_coeff_pred", x0_coeff_pred)

        # DDPM reverse process coefficients
        super().register_buffer("ddpm_posterior_variance", beta_tilde_t)
        super().register_buffer("ddpm_posterior_std", sqrt_beta_tilde_t)
        super().register_buffer("ddpm_eps_coeff_xt", eps_coeff_xt)
        super().register_buffer("ddpm_eps_coeff_pred", eps_coeff_pred)
        super().register_buffer("ddpm_x0_coeff_xt", x0_coeff_xt)
        super().register_buffer("ddpm_x0_coeff_pred", x0_coeff_pred)

    # =========================================================================
    # Forward process
    # =========================================================================

    def __gather(self, buf: torch.Tensor, t: torch.Tensor):
        return buf.take(t).reshape(-1, 1, 1, 1)

    def forward_process(
        self, x_0: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor
    ):  # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        x_t = (
            self.__gather(self.sqrt_alpha_bar_t, t) * x_0
            + self.__gather(self.sqrt_one_minus_alpha_bar_t, t) * epsilon
        )
        return x_t

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        epsilon = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, epsilon)

        prediction = self.model.forward(x_t, t)

        # Compute loss in fp32 to avoid fp16 overflow in squared differences
        if self.parametrization == "epsilon":
            target = epsilon
        else:
            target = x_0
        loss = F.mse_loss(prediction.float(), target.float())

        metrics = {"loss": loss.detach().item()}
        metrics["pred_mean"] = prediction.mean().item()
        metrics["pred_std"] = prediction.std().item()
        metrics["target_std"] = target.std().item()

        with torch.no_grad():
            # Create 4 buckets: 0-249, 250-499, 500-749, 750-999
            bucket_size = self.num_timesteps // 4
            for i in range(4):
                bucket_mask = (t >= i * bucket_size) & (t < (i + 1) * bucket_size)
                if bucket_mask.any():
                    bucket_loss = F.mse_loss(
                        prediction[bucket_mask].float(), target[bucket_mask].float()
                    )
                    metrics[f"loss_t_bucket_{i}"] = bucket_loss.item()

        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sampler: str = "ddpm",
        t_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step of the reverse process, dispatching on the sampler type.

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: Current timestep
            sampler: 'ddpm' or 'ddim'
            t_prev: Previous timestep in the subsequence (needed by DDIM)

        Returns:
            x_prev: Denoised samples at the previous timestep
        """
        prediction = self.model.forward(x_t, t)

        if sampler == "ddpm":
            sigma_t = self.__gather(self.ddpm_posterior_std, t)
            z = torch.randn_like(x_t) * (t != 0).float().reshape(
                -1, 1, 1, 1
            )  # reshaping to align with B, C, H, W

            if self.parametrization == "epsilon":
                mu = (
                    self.__gather(self.ddpm_eps_coeff_xt, t) * x_t
                    + self.__gather(self.ddpm_eps_coeff_pred, t) * prediction
                )
            else:
                mu = (
                    self.__gather(self.ddpm_x0_coeff_xt, t) * x_t
                    + self.__gather(self.ddpm_x0_coeff_pred, t) * prediction
                )
            x_prev = mu + sigma_t * z

        elif sampler == "ddim":
            # TODO: Implement DDIM reverse step here.
            # Use self.alpha_bar_t looked up at t and t_prev to compute
            # the DDIM update. Coefficients must be computed on the fly
            # because t_prev depends on the timestep subsequence.
            x_0_hat = (x_t - self.__gather(
                self.sqrt_one_minus_alpha_bar_t, t
            ) * prediction) / self.__gather(self.sqrt_alpha_bar_t, t)
            x_0_hat = x_0_hat.clip(-1, 1)
            x_prev = (
                self.__gather(self.sqrt_alpha_bar_t, t_prev) * x_0_hat
                + self.__gather(self.sqrt_one_minus_alpha_bar_t, t_prev) * prediction
            )

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        sampler: str = "ddpm",
        **kwargs,
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps. If less than num_timesteps,
                       uses evenly spaced timesteps from the full schedule.
            **kwargs: Additional method-specific arguments

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x_t = torch.randn(size=(batch_size, *image_shape), device=self.device)

        if num_steps is not None and num_steps < self.num_timesteps:
            # Subsample evenly spaced timesteps from the full schedule
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1)

        for i, timestep in enumerate(timesteps):
            t = timestep * torch.ones(size=(batch_size,), device=self.device, dtype=torch.int64)
            # t_prev is the next timestep in the list or 0 at the end
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = torch.tensor(0, dtype=torch.long)
            t_prev = t_prev_val * torch.ones(
                size=(batch_size,), device=self.device, dtype=torch.int64
            )
            x_t = self.reverse_process(x_t, t, sampler=sampler, t_prev=t_prev)
        x_t = x_t.clamp(-1, 1)
        self.train_mode()
        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
            # parametrization=ddpm_config["parametrization"],
        )
