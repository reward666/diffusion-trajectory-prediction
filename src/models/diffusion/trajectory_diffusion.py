from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.diffusion.denoiser import MLPDenoiser
from src.models.diffusion.scheduler import DDPMScheduler
from src.models.diffusion.time_embedding import TimeEmbeddingMLP
from src.models.encoders.trajectory_encoder import EgoLeaderEncoder


class TrajectoryDiffusion(nn.Module):
    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...],
        pred_len: int = 50,
        future_dim: int = 2,
        condition_dim: int = 128,
        time_dim: int = 128,
        denoiser_hidden_dim: int = 512,
        denoiser_num_layers: int = 4,
        num_train_timesteps: int = 100,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.future_dim = future_dim

        self.encoder = EgoLeaderEncoder(
            feature_names=feature_names,
            output_dim=condition_dim,
        )
        self.time_embedding = TimeEmbeddingMLP(
            output_dim=time_dim,
        )
        self.denoiser = MLPDenoiser(
            pred_len=pred_len,
            future_dim=future_dim,
            time_dim=time_dim,
            condition_dim=condition_dim,
            hidden_dim=denoiser_hidden_dim,
            num_layers=denoiser_num_layers,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs) -> "TrajectoryDiffusion":
        module = super().to(*args, **kwargs)
        self.scheduler.to(self.device)
        return module

    def predict_noise(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        condition_emb: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)
        return self.denoiser(noisy_future, time_emb, condition_emb)

    def forward(
        self,
        past: torch.Tensor,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        condition_emb = self.encoder(past)
        return self.predict_noise(noisy_future, timesteps, condition_emb)

    def training_loss(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if future.ndim != 3:
            raise ValueError(f"Expected future shape [batch, pred_len, future_dim], got {tuple(future.shape)}.")
        if future.shape[1] != self.pred_len or future.shape[2] != self.future_dim:
            raise ValueError(
                f"Expected future shape [batch, {self.pred_len}, {self.future_dim}], got {tuple(future.shape)}."
            )

        batch_size = future.shape[0]
        timesteps = self.scheduler.sample_timesteps(batch_size, device=future.device)
        noise = torch.randn_like(future)
        noisy_future = self.scheduler.add_noise(future, noise, timesteps)
        pred_noise = self.forward(past, noisy_future, timesteps)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, past: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        batch_size = past.shape[0]
        repeated_past = past.repeat_interleave(num_samples, dim=0)
        condition_emb = self.encoder(repeated_past)

        def denoise_fn(noisy_future: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return self.predict_noise(noisy_future, timesteps, condition_emb)

        generated = self.scheduler.sample(
            denoise_fn=denoise_fn,
            shape=(batch_size * num_samples, self.pred_len, self.future_dim),
            device=past.device,
        )
        return generated.reshape(batch_size, num_samples, self.pred_len, self.future_dim)

