from __future__ import annotations

import torch


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str | None = None,
    ):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("Expected 0 < beta_start < beta_end < 1.")

        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        device = torch.device(device)
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.device = device
        return self

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        timesteps = timesteps.to(device=values.device, dtype=torch.long)
        out = values.gather(0, timesteps)
        while out.ndim < len(target_shape):
            out = out.unsqueeze(-1)
        return out

    def sample_timesteps(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device(device) if device is not None else self.device
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(
        self,
        clean_future: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_future.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_future.shape)
        return sqrt_alpha_bar * clean_future + sqrt_one_minus_alpha_bar * noise

    def step(
        self,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        beta_t = self._extract(self.betas, timesteps, sample.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, sample.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, sample.shape)

        model_mean = sqrt_recip_alpha * (sample - beta_t * pred_noise / sqrt_one_minus_alpha_bar)
        noise = torch.randn_like(sample)
        nonzero_mask = (timesteps != 0).to(sample.dtype)
        while nonzero_mask.ndim < sample.ndim:
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        posterior_variance = self._extract(self.posterior_variance, timesteps, sample.shape)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        device: torch.device | str,
    ) -> torch.Tensor:
        device = torch.device(device)
        sample = torch.randn(shape, device=device)
        for t in reversed(range(self.num_train_timesteps)):
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = denoise_fn(sample, timesteps)
            sample = self.step(pred_noise, timesteps, sample)
        return sample

