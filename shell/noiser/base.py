from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
from torch_scatter import scatter_mean


@dataclass
class NoiserOutput:
    alpha: torch.Tensor
    sigma: torch.Tensor
    
    gamma: torch.Tensor = None


class Noiser(nn.Module, ABC):
    @abstractmethod
    def forward(self, t: torch.Tensor) -> NoiserOutput:
        pass
    
    def forward_batch(
        self, 
        t: torch.Tensor, 
        batch: torch.Tensor
    ) -> NoiserOutput:
        output = self.forward(t)
        for attr, value in output.__dict__.items():
            if value is not None:
                setattr(output, attr, value[batch])
        return output
    
    # def to_device(self, device: torch.device):
    #     self.alpha = self.alpha.to(device)
    #     self.sigma = self.sigma.to(device)
    #     self.gamma = self.gamma.to(device)
    #     return self
    
    def noise(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        remove_com: bool = False
    ) -> torch.Tensor:
        output = self.forward_batch(t, batch)
        eps = torch.randn_like(x)
        if remove_com:
            com = scatter_mean(x, batch, dim=0)
            eps -= com[batch]
        return output.alpha * x + output.sigma * eps
    
    def noise_traj(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        timesteps: int = 1000,
        remove_com: bool = True
    ) -> torch.Tensor:
        x_traj = [x]
        for t in range(timesteps + 1):
            t = torch.tensor([t / timesteps], dtype=torch.float32)
            x = self.noise(t, x, batch, remove_com)
            x_traj.append(x)
        return torch.stack(x_traj, dim=0)
    
    def _calc_gamma(
        self,
        alpha2: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        log_alpha2 = torch.log(alpha2)
        log_sigma2 = torch.log(sigma2)
        log_alphas2_to_sigmas2 = log_alpha2 - log_sigma2
        return -log_alphas2_to_sigmas2
    
    @property
    def init_params(self):
        return getattr(self, '_init_params')