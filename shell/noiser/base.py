from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


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