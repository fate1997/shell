from abc import ABC, abstractmethod
from torch import nn


class Denoiser(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @property
    def init_params(self):
        return getattr(self, '_init_params')