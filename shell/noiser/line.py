import torch

from shell.noiser.base import Noiser, NoiserOutput
from shell.utils.decorator import register_init_params


@register_init_params
class LineNoiser(Noiser):
    def __init__(self, timesteps: int, sigma: float):
        self.timesteps = timesteps
        
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps)
        alpha = 1 - x / steps
        self.alpha = alpha
        self.sigma = torch.full((steps,), sigma)
        self.gamma = self._calc_gamma(alpha**2, self.sigma**2)
    
    def forward(self, t: torch.Tensor) -> NoiserOutput:
        t_int = torch.round(t * self.timesteps).long()
        gamma = self.gamma[t_int]

        alpha = torch.sqrt(torch.sigmoid(-gamma))
        sigma = torch.sqrt(torch.sigmoid(gamma))
        
        return NoiserOutput(alpha=alpha, sigma=sigma, gamma=gamma)