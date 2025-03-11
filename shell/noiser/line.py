import torch

from shell.noiser.base import Noiser, NoiserOutput
from shell.utils.decorator import register_init_params


@register_init_params
class LineNoiser(Noiser):
    def __init__(self, timesteps: int, sigma: float):
        super(LineNoiser, self).__init__()
        self.timesteps = timesteps
        
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps)
        alpha = 1 - x / steps
        self.alpha = torch.nn.Parameter(alpha, requires_grad=False)
        self.sigma = torch.nn.Parameter(torch.full((steps,), sigma), requires_grad=False)
        gamma = self._calc_gamma(alpha**2, self.sigma**2)
        self.gamma = torch.nn.Parameter(gamma, requires_grad=False)
    
    def forward(self, t: torch.Tensor) -> NoiserOutput:
        t_int = torch.round(t * self.timesteps).long()
        gamma = self.gamma[t_int]
        alpha = self.alpha[t_int]
        sigma = self.sigma[t_int]
        
        return NoiserOutput(alpha=alpha, sigma=sigma, gamma=gamma)