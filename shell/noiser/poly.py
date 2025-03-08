import numpy as np
import torch

from shell.noiser.base import Noiser, NoiserOutput
from shell.utils.decorator import register_init_params


@register_init_params
class PolyNoiser(Noiser):
    def __init__(
        self, 
        timesteps: int, 
        precision: float,
        power: float = 2.0
    ):
        super(PolyNoiser, self).__init__()
        self.timesteps = timesteps

        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, power))**2
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
        precision = 1 - 2 * precision
        alphas2 = precision * alphas2 + precision

        sigmas2 = 1 - alphas2

        gamma = self._calc_gamma(
            torch.from_numpy(alphas2).float(),
            torch.from_numpy(sigmas2).float()
        )

        self.gamma = torch.nn.Parameter(gamma, requires_grad=False)

    def forward(self, t: torch.Tensor) -> NoiserOutput:
        t_int = torch.round(t * self.timesteps).long()
        gamma = self.gamma[t_int]
        
        # See https://arxiv.org/abs/2107.00630 Eq(4) and Eq(3)
        alpha = torch.sqrt(torch.sigmoid(-gamma))
        sigma = torch.sqrt(torch.sigmoid(gamma))
        
        return NoiserOutput(alpha=alpha, sigma=sigma, gamma=gamma)


def clip_noise_schedule(
    alphas2: np.ndarray, 
    clip_value: float = 0.001
) -> np.ndarray:
    """For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. 
    This may help improve stability during sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2