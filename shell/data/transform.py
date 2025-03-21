from typing import Tuple

import torch
import torch.nn.functional as F

from shell.utils.settings import EPS, MID_RADIUS


class GeometryTransform:
    def __init__(
        self,
        mid_radius: float = MID_RADIUS,
        scale: float = 5.0,
    ):
        self.mid_radius = mid_radius
        self.scale = scale
    
    def to_sphere(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = pos.norm(dim=-1, keepdim=True)
        v = pos / (r + EPS)
        r = torch.log(r / self.mid_radius + EPS) * self.scale
        r[0] = 0.0
        return v, r
    
    def to_cartes(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r = r / self.scale
        r = self.mid_radius * torch.exp(r)
        pos = v * r
        return pos


class AtomNumTransform:
    def __init__(
        self,
        unique_atom_nums: torch.Tensor,
    ):
        self.unique_atom_nums = unique_atom_nums
        self.num_atom_types = len(unique_atom_nums)
    
    def to_onehot(self, n: torch.Tensor) -> torch.Tensor:
        return F.one_hot(n, num_classes=self.num_atom_types).float()
    
    def to_num(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1)
    
    def to_atom_num(self, x: torch.Tensor) -> torch.Tensor:
        return self.unique_atom_nums[self.to_num(x)]