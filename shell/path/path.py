from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from flow_matching.path import (AffineProbPath, GeodesicProbPath,
                                MixtureDiscreteProbPath)
from flow_matching.path.scheduler import (CondOTScheduler,
                                          PolynomialConvexScheduler, Scheduler)
from flow_matching.utils.manifolds import Sphere

from shell.data.mol import SphMol


def get_scheduler(scheduler: Literal['ot', 'poly']) -> Scheduler:
    if scheduler == 'ot':
        return CondOTScheduler()
    elif scheduler == 'poly':
        return PolynomialConvexScheduler(n=2.0)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")
    

class SphMolPath:
    def __init__(
        self,
        x_scheduler: Literal['ot', 'poly'] = 'poly',
        v_scheduler: Literal['ot', 'poly'] = 'ot',
        r_scheduler: Literal['ot', 'poly'] = 'ot',
    ):
        self.x_path = MixtureDiscreteProbPath(get_scheduler(x_scheduler))
        self.v_path = GeodesicProbPath(get_scheduler(v_scheduler), Sphere())
        self.r_path = AffineProbPath(get_scheduler(r_scheduler))
        
    def  sample(
        self,
        sphmol0: SphMol,
        sphmol1: SphMol,
        t: torch.Tensor
    ) -> Tuple[SphMol, torch.Tensor, torch.Tensor]:
        num_atom_types = sphmol0.x.shape[1]        
        x_sample = self.x_path.sample(sphmol0.x.argmax(-1), sphmol1.x.argmax(-1), t)
        x_t = F.one_hot(x_sample.x_t, num_classes=num_atom_types).float()
        
        v_sample = self.v_path.sample(sphmol0.v, sphmol1.v, t)
        r_sample = self.r_path.sample(sphmol0.r, sphmol1.r, t)
        
        sphmol = SphMol(x_t, v_sample.x_t, r=r_sample.x_t, b=sphmol0.b)
        dvdt = v_sample.dx_t
        drdt = r_sample.dx_t
        return sphmol, dvdt, drdt