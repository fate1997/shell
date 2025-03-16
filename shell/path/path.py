from typing import Literal, Tuple

import torch
from flow_matching.path import (AffineProbPath, GeodesicProbPath,
                                MixtureDiscreteProbPath)
from flow_matching.path.scheduler import (CondOTScheduler,
                                          PolynomialConvexScheduler, Scheduler)

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
        self.v_path = GeodesicProbPath(get_scheduler(v_scheduler))
        self.r_path = AffineProbPath(get_scheduler(r_scheduler))
        
    def sample(
        self,
        sphmol0: SphMol,
        sphmol1: SphMol,
        t: torch.Tensor
    ) -> Tuple[SphMol, torch.Tensor, torch.Tensor]:
        x_sample = self.x_path.sample(sphmol0.x, sphmol1.x, t)
        v_sample = self.v_path.sample(sphmol0.v, sphmol1.v, t)
        r_sample = self.r_path.sample(sphmol0.r, sphmol1.r, t)
        
        sphmol = SphMol(x_sample.x_t, v_sample.x_t, r=r_sample.x_t)
        dvdt = v_sample.dx_t
        drdt = r_sample.dx_t
        return sphmol, dvdt, drdt