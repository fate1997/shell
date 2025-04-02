from typing import Dict, List, Literal

import torch
import torch.nn.functional as F
from flow_matching.path import (AffineProbPath, GeodesicProbPath,
                                MixtureDiscreteProbPath)
from flow_matching.path.scheduler import (CondOTScheduler,
                                          PolynomialConvexScheduler, Scheduler)
from flow_matching.utils.manifolds import Sphere

from shell.data.mol import TMC
from shell.data.transform import AtomNumTransform, GeometryTransform


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
        p_scheduler: Literal['ot', 'poly'] = 'ot',
    ):
        self.x_path = MixtureDiscreteProbPath(get_scheduler(x_scheduler))
        self.p_path = AffineProbPath(get_scheduler(p_scheduler))
        
    def sample(
        self,
        sphmol0: TMC,
        sphmol1: TMC,
        t: torch.Tensor,
        unique_atom_nums: torch.Tensor,
        unchanged_vars: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        mask = (1 - sphmol1.ligand_mask).bool().squeeze(-1)
        
        xt, pt = sphmol1.x, sphmol1.pos
        dvdt, drdt = None, None
        if unchanged_vars is None or 'x' not in unchanged_vars:
            n0 = AtomNumTransform(unique_atom_nums).to_num(sphmol0.x)
            n1 = AtomNumTransform(unique_atom_nums).to_num(sphmol1.x)
            nt = self.x_path.sample(n0, n1, t).x_t
            xt = AtomNumTransform(unique_atom_nums).to_onehot(nt)
            xt[mask] = sphmol1.x.float()[mask]
        if unchanged_vars is None or 'p' not in unchanged_vars:
            v_sample = self.p_path.sample(sphmol0.pos, sphmol1.pos, t)
            vt = v_sample.x_t
            vt[mask] = sphmol1.pos[mask]
            dvdt = v_sample.dx_t
            dvdt[mask] = 0.0
        
        return {
            'xt': xt,
            'pt': vt,
            'dpdt': dvdt,
        }