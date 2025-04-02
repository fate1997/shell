import torch
import torch.nn.functional as F
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical
from flow_matching.utils.manifolds import Sphere
from tqdm import tqdm

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.data import TMC
from shell.model import VectorField
from shell.data.transform import GeometryTransform, AtomNumTransform


class SphMolSolver(Solver):
    def __init__(
        self, 
        vf: VectorField,
        x_path: MixtureDiscreteProbPath,
        unique_atom_nums: torch.Tensor,
        n_steps: int = 1000,
    ):
        super().__init__()
        self.vf = vf
        self.x_path = x_path
        self.manifold = Sphere()
        self.n_steps = n_steps
        self.unique_atom_nums = unique_atom_nums
    
    @torch.no_grad()
    def sample(
        self,
        sphmol0: TMC,
        return_traj: bool = False,
        unchanged_vars: list = None,
    ) -> MolSampleList:
        vocab_size = sphmol0.x.shape[1]
        mask = (1 - sphmol0.ligand_mask).bool().squeeze(-1)
        sphmolt = sphmol0
        t = torch.linspace(0, 1, self.n_steps, device=sphmolt.x.device)
        x_traj = [self._x2atom_num(sphmol0.x).detach().cpu()]
        pos_traj = [sphmol0.pos.detach().cpu()]
        xt = sphmolt.x
        pt = sphmolt.pos
        for t0, t1 in tqdm(zip(t[:-1], t[1:]), desc='Sampling', total=self.n_steps - 1):
            dt = t1 - t0
            xt_pred, dpdt = self.vf(
                x=xt, 
                pos=pt,
                t=t.unsqueeze(-1),
                atom_mask=sphmol0.ligand_mask,
                batch=sphmol0.batch
            )
            last_step = t1 == t[-1]
            if unchanged_vars is None or 'x' not in unchanged_vars:
                xt_pred = self._step_x(xt_pred.argmax(-1), xt_pred.softmax(-1), t0, dt, last_step, vocab_size)
                xt_pred = F.one_hot(xt_pred, num_classes=vocab_size).float()
                xt_pred[mask] = sphmol0.x.float()[mask]
                xt = xt_pred
            if unchanged_vars is None or 'p' not in unchanged_vars:
                pt = self._step_p(dt, pt, dpdt)
                pt[mask] = sphmol0.pos[mask]
        
            if return_traj and not last_step:
                x_traj.append(self._x2atom_num(xt).detach().cpu())
                pos_traj.append(pt.detach().cpu())

        return MolSampleList.from_batch(
            pos=pt.detach().cpu(),
            atom_num=self._x2atom_num(xt).detach().cpu(),
            batch=sphmol0.batch.detach().cpu(),
            pos_traj=pos_traj,
            atom_traj=x_traj,
        )
    
    def _step_x(
        self,
        xt: torch.Tensor,
        p_1t: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        last_step: bool = False,
        vocab_size: int = 5,
    ) -> torch.Tensor:
        x_1 = categorical(p_1t)
        if last_step:
            xt = x_1
        else:
            # Compute u_t(x|x_t,x_1)
            scheduler_output = self.x_path.scheduler(t=t)

            k_t = scheduler_output.alpha_t
            d_k_t = scheduler_output.d_alpha_t

            delta_1 = F.one_hot(x_1, num_classes=vocab_size).to(k_t.dtype)
            u = d_k_t / (1 - k_t) * delta_1

            # Set u_t(x_t|x_t,x_1) = 0
            delta_t = F.one_hot(xt, num_classes=vocab_size)
            u = torch.where(
                delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
            )

            # Sample x_t ~ u_t( \cdot |x_t,x_1)
            intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
            mask_jump = torch.rand(
                size=xt.shape, device=xt.device
            ) < 1 - torch.exp(-dt * intensity)

            if mask_jump.sum() > 0:
                xt[mask_jump] = categorical(u[mask_jump].to(dtype=p_1t.dtype))
        return xt
    
    def _step_p(
        self,
        dt: torch.Tensor,
        pt: torch.Tensor,
        dpdt: torch.Tensor,
    ) -> torch.Tensor:
        pt[0] = 0.0
        pt = pt + dpdt * dt
        pt[0] = 0.0
        return pt

    def _x2atom_num(self, x: torch.Tensor) -> torch.Tensor:
        unique_atom_nums = self.unique_atom_nums
        x = x.argmax(dim=-1).detach().cpu()
        atom_num = unique_atom_nums[x]
        return atom_num