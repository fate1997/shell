import torch
import torch.nn.functional as F
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical
from flow_matching.utils.manifolds import Sphere
from tqdm import tqdm

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.data import Mol, SphMol
from shell.model import VectorField


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
        sphmol0: SphMol,
        return_traj: bool = False,
    ) -> MolSampleList:
        vocab_size = sphmol0.x.shape[1]
        
        sphmolt = sphmol0
        t = torch.linspace(0, 1, self.n_steps, device=sphmolt.x.device)
        x_traj = [self._x2atom_num(sphmol0.to_mol().x).detach().cpu()]
        pos_traj = [sphmol0.to_mol().pos.detach().cpu()]
        for t0, t1 in tqdm(zip(t[:-1], t[1:]), desc='Sampling', total=self.n_steps - 1):
            dt = t1 - t0
            xt, dvdt, drdt = self.vf(sphmolt.to_mol(), t0.repeat(len(sphmol0)).unsqueeze(1))
            last_step = t1 == t[-1]
            xt = self._step_x(sphmolt.x.argmax(-1), xt.softmax(-1), t0, dt, last_step, vocab_size)
            vt = self._step_v(dt, sphmolt.v, dvdt)
            rt = self._step_r(dt, sphmolt.r, drdt)
            xt = F.one_hot(xt, num_classes=vocab_size).float()
            sphmolt = SphMol(xt, vt, rt, sphmolt.b)
            if return_traj and not last_step:
                mol = sphmolt.to_mol()
                x_traj.append(self._x2atom_num(mol.x).detach().cpu())
                pos_traj.append(mol.pos.detach().cpu())
        mol = sphmolt.to_mol()
        return MolSampleList.from_batch(
            pos=mol.pos.detach().cpu(),
            atom_num=self._x2atom_num(mol.x).detach().cpu(),
            batch=mol.batch.detach().cpu(),
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
    
    def _step_v(
        self,
        dt: torch.Tensor,
        vt: torch.Tensor,
        dvdt: torch.Tensor,
        projx: bool = True,
        proju: bool = True,
    ) -> torch.Tensor:
        dvdt = self.manifold.proju(vt, dvdt) if proju else dvdt
        projx_fn = lambda x: self.manifold.projx(x) if projx else x
        vt = vt + dvdt * dt
        return projx_fn(vt)
    
    def _step_r(
        self,
        dt: torch.Tensor,
        rt: torch.Tensor,
        drdt: torch.Tensor,
    ) -> torch.Tensor:
        return rt + drdt * dt

    def _x2atom_num(self, x: torch.Tensor) -> torch.Tensor:
        unique_atom_nums = torch.LongTensor(self.unique_atom_nums)
        x = x.argmax(dim=-1).detach().cpu()
        atom_num = unique_atom_nums[x]
        return atom_num