import math
import os
from dataclasses import dataclass
from typing import Dict, List, Literal

import torch
from torch.nn import functional as F
from torch.nn.functional import logsigmoid, softplus
from torch_scatter import scatter_add, scatter_sum

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.denoiser import EGNNDenoiser
from shell.diffusion.loss import NormValues
from shell.noiser import LineNoiser, PolyNoiser
from shell.utils.geometry import get_mid_shell
from shell.utils.settings import (DEFAULT_POLY_PRECISION, DEFAULT_SIGMA_MIN,
                                  QM9_SHELL_RADIUS)


class EDMSampler:
    def __init__(
        self,
        denoiser: EGNNDenoiser,
        timesteps: int,
        num_shells: int,
        shell_radius: List[float] = QM9_SHELL_RADIUS,
        norm_values: NormValues = NormValues(p=1.0, x=4.0, n=10.0),
        unique_atom_types: List[int] = [1, 6, 7, 8, 9],
        device: Literal['cpu', 'cuda'] = 'cpu',
        annealing_rate: float = 0.5,
    ):
        line_noiser = LineNoiser(timesteps, sigma=DEFAULT_SIGMA_MIN).to(device)
        poly_noiser = PolyNoiser(timesteps, precision=DEFAULT_POLY_PRECISION).to(device)
        self.line_noiser = line_noiser
        self.poly_noiser = poly_noiser
        self.denoiser = denoiser.to(device)
        self.timesteps = timesteps
        self.norm_values = norm_values
        self.num_atom_types = len(unique_atom_types) + 1
        if not isinstance(unique_atom_types, torch.Tensor):
            unique_atom_types = torch.tensor(unique_atom_types).to(device)
        self.unique_atom_types = unique_atom_types
        self.num_shells = num_shells
        self.mid_shell = get_mid_shell(shell_radius).to(device)
        self.annealing_rate = annealing_rate
    
    @torch.no_grad()
    def sample(
        self, 
        num_nodes: torch.Tensor,
        record_traj: bool = False,
        context: torch.Tensor = None,
    ) -> MolSampleList:
        self.denoiser.eval()
        device = next(self.denoiser.parameters()).device
        
        num_nodes = num_nodes.to(device)
        timesteps = self.timesteps
        num_samples = num_nodes.shape[0]
        indptr = torch.zeros(num_samples+1, device=device, dtype=torch.long)
        indptr[1:] = num_nodes.cumsum(0)
        
        if record_traj:
            pos_traj, atom_traj = [], []
        
        for focus_shell in range(self.num_shells):
            x, r, v, batch = self._prep_prior(num_nodes, focus_shell)
            atom_mask = torch.ones((num_nodes.sum().item(), 1), device=device)
            p, x, traj = self._sample_step(x, r, v, batch, atom_mask, focus_shell, context)
            if record_traj:
                pos_traj.extend(traj['p'])
                atom_traj.extend(traj['x'])
            break
        
        pos_traj = None if not record_traj else pos_traj
        atom_traj = None if not record_traj else atom_traj
        sample_list = MolSampleList.from_batch(
            pos=p.detach().cpu(),
            atom_num=self._n2atom_num(x),
            batch=batch.cpu(),
            pos_traj=pos_traj,
            atom_traj=atom_traj,
        )
        return sample_list
    
    def _sample_step(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        batch: torch.Tensor,
        atom_mask: torch.Tensor,
        focus_shell: int,
        context: torch.Tensor = None,
    ):
        device = x.device
        batch_size = batch.max().item() + 1
        focus_shell = torch.full((batch_size, 1), fill_value=focus_shell, device=device)
        traj = {
            'p': [r * v * self.norm_values.p],
            'x': [self._n2atom_num(x * self.norm_values.x)],
        }
        for s in reversed(range(0, self.timesteps)):
            s = torch.full((batch_size, 1), fill_value=s, device=device)
            t = s + 1
            s = s / self.timesteps
            t = t / self.timesteps
            x, r, v = self._sample_zs(s, t, x, r, v, batch, atom_mask, focus_shell, context)
            p = r * v
            
            traj['p'].append(p * self.norm_values.p)
            traj['x'].append(self._n2atom_num(x * self.norm_values.x))
    
        x, v = self._sample_x(x, r, v, batch, atom_mask, focus_shell, context)
        p = r * v
        traj['p'].append(p * self.norm_values.p)
        traj['x'].append(self._n2atom_num(x * self.norm_values.x))
    
        return p, x, traj
        
    def _sample_zs(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        batch: torch.Tensor,
        atom_mask: torch.Tensor,
        focus_shell: torch.Tensor,
        context: torch.Tensor = None,
    ):
        p = r * v
        noise_s = self.poly_noiser.forward_batch(s, batch)
        noise_t = self.poly_noiser.forward_batch(t, batch)
        sigma_s, gamma_s = noise_s.sigma, noise_s.gamma
        sigma_t, gamma_t = noise_t.sigma, noise_t.gamma
        
        # 1. sigma_{t|s}^2 See https://arxiv.org/abs/2107.00630 Eq. 63
        sigma_ts2 = -torch.expm1(softplus(gamma_s) - softplus(gamma_t))
        sigma_ts = sigma_ts2.sqrt()
        
        # 2. alpha_{t|s} = alpha_t / alpha_s
        log_alpha_ts2 = logsigmoid(-gamma_t) - logsigmoid(-gamma_s)
        alpha_ts = torch.exp(0.5 * log_alpha_ts2)

        # 3. Calculate mu and sigma for p(z_s|z_t)
        # See Eq. (29) in https://arxiv.org/abs/2107.00630
        p_eps_hat, x_eps_hat = self.denoiser(
            t, p, x, focus_shell, atom_mask=atom_mask, context=context, batch=batch
        )
        x_eps_hat = x_eps_hat * atom_mask
        r_eps_hat = p_eps_hat.norm(dim=1, keepdim=True)
        v_eps_hat = p_eps_hat / r_eps_hat

        xv = torch.cat([x, v], dim=1)
        xv_eps_hat = torch.cat([x_eps_hat, v_eps_hat], dim=1)
        mu = xv / alpha_ts - (sigma_ts2 / alpha_ts / sigma_t) * xv_eps_hat
        sigma = sigma_ts * sigma_s / sigma_t
        
        xv_s = self.sample_normal(mu, atom_mask, sigma, batch)
        xv_s = xv * (1 - atom_mask) + xv_s * atom_mask
        
        x_zs, v_zs = xv_s.split([x.size(1), v.size(1)], dim=1)
        v_zs = v_zs / v_zs.norm(dim=1, keepdim=True)

        # 4. Calculate r_zs
        delta_t = t - s
        r_zs = r
        if (t - 1.0) > 1e-5:
            r_zs = (1 - (t - delta_t) / (1 - t)) * (r - r_eps_hat * DEFAULT_SIGMA_MIN)
        if (t - delta_t) > 1e-5:
            eps = torch.randn_like(r_zs)
            r_zs = r_zs + t ** self.annealing_rate * math.sqrt(2) * eps * DEFAULT_SIGMA_MIN
        r_zs = r_zs * atom_mask + r * (1 - atom_mask)

        return x_zs, r_zs, v_zs
    
    def _sample_x(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        batch: torch.Tensor,
        atom_mask: torch.Tensor,
        focus_shell: torch.Tensor,
        context: torch.Tensor = None,
    ):
        device = x.device
        batch_size = batch.max().item() + 1
        xv = torch.cat([x, v], dim=1)
        p = r * v
        
        zeros = torch.zeros(batch_size, 1, device=device)
        zero_output = self.poly_noiser.forward_batch(zeros, batch)
        p_eps_hat, x_eps_hat = self.denoiser(
            zeros, p, x, focus_shell, atom_mask=atom_mask, context=context, batch=batch
        )
        r_eps_hat = p_eps_hat.norm(dim=1, keepdim=True)
        v_eps_hat = p_eps_hat / r_eps_hat
        
        
        xv_eps_hat = torch.cat([x_eps_hat, v_eps_hat], dim=1)
        xv_eps_hat *= atom_mask
        
        mu_x = 1 / zero_output.alpha * (xv - zero_output.sigma * xv_eps_hat)
        sigma_x = torch.exp(zero_output.gamma * 0.5) # EDM Eq. (19)
        xv_eps = self.sample_normal(mu_x, atom_mask, sigma_x, batch)
        xv = xv_eps * atom_mask + xv * (1 - atom_mask)
        
        x, v = xv.split([x.size(1), v.size(1)], dim=1)
        v = v / v.norm(dim=1, keepdim=True)
        
        # Un-normalize p, x, n
        x = x * self.norm_values.x
        
        x = F.one_hot(torch.argmax(x, dim=1), self.num_atom_types)
        
        return x, v
    
    def sample_normal(self, mu_pxn, atom_mask, sigma, batch_seg):
        pxn_noise = torch.randn_like(mu_pxn) * atom_mask
        out_pxn = mu_pxn + sigma[batch_seg] * pxn_noise
        return out_pxn
    
    def _prep_prior(
        self,
        num_nodes: torch.Tensor,
        focus_shell: int,
    ) -> Dict[str, torch.Tensor]:
        num_samples = num_nodes.shape[0]
        num_node_types = self.num_atom_types
        total_nodes = num_nodes.sum().item()
        batch = torch.repeat_interleave(
            torch.arange(num_samples, device=num_nodes.device), num_nodes
        )
        mid_radius = self.mid_shell[focus_shell]
        
        # 1. Sample the initial radius
        r_eps = torch.randn((total_nodes, 1), device=num_nodes.device)
        r = mid_radius + r_eps * DEFAULT_SIGMA_MIN
        
        # 2. Sample the initial angle
        v_eps = torch.randn((total_nodes, 3), device=num_nodes.device)
        v = v_eps / v_eps.norm(dim=1, keepdim=True)
        
        # 3. Sample the initial atom numbers
        x = torch.randn((total_nodes, num_node_types), device=num_nodes.device)
        return x, r, v, batch
    
    def _merge_data(
        self,
        x: torch.Tensor,
        
    ):
        pass

    def _n2atom_num(self, x: torch.Tensor) -> torch.Tensor:
        unique_atom_nums = torch.LongTensor(self.unique_atom_types.cpu())
        x = x.argmax(dim=-1).detach().cpu()
        x[x == len(unique_atom_nums)] = 1
        atom_num = unique_atom_nums[x]
        return atom_num.detach().cpu()
