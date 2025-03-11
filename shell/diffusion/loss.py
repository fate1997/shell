from dataclasses import dataclass
from typing import List, Literal, NamedTuple

import torch
from torch.distributions import Normal
from torch_scatter import scatter_add, scatter_mean

from shell.denoiser import Denoiser
from shell.noiser import LineNoiser, NoiserOutput, PolyNoiser
from shell.utils.geometry import get_mid_shell
from shell.utils.settings import (DEFAULT_POLY_PRECISION, DEFAULT_SIGMA_MIN,
                                  QM9_SHELL_RADIUS)

NormValues = NamedTuple(
    "NormValues", [("p", float), ("x", float), ("n", float)],
)

@dataclass(frozen=True)
class EDMPredictionOutput:
    x_eps_true: torch.Tensor
    r_eps_true: torch.Tensor
    v_eps_true: torch.Tensor
    
    x_eps_pred: torch.Tensor
    r_eps_pred: torch.Tensor
    v_eps_pred: torch.Tensor
    
    x: torch.Tensor
    r: torch.Tensor
    v: torch.Tensor
    
    x_zt: torch.Tensor
    p_zt: torch.Tensor
    
    line_noiser_output: NoiserOutput
    poly_noiser_output: NoiserOutput
    num_nodes: torch.Tensor
    atom_mask: torch.Tensor


class EDMLoss:
    def __init__(
        self,
        denoiser: Denoiser,
        timesteps: int,
        num_shells: int,
        shell_radius: List[float] = QM9_SHELL_RADIUS,
        norm_values: NormValues = NormValues(p=1.0, x=4.0, n=10.0),
        unique_atom_types: List[int] = [1, 6, 7, 8, 9],
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        line_noiser = LineNoiser(timesteps, sigma=DEFAULT_SIGMA_MIN).to(device)
        poly_noiser = PolyNoiser(timesteps, precision=DEFAULT_POLY_PRECISION).to(device)
        self.line_noiser = line_noiser
        self.poly_noiser = poly_noiser
        self.denoiser = denoiser.to(device)
        self.timesteps = timesteps
        self.norm_values = norm_values
        self.num_atom_types = len(unique_atom_types)
        if not isinstance(unique_atom_types, torch.Tensor):
            unique_atom_types = torch.tensor(unique_atom_types).to(device)
        self.unique_atom_types = unique_atom_types
        self.num_shells = num_shells
        self.mid_shell = get_mid_shell(shell_radius).to(device)
    
    def __call__(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        shell_id: torch.Tensor,
        focus_shell_id: torch.Tensor,
        batch: torch.Tensor,
        return_parts: bool = False,
        context: torch.Tensor = None,
    ) -> float:
        """Calculate the loss for the EDM model.
        
        Args:
            x (torch.Tensor): The input atom features. In EDM, this is the 
                one-hot encoded atom type. Shape: [n_nodes, n_atom_types]
            p (torch.Tensor): The positions of the atoms. Shape: [n_nodes, 3]
            edge_index (torch.Tensor): The edge indices of the graph.
                Shape: [2, n_edges]
            shell_id (torch.Tensor): The shell indices of the atoms.
                Shape: [n_nodes]
            focus_shell_id (torch.Tensor): The focus shell indices of the atoms.
                Shape: [batch_size]
            batch (torch.Tensor): The batch indices of the atoms.
                Shape: [n_nodes]
            context (torch.Tensor, optional): The context tensor.
                Shape: [batch_size, context_node_nf]
        Returns:
            loss (float): The loss for the EDM model.
        """
        atom_mask = (shell_id[:, None] == focus_shell_id[batch]).float()
        # 1. Normalize the input data
        p = p / self.norm_values.p
        x = x / self.norm_values.x
        
        # 2. Sample a random timestep t and get its previous timestep s
        batch_size = batch.max().item() + 1
        t = torch.randint(
            0, self.timesteps + 1, size=(batch_size, 1), device=x.device
        ).float()
        s = t - 1
        t = t / self.timesteps
        s = s / self.timesteps
        t_is_zero = (t == 0).float().squeeze(1)
        
        # 3. Get the prediction output at timestep t
        output = self.step(
            t, p, x, shell_id, edge_index, batch, focus_shell_id, atom_mask, context
        )

        # 4. Calculate the loss for t > 1. Calculate the weight first, which 
        # is shown in Eq. (17) in the paper. Then calculate the loss.
        loss_t = self._loss_t(output, batch)
        loss_t = loss_t * 0.5
        
        # 5. Calculate the rest loss terms
        # kl_loss = self._kl_loss(output, batch)
        loss_t0 = self._loss_t0(output, batch)
        loss_t = loss_t * (1 - t_is_zero)
        loss_t0 = loss_t0 * t_is_zero
        
        # print(output.num_nodes)
        # loss -= model.node_distr.log_prob(output.num_nodes)
        mask = (output.num_nodes == 0).float().squeeze(-1)
        loss_t = loss_t * (1 - mask)
        loss_t0 = loss_t0 * (1 - mask)
        # kl_loss = kl_loss * (1 - mask)
        num_nodes = output.num_nodes.sum()
        # print(loss_t.shape, loss_t0.shape, kl_loss.shape)
        loss = loss_t + loss_t0 # + kl_loss
        if return_parts:
            return loss.sum() / num_nodes, {
                'loss_t': loss_t.sum() / num_nodes,
                'loss_t0': loss_t0.sum() / num_nodes,
                # 'kl_loss': kl_loss.sum() / num_nodes,
            }
        return loss.sum() / num_nodes
    
    def step(
        self,
        t: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        shell_id: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        focus_shell_id: torch.Tensor,
        atom_mask: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> EDMPredictionOutput:
        device = p.device
        
        # 1 Obtain the noisy atom types
        x_eps = torch.randn(x.size(), device=device) * atom_mask
        poly_noiser_output = self.poly_noiser.forward_batch(t, batch)
        x_zt = poly_noiser_output.alpha * x + poly_noiser_output.sigma * x_eps
        x_zt = x_zt * atom_mask + x * (1 - atom_mask)
        
        # 2. Obtain the noisy positions
        # 2.1 Obtain the noisy radius
        r = p.norm(dim=1, keepdim=True)
        r_eps = torch.randn(r.size(), device=device) * atom_mask
        line_noiser_output = self.line_noiser.forward_batch(t, batch)
        mid_radius = self.mid_shell[shell_id].unsqueeze(1)
        alpha = line_noiser_output.alpha
        force = alpha * (r - mid_radius) + mid_radius
        r_t = force + line_noiser_output.sigma * r_eps
        r_t = r_t * atom_mask + r * (1 - atom_mask)

        # 2.2 Obtain the noisy angle
        v = p / r
        v_eps = torch.randn(v.size(), device=device)
        v_eps /= v_eps.norm(dim=1, keepdim=True)
        v_eps = v_eps * atom_mask
        v_t = poly_noiser_output.alpha * v + poly_noiser_output.sigma * v_eps
        v_t = v_t / v_t.norm(dim=1, keepdim=True)
        v_t = v_t * atom_mask + v * (1 - atom_mask)
        p_zt = r_t * v_t

        # 3. Predict noise by the denoiser
        p_pred, x_pred = self.denoiser(
            t, p_zt, x_zt, focus_shell_id, edge_index, atom_mask, context, batch
        )
        x_pred = x_pred * atom_mask
        r_pred = p_pred.norm(dim=1, keepdim=True) * atom_mask
        v_pred = p_pred / (r_pred + 1e-15) * atom_mask
        
        # 4. Compute position freedom
        num_nodes = scatter_add(atom_mask, batch, dim=0)
        # p_freedom = (num_nodes) * p.size(1)
        
        return EDMPredictionOutput(
            x_eps_true=x_eps, r_eps_true=r_eps, v_eps_true=v_eps,
            x_eps_pred=x_pred, r_eps_pred=r_pred, v_eps_pred=v_pred,
            x=x, r=r, v=v,
            x_zt=x_zt, p_zt=p_zt,
            line_noiser_output=line_noiser_output,
            poly_noiser_output=poly_noiser_output,
            num_nodes=num_nodes,
            atom_mask=atom_mask
        )
    
    def _loss_t(
        self, 
        output: 'EDMPredictionOutput', 
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss at timestep t (t > 0). See Eq (17) in the paper.
        """
        xrv_true = torch.cat(
            [output.x_eps_true, output.r_eps_true, output.v_eps_true], dim=1
        )
        xrv_pred = torch.cat(
            [output.x_eps_pred, output.r_eps_pred, output.v_eps_pred], dim=1
        )
        num_nodes = output.num_nodes
        n_dims = (3 + self.num_atom_types)
        loss = self._calc_batch_mse(xrv_pred, xrv_true, batch, n_dims, num_nodes)
        return loss
    
    def _loss_t0(
        self,
        output: 'EDMPredictionOutput', 
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss at timestep t = 0.
        """
        # 1. Rescale sigma for the atom types and one-hot encoded atom types
        sigma_x = output.line_noiser_output.sigma * self.norm_values.x

        # 2. Calculate the loss_t0 for the positions. See Eq. (19) in the paper.
        loss_t0_r = self._calc_batch_mse(
            output.r_eps_pred, output.r_eps_true, batch, 3, output.num_nodes,
        ) * 0.5
        loss_t0_v = self._calc_batch_mse(
            output.v_eps_pred, output.v_eps_true, batch, 3, output.num_nodes,
        ) * 0.5

        # 3. Calculate the loss_t0 for the one-hot encoded atom types. Similar
        # to the atom type loss, but should be normalized over categories.
        x_true = output.x * self.norm_values.x
        x_pred = output.x_zt * self.norm_values.x
        x_diff = x_pred - 1 # Due to one-hot encoded
        normal = Normal(0, sigma_x)
        lower, upper = x_diff - 0.5, x_diff + 0.5
        log_x = torch.log(normal.cdf(upper) - normal.cdf(lower) + 1e-10)
        
        log_norm = torch.logsumexp(log_x, dim=1, keepdim=True)
        log_x = log_x - log_norm
        log_x = log_x * output.atom_mask
        loss_t0_x = scatter_add((log_x * x_true).sum(-1), batch, dim=0) * -1
        
        return loss_t0_r + loss_t0_x + loss_t0_v
    
    def _kl_loss(
        self,
        output: 'EDMPredictionOutput',
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x, r, v = output.x, output.r, output.v

        # 1. Compute mu and sigma for z_T ~ N(mu, sigma)
        batch_size = batch.max().item() + 1
        ones = torch.ones(batch_size, 1, device=x.device)
        line_noiser_output = self.line_noiser.forward(ones)
        poly_noiser_output = self.poly_noiser.forward(ones)
        x_mu, x_sigma = x * poly_noiser_output.alpha[batch], poly_noiser_output.sigma
        r_mu, r_sigma = r * line_noiser_output.alpha[batch], line_noiser_output.sigma
        v_mu, v_sigma = v * poly_noiser_output.alpha[batch], poly_noiser_output.sigma
        
        # 2. Compute the KL divergence for atom types
        ones_x = torch.ones_like(x_sigma)
        mu_norm2 = scatter_add(((x_mu ** 2) * output.atom_mask).sum(-1), batch, dim=0)
        kl_distance_x = self.gaussian_kl(mu_norm2.unsqueeze(-1), x_sigma, ones_x, d=1)

        # 3. Compute the KL divergence for angle. See Apendix A in the paper.
        ones_v = torch.ones_like(v_sigma)
        mu_norm2 = scatter_add(((v_mu ** 2) * output.atom_mask).sum(-1), batch, dim=0)
        d = 2 * (output.num_nodes - 1)
        kl_distance_v = self.gaussian_kl(mu_norm2.unsqueeze(-1), v_sigma, ones_v, d=d)
        
        # 4. Compute the KL divergence for radius. See Apendix A in the paper.
        ones_r = torch.ones_like(r_sigma)
        mu_norm2 = scatter_add(((r_mu ** 2) * output.atom_mask), batch, dim=0)
        d = 1 * (output.num_nodes - 1)
        kl_distance_r = self.gaussian_kl(mu_norm2, r_sigma, ones_r, d=d)
        kl = (kl_distance_x + kl_distance_v + kl_distance_r).squeeze(-1)
        return kl

    def _calc_batch_mse(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        batch: torch.Tensor,
        n_dims: int,
        num_nodes: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = scatter_add(((pred - true) ** 2).sum(-1), batch, dim=0)
        denom = num_nodes.squeeze(1) * n_dims
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        loss /= denom
        return loss
    
    def gaussian_kl(self, q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d