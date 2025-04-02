# https://github.com/atomicarchitects/equiformer_v2/blob/main/nets/equiformer_v2/equiformer_v2_oc20.py
# https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/base.py

import math
from typing import Any, Dict, Tuple
import numpy as np
import torch

from e3nn import o3
from shell.model.equiformer_v2.components.edge_rot_mat import init_edge_rot_mat
from shell.model.equiformer_v2.components.gaussian_rbf import GaussianRadialBasisLayer
from shell.model.equiformer_v2.components.input_block import EdgeDegreeEmbedding
from shell.model.equiformer_v2.components.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from shell.model.equiformer_v2.components.module_list import ModuleListInfo
from shell.model.equiformer_v2.components.radial_function import RadialFunction
from shell.model.equiformer_v2.components.so2_ops import SO2_Convolution
from shell.model.equiformer_v2.components.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from shell.model.equiformer_v2.components.transformer_block import (
    FeedForwardNetwork,
    TransBlockV2,
)

from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter


class GaussianSmearing(torch.nn.Module):
    """RBF distance expansion used in eSCN and Equiformer-V2."""

    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EquiformerEncoder(nn.Module):
    """Equiformer with graph attention built upon SO(2) convolution and feedforward network built
    upon S2 activation. Used as encoder in Equivariant VAEs.

    Adapted from: https://github.com/atomicarchitects/equiformer_v2/

    Args:
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        num_distance_basis (int):           Number of RBF functions used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
        self,
        otf_graph=True,
        max_neighbors=500,
        max_radius=9.0,
        max_num_elements=100, # should not be 100
        
        num_layers=4,
        sphere_channels=64,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        norm_type="layer_norm_sh",
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        
        num_sphere_samples=128,
        
        edge_channels=64,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,
        
        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=True,
        use_sep_s2_act=True,
        
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        
        weight_init="normal",
    ):
        super().__init__()

        ###########################################
        # Initializations from EquiformerV2 code
        ###########################################

        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        assert self.num_resolutions == 1, "Only one resolution is supported for now."
        self.d_model = self.sphere_channels * (self.lmax_list[0] + 1) ** 2
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels - 1

        # Weights for message initialization
        self.atom_type_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)

        # Initialize the function used to measure the distances between atoms
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0, self.cutoff, self.num_distance_basis, 2.0
            )
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(f"({max(self.lmax_list)}, {max(self.lmax_list)})")
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=self.grid_resolution, normalization="component")
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=1.0,  # TODO dataset-wide avg. degree
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks
        self.norm = get_normalization_layer(
            self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels
        )
        self.v_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            3,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        self.x_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            self.max_num_elements,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        self.r_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        t: torch.Tensor, 
        # focus_shell_id: torch.Tensor,
        atom_mask: torch.Tensor = None,
        # context: torch.Tensor = None,
        batch: torch.Tensor = None,
    ):
        """
        Args:
            batch: Data object with the following attributes:
                atom_types (torch.Tensor): Atomic numbers of atoms in the batch  # this is z
                pos (torch.Tensor): Cartesian coordinates of atoms in the batch
                num_atoms (torch.Tensor): Number of atoms in the batch
                batch (torch.Tensor): Batch index for each atom

        Returns:
            Dict[str, torch.Tensor]: Dictionary with the following keys:
                x (SO3_Embedding): Node embeddings
                num_atoms (torch.Tensor): Number of total atoms in the batch
                batch (torch.Tensor): Batch index for each atom
        """

        self.batch_size = batch.max().item() + 1  #  should be 64, the batch size, not a batch of nodes
        self.dtype = pos.dtype
        self.device = pos.device

        atom_types = x.argmax(-1)

        # Create graph/edge index tensor
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
        ) = self.generate_graph(
            pos,
            batch,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        self.edge_index = edge_index  # save on-the-fly edges for logging
        edge2graph = batch[edge_index[0]]

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            x.shape[0],
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        # Atom embedding
        atom_emb = self.atom_type_embedding(atom_types)  # (n, d)
        atom_emb = torch.cat([atom_emb, t[batch]], dim=1)  # (n, d)
        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = atom_emb
            else:
                x.embedding[:, offset_res, :] = atom_emb[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)


        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:  # here will not functions
            source_element = atom_types[edge_index[0]]  # Source atom atomic number
            target_element = atom_types[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(atom_types, edge_distance, edge_index)  # problem here
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atom_types,
                edge_distance,
                edge_index,
                batch=batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        
        #return (x.embedding.view(-1, self.d_model),
        #        num_atoms_batch,
        #        batch.batch)
        v = self.v_block(x).embedding.narrow(1, 0, 1).squeeze(1)
        r = self.r_block(x).embedding.narrow(1, 0, 1).squeeze(1)
        x = self.x_block(x).embedding.narrow(1, 0, 1).squeeze(1)
        return x, v

    def generate_graph(
        self,
        pos,  # 3D Cartesian coordinates: (n, 3)
        batch,  # node2graph: (n,)
        cutoff=None,
        max_neighbors=None,

    ):
        """Generate radial cutoff graphs with PBCs for a batch of crystal structures.

        Adapted from: https://github.com/FAIR-Chem/fairchem

        Args:
            pos: (n, 3) - 3D Cartesian coordinates
            num_atoms: (bsz,) - Number of atoms per crystal
            batch: (n,) - Batch index for each atom
            cutoff: (float) - Cutoff radius for pairwise distances
            max_neighbors: (int) - Maximum number of neighbors per atom

        Returns:
            edge_index: (2, e) - Pairs of edges (i, j)
            edge_dist: (e,) - Pairwise distances
            distance_vec: (e, 3) - Pairwise distance vectors
        """
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors

        edge_index = radius_graph(
            pos,
            r=cutoff,
            batch=batch,
            max_num_neighbors=max_neighbors,
        )
        j, i = edge_index
        distance_vec = pos[j] - pos[i]
        edge_dist = distance_vec.norm(dim=-1)

        return (
            edge_index,
            edge_dist,
            distance_vec,
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(module, SO3_LinearV2):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)