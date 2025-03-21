from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, MLP

from shell.model.base import VectorField
from shell.model.submodule import (DenseLayer, SinEmbedding, coord2diff,
                                      unsorted_segment_sum)
from shell.utils.decorator import register_init_params
from shell.data import TMC


class GCL(nn.Module):
    def __init__(
        self, 
        input_nf: int, 
        output_nf: int, 
        hidden_nf: int, 
        edges_in_d: int = 0, 
        nodes_att_dim: int = 0, 
        act_fn: str = 'silu', 
        attention: bool = False,
        normal_factor: float = 100.0
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.attention = attention
        self.normal_factor = normal_factor

        self.edge_mlp = nn.Sequential(
            DenseLayer(input_edge + edges_in_d, hidden_nf, activation=act_fn),
            nn.BatchNorm1d(hidden_nf),
            DenseLayer(hidden_nf, hidden_nf, activation=act_fn)
        )

        node_in_dim = hidden_nf + input_nf + nodes_att_dim
        self.node_mlp = nn.Sequential(
            DenseLayer(node_in_dim, hidden_nf, activation=act_fn),
            DenseLayer(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = agg / self.normal_factor
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out

    def forward(
        self, 
        h: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor = None, 
        node_attr: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GCL model.
        Args:
            h: [n_nodes, hidden_nf]
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_feat_nf]
            node_attr: [n_nodes, node_feat_nf]
        """
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr)
        h = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, mij


class GCL_Equivariant(nn.Module):
    def __init__(
        self, 
        hidden_nf: int, 
        edges_in_d: int = 1, 
        act_fn: str = 'silu', 
        tanh: bool = False, 
        coords_range: float = 10.0,
        nomal_factor: float = 100.0
    ):
        super(GCL_Equivariant, self).__init__()
        
        self.tanh = tanh
        self.coords_range = coords_range
        self.normal_factor = nomal_factor
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        
        self.coord_mlp = nn.Sequential(
            DenseLayer(input_edge, hidden_nf, activation=act_fn),
            nn.BatchNorm1d(hidden_nf),
            DenseLayer(hidden_nf, hidden_nf, activation=act_fn),
            nn.BatchNorm1d(hidden_nf),
            layer)

    def forward(
        self, 
        h: torch.Tensor, 
        coord: torch.Tensor, 
        edge_index: torch.Tensor, 
        coord_diff: torch.Tensor, 
        edge_attr: torch.Tensor = None,
        atom_mask: torch.Tensor = None
    ) -> torch.Tensor:
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            input_tensor = torch.tanh(self.coord_mlp(input_tensor))
            trans = coord_diff * input_tensor * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        agg = agg / self.normal_factor
        if atom_mask is not None:
            agg = agg * atom_mask
        coord = coord + agg

        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, 
        hidden_nf: int, 
        edge_feat_nf: int = 2, 
        act_fn: str = 'silu', 
        n_layers: int = 2, 
        attention: bool = True,
        norm_diff: bool = True, 
        tanh: bool = False, 
        coords_range: float = 15, 
        norm_constant: float = 1, 
        sin_embedding: nn.Module = None,
        normal_factor: float = 100.0
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normal_factor = normal_factor
        
        edge_feat_nf += 1 # Distance

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, 
                self.hidden_nf, 
                self.hidden_nf, 
                edges_in_d=edge_feat_nf, 
                act_fn=act_fn, 
                attention=attention,
                normal_factor=self.normal_factor
            ))
        self.add_module("gcl_equiv", GCL_Equivariant(
            hidden_nf, 
            edges_in_d=edge_feat_nf, 
            act_fn=act_fn, 
            tanh=tanh,
            coords_range=self.coords_range_layer,
            nomal_factor=self.normal_factor
        ))

    def forward(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor = None,
        atom_mask: torch.Tensor = None
    ):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, atom_mask=atom_mask)

        return h, x


class EGNN(nn.Module):
    def __init__(self, 
        in_node_nf: int, 
        hidden_nf: int, 
        act_fn: str = 'silu', 
        n_layers: int = 3, 
        attention: bool = False,
        norm_diff: bool = True, 
        out_node_nf: int = None, 
        tanh: bool = False, 
        coords_range: float = 15, 
        norm_constant: float = 1, 
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normal_factor: float = 100.0
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normal_factor = normal_factor

        if sin_embedding:
            self.sin_embedding = SinEmbedding()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 1

        self.embedding = DenseLayer(in_node_nf, self.hidden_nf, activation=act_fn)
        self.embedding_out = DenseLayer(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(
                hidden_nf, 
                edge_feat_nf=edge_feat_nf, 
                act_fn=act_fn, 
                n_layers=inv_sublayers,
                attention=attention, 
                norm_diff=norm_diff, 
                tanh=tanh,
                coords_range=coords_range, 
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normal_factor=self.normal_factor
            ))

    def forward(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        atom_mask: torch.Tensor = None
    ):
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, distances, atom_mask=atom_mask)
        h = self.embedding_out(h)
        return h, x


@register_init_params
class EGNNVectorField(VectorField):
    def __init__(
        self, 
        in_node_nf: int, 
        context_node_nf: int,
        hidden_nf: int = 64, 
        act_fn: str = 'silu', 
        n_layers: int = 4, 
        attention: bool = False,
        tanh: bool = False, 
        norm_constant: float = 0,
        inv_sublayers: int = 2, 
        sin_embedding: bool = False, 
        normal_factor: float = 100.0,
        num_shells: int = 5
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, 
            hidden_nf=hidden_nf, 
            act_fn=act_fn,
            n_layers=n_layers, 
            attention=attention, 
            tanh=tanh, 
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, 
            sin_embedding=sin_embedding,
            normal_factor=normal_factor,
            out_node_nf=hidden_nf
        )
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        # self.shell_embedding = nn.Embedding(num_shells, in_node_nf - 1)
        self.radius_pred = MLP([hidden_nf, hidden_nf, 1], act_fn=act_fn)
        self.x_pred = MLP([hidden_nf, hidden_nf, in_node_nf - 1], act_fn=act_fn)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        t: torch.Tensor, 
        # focus_shell_id: torch.Tensor,
        atom_mask: torch.Tensor = None,
        # context: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the EGNNDenoiser model.
        Args:
            t: Time. [batch_size, 1]
            x: Positions. [n_nodes, 3]
            h: Features. [n_nodes, h_dims]
            focus_shell_id: Focus shell id. [batch_size, 1]
            edge_index: [2, n_edges]
            atom_mask: [n_nodes, 1]
            context: [batch_size, context_node_nf]
            batch: [n_nodes]
        """
        h = x
        if atom_mask is None:
            atom_mask = torch.ones((h.shape[0], 1), device=h.device)
        
        # 1. Concatenate time and context (if provided) to h
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        # shell_emb = self.shell_embedding(focus_shell_id.squeeze(1)) # [batch_size, in_node_nf]
        # h = h + shell_emb[batch]
        h = torch.cat([h, t[batch]], dim=1)
        # if context is not None:
        #     h = torch.cat([h, context], dim=1)
        edge_index = radius_graph(pos, r=1e+50, batch=batch, max_num_neighbors=100) #!NOTICE

        # 2. Forward pass through EGNN
        h_final, pos_final = self.egnn(h, pos, edge_index, atom_mask=atom_mask)
        
        # 3. Post-process outputs
        # if context is not None:
        #     h_final = h_final[:, :-self.context_node_nf]
        # h_final = h_final[:, :-1]
        x = self.x_pred(h_final)
        r = self.radius_pred(h_final)
        
        pos_final = pos_final - pos
        # may not be necessary
        v = pos_final / (pos_final.norm(dim=-1, keepdim=True) + 1e-12)
        
        return x, v, r
