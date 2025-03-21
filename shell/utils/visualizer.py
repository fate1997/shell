from typing import Tuple

import py3Dmol
import torch

from shell.utils.writer import get_xyz_str
from shell.utils.settings import COLOR_CONTRAST


def visualize_traj(
    atom_nums: torch.Tensor, 
    positions: torch.Tensor,
    shell_radius: Tuple[float, float] = None,
    atom_scale: float = 0.2,
    interval: int = 100,
):
    xyz_blocks = [get_xyz_str(z, x) for z, x in zip(atom_nums, positions)]
    xyz_block = ''.join(xyz_blocks)
    
    view = py3Dmol.view(width=800, height=400)
    view.addModelsAsFrames(xyz_block, format='xyz')
    
    if shell_radius is not None:
        shell_radius = sorted(shell_radius)
        for i, radius in enumerate(shell_radius):
            view.addSphere({
                "center": {"x": 0, "y": 0, "z": 0},
                "radius": radius,
                "color": COLOR_CONTRAST[i],
                "opacity": 1.0 if i == 0 else 0.5
            })
    
    view.setStyle({'stick':{}, 'sphere':{'scale': atom_scale}},)
    view.animate({'loop': 'forward', 'interval': interval})
    view.zoomTo()
    view.show()


def visualize_mol(
    atom_num: torch.Tensor,
    position: torch.Tensor,
    force: torch.Tensor = None
):
    xyz_str = get_xyz_str(atom_num, position)
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_str, format='xyz')
    view.setStyle({'stick':{}, 'sphere':{'scale': 0.2}},)
    
    if force is not None:
        force = force.cpu().numpy().tolist()
        position = position.cpu().numpy().tolist()
        view.setStyle({'stick':{'opacity': 0.6}, 'sphere':{'scale': 0.2}},)
        for i, (x, y, z) in enumerate(position):
            fx, fy, fz = force[i]
            view.addArrow({
                'start': {'x': x, 'y': y, 'z': z},
                'end': {'x': x + fx, 'y': y + fy, 'z': z + fz},
                'radius': 0.05,
                'color': 'red'
            })
    
    view.zoomTo()
    view.show()