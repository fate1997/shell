import numpy as np
import torch


def identify_intervals(
    n: int,
    distance: torch.Tensor,
    max_val: float = None, 
    min_val: float = 0.0, 
    num_bins: int = 100
) -> torch.Tensor:
    if max_val is None:
        max_val = torch.max(distance)
    
    hist = torch.histogram(
        distance, bins=num_bins, range=(min_val, max_val), density=True
    )
    bin_widths = hist.bin_edges[1:] - hist.bin_edges[:-1]
    bin_areas = hist.hist * bin_widths
    cumu_areas = torch.cumsum(bin_areas, dim=0)
    total_area = cumu_areas[-1]
    target_areas = torch.linspace(0, total_area, steps=n + 1)[1:-1]
    intervals = np.interp(target_areas, cumu_areas, hist.bin_edges[1:]).tolist()
    intervals = [min_val] + intervals + [max_val]
    return intervals


def assign_shell_id(
    distance: torch.Tensor,
    intervals: torch.Tensor
) -> torch.Tensor:
    shell_id = torch.bucketize(distance, intervals, right=True) - 1
    return shell_id


def get_mid_shell(
    shell_radius: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(shell_radius, torch.Tensor):
        shell_radius = torch.tensor(shell_radius)
    mid_radius = (shell_radius[1:] + shell_radius[:-1]) / 2
    return mid_radius