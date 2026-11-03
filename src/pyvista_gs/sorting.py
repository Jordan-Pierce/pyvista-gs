"""Back-to-front depth sorting of Gaussians.

Backend is chosen at import time: torch (CUDA) > cupy > NumPy CPU.
"""
from __future__ import annotations

import importlib.util

import numpy as np

_sort_buffer_xyz = None
_sort_buffer_gausid = None


def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp  # type: ignore[reportMissingImports]

    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index)
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


_sort_gaussian = None
try:
    import torch  # type: ignore[reportMissingImports]
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    if importlib.util.find_spec("cupy") is not None:
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    else:
        _sort_gaussian = _sort_gaussian_cpu
