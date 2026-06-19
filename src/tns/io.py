"""
io.py: Save and load SimResult to/from HDF5 files.

Decouples data generation from analysis and plotting.
Run run_a4.py once to produce data files, then load them in any script.

File format: HDF5 (via h5py)
  /times        float64 array (n_steps,)
  /lambda_t     float64 array (n_steps,)
  /entropy_A    float64 array (n_steps,)
  /entropy_B    float64 array (n_steps,)
  /sz_A         float64 array (n_steps,)
  /sz_B         float64 array (n_steps,)
  /trunc_err    float64 array (n_steps,)
  /chi_A        int64 array   (n_steps,)
  /params/J         scalar attribute
  /params/h1        scalar attribute
  /params/chi_max   scalar attribute
  /params/dt        scalar attribute
  /params/n_steps   scalar attribute
"""

from __future__ import annotations

import h5py
import numpy as np

from tns.simulate import SimResult


def save_result(result: SimResult, path: str) -> None:
    """Save SimResult to an HDF5 file at the given path."""
    with h5py.File(path, 'w') as f:
        for key in ('times', 'lambda_t', 'entropy_A', 'entropy_B',
                    'sz_A', 'sz_B', 'trunc_err', 'chi_A'):
            f.create_dataset(key, data=getattr(result, key))
        grp = f.create_group('params')
        for k, v in result.params.items():
            grp.attrs[k] = v


def load_result(path: str) -> SimResult:
    """Load a SimResult from an HDF5 file."""
    with h5py.File(path, 'r') as f:
        arrays = {key: f[key][()] for key in
                  ('times', 'lambda_t', 'entropy_A', 'entropy_B',
                   'sz_A', 'sz_B', 'trunc_err', 'chi_A')}
        params = dict(f['params'].attrs)
    return SimResult(**arrays, params=params)
