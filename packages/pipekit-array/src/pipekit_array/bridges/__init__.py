"""Other libraries on named carriers, one module each.

- `pipekit_array.bridges.xarray` (``[xarray]``): the ``xr.DataArray`` adapter.
  It loads the first time a DataArray reaches `pipekit_ops.get_adapter`.
- `pipekit_array.bridges.einx` (``[einx]``): tensor algebra with dim names as
  pattern axes; ``from pipekit_array.bridges import einx``.
- `pipekit_array.bridges.sklearn` (``[sklearn]``): `Estimator`, a
  scikit-learn-style estimator fitted and applied on a named carrier.

Importing this package imports none of them.
"""
