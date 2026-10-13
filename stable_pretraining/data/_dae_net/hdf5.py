"""Optional HDF5 dependency for direct DAE-Net data reads."""

from typing import Any


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "DAENetDataset requires the optional dependency 'h5py'"
        ) from exc
    return h5py
