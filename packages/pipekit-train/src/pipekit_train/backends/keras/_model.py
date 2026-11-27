"""`KerasModelOp`, a Keras model as a pipekit operator with a weight blob (ADR D20)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
from pipekit_ops import Operator
from pipekit_ops.core import record_under

from pipekit_train.backends.keras._backend import import_keras


#: the first bytes of every blob, so a blob of another kind is refused
_MAGIC = b"PKKRW1\n"


class KerasModelOp(Operator):
    """Wrap a Keras model as a `pipekit_ops.Operator`.

    Calling the operator calls the model in inference mode. Its weights are
    written by `serialize_weights` as one blob: a header that lists each
    weight's name, shape and dtype, then the raw arrays in that order. The
    bytes depend only on the weights, so a registry gives one trained model one
    hash; a ``.keras`` file would not, because it records the time it was saved.
    The names are the weights' own (``kernel``, ``bias``), without the layer
    names Keras numbers per process (``dense_2``).

    A blob loads back into a model of the same architecture, the skeleton, with
    `with_weights` or `from_registry`: the architecture is Python, not bytes.

    Args:
        model: A ``keras.Model``.

    Raises:
        TypeError: If ``model`` is not a ``keras.Model``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, model: Any) -> None:
        keras = import_keras()
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"KerasModelOp.model must be a keras.Model; got {type(model).__name__}."
            )
        self.model = model

    def _apply(self, x: Any) -> Any:
        return self.model(x, training=False)

    def serialize_weights(self) -> bytes:
        """Return the model's weights as a blob.

        Layout: ``b"PKKRW1\\n"``, the header's length as 8 little-endian bytes,
        the header as JSON (one ``{"name", "shape", "dtype"}`` per weight, in
        the order of ``model.weights``), then each array's bytes in C order.

        Raises:
            ValueError: If the model has no weights yet (it is not built).
        """
        weights = list(self.model.weights)
        if not weights:
            raise ValueError(
                "The Keras model has no weights; build it (call it on an input) "
                "before writing them."
            )
        # np.asarray(order='C') keeps a scalar weight 0-d; ascontiguousarray would not
        arrays = [np.asarray(w, order="C") for w in self.model.get_weights()]
        header = json.dumps(
            [_describe(v.path, a) for v, a in zip(weights, arrays, strict=True)],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        body = b"".join(a.tobytes() for a in arrays)
        return _MAGIC + len(header).to_bytes(8, "little") + header + body

    def with_weights(self, blob: bytes, *, copy: bool = True) -> KerasModelOp:
        """Return the skeleton ``self`` with the weights of ``blob`` loaded.

        Args:
            blob: Bytes from `serialize_weights`.
            copy: Load into a copy of the model, leaving ``self`` as it is.
                With ``False`` the blob is loaded into ``self.model``, for a
                model that ``keras.models.clone_model`` cannot copy.

        Returns:
            A `KerasModelOp` holding the loaded model.

        Raises:
            TypeError: If ``blob`` is not bytes, or the model cannot be copied.
            ValueError: If ``blob`` is not a blob of this kind, or its weights
                do not match the skeleton's in number, name, shape or dtype.
        """
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(f"with_weights expects bytes; got {type(blob).__name__}.")
        header, arrays = _read_blob(bytes(blob))
        model = _copy_model(self.model) if copy else self.model
        _check_matches(header, list(model.weights))
        model.set_weights(arrays)
        return KerasModelOp(model)

    def from_registry(self, registry: Any, ref: str) -> KerasModelOp:
        """Load the weights stored under ``ref`` into a copy of the skeleton ``self``.

        The inverse of ``registry.store(op, weights=op.serialize_weights())``.

        Args:
            registry: Anything with ``load_weights(ref) -> bytes | None``, such
                as a ``pipekit_experiment.ModelRegistry``.
            ref: The hash or tag of the stored model.

        Raises:
            TypeError: If ``registry`` cannot load weights.
            ValueError: If nothing is stored under ``ref`` with weights.
        """
        loader = getattr(registry, "load_weights", None)
        if loader is None:
            raise TypeError(
                "KerasModelOp.from_registry needs a registry with load_weights; "
                f"got {type(registry).__name__}."
            )
        blob = loader(ref)
        if not blob:
            raise ValueError(f"No weights are stored under {ref!r}.")
        return self.with_weights(blob)

    def get_config(self) -> dict[str, Any]:
        return {"model_class": type(self.model).__qualname__}


def _weight_name(path: str) -> str:
    """The last part of a weight's path: ``kernel`` of ``sequential/dense_2/kernel``."""
    return path.rsplit("/", 1)[-1]


def _describe(path: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": _weight_name(path),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def _dtype(name: str) -> np.dtype:
    """NumPy's dtype of ``name``, including the ml_dtypes ones such as bfloat16."""
    try:
        return np.dtype(name)
    except TypeError:
        import ml_dtypes

        return np.dtype(getattr(ml_dtypes, name))


def _read_blob(blob: bytes) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if not blob.startswith(_MAGIC):
        raise ValueError("The bytes are not a KerasModelOp weight blob.")
    start = len(_MAGIC)
    size = int.from_bytes(blob[start : start + 8], "little")
    header = json.loads(blob[start + 8 : start + 8 + size])
    offset = start + 8 + size
    arrays = []
    for entry in header:
        dtype = _dtype(entry["dtype"])
        count = int(np.prod(entry["shape"], dtype=np.int64))
        array = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
        arrays.append(array.reshape(entry["shape"]))
        offset += array.nbytes
    if offset != len(blob):
        raise ValueError("The weight blob is longer than its header says.")
    return header, arrays


def _check_matches(header: list[dict[str, Any]], weights: list[Any]) -> None:
    """Raise unless the skeleton's weights match the header, one by one."""
    if not weights:
        raise ValueError(
            "The skeleton has no weights; build it (call it on an input) first."
        )
    if len(header) != len(weights):
        raise ValueError(
            f"The blob holds {len(header)} weights and the skeleton {len(weights)}."
        )
    for entry, variable in zip(header, weights, strict=True):
        if (
            _weight_name(variable.path) != entry["name"]
            or list(variable.shape) != entry["shape"]
            or np.dtype(variable.dtype).name != entry["dtype"]
        ):
            raise ValueError(
                f"The blob's weight {entry['name']} {entry['shape']} "
                f"{entry['dtype']} does not match the skeleton's {variable.path} "
                f"{list(variable.shape)} {np.dtype(variable.dtype).name}."
            )


def _clone_layer(layer: Any) -> Any:
    """Copy one layer: through its own ``_pipekit_clone`` if it has one."""
    clone = getattr(layer, "_pipekit_clone", None)
    if clone is not None:
        return clone()
    return layer.__class__.from_config(layer.get_config())


def _functional_class() -> type:
    """Keras's class of models built from inputs and outputs, ``keras.Model(i, o)``.

    Keras 3 exports no name for it, and does not set the ``_is_graph_network``
    flag that Keras 2 did, so it is read from where Keras 3 defines it.
    """
    from keras.src.models.functional import Functional  # ty: ignore[unresolved-import]

    return Functional


def _copy_model(model: Any) -> Any:
    """A copy of ``model`` with its own weights, built like ``model``.

    Raises:
        TypeError: If Keras cannot copy the model, naming ``copy=False``.
    """
    keras = import_keras()
    is_graph = isinstance(model, (keras.Sequential, _functional_class()))
    try:
        if is_graph:
            clone = keras.models.clone_model(model, clone_function=_clone_layer)
        else:
            clone = keras.models.clone_model(model)
        if not clone.built and model.built:
            clone.build_from_config(model.get_build_config())
    except Exception as exc:
        raise TypeError(
            f"Keras cannot copy this {type(model).__name__}: {exc}. Load the "
            "weights into the skeleton itself with with_weights(blob, copy=False)."
        ) from exc
    return clone


record_under("pipekit_train.backends.keras", KerasModelOp)
