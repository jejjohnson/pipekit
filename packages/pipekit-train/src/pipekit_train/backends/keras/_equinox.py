"""`EquinoxLayer`, an Equinox module inside Keras (ADR D21)."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import numpy as np

from pipekit_train._signatures import accepts_keyword
from pipekit_train.backends.keras._backend import import_keras


keras = import_keras()


class EquinoxLayer(keras.layers.JaxLayer):
    """An Equinox module as a Keras layer, trained by Keras.

    The module's inexact arrays (its weights) become the layer's trainable
    variables, and everything else in it (activations, sizes, flags) stays as
    Python. The layer calls the module on one sample at a time and maps it over
    the batch, the convention of the Equinox backend, unless ``per_sample`` is
    ``False``. Dropout and other layers with an inference flag are switched off
    outside training, and a module whose call takes ``key`` gets one key per
    sample while training.

    Keras frees the arrays it trains, so the layer trains copies, and the module
    it was given stays usable. `module` rebuilds the module from the layer's
    current weights.

    Args:
        module: The ``eqx.Module``.
        per_sample: Map the module over the leading axis of the inputs.
        seed: The seed of the layer's random keys.
        **kwargs: Passed to ``keras.layers.JaxLayer`` (``name``, …).
    """

    def __init__(
        self,
        module: eqx.Module,
        *,
        per_sample: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(module, eqx.Module):
            raise TypeError(
                f"EquinoxLayer.module must be an eqx.Module; got {type(module).__name__}."
            )
        params, static = eqx.partition(module, eqx.is_inexact_array)
        leaves, treedef = jax.tree.flatten(params)
        takes_key = accepts_keyword(module, "key")

        def call_fn(params: Any, rng: Any, inputs: Any, training: Any) -> Any:
            net = eqx.combine(jax.tree.unflatten(treedef, list(params)), static)
            net = eqx.nn.inference_mode(net, not training)
            keyed = takes_key and training and rng is not None
            if not per_sample:
                return net(inputs, key=rng) if keyed else net(inputs)
            if not keyed:
                return jax.vmap(net)(inputs)
            size = jax.tree.leaves(inputs)[0].shape[0]
            keys = jax.random.split(rng, size)
            return jax.vmap(lambda row, key: net(row, key=key))(inputs, keys)

        # copies: Keras donates the buffers it trains
        copies = [np.array(leaf) for leaf in leaves]
        super().__init__(call_fn, params=copies, seed=seed, **kwargs)
        self._module_static = static
        self._module_treedef = treedef
        self._per_sample = per_sample
        self._module_seed = seed

    @property
    def module(self) -> eqx.Module:
        """The module rebuilt from the layer's current weights."""
        values = [jax.numpy.asarray(variable.value) for variable in self.params]
        return eqx.combine(
            jax.tree.unflatten(self._module_treedef, values), self._module_static
        )

    def _pipekit_clone(self) -> EquinoxLayer:
        """A new layer holding a copy of this one's module, for `KerasModelOp`."""
        return EquinoxLayer(
            self.module, per_sample=self._per_sample, seed=self._module_seed
        )
