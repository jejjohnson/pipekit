# Guards

NumPy and JAX interoperate by default: a NumPy array next to a JAX one joins it, and nothing warns.
When you build an app, and above all one on a GPU, you may want to know what moves.
A guard tells you, and it is opt-in.

```python
import pipekit_ops as pk

# today (pipekit-ops)
with pk.guard("warn"):  # "off" (the default), "warn" or "raise"
    app(scene)
```

`"warn"` gives a `BackendWarning` for each move, at your line and with the name of the step that moved data.
`"raise"` gives a `BackendError` at the first one, which suits a test.

| Reported | Not reported |
| --- | --- |
| a NumPy array of one dimension or more joins JAX | Python numbers, NumPy scalars and 0-d arrays |
| JAX data is copied to the host | a NumPy constant inside `jax.jit` |
| joining widens the dtype, such as float32 to float64 | |

The level is a context variable.
It holds inside a `with` block, in pipekit's thread and async executors, and not in a process or in a thread pool of your own unless that runs `contextvars.copy_context().run`.

Code that moves data calls `report_move`; with no guard that costs one read of the variable:

```python
# today (pipekit-ops)
pk.report_move("Smooth", "JAX data copied to the host")
```

JAX has a guard of its own, `jax.transfer_guard`, and it answers another question.
At `"disallow"` it also refuses `x * 2.0`, because the number is transferred, and on CPU it does not see a copy from JAX to NumPy.
