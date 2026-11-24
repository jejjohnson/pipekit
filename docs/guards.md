# Guards

NumPy and JAX interoperate by default: a NumPy array next to a JAX one joins it, and nothing warns.
An operator that runs on NumPy only (`backends = ("numpy",)`) takes JAX data too: it runs on the host and returns JAX.
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

## Which steps are jittable

`jit_report` runs a pipeline once and tries each step under a JAX trace, without compiling anything.

```python
# today (pipekit-ops)
report = pk.jit_report(app, scene)
print(report)
# [0] Roll: jittable
# [1] HostSort: no (BackendError: HostSort runs on numpy only, …)
# [2] Branch: no (TracerBoolConversionError: …)
# [3] Cache: no (Cache declares traceable = False)
report.require()  # BackendError, naming every step that is not
```

A step is not jittable when the trace raises, when it stores a traced value, or when it, or an operator inside it, declares `traceable = False`.
Set that on an operator of your own that writes a file or keeps what it computes.

The run is eager, so a step that writes a file writes it.
Steps are deep copies where Python can copy them, so the pipeline keeps nothing of the run; a closure is not copied, so a `Tap` calls its function twice, once with a traced value.
