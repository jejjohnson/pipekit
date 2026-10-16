# Tutorial — Build a bigger graph

Builds an anomaly detector for a stack of multi-channel images as one
`Graph`: satellite scenes, microscope fields or telescope exposures read
the same way. Six small domain operators do the science; pipekit's own
operators do the plumbing (parallel reads, retries, a cache, shape
checks, a branch, a second input, an observer and a summary).

**Time:** ~15 minutes.

**Prerequisites:**

```bash
uv add pipekit pipekit-array numpy
```

![The anomaly detector as one pipekit Graph: sources are read in parallel (ThreadMap around Cache around Retry around ReadTile), stacked and checked, composited by a Branch (median for few exposures, mean for many), turned into a z-score against a background input and a channel ratio, scored, thresholded with a Tap, and summarised by a Fanout; composite, score, mask and summary are its outputs](../assets/graph-light.svg#only-light){ width="640" }
![The anomaly detector as one pipekit Graph](../assets/graph-dark.svg#only-dark){ width="640" }

Every box in the figure is one line of code below. The code blocks run
in order as one script.

## Step 1 — The domain operators

These are the only classes you write. Each takes its configuration as
keyword-only constructor arguments and does one thing in `_apply`.
`ReadTile` is a stand-in for your reader: it synthesises a noisy
exposure with a small bright patch in channel 0, so the detector has
something to find.

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pipekit import (
    AssertShape,
    Branch,
    Cache,
    Fanout,
    Graph,
    Input,
    Lambda,
    Operator,
    Retry,
    Tap,
    ThreadMap,
)
from pipekit_array import StackAlong

Array = NDArray[np.float64]
C, H, W = 3, 64, 64  # channels, rows, columns


class ReadTile(Operator):
    """source id ↦ one exposure (C, H, W). A stand-in for your reader."""

    def _apply(self, source: str) -> Array:
        rng = np.random.default_rng(sum(map(ord, source)))
        x = 100.0 + rng.normal(0.0, 1.0, (C, H, W))  #              str → (C, H, W)
        x[0, 20:26, 30:36] += 8.0  # something real, in every exposure
        return x


class Composite(Operator):
    """x̄_chw = reduce_t x_tchw, by mean or median over the stack."""

    def __init__(self, *, how: str) -> None:
        self.how = how

    def _apply(self, stack: Array) -> Array:
        reduce = np.median if self.how == "median" else np.mean
        return reduce(stack, axis=0)  #                       (T, C, H, W) → (C, H, W)


class ZScore(Operator):
    """z = (x̄ − μ_b) / σ_b, against a background (μ_b, σ_b)."""

    def __init__(self, *, sigma: float) -> None:
        self.sigma = sigma  # σ_b

    def _apply(self, composite: Array, background: Array) -> Array:
        return (composite - background) / self.sigma  # (C, H, W)² → (C, H, W)


class NormalizedDifference(Operator):
    """r = (x_a − x_b) / (x_a + x_b), a ratio between two channels."""

    def __init__(self, *, a: int, b: int) -> None:
        self.a, self.b = a, b

    def _apply(self, x: Array) -> Array:
        xa, xb = x[self.a], x[self.b]
        return (xa - xb) / (xa + xb)  #                           (C, H, W) → (H, W)


class Score(Operator):
    """s = max_c |z_c| + w · |r − r̄|: how unusual each pixel is."""

    def __init__(self, *, weight: float) -> None:
        self.weight = weight  # w

    def _apply(self, z: Array, ratio: Array) -> Array:
        return np.abs(z).max(axis=0) + self.weight * np.abs(ratio - ratio.mean())


class Threshold(Operator):
    """m = 1[s > k]: the anomaly mask."""

    def __init__(self, *, k: float) -> None:
        self.k = k

    def _apply(self, score: Array) -> NDArray[np.bool_]:
        return score > self.k  #                                     (H, W) → (H, W)
```

## Step 2 — Read in parallel, safely

A graph starts from named `Input`s. Calling an operator on an `Input`
(or on the result of another call) records a node instead of running.

Reading is where real pipelines fail: a flaky disk or network, the same
file asked for twice. Wrappers take an operator and return one, so they
nest: `Retry` re-runs a failed read, `Cache` keeps each source's result
so it is read once, and `ThreadMap` maps the whole thing over the list
of sources on eight threads. `StackAlong` (from `pipekit-array`) stacks
the exposures, and `AssertShape` fails early if a reader returned the
wrong shape.

```python
sources, background = Input("sources"), Input("background")

# read every source in parallel; retry flaky reads; never read one twice
read = ThreadMap(
    Cache(Retry(ReadTile(), attempts=3, base_delay=0.5, on=(OSError,))), n_workers=8
)
# list of (C, H, W) → (T, C, H, W), checked
stack = (read | StackAlong(axis=0) | AssertShape((None, C, H, W)))(sources)
```

## Step 3 — Choose the composite

`Branch` routes on a predicate of its input. With few exposures the
median resists an outlier; with many the mean is cheaper and as good.

```python
composite = Branch(
    lambda s: s.shape[0] < 5,
    if_true=Composite(how="median"),
    if_false=Composite(how="mean"),
)(stack)
```

## Step 4 — Join a second input and score

An operator with two arguments joins two branches of the graph. Here
`ZScore` takes the composite and the `background` input; the channel
ratio branches off the same composite, and `Score` joins the two again.

```python
z = ZScore(sigma=1.0)(composite, background)  # (C, H, W)
ratio = NormalizedDifference(a=0, b=1)(composite)  # (H, W)
score = Score(weight=10.0)(z, ratio)  # (H, W)
```

## Step 5 — Observe and summarise

`Tap` runs a side effect (here, a log line) and passes its input on
unchanged. `Fanout` runs several operators on the same input and returns
a dict of their results.

```python
mask = (Threshold(k=3.0) | Tap(lambda m: print(f"{int(m.sum())} pixels flagged")))(
    score
)
summary = Fanout(
    {
        "flagged": Lambda(lambda m: int(m.sum())),
        "fraction": Lambda(lambda m: float(m.mean())),
    }
)(mask)
```

## Step 6 — Wrap it in a `Graph` and run it

`Graph` names the inputs and the outputs; everything between them is
what you recorded in steps 2–5. Run it with keyword arguments, one per
input; it returns a dict, one entry per output.

```python
detector = Graph(
    inputs={"sources": sources, "background": background},
    outputs={"composite": composite, "score": score, "mask": mask, "summary": summary},
)

out = detector(
    sources=[f"scene-{i:02d}" for i in range(6)],  # T = 6
    background=np.full((C, H, W), 100.0),  # μ_b
)
# {"composite": (C, H, W), "score": (H, W), "mask": (H, W), "summary": {...}}
print({k: getattr(v, "shape", v) for k, v in out.items()})
```

```text
36 pixels flagged
{'composite': (3, 64, 64), 'score': (64, 64), 'mask': (64, 64), 'summary': {'flagged': 36, 'fraction': 0.0087890625}}
```

The 36 flagged pixels are the 6 × 6 patch `ReadTile` put in channel 0.

## Make it yours

- **Another science, another reader.** Replace `ReadTile` with whatever
  returns your carrier: a FITS cube, a microscope field, a satellite
  scene through geotoolz. Nothing downstream changes as long as the
  shape does; `AssertShape` tells you when it does not.
- **Change a setting, not the code.** Every operator keeps its
  configuration (`Threshold(k=3.0)`, `Composite(how="median")`), so a
  different threshold or composite is a constructor argument.
- **Save an operator's configuration.** `pipekit.dumps(op)` writes any
  operator's configuration as JSON; `pipekit.loads` rebuilds single
  operators (nested ones need a YAML or Hydra loader).

## Where next

- [Concepts](../concepts.md): the model behind `Operator`, `Sequential`
  and `Graph`.
- [Run a DA cycle](da-cycle.md): the same interface stepping through
  time.
- [Train an emulator](train-emulator.md): an operator that learns.
