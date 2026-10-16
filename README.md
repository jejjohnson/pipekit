<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="pipekit" src="docs/assets/logo-light.svg" width="320">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml"><img alt="Type Check" src="https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="docs/concepts.md"><b>Concepts</b></a> ·
  <a href="docs/protocols.md"><b>Protocols</b></a> ·
  <a href="docs/tutorials/train-emulator.md"><b>Tutorials</b></a>
</p>

**Composable scientific pipelines: one operator interface, from loading
data to training, assimilating and deploying.**

Every step of a scientific workflow becomes an `Operator`: one method,
`_apply`, that takes a carrier and returns one. The carrier is whatever
your science uses (a NumPy array, an xarray Dataset, a DataFrame, an
AnnData, a dict of arrays) and pipekit never looks inside it. Operators
compose with `|`, wire into graphs, wrap in retries, caches and
observers, and carry state through time steps and training loops, and
each of those is itself an Operator. pipekit was built for the
[GeoStack](https://github.com/jejjohnson/geostack), but the core is pure
Python with no dependencies and nothing in it is about geoscience.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/operator-dark.svg">
    <img alt="A pipekit Operator takes a carrier and returns one through a single _apply method; it composes into a Sequential (a | b | c), a Graph with named inputs and outputs, wrappers such as Retry, Cache and Tap, and stateful steps such as Cycle and TrainingLoop, each itself an Operator" src="docs/assets/operator-light.svg" width="100%">
  </picture>
</p>

## Quick start

```bash
uv add "pipekit @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit"
uv add numpy jaxtyping
```

```python
import numpy as np
from jaxtyping import Float

from pipekit import Graph, Input, Operator, Sequential, Tap, dumps, loads

# ── shapes ──
# N  samples
Vector = Float[np.ndarray, " N"]


class Standardize(Operator):
    """Centre and scale by a fixed mean μ and spread σ."""

    def __init__(self, *, mean: float, std: float) -> None:
        # μ
        self.mean = mean
        # σ
        self.std = std

    def _apply(self, x: Vector) -> Vector:
        # z = (x − μ) / σ
        return (x - self.mean) / self.std


class Clip(Operator):
    """Clip to the interval [lo, hi]."""

    def __init__(self, *, lo: float, hi: float) -> None:
        self.lo = lo
        self.hi = hi

    def _apply(self, z: Vector) -> Vector:
        # z̃ = min(max(z, lo), hi)
        return np.clip(z, self.lo, self.hi)


class Difference(Operator):
    """Subtract the second input from the first."""

    def _apply(self, a: Vector, b: Vector) -> Vector:
        # d = a − b
        return a - b


def show(v: Vector) -> None:
    print("clean:", v)


# 1 · compose with |: a Sequential, itself an Operator

# z = (x − 10) / 2
standardize = Standardize(mean=10.0, std=2.0)
# z̃ = min(max(z, −3), 3)
clip = Clip(lo=-3.0, hi=3.0)
clean: Sequential = standardize | clip

x: Vector = np.array([4.0, 10.0, 30.0])
# [-3., 0., 3.]
z: Vector = clean(x)

# 2 · wire a Graph: named inputs, shared operators, named outputs

difference = Difference()
a = Input("a")
b = Input("b")
a_clean = clean(a)
b_clean = clean(b)
anomaly = difference(a_clean, b_clean)
graph = Graph(inputs={"a": a, "b": b}, outputs={"anomaly": anomaly})

# {"anomaly": d}, one value per sample
out: dict[str, Vector] = graph(a=x, b=x + 2.0)

# 3 · observe without changing the flow

traced: Sequential = clean | Tap(show)

# 4 · configuration is data: an operator round-trips through JSON

text: str = dumps(standardize)
# Standardize(mean=10.0, std=2.0)
same: Operator = loads(text)
```

The same interface steps through time. With `pipekit-cycle`, a forward
model is an operator and a `Cycle` runs it:

```python
import numpy as np
from jaxtyping import Float

import pipekit_cycle as pc

# N  state variables
State = Float[np.ndarray, " N"]


def decay(x: State, dt: float) -> State:
    """One explicit Euler step of dx/dt = −k x, with k = 0.1."""
    # x_{t+1} = x_t − Δt · k · x_t
    return x - dt * 0.1 * x


# x_t ↦ x_{t+1}, Δt = 1
step = pc.CallableForward(decay, dt=1.0)
# x_0 ↦ x_24
forecast = pc.Cycle(step_op=step, n_steps=24)

# x₂₄ = (1 − 0.1)²⁴ x₀ ≈ 0.080 x₀
x24, _ = forecast(np.ones(3), None)
```

Swap `CallableForward` for `NeuralForward` around a trained model from
the registry and the same `Cycle` forecasts with an emulator; add an
observation operator and an analysis step and it assimilates
(`DACycle`, `EnsembleDACycle`).

## A bigger graph

Everything above composes. This is an anomaly detector for a stack of
multi-channel images: satellite scenes, microscope fields or telescope
exposures read the same way. Six small domain operators do the science;
pipekit does the rest: reading in parallel with retries and a cache,
stacking and checking shapes, choosing a composite by the size of the
stack, joining a second input, observing, and summarising, with four
named outputs.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/graph-dark.svg">
    <img alt="An anomaly detector as one pipekit Graph: sources are read in parallel (ThreadMap around Cache around Retry around ReadTile), stacked and checked, composited by a Branch (median for few exposures, mean for many), turned into a z-score against a background input and a channel ratio, scored, thresholded with a Tap, and summarised by a Fanout; composite, score, mask and summary are its outputs" src="docs/assets/graph-light.svg" width="760">
  </picture>
</p>

The wiring, once the six domain operators are written:

```python
# ReadTile … Threshold, CountFlagged and FlaggedFraction are the tutorial's
# operators; few_exposures and log_flagged are its two named functions.

# the steps

# on OSError, read again: up to 3 attempts
retried_read = Retry(ReadTile(), attempts=3, base_delay=0.5, on=(OSError,))
# keep each source's exposure, so no source is read twice
cached_read = Cache(retried_read)
# [s_1, …, s_T] ↦ [x_1, …, x_T], on 8 threads
read = ThreadMap(cached_read, n_workers=8)
# [x_1, …, x_T] ↦ X ∈ ℝ^{T×C×H×W}
stack_exposures = StackAlong(axis=0)
check_shape = AssertShape((None, C, H, W))
# x̄ = median_t X when T < 5, else mean_t X
choose_composite = Branch(
    few_exposures,
    if_true=Composite(how="median"),
    if_false=Composite(how="mean"),
)
# z = (x̄ − μ_b) / σ_b
z_score = ZScore(sigma=1.0)
# r = (x̄_0 − x̄_1) / (x̄_0 + x̄_1)
channel_ratio = NormalizedDifference(a=0, b=1)
# s = max_c |z_c| + w · |r − r̄|
score_pixels = Score(weight=10.0)
# m = 1[s > k], logged on the way through
threshold = Threshold(k=3.0)
report = Tap(log_flagged)
# m ↦ {"flagged": n, "fraction": f}
summarise = Fanout({"flagged": CountFlagged(), "fraction": FlaggedFraction()})

# the bundles

load: Sequential = read | stack_exposures | check_shape
flag: Sequential = threshold | report

# wired as a graph

sources_in = Input("sources")
background_in = Input("background")
stack_node = load(sources_in)
composite_node = choose_composite(stack_node)
z_node = z_score(composite_node, background_in)
ratio_node = channel_ratio(composite_node)
score_node = score_pixels(z_node, ratio_node)
mask_node = flag(score_node)
summary_node = summarise(mask_node)

detector = Graph(
    inputs={"sources": sources_in, "background": background_in},
    outputs={
        "composite": composite_node,
        "score": score_node,
        "mask": mask_node,
        "summary": summary_node,
    },
)
```

The full, runnable version, with the six operators and each step
explained, is the tutorial
[Build a bigger graph](docs/tutorials/anomaly-graph.md).

More end-to-end examples, each with its pre- and post-processing and a
script that runs as printed:
[train, register and forecast](docs/tutorials/train-register-forecast.md),
[inference](docs/tutorials/inference.md),
[ensemble data assimilation](docs/tutorials/ensemble-da.md) and
[ETL over many regions](docs/tutorials/etl-regions.md).

## The packages

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/packages-dark.svg">
    <img alt="The pipekit workspace: six sister packages (cycle, train, experiment, evaluate, array, jax) on the dependency-free pipekit core" src="docs/assets/packages-light.svg" width="100%">
  </picture>
</p>

A uv workspace of seven packages. Install only what you need: each one
builds on the core and on none of the others, except `pipekit-jax` on
`pipekit-cycle`; tool integrations are opt-in extras.

| Package | Status | What it gives you |
|---|---|---|
| `pipekit` | implemented | The core: `Operator`, `Sequential`, `Graph`; control flow (`Branch`, `Switch`, `Try`, `Retry`, `Coalesce`), observers (`Tap`, `Snapshot`, `Profile`), `Cache`, QC asserts, parallel maps, JSON `dumps` / `loads`, `StatefulOperator` |
| `pipekit-cycle` | implemented | Time stepping and data assimilation: `Cycle`, `DACycle`, `EnsembleDACycle`, `SmootherCycle`, and the `ForwardModel`, `ObservationOperator` and `AnalysisStep` protocols |
| `pipekit-train` | implemented | Training as an operator: `TrainingLoop`, datasets, losses, callbacks, `HyperSweep`; Equinox, NumPyro and BlackJAX backends behind extras |
| `pipekit-experiment` | implemented | A content-addressed model registry (`LocalModelRegistry`, `S3ModelRegistry`, keyed by a hash of config and weights) and tracker protocols, with DVC, Hydra, Metaflow and MLflow adapters |
| `pipekit-jax` | implemented | `JaxModelOp` (an `eqx.Module` with registry round-trip), `DiffraxForwardModel`, adjoints; builds on `pipekit-cycle` |
| `pipekit-array` | partial | Array API dispatch: one operator on NumPy, JAX, CuPy, PyTorch or dask (`array_namespace`, `StackAlong`, `ConcatenateAlong`) |
| `pipekit-evaluate` | partial | A benchmark ladder: `Benchmark`, `MetricScorer`, `EvaluationReport` |

Each package has its own README under [`packages/`](packages/) and,
for the larger ones, design notes under `packages/<pkg>/docs/design/`.

## Who builds on it

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/ecosystem-dark.svg">
    <img alt="pipekit in the middle: on the left the GeoStack packages that build on it today (geotoolz, xrtoolz, vardax, plumax, radtrax's geo extra); on the right, illustrative carriers from other sciences (genomics, astronomy, climate, imaging, simulation)" src="docs/assets/ecosystem-light.svg" width="100%">
  </picture>
</p>

In the GeoStack, [geotoolz](https://github.com/jejjohnson/geotoolz) and
[xrtoolz](https://github.com/jejjohnson/xrtoolz) narrow the carrier to a
`GeoTensor` and an xarray object;
[vardax](https://github.com/jejjohnson/vardax) and
[plumax](https://github.com/jejjohnson/plumax) run their assimilation and
models through `pipekit-cycle`; [radtrax](https://github.com/jejjohnson/radtrax)
ships its retrievals as operators behind its `[geo]` extra.

Nothing there is specific to the Earth. A genomics pipeline on AnnData,
an astronomy reduction on astropy tables, or a microscopy workflow on
image arrays gets the same composition, observation, caching and
serialisation by writing its own operators; the right-hand side of the
figure is that invitation, not a list of integrations. Libraries can
also plug in without importing pipekit at all, by satisfying its
runtime-checkable protocols (`ForwardModel`, `AnalysisStep`,
`ObservationNoise`, `ModelRegistry`, …; see
[docs/protocols.md](docs/protocols.md)).

## The six things to know

1. **An `Operator` has one method**, `_apply(self, carrier) -> carrier`.
   pipekit never inspects the carrier; domain packages narrow it.
2. **Composition is `|`.** `a | b | c` is a `Sequential`; branching and
   merging flows are a `Graph` built by calling operators on `Input`s.
3. **Wrappers are operators too.** `Retry`, `Try`, `Cache`, `Tap`,
   `ThreadMap` and the QC asserts take an operator and return one, so
   they nest anywhere.
4. **State is explicit.** A `StatefulOperator` maps
   `(carrier, state) → (carrier, state)`; `Cycle`, `DACycle` and
   `TrainingLoop` are built on it.
5. **Configuration is data.** An operator's keyword-only constructor
   arguments are its config; `dumps` writes it as JSON and `loads`
   rebuilds the operator. The model registry hashes config and weights
   together, so the same model has the same key.
6. **Libraries plug in by protocol.** Algorithm libraries satisfy
   runtime-checkable protocols structurally and never have to import
   pipekit ([docs/protocols.md](docs/protocols.md)).

The long form is [docs/concepts.md](docs/concepts.md).

## Where to start

| If you want to… | Read |
|---|---|
| Understand the model in ten minutes | [docs/concepts.md](docs/concepts.md) |
| Plug a library in without depending on pipekit | [docs/protocols.md](docs/protocols.md) |
| Go from `git clone` to a running pipeline | [docs/getting-started.md](docs/getting-started.md) |
| Install only the parts you need | [docs/installation.md](docs/installation.md) |
| See every core operator, group by group | [docs/core-tour.md](docs/core-tour.md) |
| Pick a shape for a deployment (batch, service, orchestrator, audit) | [docs/patterns.md](docs/patterns.md) |
| Know why operators are classes, not functions | [docs/background/toolz.md](docs/background/toolz.md) |
| Train an emulator end to end | [docs/tutorials/train-emulator.md](docs/tutorials/train-emulator.md) |
| Run a data-assimilation cycle | [docs/tutorials/da-cycle.md](docs/tutorials/da-cycle.md) |
| Build a bigger graph, step by step | [docs/tutorials/anomaly-graph.md](docs/tutorials/anomaly-graph.md) |
| Clean an archive, train, register and forecast | [docs/tutorials/train-register-forecast.md](docs/tutorials/train-register-forecast.md) |
| Put a model in a pre- and post-processing chain | [docs/tutorials/inference.md](docs/tutorials/inference.md) |
| Assimilate observations with an ensemble | [docs/tutorials/ensemble-da.md](docs/tutorials/ensemble-da.md) |
| Run an ETL batch over many regions | [docs/tutorials/etl-regions.md](docs/tutorials/etl-regions.md) |
| Browse the API | [jejjohnson.github.io/pipekit](https://jejjohnson.github.io/pipekit/) |

## Develop

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install      # uv sync --all-groups + pre-commit hooks
make test         # every package's tests
make lint         # ruff check . (the whole repo)
make typecheck    # ty on every package
make docs-serve   # MkDocs preview
```

Conventional Commits drive releases (release-please, one version per
package). The rules for contributors and coding agents are in
[`AGENTS.md`](AGENTS.md), review follows [`CODE_REVIEW.md`](CODE_REVIEW.md),
and the local workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md). The
figures are generated: edit `scripts/make_readme_assets.py` and run it.

## License

MIT; see [LICENSE](LICENSE).
