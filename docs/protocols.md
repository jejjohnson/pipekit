# Protocols — the plug-in seam

pipekit's interoperability story is structural: algorithm and domain
libraries plug in by *satisfying a Protocol*, never by inheriting from
a pipekit base class or importing pipekit at all. Every protocol is
`@runtime_checkable`, so `isinstance(obj, ForwardModel)` succeeds on
any structurally compatible class — adapters in filterax, vardax,
geotoolz, and xrtoolz stay dependency-free.

This page is the inventory: what each protocol is for, where it lives,
and who satisfies it.

## Core (`pipekit.protocols`)

Carrier-agnostic seams for model-inference wrappers.

| Protocol | Contract | Satisfied by |
|----------|----------|--------------|
| `Predictor` | `predict(x)` | scikit-learn estimators, Keras models, geotoolz `GeoTensorEstimator`, xrtoolz `XarrayEstimator` |
| `FittableTransformer` | `fit(x)` + `transform(x)` | scikit-learn transformers, the carrier-aware sklearn wrappers in geotoolz / xrtoolz |

Models invoked via bare `__call__` (PyTorch modules, JAX functions) are
covered by `callable(model)` — a `__call__`-only Protocol would match
nearly everything, so pipekit deliberately doesn't define one.

## Data assimilation (`pipekit_cycle.protocols`)

### The standard-form triple

| Protocol | Contract | Role |
|----------|----------|------|
| `ForwardModel` | `step(state, dt)`, `dt`, `state_signature` | Advance the state — predict |
| `ObservationOperator` | `__call__(state)`, `linearize(state)` | State → predicted observations — compare |
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov)` | Forecast + observations → analysis — update |

vardax exposes every Layer-2 method (OI, 3D-Var, strong/weak/incremental
4D-Var, FourDVarNet, amortized posteriors) as an `AnalysisStep` via
`.as_analysis_step()`; its observation operators (`LinearObs`,
`MaskedIdentity`, `AveragingKernel`) satisfy `ObservationOperator`
directly.

### Reduced-order seams

| Protocol | Contract | Role |
|----------|----------|------|
| `ReducedBasis` | `operg(t, X, state)`, `prior_inv(X)`, `nbasis` | Reduced control basis with a prior (4D-Var on SSH, …) |
| `TangentLinearModel` | `tangent(state, dx, dt)`, `adjoint(state, dz, dt)` | Explicit M′ / M* of the dynamics |
| `ErrorSubspace` | `propagate(model, state, dt)`, `modes`, `rank` | Propagating low-rank covariance factor (SEEK / SEIK) |
| `ReducedOrderModel` | `encode`, `decode`, `step(coords, dt)`, `latent_dim` | Galerkin ROM — projection maps + latent dynamics |
| `LatentMap` | `encode(state)`, `decode(coords)` | The stateless subset of `ReducedOrderModel` — what latent-space filters and autoencoder priors need |

### Ensemble-method seams

| Protocol | Contract | Role |
|----------|----------|------|
| `ObservationNoise` | `covariance()`, `sample(key, shape)` | Error model usable by both deterministic and stochastic analyses |
| `CovarianceLocalizer` | `__call__(covariance, coords)` | Distance-based covariance tapering (Gaspari-Cohn, …) |
| `EnsembleInflator` | `__call__(particles, forecast_particles=None)` | Post-analysis inflation (multiplicative, RTPS / RTPP, additive) |

filterax's `AbstractNoise`, `AbstractLocalizer`, and `AbstractInflator`
hierarchies satisfy these structurally.

### Iterative-inversion seams

| Protocol | Contract | Role |
|----------|----------|------|
| `IterativeProcess` | `init(particles, obs, noise_cov)`, `update(state, forward_evals)` | Ensemble Kalman inversion / sampling (EKI, EKS, UKI) |
| `StepScheduler` | `get_dt(state)` | Pseudo-time step policy for the process loop |

## Experiment tracking (`pipekit_experiment.protocols`)

| Protocol | Contract | Satisfied by |
|----------|----------|--------------|
| `ExperimentTracker` | `start_run`, `log_metrics`, `log_artifact`, `end_run` | MLflow / W&B / ClearML adapters, ad-hoc writers |
| `ModelRegistry` | `store`, `load`, `list`, `tag` | `LocalModelRegistry`, `S3ModelRegistry`, MLflow-registry adapters |

## Training (`pipekit_train`)

| Protocol | Module | Contract |
|----------|--------|----------|
| `Loss` | `pipekit_train.loss` | Carrier-agnostic loss callable |
| `Callback` | `pipekit_train.callbacks` | Training-loop hook points |
| `MetricWriter` | `pipekit_train.writer` | Metric sink (`JSONLWriter` is the reference impl) |

## Benchmarking (`pipekit_evaluate.benchmark.protocols`)

| Protocol | Contract | Role |
|----------|----------|------|
| `Prior` | `sample(n, key)`, `log_prob(theta)` | Prior over task parameters θ |
| `Task` | `name`, `domain`, `prior()`, `simulator()`, `observation_operator()`, `datasets()` | One benchmark scenario — reuses `ForwardModel` / `ObservationOperator` as its simulator |
| `Estimate` | `point()`, `sample(n, key)`, `log_prob(theta)` | A point and/or posterior produced by an estimator |
| `Estimator` | `rung`, `complexity`, `fit(task)`, `__call__(obs)` | Any rung mapping observations → an `Estimate` |
| `Oracle` | `reference_for(task)` | The designated reference estimator for a task |
| `Scorer` | `name`, `__call__(prediction, reference)` | One `(Unit, Lens)` scorer — named metric values |

## Conventions

- Protocols are **minimal**: only the members the consuming cycle or
  loop actually calls. Optional capabilities (e.g.
  `ObservationOperator.linearize`) may raise `NotImplementedError`.
- `runtime_checkable` checks method *presence*, not signatures — treat
  `isinstance` as a structural sniff, not a type-safety guarantee.
- Domain-specific interfaces stay in domain libraries: vardax keeps its
  `Prior` / `GradModulator` / `CostFunction` / `Minimiser` protocols,
  geotoolz keeps band resolution and tiled inference. Only seams shared
  by two or more consumers are promoted here.
