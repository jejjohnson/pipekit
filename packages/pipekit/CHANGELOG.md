# Changelog

## [0.3.0](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.2.0...pipekit-v0.3.0) (2026-11-25)


### ⚠ BREAKING CHANGES

* **pipekit-jax:** pipekit-jax no longer depends on diffrax and integrix ([#289](https://github.com/jejjohnson/pipekit/issues/289))

### Code Refactoring

* **pipekit-jax:** pipekit-jax no longer depends on diffrax and integrix ([#289](https://github.com/jejjohnson/pipekit/issues/289)) ([e600b8c](https://github.com/jejjohnson/pipekit/commit/e600b8ce9555f35bc345a88af3903b7785693884))

## [0.2.0](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.1.5...pipekit-v0.2.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* `from pipekit import X` no longer works; use `from pipekit_ops import X`. The pipekit distribution now holds no code and installs the whole workspace, JAX included, so libraries should depend on pipekit-ops. pipeline_hash changes for every pipeline, because the hash covers each operator's module.

### Features

* pipekit-ops is the core package, pipekit becomes the umbrella ([#199](https://github.com/jejjohnson/pipekit/issues/199)) ([903b0c3](https://github.com/jejjohnson/pipekit/commit/903b0c38202173418cd7594ba42fe3c622876574))

## Changelog
