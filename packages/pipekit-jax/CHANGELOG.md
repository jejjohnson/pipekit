# Changelog

## [0.4.1](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.4.0...pipekit-jax-v0.4.1) (2026-11-26)


### Bug Fixes

* **deps:** each pipekit package asks for the releases whose areas it imports ([#306](https://github.com/jejjohnson/pipekit/issues/306)) ([e0a1384](https://github.com/jejjohnson/pipekit/commit/e0a1384186ce8523ba23ed4b9d0faf2f7732c730))

## [0.4.0](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.3.0...pipekit-jax-v0.4.0) (2026-11-26)


### ⚠ BREAKING CHANGES

* **cycle:** pipekit-cycle is four areas, with the protocols where they were ([#303](https://github.com/jejjohnson/pipekit/issues/303))
* **train:** pipekit-train is five areas, and each backend is an area of its own ([#297](https://github.com/jejjohnson/pipekit/issues/297))
* **ops:** pipekit-ops is seven areas, each with one job ([#295](https://github.com/jejjohnson/pipekit/issues/295))

### Code Refactoring

* **cycle:** pipekit-cycle is four areas, with the protocols where they were ([#303](https://github.com/jejjohnson/pipekit/issues/303)) ([7268963](https://github.com/jejjohnson/pipekit/commit/7268963ddfe3efdd69b6804dc1426a754de46ee5))
* **ops:** pipekit-ops is seven areas, each with one job ([#295](https://github.com/jejjohnson/pipekit/issues/295)) ([780fd27](https://github.com/jejjohnson/pipekit/commit/780fd27da51dbfe452da13fbb4268ab76cad6b26))
* **train:** pipekit-train is five areas, and each backend is an area of its own ([#297](https://github.com/jejjohnson/pipekit/issues/297)) ([f2a144e](https://github.com/jejjohnson/pipekit/commit/f2a144e056849fde8a6f0c629f37a6a3f3948272))

## [0.3.0](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.2.0...pipekit-jax-v0.3.0) (2026-11-25)


### ⚠ BREAKING CHANGES

* **array:** pipekit-array is six areas, each with one job ([#292](https://github.com/jejjohnson/pipekit/issues/292))

### Code Refactoring

* **array:** pipekit-array is six areas, each with one job ([#292](https://github.com/jejjohnson/pipekit/issues/292)) ([3aab7d8](https://github.com/jejjohnson/pipekit/commit/3aab7d85306ce667f96001b2e2331cacd6569d43))

## [0.2.0](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.1.0...pipekit-jax-v0.2.0) (2026-11-25)


### ⚠ BREAKING CHANGES

* **pipekit-jax:** pipekit-jax no longer depends on diffrax and integrix ([#289](https://github.com/jejjohnson/pipekit/issues/289))

### Code Refactoring

* **pipekit-jax:** pipekit-jax no longer depends on diffrax and integrix ([#289](https://github.com/jejjohnson/pipekit/issues/289)) ([e600b8c](https://github.com/jejjohnson/pipekit/commit/e600b8ce9555f35bc345a88af3903b7785693884))

## [0.1.0](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.6...pipekit-jax-v0.1.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* `from pipekit import X` no longer works; use `from pipekit_ops import X`. The pipekit distribution now holds no code and installs the whole workspace, JAX included, so libraries should depend on pipekit-ops. pipeline_hash changes for every pipeline, because the hash covers each operator's module.

### Features

* pipekit-ops is the core package, pipekit becomes the umbrella ([#199](https://github.com/jejjohnson/pipekit/issues/199)) ([903b0c3](https://github.com/jejjohnson/pipekit/commit/903b0c38202173418cd7594ba42fe3c622876574))

## [0.0.6](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.5...pipekit-jax-v0.0.6) (2026-10-24)


### Features

* **jax:** pipekit_jax.to_jax(op) builds a JaxOperator from a KernelOp ([#115](https://github.com/jejjohnson/pipekit/issues/115)) ([0b104c8](https://github.com/jejjohnson/pipekit/commit/0b104c8f1a6c6d810197752a67b2e6753c0f2a90))

## [0.0.5](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.4...pipekit-jax-v0.0.5) (2026-10-24)


### Bug Fixes

* **experiment:** verify the content hash when loading from a registry ([#93](https://github.com/jejjohnson/pipekit/issues/93)) ([3374a57](https://github.com/jejjohnson/pipekit/commit/3374a570cfb7b3a72a21bb71f73647ac09ea9d0e))

## [0.0.4](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.3...pipekit-jax-v0.0.4) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.0.3](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.2...pipekit-jax-v0.0.3) (2026-10-17)


### Features

* **jax:** add EnsembleScanCycle and a "pipekit under jit" tutorial ([#70](https://github.com/jejjohnson/pipekit/issues/70)) ([d1eb41d](https://github.com/jejjohnson/pipekit/commit/d1eb41d4ac98c371cb198ee72aedfdf45046127c))
* **jax:** add JaxOperator, check_jax_operator and a shared weights mixin ([#68](https://github.com/jejjohnson/pipekit/issues/68)) ([232014a](https://github.com/jejjohnson/pipekit/commit/232014ae01fe931620c3e13d806845a10b773b17))
* **jax:** add ScanCycle and Rollout — lax.scan-based time stepping ([#69](https://github.com/jejjohnson/pipekit/issues/69)) ([10ddaae](https://github.com/jejjohnson/pipekit/commit/10ddaae0016a50fc2f207cb17e09248239ef6602))
* **jax:** register Sequential and Graph as JAX pytrees ([#67](https://github.com/jejjohnson/pipekit/issues/67)) ([10d2f21](https://github.com/jejjohnson/pipekit/commit/10d2f21bbae63c57af6e929ab0087025c7a823a6))


### Bug Fixes

* **jax:** Rollout rejects a Sequential holding a stateful step ([#81](https://github.com/jejjohnson/pipekit/issues/81)) ([cc88e80](https://github.com/jejjohnson/pipekit/commit/cc88e801b6b2c0e41b4875846830f1ae49723a73))
* **jax:** use the nested round-trip in JaxOperator and check_jax_operator ([#80](https://github.com/jejjohnson/pipekit/issues/80)) ([0b58b8e](https://github.com/jejjohnson/pipekit/commit/0b58b8e052800c70aa1f86f20e97b1a7baf51494))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-jax-v0.0.1...pipekit-jax-v0.0.2) (2026-07-11)


### Features

* **adjoints:** shared adjoint-strategy specs and diffrax adapters ([#46](https://github.com/jejjohnson/pipekit/issues/46)) ([c88c6f9](https://github.com/jejjohnson/pipekit/commit/c88c6f97c18da1bce7735cd4ee322bcba2828591))
* **pipekit-jax:** new workspace package — JaxModelOp with weight round-trip (closes [#19](https://github.com/jejjohnson/pipekit/issues/19)) ([#23](https://github.com/jejjohnson/pipekit/issues/23)) ([b18022b](https://github.com/jejjohnson/pipekit/commit/b18022b56ed6f479dde105b04fed8452a84b298b))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## Changelog
