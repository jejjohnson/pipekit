# Changelog

## [0.0.6](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.5...pipekit-array-v0.0.6) (2026-10-27)


### Features

* **array:** padded batches with batch, Padded, stack_items and reduce(where=) ([#121](https://github.com/jejjohnson/pipekit/issues/121)) ([9bc9e1d](https://github.com/jejjohnson/pipekit/commit/9bc9e1dbbdc2d8cce5db9bbcbcd81b7076bcbb5f))

## [0.0.5](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.4...pipekit-array-v0.0.5) (2026-10-25)


### Features

* **core:** on_import runs code when a module is imported; pipekit-array registers DimArray lazily ([#117](https://github.com/jejjohnson/pipekit/issues/117)) ([3446f13](https://github.com/jejjohnson/pipekit/commit/3446f13ef2a5feb70109aaf9df1954052b04f8f2))

## [0.0.4](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.3...pipekit-array-v0.0.4) (2026-10-24)


### Features

* **array:** add DimArray and its reference adapter ([#109](https://github.com/jejjohnson/pipekit/issues/109)) ([df7ed83](https://github.com/jejjohnson/pipekit/commit/df7ed8387e720e319d984802e64caa9c7c2d472f))
* **array:** add KernelSpec and KernelOp ([#111](https://github.com/jejjohnson/pipekit/issues/111)) ([c659ca8](https://github.com/jejjohnson/pipekit/commit/c659ca8b882bef6d35d8be7847ec1c288329047a))
* **array:** dim-level functions over the adapter ([#110](https://github.com/jejjohnson/pipekit/issues/110)) ([18a2dff](https://github.com/jejjohnson/pipekit/commit/18a2dfffd754db211b3f9c481698fa13ff739c72))
* **array:** the [xarray] adapter and the apply_ufunc path ([#112](https://github.com/jejjohnson/pipekit/issues/112)) ([fa0e924](https://github.com/jejjohnson/pipekit/commit/fa0e924dbe55dd05513f1b0bd14ff3e696cab78f))
* **jax:** pipekit_jax.to_jax(op) builds a JaxOperator from a KernelOp ([#115](https://github.com/jejjohnson/pipekit/issues/115)) ([0b104c8](https://github.com/jejjohnson/pipekit/commit/0b104c8f1a6c6d810197752a67b2e6753c0f2a90))

## [0.0.3](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.2...pipekit-array-v0.0.3) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.1...pipekit-array-v0.0.2) (2026-07-11)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
* **pipekit-array:** namespace dispatch + MeanScalar + Stack/Concat (Phase A) ([#26](https://github.com/jejjohnson/pipekit/issues/26)) ([d69a101](https://github.com/jejjohnson/pipekit/commit/d69a101d351fc6a80121fd6004f9a273584b8f11))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## 0.0.1 (2026-05-22)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
