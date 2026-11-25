# Changelog

## [0.2.0](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.6...pipekit-array-v0.2.0) (2026-11-25)


### ⚠ BREAKING CHANGES

* **array:** pipekit-array is six areas, each with one job ([#292](https://github.com/jejjohnson/pipekit/issues/292))

### Bug Fixes

* **ops:** import_optional names the extra behind a library's own error, and concrete copies CuPy arrays ([#293](https://github.com/jejjohnson/pipekit/issues/293)) ([d09db62](https://github.com/jejjohnson/pipekit/commit/d09db62524706a8c99b0ae5fb5cee3bc2e9bd3f3))


### Code Refactoring

* **array:** pipekit-array is six areas, each with one job ([#292](https://github.com/jejjohnson/pipekit/issues/292)) ([3aab7d8](https://github.com/jejjohnson/pipekit/commit/3aab7d85306ce667f96001b2e2331cacd6569d43))

## [0.1.6](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.5...pipekit-array-v0.1.6) (2026-11-25)


### Features

* **array:** check_values and require_floating also work under jit ([#287](https://github.com/jejjohnson/pipekit/issues/287)) ([1558bd5](https://github.com/jejjohnson/pipekit/commit/1558bd5639231305f4bf27abfec268502f1fa568))
* **array:** indexed updates, vmap, scan and pad run on any backend ([#286](https://github.com/jejjohnson/pipekit/issues/286)) ([2944f7c](https://github.com/jejjohnson/pipekit/commit/2944f7c3e5a7548218319aa6e7d9dac88148c6b1))
* **array:** safe roots, as_inexact, constants_like and the inputs of a reduction ([#285](https://github.com/jejjohnson/pipekit/issues/285)) ([517e7f5](https://github.com/jejjohnson/pipekit/commit/517e7f517e48fa70a04b6a1c425671b34d1073ce))
* **array:** shared test checks for backends, transforms and gradients ([#288](https://github.com/jejjohnson/pipekit/issues/288)) ([b612af3](https://github.com/jejjohnson/pipekit/commit/b612af37ef7902928456bc8468a32ece9b00cb16))
* **ops:** is_traced, is_concrete and import_optional are public ([#284](https://github.com/jejjohnson/pipekit/issues/284)) ([4ba4eba](https://github.com/jejjohnson/pipekit/commit/4ba4eba44d4e4d1677e9633fbd8e5e60f5de6a69))

## [0.1.5](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.4...pipekit-array-v0.1.5) (2026-11-24)


### Features

* **array:** masked reductions, static groups and records for plain functions ([#269](https://github.com/jejjohnson/pipekit/issues/269)) ([5fa2894](https://github.com/jejjohnson/pipekit/commit/5fa2894c1c0d4968841ecb03a634ee09eba60000))

## [0.1.4](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.3...pipekit-array-v0.1.4) (2026-11-24)


### Features

* **array:** the dim functions, KernelOp and elementwise join NumPy to JAX ([#262](https://github.com/jejjohnson/pipekit/issues/262)) ([cf5d1d1](https://github.com/jejjohnson/pipekit/commit/cf5d1d11e079b6eec382a4b34b1fb6e4004ad56e))
* **ops:** a NumPy-only operator takes JAX data and returns JAX ([#264](https://github.com/jejjohnson/pipekit/issues/264)) ([05812b7](https://github.com/jejjohnson/pipekit/commit/05812b7dc6cf9bae72c5ae6f8b5d9128ea895b48))
* **ops:** add guard, an opt-in report of data moving between NumPy and JAX ([#261](https://github.com/jejjohnson/pipekit/issues/261)) ([28ded27](https://github.com/jejjohnson/pipekit/commit/28ded27cc7a648a6c13a33858fc1fd9358025df1))


### Bug Fixes

* **array:** a scan stack of a DimArray has the tree structure of one DimArray ([#266](https://github.com/jejjohnson/pipekit/issues/266)) ([82dae66](https://github.com/jejjohnson/pipekit/commit/82dae667ef86c4b94a80d400a41556520ea324a0))
* **array:** elementwise keeps a carrier's missing elements missing ([#268](https://github.com/jejjohnson/pipekit/issues/268)) ([2543eef](https://github.com/jejjohnson/pipekit/commit/2543eef7b1287d56cb8a69b500674291073b4197))


### Performance Improvements

* **array:** map_over runs one vmap on JAX data ([#267](https://github.com/jejjohnson/pipekit/issues/267)) ([399a67d](https://github.com/jejjohnson/pipekit/commit/399a67d4fb2e9ea0e66e55d595fbf3a31120fc87))

## [0.1.3](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.2...pipekit-array-v0.1.3) (2026-11-23)


### Features

* **array:** elementwise takes labelled arrays in variadic arguments ([#251](https://github.com/jejjohnson/pipekit/issues/251)) ([acbdb96](https://github.com/jejjohnson/pipekit/commit/acbdb961f9ab53dce45e7e17b344428a3917b899))

## [0.1.2](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.1...pipekit-array-v0.1.2) (2026-11-23)


### Features

* **array:** plain functions get one array backend, pipekit_array.functions ([#248](https://github.com/jejjohnson/pipekit/issues/248)) ([68532bd](https://github.com/jejjohnson/pipekit/commit/68532bde543f4cdb9caba08528c0668e76e59a1a)), closes [#245](https://github.com/jejjohnson/pipekit/issues/245)


### Bug Fixes

* **array:** every KernelOp output has all the loop dims, and on DataArrays no name or attributes ([#250](https://github.com/jejjohnson/pipekit/issues/250)) ([b07494d](https://github.com/jejjohnson/pipekit/commit/b07494d3703d373e9519f3a27888bc669d4480ab)), closes [#246](https://github.com/jejjohnson/pipekit/issues/246)

## [0.1.1](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.1.0...pipekit-array-v0.1.1) (2026-11-07)


### Features

* **pipekit-array:** dim-level functions phase 2, pad, roll, shift, diff, cumreduce, where ([#205](https://github.com/jejjohnson/pipekit/issues/205)) ([e5f6e36](https://github.com/jejjohnson/pipekit/commit/e5f6e36ab61e054961037f68c73ba99b003f9c63))
* **pipekit-array:** dim-level functions phase 4, quantile and dot ([#207](https://github.com/jejjohnson/pipekit/issues/207)) ([8742e49](https://github.com/jejjohnson/pipekit/commit/8742e4987484a0ac0cb02f5af1a5fe5271b22605))
* **pipekit-array:** dim-level functions, phase 1 (dims and relabel) ([#204](https://github.com/jejjohnson/pipekit/issues/204)) ([8f5eb32](https://github.com/jejjohnson/pipekit/commit/8f5eb3205e8185baaf74c3013c79cf6c02f30b15))
* **pipekit-array:** dim-level functions, phase 3 (windows) ([#206](https://github.com/jejjohnson/pipekit/issues/206)) ([18b18f8](https://github.com/jejjohnson/pipekit/commit/18b18f8efb7624743b5668b7ecdce003648bbe94))

## [0.1.0](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.9...pipekit-array-v0.1.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* `from pipekit import X` no longer works; use `from pipekit_ops import X`. The pipekit distribution now holds no code and installs the whole workspace, JAX included, so libraries should depend on pipekit-ops. pipeline_hash changes for every pipeline, because the hash covers each operator's module.

### Features

* pipekit-ops is the core package, pipekit becomes the umbrella ([#199](https://github.com/jejjohnson/pipekit/issues/199)) ([903b0c3](https://github.com/jejjohnson/pipekit/commit/903b0c38202173418cd7594ba42fe3c622876574))

## [0.0.9](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.8...pipekit-array-v0.0.9) (2026-10-30)


### Features

* **array:** nan_policy="mask" on Estimator, mask_features then mask_samples ([#136](https://github.com/jejjohnson/pipekit/issues/136)) ([9554709](https://github.com/jejjohnson/pipekit/commit/9554709f0c121e4464d901472087342c82bc642a))

## [0.0.8](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.7...pipekit-array-v0.0.8) (2026-10-29)


### Features

* **array:** one scikit-learn Estimator on any named carrier ([#133](https://github.com/jejjohnson/pipekit/issues/133)) ([97ee31e](https://github.com/jejjohnson/pipekit/commit/97ee31e33f71e173052646d7c6013a77aefe9e50))
* **array:** pipekit_array.einx, a named-dims einx bridge on any carrier ([#134](https://github.com/jejjohnson/pipekit/issues/134)) ([e6ee61d](https://github.com/jejjohnson/pipekit/commit/e6ee61d5b9d0a295e494d5a6e68782349999f2a8))
* **array:** to_rows and from_rows, carriers to sample tables and back by dim name ([#132](https://github.com/jejjohnson/pipekit/issues/132)) ([3fca1dc](https://github.com/jejjohnson/pipekit/commit/3fca1dc4a389273cd4b96f9fb23ceb37ed6dc95b))

## [0.0.7](https://github.com/jejjohnson/pipekit/compare/pipekit-array-v0.0.6...pipekit-array-v0.0.7) (2026-10-27)


### Features

* **pipekit:** operators declare the array backends they run on ([#126](https://github.com/jejjohnson/pipekit/issues/126)) ([040e7a0](https://github.com/jejjohnson/pipekit/commit/040e7a01dcd6889b5a51dfd0b19250af1f539da8))

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
