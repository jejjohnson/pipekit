# Changelog

## [0.1.4](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.1.3...pipekit-v0.1.4) (2026-10-27)


### Features

* **pipekit:** operators declare the array backends they run on ([#126](https://github.com/jejjohnson/pipekit/issues/126)) ([040e7a0](https://github.com/jejjohnson/pipekit/commit/040e7a01dcd6889b5a51dfd0b19250af1f539da8))

## [0.1.3](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.1.2...pipekit-v0.1.3) (2026-10-25)


### Features

* **core:** on_import runs code when a module is imported; pipekit-array registers DimArray lazily ([#117](https://github.com/jejjohnson/pipekit/issues/117)) ([3446f13](https://github.com/jejjohnson/pipekit/commit/3446f13ef2a5feb70109aaf9df1954052b04f8f2))

## [0.1.2](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.1.1...pipekit-v0.1.2) (2026-10-24)


### Features

* **array:** dim-level functions over the adapter ([#110](https://github.com/jejjohnson/pipekit/issues/110)) ([18a2dff](https://github.com/jejjohnson/pipekit/commit/18a2dfffd754db211b3f9c481698fa13ff739c72))
* **array:** the [xarray] adapter and the apply_ufunc path ([#112](https://github.com/jejjohnson/pipekit/issues/112)) ([fa0e924](https://github.com/jejjohnson/pipekit/commit/fa0e924dbe55dd05513f1b0bd14ff3e696cab78f))
* **core:** add the NamedArrayAdapter protocol and register_adapter registry ([#108](https://github.com/jejjohnson/pipekit/issues/108)) ([242ba52](https://github.com/jejjohnson/pipekit/commit/242ba52f3170e8f6346aa86325ad06e865812e65))
* **core:** Signature.matches and Signature.from_pattern ([#107](https://github.com/jejjohnson/pipekit/issues/107)) ([ea6e4e3](https://github.com/jejjohnson/pipekit/commit/ea6e4e3cd0f4cff78a1021977288655fa54ec633)), closes [#99](https://github.com/jejjohnson/pipekit/issues/99)

## [0.1.1](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.1.0...pipekit-v0.1.1) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.1.0](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.0.2...pipekit-v0.1.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* **train:** `from pipekit_train import XarrayWindowDataset` and `pipekit_train.xarray_window` are gone; use `from geopatcher.integrations.train import XarrayWindowDataset` with `geotoolz-patcher[train]`. The `pipekit-train[xreader]` extra is removed.

### Features

* **experiment:** load untrusted _target_ YAML through the allowlist ([#76](https://github.com/jejjohnson/pipekit/issues/76)) ([0fdbce3](https://github.com/jejjohnson/pipekit/commit/0fdbce374b7d95873dedb08ac615aa408ee95357))
* **graph:** thread carry state through a Graph's stateful nodes ([#83](https://github.com/jejjohnson/pipekit/issues/83)) ([e401adc](https://github.com/jejjohnson/pipekit/commit/e401adc8e414c96351f637361630412911cb544b))
* **signature:** Graph.summary, and containers report what they return ([#85](https://github.com/jejjohnson/pipekit/issues/85)) ([59cfa7d](https://github.com/jejjohnson/pipekit/commit/59cfa7da66d164a381f9d6c0ff349ad0b16f12f5))


### Bug Fixes

* **core:** AsyncMap in a running loop, ProcessMap logging, a ShapeTrace hook ([#86](https://github.com/jejjohnson/pipekit/issues/86)) ([132b53e](https://github.com/jejjohnson/pipekit/commit/132b53e51e55cc439269307f6e9d3d6c6b9ceafc))
* **graph:** record every node in Graph.state ([#73](https://github.com/jejjohnson/pipekit/issues/73)) ([d0285b4](https://github.com/jejjohnson/pipekit/commit/d0285b46dd9f5fb2bb797f1ce3a36dd49dcac0cd))
* **state:** make CarryState unhashable ([#84](https://github.com/jejjohnson/pipekit/issues/84)) ([1d93a3a](https://github.com/jejjohnson/pipekit/commit/1d93a3ad63173718df93a59e6c7a7dcdc6d331b6))


### Code Refactoring

* **train:** move XarrayWindowDataset to geopatcher.integrations.train ([#57](https://github.com/jejjohnson/pipekit/issues/57)) ([96c8c6b](https://github.com/jejjohnson/pipekit/commit/96c8c6b9436f12ba9e3d2d2dc92a51afcb2512a1))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-v0.0.1...pipekit-v0.0.2) (2026-07-11)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
* cross-library protocols, shared hashing utils, docs + CI hardening ([#43](https://github.com/jejjohnson/pipekit/issues/43)) ([6405dcf](https://github.com/jejjohnson/pipekit/commit/6405dcfc5ca6d7e3ec13b843b2a9211f86f360e8))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## 0.0.1 (2026-05-22)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
