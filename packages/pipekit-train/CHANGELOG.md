# Changelog

## [0.1.5](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.1.4...pipekit-train-v0.1.5) (2026-11-06)


### Features

* **train:** add a sample-based fallback to the KL loss ([#195](https://github.com/jejjohnson/pipekit/issues/195)) ([1c6fbe2](https://github.com/jejjohnson/pipekit/commit/1c6fbe2ede2b41a9242ce8f00a4e3c1a9e2b8492))
* **train:** add opt-in padded evaluation through TrainTask.eval_fn ([#192](https://github.com/jejjohnson/pipekit/issues/192)) ([c506375](https://github.com/jejjohnson/pipekit/commit/c5063757a0ee97a4e5156e64eb6df296deaca746)), closes [#125](https://github.com/jejjohnson/pipekit/issues/125)
* **train:** add SimulationDataset(invert=True) and an amortized recipe ([#196](https://github.com/jejjohnson/pipekit/issues/196)) ([9c29999](https://github.com/jejjohnson/pipekit/commit/9c29999e6803bbd121700ac4ff114aa7037eb36d))
* **train:** support unsupervised NumPyro fits and full posterior draws ([#194](https://github.com/jejjohnson/pipekit/issues/194)) ([ff6160b](https://github.com/jejjohnson/pipekit/commit/ff6160b473f20486b18288c1b946ace5679af1f7))

## [0.1.4](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.1.3...pipekit-train-v0.1.4) (2026-11-04)


### Features

* **train:** jit the Equinox eval step ([#173](https://github.com/jejjohnson/pipekit/issues/173)) ([e902ca9](https://github.com/jejjohnson/pipekit/commit/e902ca975e8ce487f3886c1472ebe2258fd2fbc6)), closes [#122](https://github.com/jejjohnson/pipekit/issues/122)


### Bug Fixes

* **train:** compile the numpyro-svi step ([#170](https://github.com/jejjohnson/pipekit/issues/170)) ([5b6e497](https://github.com/jejjohnson/pipekit/commit/5b6e49742efed2def704bd26248faf897d7364c8)), closes [#160](https://github.com/jejjohnson/pipekit/issues/160)
* **train:** hash the simulator, so existing SimulationDataset caches miss once ([#176](https://github.com/jejjohnson/pipekit/issues/176)) ([8363a5f](https://github.com/jejjohnson/pipekit/commit/8363a5f562dd3df4749f53c89329601be594317a)), closes [#155](https://github.com/jejjohnson/pipekit/issues/155)
* **train:** honour Checkpoint.save_dir and save the final step ([#179](https://github.com/jejjohnson/pipekit/issues/179)) ([56b7d8b](https://github.com/jejjohnson/pipekit/commit/56b7d8b3b5dd3bcb58a64aec790871ccea2f7382)), closes [#158](https://github.com/jejjohnson/pipekit/issues/158)
* **train:** let a trained Equinox model round-trip through the registry ([#181](https://github.com/jejjohnson/pipekit/issues/181)) ([197d65e](https://github.com/jejjohnson/pipekit/commit/197d65e966b5485652f2ddce6024d5a09edcb572))
* **train:** make SimulationDataset work with a real cycle ([#175](https://github.com/jejjohnson/pipekit/issues/175)) ([90b8d44](https://github.com/jejjohnson/pipekit/commit/90b8d44633eb861c4454f60cb861c3bbc0214950))
* **train:** rank HyperSweep trials on validation metrics ([#182](https://github.com/jejjohnson/pipekit/issues/182)) ([507b2aa](https://github.com/jejjohnson/pipekit/commit/507b2aa784f300fac9780ec155e24ede64b3f3ef)), closes [#153](https://github.com/jejjohnson/pipekit/issues/153)
* **train:** stop CachedDataset reopening its zarr store on every item ([#142](https://github.com/jejjohnson/pipekit/issues/142)) ([7c446f1](https://github.com/jejjohnson/pipekit/commit/7c446f141c23164918e2e8b708428b860206120c)), closes [#141](https://github.com/jejjohnson/pipekit/issues/141)
* **train:** stop SimulationDataset swallowing a TypeError from the prior ([#174](https://github.com/jejjohnson/pipekit/issues/174)) ([94abffd](https://github.com/jejjohnson/pipekit/commit/94abffd0d370fd7a0627e9917867373d37e2e557)), closes [#156](https://github.com/jejjohnson/pipekit/issues/156)

## [0.1.3](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.1.2...pipekit-train-v0.1.3) (2026-10-27)


### Bug Fixes

* **train:** weight eval metrics by batch size, not per batch ([#123](https://github.com/jejjohnson/pipekit/issues/123)) ([3d95a09](https://github.com/jejjohnson/pipekit/commit/3d95a09b19f17c2620a77d9a6c0b9ff40a469543))

## [0.1.2](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.1.1...pipekit-train-v0.1.2) (2026-10-24)


### Bug Fixes

* **experiment:** verify the content hash when loading from a registry ([#93](https://github.com/jejjohnson/pipekit/issues/93)) ([3374a57](https://github.com/jejjohnson/pipekit/commit/3374a570cfb7b3a72a21bb71f73647ac09ea9d0e))

## [0.1.1](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.1.0...pipekit-train-v0.1.1) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.1.0](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.0.2...pipekit-train-v0.1.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* **train:** `from pipekit_train import XarrayWindowDataset` and `pipekit_train.xarray_window` are gone; use `from geopatcher.integrations.train import XarrayWindowDataset` with `geotoolz-patcher[train]`. The `pipekit-train[xreader]` extra is removed.

### Code Refactoring

* **train:** move XarrayWindowDataset to geopatcher.integrations.train ([#57](https://github.com/jejjohnson/pipekit/issues/57)) ([96c8c6b](https://github.com/jejjohnson/pipekit/commit/96c8c6b9436f12ba9e3d2d2dc92a51afcb2512a1))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-train-v0.0.1...pipekit-train-v0.0.2) (2026-07-11)


### Features

* cross-library protocols, shared hashing utils, docs + CI hardening ([#43](https://github.com/jejjohnson/pipekit/issues/43)) ([6405dcf](https://github.com/jejjohnson/pipekit/commit/6405dcfc5ca6d7e3ec13b843b2a9211f86f360e8))
* **pipekit-jax:** new workspace package — JaxModelOp with weight round-trip (closes [#19](https://github.com/jejjohnson/pipekit/issues/19)) ([#23](https://github.com/jejjohnson/pipekit/issues/23)) ([b18022b](https://github.com/jejjohnson/pipekit/commit/b18022b56ed6f479dde105b04fed8452a84b298b))
* **pipekit-train:** fsspec CachedDataset + Grain iterator-state checkpointing (closes [#17](https://github.com/jejjohnson/pipekit/issues/17), [#18](https://github.com/jejjohnson/pipekit/issues/18)) ([#24](https://github.com/jejjohnson/pipekit/issues/24)) ([c5922da](https://github.com/jejjohnson/pipekit/commit/c5922da1d5277a898913b4da98c423c1188711a9))
* **pipekit-train:** scaffold package + multi-file design doc ([#10](https://github.com/jejjohnson/pipekit/issues/10)) ([9c974d7](https://github.com/jejjohnson/pipekit/commit/9c974d712fe1a85709d07613c07e1988d1f000f7))
* **pipekit-train:** sweep.py — HyperSweep + ParameterGrid ([#22](https://github.com/jejjohnson/pipekit/issues/22)) ([e108e9a](https://github.com/jejjohnson/pipekit/commit/e108e9ad35409863e10684b944fe4a37dda728a6))
* **pipekit-train:** v0.1 — protocols, losses, callbacks, loop, Equinox adapter ([#11](https://github.com/jejjohnson/pipekit/issues/11)) ([3875835](https://github.com/jejjohnson/pipekit/commit/38758353b8acbb3705df45203af50af0b0c0ecbd))
* **pipekit-train:** v0.1 polish — integration tests, API docs, notebooks ([#12](https://github.com/jejjohnson/pipekit/issues/12)) ([24b204e](https://github.com/jejjohnson/pipekit/commit/24b204eea846cc495dc2d2a89a142c7eb5ef0219))
* **train:** add lazy, indexable windowed-Zarr dataset (XarrayWindowDataset) ([#34](https://github.com/jejjohnson/pipekit/issues/34)) ([12f7f66](https://github.com/jejjohnson/pipekit/commit/12f7f66da429a21da4e88f8540fca9053e561f66)), closes [#33](https://github.com/jejjohnson/pipekit/issues/33)
* **train:** implement the BlackJAX backend (NUTS) ([#38](https://github.com/jejjohnson/pipekit/issues/38)) ([7670c3a](https://github.com/jejjohnson/pipekit/commit/7670c3a06ea37d5e905a67bfbbd718078c00b2f8))
* **train:** implement the NumPyro backends (numpyro-svi + numpyro-mcmc) ([#37](https://github.com/jejjohnson/pipekit/issues/37)) ([5008de7](https://github.com/jejjohnson/pipekit/commit/5008de7423334aaf3e9516fae079ef179a85e624))
* **train:** multi-device / multi-host sharding for the Equinox adapter ([#32](https://github.com/jejjohnson/pipekit/issues/32)) ([e0336de](https://github.com/jejjohnson/pipekit/commit/e0336defd1bb55de5ac1363cae3d6773b486a4bf))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## Changelog
