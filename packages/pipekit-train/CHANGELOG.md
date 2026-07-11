# Changelog

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
