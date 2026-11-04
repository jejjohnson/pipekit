# Changelog

## [0.0.4](https://github.com/jejjohnson/pipekit/compare/pipekit-cycle-v0.0.3...pipekit-cycle-v0.0.4) (2026-11-04)


### Features

* **cycle:** add obs_err_source for a per-step observation-error covariance ([#185](https://github.com/jejjohnson/pipekit/issues/185)) ([2c78c79](https://github.com/jejjohnson/pipekit/commit/2c78c79e0386145fc26c65807f7721e205386d2e)), closes [#164](https://github.com/jejjohnson/pipekit/issues/164)


### Bug Fixes

* **cycle:** accept a stateful Sequential as a cycle step ([#183](https://github.com/jejjohnson/pipekit/issues/183)) ([8b1620f](https://github.com/jejjohnson/pipekit/commit/8b1620f1b80548888e1f0d65c6ac5d9ab6ceaa46)), closes [#162](https://github.com/jejjohnson/pipekit/issues/162)
* **cycle:** let a resumed DA cycle read the next observations ([#184](https://github.com/jejjohnson/pipekit/issues/184)) ([7778186](https://github.com/jejjohnson/pipekit/commit/777818634caed05bd9598e4ace72dd6634c70244)), closes [#163](https://github.com/jejjohnson/pipekit/issues/163)

## [0.0.3](https://github.com/jejjohnson/pipekit/compare/pipekit-cycle-v0.0.2...pipekit-cycle-v0.0.3) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-cycle-v0.0.1...pipekit-cycle-v0.0.2) (2026-07-11)


### Features

* **adjoints:** shared adjoint-strategy specs and diffrax adapters ([#46](https://github.com/jejjohnson/pipekit/issues/46)) ([c88c6f9](https://github.com/jejjohnson/pipekit/commit/c88c6f97c18da1bce7735cd4ee322bcba2828591))
* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
* cross-library protocols, shared hashing utils, docs + CI hardening ([#43](https://github.com/jejjohnson/pipekit/issues/43)) ([6405dcf](https://github.com/jejjohnson/pipekit/commit/6405dcfc5ca6d7e3ec13b843b2a9211f86f360e8))
* **cycle:** add ReducedBasis protocol + BFNCycle (adjoint-free DA) ([#41](https://github.com/jejjohnson/pipekit/issues/41)) ([de68dba](https://github.com/jejjohnson/pipekit/commit/de68dbaf2f06ab87ccc115fcb489ce2d8b94aace))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## 0.0.1 (2026-05-22)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
