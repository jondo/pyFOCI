# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added parallel candidate scoring through the new `FOCISelector` parameter `n_jobs`.
  `None` and `1` score sequentially, while `-1` uses all available processors.
- Added a local benchmark, runnable from source checkout or installed package.

### Changed
- With an integer `random_state`, FOCI now uses deterministic per-candidate random
  streams, yielding identical selections across sequential and parallel worker counts.

## [0.5.0]

### Added
- Made the target rank tie handling configurable. For this, the new keyword-only `FOCISelector` parameter
  `rank_method` is introduced, with default value `"max"` for using the maximum rank as before,
  and value `"average"` for using the average rank instead. With the maximum rank, the
  implemented conditional dependence coefficient estimator is proven to be consistent in the case of a
  continuous target. The average rank however empirically makes the estimator work better in case of
  tied target values, see the provided example.
- Added the estimator formulas to the user guide.

## [0.4.0]

### Added
- Introduced an alternative method to deal with nearest neighbor ties:
  Instead of selecting one of the tied neighbors randomly and then using its target rank,
  one can now switch to deterministically using the mean target rank of all tied neighbors instead.
  For this, the new keyword-only `FOCISelector` parameter `nn_tie_breaking` is introduced,
  with default value `"random"` for the usual behavior, and value `"mean"` for the
  new deterministic tie breaking.

### Changed
- The `FOCISelector` parameters `standardize`, `nn_strategy`, `nn_tie_breaking`, and `random_state`
  are now keyword-only.

## [0.3.2]

### Added
- The documentation now contains an example comparing FOCI and Lasso on a small non-linear real-world dataset.

### Changed
- Loosened the restriction to scikit-learn 1.8.

## [0.3.1] - 2026-07-03

### Added
- The documentation now contains an example comparing FOCI with some scikit-learn feature selectors
  on a small artificial redundant nonlinear dataset.

### Changed
- The default nearest neighbors strategy `nn_strategy="grouping"` is now
  faster than quadratic in the number of unique rows.
- Removed the restriction to Python < 3.13.

## [0.3.0] - 2026-06-26

### Added
- Introduced an alternative nearest neighbors selection algorithm, similar to the R reference implementation.
  It is exposed with the new `FOCISelector` parameter `nn_strategy="grouping"`.
  The original algorithm remains available with `nn_strategy="radius"`.

## Changed
- Made `nn_strategy="grouping"` the new default, because it is faster.

## [0.2.3] - 2026-06-23

### Changed
- Updated Action for creating GitHub releases.

## [0.2.2] - 2026-06-17

### Added
- There is now a changelog, which is also used for the GitHub releases.

## [0.2.1] - 2026-06-17

### Changed
- Input features are now N(0,1)-normalized by default.
  This can be switched off with the new parameter `standardize` by setting it to `None`.

## [0.2.0] - 2026-06-16

### Added
- FOCI feature selection with Fuchs Tn formula and radius_neighbors tie breaking.

## [0.1.2] - 2026-05-22

### Added
- Dummy release to test PyPI publishing.

