# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-07-03

### Added
- The documentation now contains an example comparing FOCI with some Scikit-Learn Feature selectors
  on a small artificial redundant nonlinear dataset.

### Changed
- The default Nearest Neighbors strategy `nn_strategy="grouping"` is now
  faster than quadratic in the number of unique rows.

## [0.3.0] - 2026-06-26

### Added
- Introduced an alternative nearest neighbors selection algorithm, similar to the R reference implementation.
  It is exposed with the new keyword-only `FOCISelector` parameter `nn_strategy="grouping"`.
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

