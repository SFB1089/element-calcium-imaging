# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.


## [0.1.0b2] - unreleased
### Added
+ `imaging.RoiQCCriterion` (Lookup) and `imaging.RoiQC` (+ `.Rejected`): ROIs rejected per QC criterion and curation; first criterion `empty_f` = fluorescence 0 (or non-finite) in every frame (suite2p: every pixel overlaps another ROI). Exclude `RoiQC.Rejected` in queries
+ `dff.empty_rois(F)`, `dff.roi_qc_reject(criterion, F_rows)`
### Changed
+ `dff.compute_dff0` stores NaN dF/F0 for empty ROIs (and returns `f_empty`); all other rows unchanged
+ CASCADE in `Activity.make` leaves rows without a finite dF/F0 NaN instead of passing them to the model


## [0.1.0b1] - 2026-11-15
### Added
+ `dff.py`: one pure dF/F0 implementation, one recipe per parameter set (dark signal, neuropil factor/offset, detrending, baseline, F0 floor), validated and hashed
+ `imaging.DffParamSet`, `imaging.Dff0Task`, `imaging.Dff0` (+ `.Trace`): dF/F0 stored once per curation x recipe
+ `imaging.CascadeRecipe`: the dF/F0 recipe of each CASCADE extraction method
+ `imaging.Activity.CascadeInput`: the exact `Dff0`, model and code versions behind each new CASCADE activity
### Changed
+ CASCADE in `Activity.make` reads `Dff0.Trace` instead of computing dF/F0 from rules on the method name; existing methods keep the recipe they used since 2026-09-03, so their spikes are unchanged
+ `calculate_mean_darksignal` is deprecated and delegates to `dff.resolve_dark` (Decimal cast and argument order fixed)


## [0.1.0b0] - 2021-05-07
### Added
+ First beta release


## [0.1.0a4] - 2021-05-07
### Fixed 
+ Add workaround to handle DataJoint 0.13.* [issue #914](https://github.com/datajoint/datajoint-python/issues/914)


## [0.1.0a3] - 2021-05-03
### Added 
+ Added GitHub Action release process
+ `scan` and `imaging` elements
+ Readers for: `ScanImage`, `ScanBox`, `Suite2p`, `CaImAn`


[0.1.0b0]: https://github.com/datajoint/element-calcium-imaging/compare/0.1.0a4...0.1.0b0
[0.1.0a4]: https://github.com/datajoint/element-calcium-imaging/releases/tag/0.1.0a4
[0.1.0a3]: https://github.com/datajoint/element-calcium-imaging/releases/tag/0.1.0a3
