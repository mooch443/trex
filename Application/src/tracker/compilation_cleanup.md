# TRex Compilation Cleanup

## Summary
- Goal: clean the tracker compilation graph for future modules without introducing modules yet.
- Broad semantic targets now defined in CMake: `trex_core`, `trex_data`, `trex_tracking`, `trex_ml`, `trex_ui`.
- Compatibility targets remain in place: `tracker_misc`, `tracker_tracking`, `tracker_ml`, `tracker_gui`.
- `tracker_python` remains the backend/runtime carrier and keeps the Windows delay-load seam.
- macOS now has a diagnostic shared-library mode for internal libs via `TREX_ENABLE_SHARED_INTERNAL_LIBS`.
- Physical layout now matches the semantic packages:
  - `core/`
  - `data/`
  - `tracking/`
  - `ml/`
  - `ui/`
  - `python/`

## Current Ownership
- `trex.core`
  - Foundational `core/*` implementation and the bridge headers `core/PythonWrapper.h` and `core/Network.h`.
- `trex.data`
  - Shared tracker data/model files under `data/` such as `MotionRecord`, `IndividualCache`, and related cache/data types.
- `trex.tracking`
  - Core tracking algorithms and tracker orchestration under `tracking/`, including `CacheHints`, `CategorizeDatastore`, `FilterCache`, `Stuffs`, `TrackletInformation`, `TrainingData`, `Results`, and related tracker-state helpers.
- `trex.ml`
  - `VisualIdentification`, `ClosedLoop`, and `UniquenessProvider` under `ml/`.
- `trex.ui`
  - All tracker UI sources under `ui/`, plus the GUI-heavy ML files `Accumulation`, `Categorize`, and `CategorizeInterface`.
- `tracker_python`
  - Concrete Python backend sources under `python/`, including `PythonWrapper.cpp` and `Network.cpp`.

## Completed
- Added explicit tracker source manifests in `TrackerSources.cmake`.
- Replaced the folder-owned tracker libraries with semantic targets and compatibility wrapper targets.
- Kept `tracker_python` as the implementation backend instead of folding it into the semantic graph.
- Moved tracker-owned headers and sources from the legacy `misc/` and `gui/` buckets into the package-aligned `core/`, `data/`, and `ui/` directories, and moved `VisualIdentification` into `ml/`.
- Moved `FilterCache` into `tracking/` so the filter/cache helper follows the tracker-state ownership instead of the data package.
- Moved `TrainingData` into `tracking/` so the training dataset container follows tracker ownership instead of the ml package.
- Moved `CategorizeDatastore` into `tracking/` so categorization state/persistence follows tracker ownership instead of the ml package.
- Rewrote tracker and test include paths to the package prefixes (`core/`, `data/`, `tracking/`, `ml/`, `ui/`).
- Added an explicit tracker-root public include directory to the semantic targets so the package-prefixed includes resolve consistently.
- Removed the unreferenced `tracker/legacy` executable subtree and reduced `src/grabber` to the `framegrabber_misc` library surface only.
- Removed the dead top-level `BUILD_LEGACY_TGRABS` option after deleting the legacy TGrabs/grabber targets and sources.
- Added the top-level option `TREX_ENABLE_SHARED_INTERNAL_LIBS` and wired `commons`, `trex_core`, `trex_data`, `trex_tracking`, `trex_ml`, `trex_ui`, and `tracker_python` to build as shared libraries on macOS when it is enabled.
- Replaced the partial tracker-only shared-library hook with one internal-library-type policy for the tracker semantic targets.
- Relaxed symbol visibility for the internal shared-library diagnostic mode on macOS so linker failures reflect ownership/link issues instead of hidden-symbol policy.
- Narrowed the public manifests so GUI-heavy or Python-coupled tracker headers stay private unless they already fit the intended future surface.
- Moved `Segmenter` into the `ui/` package because it is the orchestration layer for export, backend selection, and GUI-triggered detection.
- Removed the `Segmenter.h -> tracking/OverlayedVideo.h` public include edge and updated `ui/ConvertScene.cpp` to include `tracking/OverlayedVideo.h` directly where it needs `BasicProcessor` and `AbstractBaseVideoSource`.
- Removed the `Outline.h -> gui/Transform.h` public include edge and updated the translation units that construct `gui::Transform` values to include `gui/Transform.h` directly.
- Reduced the `TrainingData.cpp` and `DatasetQuality.cpp` dependency on `VisualIdentification` by extracting `can_use_visual_identification(...)` into `tracking/Stuffs.*`.
- Reduced `PythonWrapper.h` coupling by forward-declaring `Python::Network`.
- Kept `PythonWrapper.cpp` in the `python/` backend and wired it into `tracker_python` after the physical move.
- Moved `TileBuffers` from `python/` into `core/` because `TileImage` and `PrecomputedDetection` already depend on it from `trex.core`.
- Extracted `find_output_name`, `infer_cm_per_pixel`, and `infer_meta_real_width_from` into `core/SettingsPaths.*` so lower layers no longer depend on `ui/SettingsInitializer` for settings/path helpers.
- Split settings initialization so the reusable load/reset/default logic now lives in `core/SettingsInitializer.*`, while `ui/SettingsInitializer.*` is only the queue-aware wrapper for GUI call sites.
- Extracted `CacheHints` into its own tracker header/source pair so it is no longer bundled with `MotionRecord`.
- Added `audit_tracker_layers.py` to check semantic-layer include edges from the manifest rather than folder names.
- Added `report_tracker_dependencies.py` to report per-file cross-package include edges from `TrackerSources.cmake`, with optional file-level detail.
- Added `core/BackgroundTask.*` as the non-UI queue bridge so lower layers can schedule background work without depending on `ui/WorkProgress`.
- Added `ml/AccumulationRuntime.*` as the non-UI accumulation bridge so `trex.ml` can consume accumulation runtime services without depending on `ui/Accumulation`.
- Restored `interactive_segmentation_prototype/main.cpp` by making its `cmn` namespace dependency explicit locally instead of relying on ambient namespace leakage.

## Remaining TODOs
- Verify both static-default and macOS-shared builds compile cleanly.
- Use the macOS shared-library build to identify and fix remaining missing link dependencies or accidental cross-package references.
- Trim additional public headers from `trex.tracking` and `trex.ui` where they are not needed outside the owning package.
- The core-oriented settings path helpers now live in `core/SettingsPaths.*`; `ui/SettingsInitializer` remains the GUI-facing workflow wrapper.
- Apply the same core-first / thin-UI-wrapper rule to future settings-related helpers we touch, instead of growing new UI-owned logic by default.
- `CacheHints` is now standalone, but it still shares the tracker settings cache pattern with `MotionRecord`; that duplication is intentional for now to keep the split small.
- Decide whether `tracker_python` should later gain one optional thin bridge target if a linker-specific issue appears on Windows.
- Decide whether `gui/DrawFish.cpp.save` and other non-build legacy artifacts should be moved or deleted now that the semantic package layout is in place.

## Shared Build Blockers
- None recorded yet from this phase because I have not run the macOS shared build locally.
- When `TREX_ENABLE_SHARED_INTERNAL_LIBS=ON`, treat every undefined symbol or missing dependency as a package-boundary issue to fix explicitly rather than by broadening link interfaces.
- Current known semantic hotspot after the latest extraction: `tracking/Tracker.cpp` still includes `ui/Accumulation.h` and therefore has a real tracking-to-UI edge unrelated to settings.
- `trex.ml -> trex.ui` has been cut at the source level by routing `VisualIdentification` and `UniquenessProvider` through `ml/AccumulationRuntime.*`.
- `tracker_python -> trex_ui` has been cut at the source/link level by removing the stray `ui/WorkProgress.h` include from `GPURecognition.cpp` and dropping the `trex_ui` link from `tracker_python`.
- `TrainingData` now lives in `tracking/`, which matches its dependency on tracker-state types and keeps `trex.ml` from owning tracker-payload code.
- `CategorizeDatastore` now lives in `tracking/`, which keeps the categorization datastore and persistence helpers with tracker-owned state instead of `trex.ml`.
- The current shared-link failure batch still reflects the pre-move `data/FilterCache` ownership in Xcode; regenerate the project after the manifest update before treating those symbols as a fresh dependency report.
- The clean Ninja configure path in `Application/tmp` currently needs either network access for `FetchContent` or pre-populated dependencies; the failure seen here is a CMake bootstrap issue, not a tracker source regression.

## Commands
- Audit the public header graph:
  - `conda run -n trex python Application/src/tracker/audit_tracker_layers.py`
- Report tracker file-to-package dependencies:
  - `conda run -n trex python Application/src/tracker/report_tracker_dependencies.py`
  - `conda run -n trex python Application/src/tracker/report_tracker_dependencies.py --package trex.ui --files`
  - `conda run -n trex python Application/src/tracker/report_tracker_dependencies.py --glob 'tracking/*' --files`

## Assumptions / Notes
- This phase now covers both target ownership cleanup and physical relocation into package-aligned directories.
- Internal `.cpp` dependencies can remain broader for now as long as the semantic/public graph stays acyclic.
- `TREX_ENABLE_SHARED_INTERNAL_LIBS` is diagnostic-only for now and is intended to surface ownership/linking mistakes on macOS, not to commit TRex to shipping shared internals everywhere.
- Translation units that touch `BasicProcessor` or `AbstractBaseVideoSource` through `Segmenter::overlayed_video()` now need to include `tracking/OverlayedVideo.h` explicitly instead of relying on `Segmenter.h` to pull it in.
- Accumulation teardown is now owned by the app/UI shutdown path instead of `Tracker::~Tracker()`.
- Translation units that instantiate or return `gui::Transform` values from tracker APIs now need to include `gui/Transform.h` explicitly instead of relying on `Outline.h` or `FilterCache.h` to pull it in.
- I have not run `cmake` or any build commands in this repository.
- I have not completed a full local build because the fresh `Application/tmp` Ninja configure tries to fetch `portable-file-dialogs` from GitHub and this environment cannot resolve that host.
