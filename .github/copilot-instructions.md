# TRex Repository Instructions

TRex is a C++ application with a standalone `commons` library for shared
utilities and GUI infrastructure. Keep changes scoped to the task and follow
the existing local patterns before introducing new abstractions.

## Repository Layout

- `Application/` is the main CMake project.
- `Application/src/commons/` is the shared commons library.
- `Application/src/tracker/` is the TRex GUI application.
- `Application/src/ProcessedVideo/` handles PV format code.
- `Application/Tests/` contains gtest/gmock unit tests.
- `docs/` contains Sphinx documentation.
- `Application/src/grabber/` is deprecated; do not use it as a reference for
  new GUI structure or best practices.

## C++ Conventions

- Prefer existing project helpers, types, and ownership patterns.
- Most source files include the precompiled header `commons.pc.h`; when adding
  a new implementation file, put `#include <commons.pc.h>` before other
  includes unless a nearby file shows a different established pattern.
- Keep edits minimal and avoid unrelated refactors, formatting churn, or broad
  metadata changes.
- Add comments only when they clarify non-obvious behavior.
- Use structured parsing or existing helper APIs rather than ad hoc string
  manipulation when the codebase already provides a suitable option.

## GUI Guidance

- Prefer the scene system for main TRex UI flows. Implement `gui::Scene`
  objects that own state, draw UI, and respond to global events.
- Use DynamicGUI for complex JSON-driven layouts and rapid iteration. Define
  variables with `dyn::VarFunc`, actions with `dyn::ActionFunc`, and load
  layouts via `file::DataLocation::parse(...)`.
- Keep GUI work on the main thread. Use `SceneManager::getInstance().gui_task_queue()`
  or the scene task queue for deferred work and long-running side effects.
- Be careful with coordinate spaces. TRex UI commonly converts between
  `HUDCoord`, `BowlCoord`, `HUDRect`, and `BowlRect` through `FindCoord`.
  Convert sizes, offsets, and hit-test bounds into the same coordinate space
  before comparing or clamping them.
- Reserve direct draw calls for small custom widgets and localized overlays.

## Tests And Validation

- When fixing a bug, first reproduce the failure with a minimal viable test or
  local repro when practical, then fix the code until that repro passes.
- Add targeted regression tests only when they materially validate the reported
  issue. Avoid broad or speculative tests.
- If modifying tests, keep changes limited to tests and the smallest necessary
  test wiring such as `Application/Tests/CMakeLists.txt`.
- Do not run commands in existing build directories or delete project files in
  build directories.

## Build Notes

- Use the repository CMake project under `Application/` for normal builds.
- For conda builds, prefer the recipe in `conda/`; GitHub Actions uses the same
  conda build flow.
- Use only the `trex` or `trex-modules` conda environments for
  environment-specific commands. For Python in the `trex` environment, prefer
  `KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python ...` on macOS.
