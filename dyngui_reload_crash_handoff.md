# DynamicGUI reload crash handoff

## Scope

This investigation was for the rare crash while reloading the DynamicGUI layout used by `choose_settings_layout`.

Observed stack:

- `cmn::gui::Layout::~Layout()` at `Application/src/commons/common/gui/types/Layout.cpp:90`
- called while `cmn::gui::dyn::DynamicGUI::reload()` clears old objects
- stack passed through `derived_ptr` / `shared_ptr` destruction

## What was checked

Files inspected:

- `Application/src/commons/common/gui/DynamicGUI.cpp`
- `Application/src/commons/common/gui/ParseLayoutTypes.cpp`
- `Application/src/commons/common/gui/LabeledField.cpp`
- `Application/src/commons/common/gui/LabeledField.h`
- `Application/src/commons/common/gui/types/Layout.cpp`
- `Application/src/commons/common/gui/types/Layout.h`
- `Application/src/commons/common/gui/types/Entangled.cpp`
- `Application/src/commons/common/gui/types/Entangled.h`
- `Application/src/commons/common/gui/types/Drawable.cpp`
- `Application/src/commons/common/gui/types/Combobox.cpp`
- `Application/src/commons/common/gui/types/Dropdown.cpp`
- `Application/src/tracker/choose_settings_layout.json`

## Verified behavior

- `choose_settings_layout.json` contains a `settings` control with `\"var\": \"\"`, so the preserved settings combobox path is involved.
- `DynamicGUI::reload()` preserves `state._last_settings_box` across reloads by moving it into a temporary `State`, clearing `objects`, resetting `state`, reparsing, then moving the temp state back.
- `Drawable::~Drawable()` already calls `set_parent(nullptr)`, so the earlier stale-parent hypothesis was not sufficient by itself.

## Tests attempted

The user requirement was: create a failing test first, confirm it fails, then fix.

Two exploratory tests were added to `Application/Tests/test_dyngui.cpp`:

1. `EntangledLifetimeTest.WrappedChildDestructionClearsParentBookkeeping`
2. `DynamicGUILifetimeTest.SettingsComboboxSurvivesRepeatedReloadStyleReparenting`

Current result:

- both tests pass
- no deterministic reproducer has been found yet
- no library fix has been applied

Important: these tests are exploratory only. They do not currently prove the bug.

## Build state

Build tree used:

- `build-bug/`

Working configure/build flow in the `trex` conda environment:

```bash
eval "$(conda shell.zsh hook)"
conda activate trex
cmake -S Application -B build-bug -DTREX_WITH_TESTS=ON -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"
cmake --build build-bug --target test_dyngui -j4
```

Useful test commands:

```bash
./build-bug/test_dyngui --gtest_filter=EntangledLifetimeTest.WrappedChildDestructionClearsParentBookkeeping
./build-bug/test_dyngui --gtest_filter=DynamicGUILifetimeTest.SettingsComboboxSurvivesRepeatedReloadStyleReparenting
```

## Strongest current hypothesis

The most credible bug class found is duplicate ownership caused by constructing `derived_ptr` / `Layout::Ptr` from raw pointers that are already owned elsewhere.

`derived_ptr(T* share)` creates a brand new `shared_ptr` control block and deletes `share` in its custom deleter. If the same raw pointer is wrapped again, that creates a second independent owner and can cause double deletion.

Relevant constructor:

- `Application/src/commons/common/misc/derived_ptr.h:117`

## High-risk aliasing sites found

These stand out because they wrap raw pointers that appear to already have owners:

1. `Application/src/commons/common/gui/LabeledField.h:446`
   - `LabeledPathArray::apply_set()` returns `Layout::Ptr(_dropdown)`
   - `_dropdown` is already a `derived_ptr<CustomDropdown>`
   - this is a likely second ownership bug

2. `Application/src/commons/common/gui/ParseLayoutTypes.cpp:283`
   - `apply_modifier_to_object(name, Layout::Ptr(ptr), ...)`
   - `ptr` comes from `state.named_entity(name)` and is already shared elsewhere
   - wrapping it again as `Layout::Ptr(ptr)` is unsafe

3. `Application/src/commons/common/gui/ParseLayoutTypes.cpp:306`
   - same issue on the unhover path

There are more raw-pointer wraps in the codebase, but the three above are in the commons GUI path and are much more relevant than legacy files.

## Secondary lifetime hazard

`Layout` keeps children in two forms:

- owning `std::vector<Layout::Ptr> _objects`
- raw pointer tracking in `Entangled::_current_children`

This dual bookkeeping is fragile during teardown. `Layout::clear_children()` copies `_objects`, clears `_objects`, then delegates to `Entangled::clear_children()`. That ordering can free `shared_ptr`-owned children before `Entangled` finishes traversing raw child pointers.

This is a plausible UB source, but it was not yet reduced to a failing test.

Relevant code:

- `Application/src/commons/common/gui/types/Layout.cpp:293`
- `Application/src/commons/common/gui/types/Entangled.cpp:569`

## Recommended next steps

1. Remove or isolate the exploratory tests in `Application/Tests/test_dyngui.cpp` if they are too noisy, then add a real failing test specifically for duplicate ownership.
2. Start with the easiest deterministic repro:
   - create one `derived_ptr` object
   - construct a second `Layout::Ptr` from its raw `.get()`
   - verify this crashes under ASan / death test or triggers a custom detection path
3. Add a focused gtest for the `LabeledPathArray::apply_set()` case, because it already returns `Layout::Ptr(_dropdown)` and looks directly wrong.
4. Audit and fix the aliasing sites by sharing the existing smart pointer instead of re-wrapping the raw pointer.
5. After that, if the original crash still exists, continue with a second failing test around `Layout` / `Entangled` teardown ordering.

## Likely code direction for the fix

Do not construct new `Layout::Ptr` / `derived_ptr` instances from raw pointers that already belong to another smart pointer.

Instead:

- return the existing smart pointer directly when possible
- or add explicit conversion helpers that share the existing control block

Examples to revisit first:

- `LabeledPathArray::apply_set()`
- hover/unhover action handling in `ParseLayoutTypes.cpp`

## Work not done

- No fix has been committed.
- `AGENTS.md` was not updated yet.
- The crash is not reproduced deterministically yet.

