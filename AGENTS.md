# AGENTS.md

This repository is TRex, a C++ application that bundles a standalone `commons`
library for shared utilities and GUI infrastructure.

## Repository layout
- `Application/`: main CMake project and C++ sources.
- `Application/src/commons/`: the commons library (utilities + GUI toolkit).
- `Application/src/tracker/`: TRex GUI app (scenes, tracking, layouts).
- `Application/src/grabber/`: frame grabber tools/utilities (deprecated; do not use as a reference for GUI patterns).
- `Application/src/ProcessedVideo/`: PV format handling.
- `Application/Tests/`: gtest/gmock unit tests.
- `docs/`: Sphinx docs.
- `website/`, `images/`, `videos/`: site + assets.

Do not use whitespace cleanup tools if they produce lots of diff, never commit huge diff that is just whitespace changes.
Write inline code comments in terms of current invariants, responsibilities,
and behavior. Do not use them to narrate code history or earlier implementations.
When creating commits, match recent repository style: use bullet-style,
past-tense messages where each line starts with `* Added`, `* Updated`,
`* Modified`, or similar wording.

## Commons library overview
- Core GUI headers live under `Application/src/commons/common/gui/`.
- The dynamic GUI system is `Application/src/commons/common/gui/DynamicGUI.h`
  and `Application/src/commons/common/gui/dyn/`.
- Do not include individual C++ standard-library headers; they are provided by
  `commons.pc.h`. When it is included, `commons.pc.h` must be the first include
  in the file for consistency.
- Start with the dynamic GUI walkthrough in `Application/src/commons/README.md`.

## Creating GUIs with commons (how it should be used)
1. Pick the UI style:
   - Dynamic, JSON-driven UI for complex layouts and rapid iteration.
   - Scene-based UI for the main app flows (multiple screens, transitions).
   - Direct draw calls for small, custom widgets.
2. Window + render loop:
   - Create an `IMGUIBase` window, render via `DrawStructure`, and keep the
     GUI event loop on the main thread.
3. Dynamic GUI workflow:
   - Define a JSON layout that uses variables (`{var}`) and actions
     (`"action":"name:arg"`).
   - Build a `dyn::Context` with `VarFunc` for data exposure and `ActionFunc`
     for side effects (scene switches, settings changes, file IO).
   - Instantiate `dyn::DynamicGUI` with the layout path, context, and the
     `SceneManager` GUI queue (`SceneManager::getInstance().gui_task_queue()`).
   - Each frame: call `dynGUI.update(graph, parent)` then process queued tasks.
   - Use `file::DataLocation::parse(...)` when layouts/assets are installed
     outside the build tree.

### Dynamic GUI capabilities in this repo
- Layout files use a top-level `"objects"` array plus optional `"defaults"`.
  Most TRex layouts live beside `Application/src/tracker/tracking_layout.json`
  and are good references for production patterns.
- Built-in object types include `vlayout`, `hlayout`, `gridlayout`,
  `collection`, `button`, `textfield`, `checkbox`, `settings`, `combobox`,
  `list`, `each`, `condition`, `text`, `stext`, `rect`, `circle`, `line`, and
  `image`. Some layouts also use custom registered module/object names; check
  the owning scene/widget before assuming a type is globally built in.
- Common object fields are `name`, `pos`, `size`, `scale`, `origin`, `pad`,
  `outer_pad`, `fill`, `line`, `corners`, `color`, `font`, `max_size`,
  `clickable`, `z-index`, and `modules`. Containers use `children`;
  `gridlayout` uses row/cell child arrays.
- `font` supports at least `size`, `style` (`regular`, `bold`, `italic`,
  `mono`), and `align` (`left`, `right`, `center`, `vcenter`).
- Text and most fields can contain dynamic expressions in `{...}`. Variables
  come from `Context` `VarFunc`s, global settings are normally exposed as
  `{global.setting_name}`, list/loop items default to `{i}`, and object fields
  can be accessed with dotted paths like `{i.name}` or `{window_size.x}`.
- Expression syntax is prefix-style and nestable. Existing layouts use
  conditionals (`{if:cond:then:else}`), boolean operators (`{&&:...}`,
  `{||:...}`, `{not:...}`), comparisons (`{equal:a:b}`, `{nequal:a:b}`,
  `{<:a:b}`, `{>:a:b}`, `{>=:a:b}`), arithmetic (`{+:...}`, `{-:...}`,
  `{*:...}`, `{/:...}`, `{mod:...}`, `{min:...}`, `{max:...}`), collection
  access (`{at:index:value}`, `{array_length:value}`, `{concat:a:b}`), and
  string/path helpers (`{lower:x}`, `{filename:x}`, `{basename:x}`,
  `{shorten:x:n}`).
- Actions are declared as strings such as `"action":"set:gui_run:true"` or
  `"action":"import_detect_annotations"`. The parser resolves expressions inside the
  action name and parameters before calling the matching `ActionFunc`.
- `button` triggers `action` on click. `textfield` triggers `action` on Enter
  and `on_text_changed` while editing; the current text is available to the
  action as `{text}`.
- `condition` uses `"var"` plus `"then"` and optional `"else"` objects. `each`
  uses `"var"` and `"do"`, with optional `"as"` to rename the loop item.
- `list` supports either dynamic `"var"` plus `"template"` or static `"items"`.
  Dynamic templates commonly expose `text`, `detail`, `tooltip`, `disabled`,
  and `action`; if a list action has no explicit parameter, the selected row
  index is passed by the list implementation.
- `settings` binds directly to an existing setting named by `"var"` and can
  render setting-specific controls. For new widget-local state, prefer exposing
  a JSON object via a `VarFunc` and mutating C++ state through actions.
- DynamicGUI is a renderer/action layer, not an ownership model. The C++
  scene/widget should own state, validate inputs, catch exceptions from actions,
  and expose derived preview/status data back to the layout via `VarFunc`.
4. Scene workflow (TRex UI):
   - Implement `gui::Scene` objects that own state, draw UI, and respond to
     global events.
   - Register all scenes once, then switch via `SceneManager::set_active`.
   - Keep scene transitions and long-running tasks off the UI thread by using
     the `SceneManager` task queue.
5. SFLoop:
   - `SFLoop` is a low-level loop helper for custom render/event hooks.
   - Prefer the scene system for the main app; reserve `SFLoop` for isolated
     tools or legacy flows that cannot use `SceneManager`.

## Minimal GUI executable (Scene + DynamicGUI)
The snippet below shows a minimal GUI app that:
- Initializes command line + settings defaults.
- Registers asset lookup for JSON/layouts/icons.
- Creates a window, sets icons, and runs the SceneManager loop.
- Renders a button, textfield, and status text via a `Scene`.
- Overlays a JSON-driven DynamicGUI with a simple text object.

```cpp
#include <commons.pc.h>
#include <file/DataLocation.h>
#include <gui/DrawStructure.h>
#include <gui/DynamicGUI.h>
#include <gui/Event.h>
#include <gui/IMGUIBase.h>
#include <gui/Scene.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/types/StaticText.h>
#include <gui/types/Textfield.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>

using namespace cmn;
using namespace cmn::gui;
using namespace cmn::gui::dyn;

class MinimalScene final : public Scene {
public:
    explicit MinimalScene(Base& window)
        : Scene(window, "minimal-scene", [this](Scene&, DrawStructure& graph) { draw(graph); })
    {}

private:
    struct GuiState {
        derived_ptr<PlaceinLayout> root;
        derived_ptr<Button> apply;
        derived_ptr<Textfield> input;
        derived_ptr<StaticText> status;
        DynamicGUI dyn_gui;
        std::once_flag init_once;
    } ui;

    void draw(DrawStructure& graph) {
        std::call_once(ui.init_once, [&]() {
            ui.root = derived_ptr<PlaceinLayout>(new PlaceinLayout());
            ui.apply = Button::MakePtr(Str("Apply"), Box(20, 20, 120, 36));
            ui.input = derived_ptr<Textfield>(new Textfield(Box(20, 70, 260, 36)));
            ui.status = derived_ptr<StaticText>(new StaticText(Str("Ready."), Box(20, 120, 600, 40)));

            ui.root->add_child(ui.apply);
            ui.root->add_child(ui.input);
            ui.root->add_child(ui.status);

            ui.input->on_text_changed([&]() {
                ui.status->set_txt("Typed: " + ui.input->text());
            });
            ui.apply->on_click([&](auto) {
                ui.status->set_txt("Clicked: " + ui.input->text());
            });

            ui.dyn_gui = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = file::DataLocation::parse("app", "minimal_gui.json"),
                .context = {},
                .base = window()
            };
        });

        graph.wrap_object(*ui.root);
        ui.dyn_gui.update(graph, ui.root.get());
    }
};

int main(int argc, char** argv) {
    CommandLine::init(argc, argv);
    CommandLine::instance().cd_home();

    SETTING(app_name) = std::string("Minimal GUI");
    SETTING(terminate) = false;

    file::DataLocation::register_path("app", [](const sprite::Map&, file::Path input) {
        if (input.is_absolute()) {
            return input;
        }
        return CommandLine::instance().wd() / input;
    });

    IMGUIBase base("Minimal GUI", Size2{800, 600},
        [&](DrawStructure& graph) -> bool {
            SceneManager::getInstance().update(&base, graph);
            return !BOOL_SETTING(terminate);
        },
        [&](auto&, const Event& event) {
            if (SceneManager::getInstance().on_global_event(event)) {
                return;
            }
            if (event.type == EventType::KEY && event.key.code == Keyboard::Escape) {
                SETTING(terminate) = true;
            }
        }
    );

    base.platform()->set_icons({
        file::DataLocation::parse("app", "gfx/TRex_16.png"),
        file::DataLocation::parse("app", "gfx/TRex_32.png"),
        file::DataLocation::parse("app", "gfx/TRex_64.png")
    });

    MinimalScene scene(base);
    auto& manager = SceneManager::getInstance();
    manager.register_scene(&scene);
    manager.set_active("minimal-scene");

    base.loop();
    return 0;
}
```

Minimal DynamicGUI JSON (place at `app: minimal_gui.json` via `DataLocation`):
```json
{
  "objects": [
    { "type": "stext", "text": "Hello from DynamicGUI", "pos": [20, 170] }
  ]
}
```

## Build and run
1) Compile this example by placing it at `Application/src/tracker/minimal_gui.cpp`
   and adding to `Application/src/tracker/CMakeLists.txt`:
```cmake
add_executable(minimal_gui minimal_gui.cpp)
target_link_libraries(minimal_gui PUBLIC tracker_gui)
```
2) Build + run (single commands):
```bash
# macOS / Linux
cmake -S Application -B build && cmake --build build --target minimal_gui
./build/src/tracker/minimal_gui
```
```powershell
# Windows (PowerShell, default generator)
cmake -S Application -B build
cmake --build build --target minimal_gui
.\build\src\tracker\Debug\minimal_gui.exe
```
3) Drop resources next to the executable:
```bash
cp -R Application/src/tracker/gfx build/src/tracker/
cp Application/src/tracker/minimal_gui.json build/src/tracker/
```
```powershell
Copy-Item Application\src\tracker\gfx build\src\tracker\ -Recurse
Copy-Item Application\src\tracker\minimal_gui.json build\src\tracker\
```

Conda build (same recipe GitHub Actions uses):
```bash
cd conda
conda build -c conda-forge .
```

## Deprecated note
- The `Application/src/grabber/` subtree is deprecated. Do not treat it as a
  reference implementation for GUI structure or best practices.

## Agent execution constraints
- When creating a new branch, use a concise descriptive name without a
  ``codex/`` or other agent-specific prefix.
- Do not run builds, CMake configure/generate commands, or CMake build commands.
  The user will run builds/tests after the agent has inspected the code and is
  confident the changes are ready.
- Optimize for fewer, higher-confidence iterations. Think and inspect longer
  before responding or editing, because each exchange has a monetary cost.
- For GitHub CI diagnosis or repair, inspect every failing check in the latest
  relevant run(s) across all operating systems before editing. List every
  distinct failure and its root cause first; do not stop after the first error.
  Then address the complete in-scope failure set together in one coordinated
  pass whenever possible.
- Do not wait for a full CI build to finish. Inspect completed failures and,
  when useful, monitor only short setup or targeted reproduction boundaries.
  Once a long build is underway, report its current status to the user and stop
  polling.
- do not run commands in the build directory and dont delete the existing project files there
- do not run commands outside the root directory of the project, or commands that affect the outside
- stay in scope for the task you were asked to do. only edit files directly relevant to that task, plus the minimal wiring required to make those edits work.
- if the task is to add or edit tests, edit tests and only the smallest necessary test wiring (for example `Application/Tests/CMakeLists.txt`). do not edit unrelated production/source files unless the user explicitly asks for that too.
- when fixing a bug, first reproduce the actual failure with a minimal viable test or local repro that matches the real issue. make sure that repro fails before changing production code, then fix the code until that same repro passes.
- do not add speculative, broad, or low-value tests just to increase coverage. prefer the smallest targeted regression test for a real bug, and skip adding tests when they do not materially validate the reported failure.
- During documentation work, when an executable behavior is confirmed, a software issue is identified, or a behavioral/boundary principle is newly stated in the documentation, locate the automated test that locks down that fact. If adequate coverage does not already exist, add the smallest focused unit or regression test before relying on the behavior in the documentation or fixing the issue. Purely editorial, navigation, external-environment, and generated-default wording changes do not require duplicate tests.
- Do not put software bug reports, temporary workarounds, or implementation-status notes in user-facing documentation. If a bug cannot be addressed in the current scope, record it in a GitHub issue instead. If the user asks for the bug to be addressed, fix the code and add the smallest adequate regression test; document only the resulting supported behavior.
- Do not treat PV metadata as a complete or self-contained settings snapshot. During conversion, `Segmenter::set_metadata()` stores `generate_delta_config(AccessLevelType::LOAD)`: a sparse delta containing eligible values that differ from `GlobalSettings::current_defaults`, plus explicitly included fields, after unconditional and detection-mode-specific exclusions. The PV header stores source, conversion range, encoding, resolution, and other format data separately from that settings delta. Generated `.settings` files are also deltas (`generate_delta_config(AccessLevelType::INIT, ...)`), not exhaustive snapshots. Tests asserting which settings were persisted must inspect the writer delta or the on-disk metadata keys while accounting for compatibility normalization performed by the PV reader; dedicated header fields must be read directly. Tests asserting an effective configuration must use the normal settings-loading layers or an appropriate runtime observable. Never parse PV metadata over registered defaults and then attribute inherited/defaulted values to the PV.
- Only use the Conda environment `trex` for environment-specific commands or instructions, or the `trex-modules` environment. Do not access or assume any other environment.
- When building in `Application/build`, always use the Conda environment `trex` from the project root. Do not use `trex-modules` for `Application/build`.
- when running Python commands in the `trex` environment for this repo, prefer `KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python ...` because duplicate `libomp` initialization can otherwise abort the process on macOS.
- Run authenticated `gh` commands with escalated access. A sandboxed `gh auth`
  failure does not establish that credentials have expired; retry with
  escalation before asking the user to authenticate again.
- When a commit fully resolves a specific GitHub issue and is intended to reach
  the default branch, include an appropriate closing keyword such as
  `Fixes #257` in the commit message so merging it closes the issue
  automatically. For partial or not-yet-verified work, use a non-closing issue
  reference instead.
- For commons monolith + modules work, run CMake/Ninja from `Application/tmp-modules-osx-tests-nolto` with the `trex-modules` Conda environment.
- For commons shared-library split testing with modules disabled, use `tmp-shared-split-osx-tests-nolto` with Ninja in the `trex-modules` Conda environment.
- For commons shared-library split testing with modules enabled, use `tmp-shared-split-osx-tests-nolto` with Ninja in the `trex-modules` Conda environment.

## Release/tag workflow notes
- Before release work, preserve any dirty user worktree state first. If the
  user approves, stash tracked and untracked files before switching branches.
- For quick-fix releases from `main`, fetch remotes and tags, fast-forward
  `main`, create an annotated `vX.Y.Z` tag, push only that tag, then create the
  GitHub release with the same title as the tag.
- Use the previous GitHub release as the naming/body template and include a
  `Full Changelog` compare link.
- If `gh` is unavailable, install/use GitHub CLI when the user approves it.
  Authenticate with `gh auth login -h github.com -w` so the user can approve
  the device-code flow.
