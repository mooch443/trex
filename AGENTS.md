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

## Commons library overview
- Core GUI headers live under `Application/src/commons/common/gui/`.
- The dynamic GUI system is `Application/src/commons/common/gui/DynamicGUI.h`
  and `Application/src/commons/common/gui/dyn/`.
- Most source files include the precompiled header `commons.pc.h`.
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
- do not run commands in the build directory and dont delete the existing project files there
- do not run commands outside the root directory of the project, or commands that affect the outside
- Only use the Conda environment `trex` for environment-specific commands or instructions, or the `trex-modules` environment. Do not access or assume any other environment.
