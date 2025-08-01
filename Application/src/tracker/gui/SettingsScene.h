#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <file/PathArray.h>

namespace cmn::gui {

class SettingsScene : public Scene {
    struct Data;
    std::unique_ptr<Data> _data;
    
public:
    SettingsScene(Base& window);
    ~SettingsScene();
    void activate() override;
    void deactivate() override;
    void _draw(DrawStructure& graph);
    bool on_global_event(Event) override;
    static void reset_last_opened_tab();
};

}

