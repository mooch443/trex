#pragma once

#include <commons.pc.h>
#include <gui/Scene.h>

namespace cmn::gui {

class CalibrateScene : public Scene {
    struct Data;
    std::unique_ptr<Data> _data;

public:
    CalibrateScene(Base& window);
    ~CalibrateScene();

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
    bool on_global_event(Event) override;
};

}
