#pragma once
#include <commons.pc.h>
#include <ui/Scene.h>

namespace cmn::gui {

class StartingScene : public Scene {
    struct Data;
    std::unique_ptr<Data> _data;

public:
    StartingScene(Base& window);
    ~StartingScene();

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
    bool on_global_event(Event) override;
    
private:
    void update_recent_items();
    void update_search_filters();
};
}
