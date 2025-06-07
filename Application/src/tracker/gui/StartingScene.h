#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>

namespace cmn::gui {

class StartingScene : public Scene {
    RecentItems _recents;
    std::string _search_text;
    std::vector<std::string> _corpus;
    PreprocessedData _preprocessed_corpus;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _recents_list, _filtered_recents;
    std::vector<sprite::Map> _data;

    // The HorizontalLayout for the two buttons and the image
    dyn::DynamicGUI dynGUI;

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
