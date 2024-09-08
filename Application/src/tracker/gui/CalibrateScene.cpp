#include "CalibrateScene.h"
#include <gui/DynamicGUI.h>
#include <misc/Coordinates.h>
#include <gui/dyn/Action.h>
#include <gui/GUIVideoAdapterElement.h>
#include <gui/GUIVideoAdapter.h>

namespace cmn::gui {

struct CalibrateScene::Data {
    dyn::DynamicGUI gui;
    std::vector<Vec2> points;
    Vec2 last_mouse;
    Layout::Ptr _adapter;
    
    std::atomic<Size2> _next_video_size;
    Size2 _video_size;
};

CalibrateScene::CalibrateScene(Base& window)
    : Scene(window, "calibrate-scene", [this](Scene&, DrawStructure& base){
        _draw(base);
    })
{
    
}

CalibrateScene::~CalibrateScene() {
    
}

void CalibrateScene::activate() {
    _data = std::make_unique<Data>();
}

void CalibrateScene::deactivate() {
    _data = nullptr;
}

void CalibrateScene::_draw(DrawStructure &graph) {
    if(not _data)
        return;
    
    if(not _data->gui) {
        using namespace dyn;
        
        _data->gui = dyn::DynamicGUI{
            .gui = nullptr,
            .path = "calibrate_layout.json",
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    ActionFunc("add", [this](const Action& action) {
                        REQUIRE_AT_LEAST(1, action);
                        for(auto &pt : action.parameters) {
                            auto v = Meta::fromStr<Vec2>(pt);
                            _data->points.push_back(v);
                        }
                    }),
                    ActionFunc("change_scene", [](Action action) {
                        REQUIRE_EXACTLY(1, action);
                        auto scene = Meta::fromStr<std::string>(action.first());
                        if(not SceneManager::getInstance().is_scene_registered(scene))
                            return false;
                        SceneManager::getInstance().set_active(scene);
                        return true;
                    }),
                    ActionFunc("remove", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        auto index = Meta::fromStr<uint32_t>(action.first());
                        _data->points.erase(_data->points.begin() + index);
                    }),
                    ActionFunc("clear", [this](const Action&) {
                        _data->points.clear();
                    })
                };

                context.variables = {
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("mouse", [this](const VarProps&) -> Vec2 {
                        return _data->last_mouse;
                    }),
                    VarFunc("video_size", [this](const VarProps&) -> Vec2 {
                        return _data->_video_size;
                    }),
                    VarFunc("points", [this](const VarProps&) {
                        return _data->points;
                    }),
                    VarFunc("point", [this](const VarProps& props) -> Vec2 {
                        REQUIRE_EXACTLY(1, props);
                        auto index = Meta::fromStr<uint32_t>(props.parameters.front());
                        return _data->points.at(index);
                    })
                };

                return context;
            }(),
            .base = window()
        };
        
        _data->gui.context.custom_elements["video"] = std::unique_ptr<GUIVideoAdapterElement>{
            new GUIVideoAdapterElement((IMGUIBase*)_window, []() -> Size2 {
                return FindCoord::get().screen_size();
            }, [this](VideoInfo info) {
                _data->_next_video_size = info.size;
            }, [this](const file::PathArray& path, IMGUIBase* window, std::function<void(VideoInfo)> callback) -> Layout::Ptr {
                if(_data->_adapter) {
                    _data->_adapter.to<GUIVideoAdapter>()->set(Str{path.source()});
                    return _data->_adapter;
                } else {
                    Layout::Ptr ptr = Layout::Make<GUIVideoAdapter>(path, window, callback);
                    _data->_adapter = ptr;
                    return ptr;
                }
            })
        };
    }
    
    _data->last_mouse = graph.mouse_position();
    _data->gui.update(graph, nullptr);
    
    _data->_video_size = _data->_next_video_size.load();
}

bool CalibrateScene::on_global_event(Event) {
    return false;
}

}
