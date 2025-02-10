#include "CalibrateScene.h"
#include <gui/DynamicGUI.h>
#include <misc/Coordinates.h>
#include <gui/dyn/Action.h>
#include <gui/GUIVideoAdapterElement.h>
#include <gui/GUIVideoAdapter.h>
#include <misc/SizeFilters.h>

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
                    }),
                    ActionFunc("configure_points", [](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        
                        auto pts = Meta::fromStr<std::vector<Vec2>>(action.first());
                        if(pts.size() == 2) {
                            static NumericTextfield<double> text(1.0, Bounds(0, 0, 200,30), arange<double>{0, infinity<double>()});
                            text.set_postfix("cm");
                            text.set_fill_color(DarkGray.alpha(50));
                            text.set_text_color(White);
                            
                            derived_ptr<Entangled> e = std::make_shared<Entangled>();
                            e->update([&](Entangled& e) {
                                e.advance_wrap(text);
                            });
                            e->auto_size(Margin{0, 0});
                            
                            auto S = pts.front();
                            auto E = pts.back();
                            auto D = euclidean_distance(S, E);
                            
                            Print("calibrating ", action.first(), " with a distance of ", D);
                            
                            SceneManager::getInstance().enqueue([D, e=std::move(e)](auto, DrawStructure& graph) mutable {
                                graph.dialog([D](Dialog::Result r) {
                                    if(r != Dialog::OKAY)
                                        return;
                                    
                                    SceneManager::getInstance().enqueue([D](auto, DrawStructure& graph) mutable {
                                        graph.dialog([D](Dialog::Result auto_change_parameters) {
                                            try {
                                                auto value = Meta::fromStr<float>(text.text());
                                                Print("Value is: ", value);
                                                
                                                if(value > 0) {
                                                    if(auto_change_parameters == Dialog::OKAY) {
                                                        auto cm_per_pixel = SETTING(cm_per_pixel).value<Float2_t>();
                                                        auto detect_size_filter = SETTING(detect_size_filter).value<SizeFilters>();
                                                        auto track_size_filter = SETTING(track_size_filter).value<SizeFilters>();
                                                        auto track_max_speed = SETTING(track_max_speed).value<Float2_t>();
                                                        
                                                        const auto new_cm_per_pixel = Float2_t(value / D);
                                                        
                                                        /// track_max_speed
                                                        SETTING(track_max_speed) = Float2_t(track_max_speed / cm_per_pixel * new_cm_per_pixel);
                                                        
                                                        /// detect_size_filter
                                                        if(not detect_size_filter.empty()) {
                                                            std::set<Range<double>> ranges;
                                                            SizeFilters filters;
                                                            for(auto &[start, end] : detect_size_filter.ranges()) {
                                                                Range<double> range{
                                                                    start / SQR(cm_per_pixel) * SQR(new_cm_per_pixel),
                                                                    end / SQR(cm_per_pixel) * SQR(new_cm_per_pixel)
                                                                };
                                                                filters.add(range);
                                                            }
                                                            SETTING(detect_size_filter) = filters;
                                                        }
                                                        
                                                        /// track_size_filter
                                                        if(not track_size_filter.empty()) {
                                                            std::set<Range<double>> ranges;
                                                            SizeFilters filters;
                                                            for(auto &[start, end] : track_size_filter.ranges()) {
                                                                Range<double> range{
                                                                    start / SQR(cm_per_pixel) * SQR(new_cm_per_pixel),
                                                                    end / SQR(cm_per_pixel) * SQR(new_cm_per_pixel)
                                                                };
                                                                filters.add(range);
                                                            }
                                                            SETTING(track_size_filter) = filters;
                                                        }
                                                        
                                                        SETTING(cm_per_pixel) = new_cm_per_pixel;
                                                        
                                                        SceneManager::getInstance().enqueue([detect_size_filter, track_max_speed, track_size_filter](auto, DrawStructure& graph)
                                                                                            {
                                                            graph.dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<Float2_t>())+"</nr> and recalculated <ref>detect_size_filter</ref> from <nr>"+Meta::toStr(detect_size_filter)+"</nr> to <nr>"+Meta::toStr(SETTING(detect_size_filter).value<SizeFilters>())+"</nr>, and <ref>track_size_filter</ref> from <nr>"+Meta::toStr(track_size_filter)+"</nr> to <nr>"+Meta::toStr(SETTING(track_size_filter).value<SizeFilters>())+"</nr> and <ref>track_max_speed</ref> from <nr>"+Meta::toStr(track_max_speed)+"</nr> to <nr>"+Meta::toStr(SETTING(track_max_speed).value<Float2_t>())+"</nr>.", "Calibration successful", "Okay");
                                                        });
                                                        
                                                    } else {
                                                        SETTING(cm_per_pixel) = Float2_t(value / D);
                                                        SceneManager::getInstance().enqueue([](auto, DrawStructure& graph)
                                                                                            {
                                                            graph.dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<Float2_t>())+"</nr>.", "Calibration successful", "Okay");
                                                        });
                                                    }
                                                }
                                                
                                            } catch(const std::exception& e) { }
                                            
                                        }, "Do you want to automatically set <ref>track_max_speed</ref>, <ref>detect_size_filter</ref>, and <ref>track_size_filter</ref> based on the given conversion factor?", "Calibrate with known length", "Yes", "No");
                                    });
                                    
                                }, "Please enter the equivalent length in centimeters for the selected distance (<nr>"+Meta::toStr(D)+"</nr>px) below. <ref>cm_per_pixel</ref> will then be recalculated based on the given value, affecting parameters such as <ref>track_max_speed</ref>, and <ref>track_size_filter</ref>, and tracking results.", "Calibrate with known length", "Okay", "Abort")->set_custom_element(std::move(e));
                            });
                        }
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
                    _data->_adapter.to<GUIVideoAdapter>()->set(path);
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
