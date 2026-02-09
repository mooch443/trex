#include "LiveSegmentation.h"
#include <gui/dyn/ParseText.h>
#include <gui/dyn/Action.h>
#include <gui/DynamicGUI.h>
#include <gui/Bowl.h>
#include <gui/DrawStructure.h>
#include <video/VideoSource.h>
#include <processing/encoding.h>

namespace cmn::gui {

using namespace dyn;

LiveSegmentation::LiveSegmentation(Base& window)
:
    Scene(window, "live-seg-scene", [this](Scene&, DrawStructure& base){
        _draw(base);
    }),
    _bowl(nullptr),
    _current_image(std::make_unique<ExternalImage>()),
    _gui(std::make_unique<dyn::DynamicGUI>())
{
    
}

void LiveSegmentation::activate() {
    Scene::activate();
    // Logic to activate the scene, e.g., initializing framePreloader
    auto source = READ_SETTING(source, file::PathArray);
    Print("Loading source = ", utils::ShortenText(source.toStr(), 1000));
    
    _video = std::make_unique<VideoSource>(source);
    _video->set_colors(ImageMode::RGB);
    
    video_length = _video->length();
    video_size = _video->size();
    
    FindCoord::set_video(video_size);
    
    _next_frame.reset();
    _terminated = false;
    
    assert(not _fetch_thread);
    _fetch_thread = std::make_unique<std::thread>(
        [this](){
            std::unique_lock guard{_next_frame_mutex};
            Frame_t index = 0_f;
            VideoFrame frame;
            const auto length = _video->length();
            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
            bool disable_waiting = false;
            
            while(not _terminated.load()) {
                /// load a frame into cache
                _video->frame(index, *frame.image);
                frame.index = index;
                
                if(not disable_waiting)
                    _condition.wait(guard, [this](){ return not _next_frame.has_value() || _terminated.load(); });
                disable_waiting = false;
                
                /// check if there was a request for a frame
                if(_requested_frame.has_value()) {
                    index = _requested_frame.value();
                    _requested_frame.reset();
                    
                    if(index >= length)
                        index = length - 1_f;
                    
                    if(frame.index != index) {
                        disable_waiting = true;
                        continue;
                    }
                }
                
                if(not _next_frame.has_value()) {
                    /// load new frame
                    _next_frame = std::move(frame);
                    
                    if(_previous_frame.has_value())
                        frame = std::move(_previous_frame.value());
                    else
                        frame.image = Image::Make(_video->size().height, _video->size().width, 4);
                    
                    ++index;
                    
                    if(index >= length)
                        index = 0_f;
                }
            }
        }
    );
}

// Deactivate method implementation
void LiveSegmentation::deactivate() {
    Scene::deactivate();
    
    // Logic to deactivate the scene
    _gui->clear();
    _bowl = nullptr;
    
    _terminated = true;
    _condition.notify_all();
    
    if(_fetch_thread) {
        _fetch_thread->join();
        _fetch_thread = nullptr;
    }
    
    std::unique_lock guard(_next_frame_mutex);
    _video = nullptr;
}

// Custom drawing implementation
void LiveSegmentation::_draw(DrawStructure& graph) {
    if(window()) {
        //auto update = FindCoord::set_screen_size(graph, *window()); //.div(graph.scale().reciprocal() * gui::interface_scale());
        //
        FindCoord::set_video(video_size);
        //if(update != window_size)
         //   window_size = update;
    }

    auto coord = FindCoord::get();
    if (not _bowl) {
        _bowl = std::make_unique<Bowl>(nullptr);
        _bowl->set_video_aspect_ratio(coord.video_size().width, coord.video_size().height);
        _bowl->fit_to_screen(coord.screen_size());
    }
    
    if(not *_gui) {
        *_gui = DynamicGUI{
            .gui = nullptr,
            .path = "live_segmentation_layout.json",
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    
                };

                context.variables = {
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("video_size", [](const VarProps&) -> Size2 {
                        return FindCoord::get().video_size();
                    }),
                    VarFunc("frame", [this](const VarProps&) {
                        return _current_frame.index;
                    }),
                    VarFunc("2hud", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Vec2>(props.parameters.front());
                        return coords.convert(BowlCoord(p));
                    }),
                    VarFunc("2bowl", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Vec2>(props.parameters.front());
                        return coords.convert(HUDCoord(p));
                    })
                };
                
                context.actions = {
                    ActionFunc("set_frame", [this](const Action& action){
                        REQUIRE_EXACTLY(1, action);
                        
                        std::unique_lock guard{_next_frame_mutex};
                        _requested_frame = Meta::fromStr<Frame_t>(action.parameters.front());
                    })
                };

                return context;
            }(),
            .base = window()
        };
    }
    
    {
        std::unique_lock guard{_next_frame_mutex};
        if(_next_frame.has_value()) {
            _current_frame.image = _current_image->update_with(nullptr);
            _previous_frame = std::move(_current_frame);
            _current_frame = std::move(_next_frame.value());
            _current_image->set_source(std::move(_current_frame.image));
            _next_frame.reset();
        }
        _condition.notify_one();
    }
    
    graph.wrap_object(*_current_image);
    
    graph.wrap_object(*_bowl);
    _bowl->update_scaling(_timer.elapsed());
    _timer.reset();
    
    _bowl->fit_to_screen(coord.screen_size());
    _bowl->set_target_focus({});
    
    auto coords = FindCoord::get();
    _bowl->update(_current_frame.index, graph, coords);
    
    _current_image->set_scale(_bowl->_current_scale);
    _current_image->set_pos(_bowl->_current_pos);
    
    graph.section("elements", [&](auto&, Section* s) {
        s->set_scale(_bowl->_current_scale);
        s->set_pos(_bowl->_current_pos);
    });
    
    graph.section("gui", [this, &graph](DrawStructure &, Section *){
        _gui->update(graph, nullptr);
    });
}

bool LiveSegmentation::on_global_event(Event event) {
    auto graph = _bowl && _bowl->stage() ? _bowl->stage() : nullptr;
    if(event.type == EventType::MMOVE
       && graph
       && graph->is_mouse_down(0)
       && not graph->selected_object())
    {
        auto p = Vec2(event.move.x, event.move.y);
        
        auto coords = FindCoord::get();
        Print(coords.convert(HUDCoord(p)));
        //_drag_box->create(Size{coords.convert(HUDCoord(p)) - pos + Vec2(1)}, FillClr{Red.alpha(50)});
    }

    if(event.type == EventType::MBUTTON
       && event.mbutton.button == 0
       && not event.mbutton.pressed)
    {
        auto p = Vec2(event.mbutton.x, event.mbutton.y);
        auto coord = FindCoord::get();
        auto original = p;
        p = coord.convert(HUDCoord(p));
        
        if(p.x >= 0 && p.y >= 0 && p.x < coord.video_size().width && p.y < coord.video_size().height) {
            Print("adding point at ", original, " => ", coord.convert(HUDCoord(p)));
        }
        
    } else if(event.type == EventType::MBUTTON
              && event.mbutton.button == 1
              && event.mbutton.pressed)
    {
        //_drag_box = nullptr;
    }
    
    if(event.type == EventType::KEY
       && event.key.pressed)
    {
        switch (event.key.code) {
            case Codes::Left:
                // retrieve a frame that is the highest frame before the current frame
                if(_current_frame.index > 0_f) {
                    /*Frame_t closestFrame;
                    for (const auto& frame : _selected_frames) {
                        if (frame < currentFrameIndex) {
                            if (not closestFrame.valid() || frame > closestFrame) {
                                closestFrame = frame;
                            }
                        }
                    }

                    if (closestFrame.valid()) {
                        Print("Navigating to frame ", closestFrame);
                        navigateToFrame(closestFrame);
                    }*/
                }
                break;
                
            case Codes::Right:
                // retrieve a frame that is the lowest frame after the current frame
                if(_current_frame.index < video_length) { // Assuming MAX_FRAME_INDEX is the upper limit for frame index
                    /*Frame_t closestFrame;
                    for (const auto& frame : _selected_frames) {
                        if (frame > currentFrameIndex) {
                            if (not closestFrame.valid() || frame < closestFrame) {
                                closestFrame = frame;
                            }
                        }
                    }

                    if (closestFrame.valid()) {
                        Print("Navigating to frame ", closestFrame);
                        navigateToFrame(closestFrame);
                    }*/
                }
                break;

                
            default:
                break;
        }
    }
    
    return false; // Return true if the event is handled
}

}
