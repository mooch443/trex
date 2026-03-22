#include "LiveSegmentation.h"
#include <file/PathArray.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/Action.h>
#include <gui/DynamicGUI.h>
#include <ui/Bowl.h>
#include <gui/DrawStructure.h>
#include <video/VideoSource.h>
#include <processing/encoding.h>
#include <python/SAM3.h>
#include <python/PythonWrapper.h>
#include <ui/ImageDisplayElement.h>
#include <ui/LabelElement.h>
#include <processing/Background.h>
#include <core/idx_t.h>

struct DetectionMeta {
    pv::bid bdx;
    float conf;
    uint16_t x, y, w, h;
    
    glz::json_t to_json() const {
        glz::json_t json;
        json["bdx"] = bdx.to_json();
        json["conf"] = conf;
        json["x"] = x;
        json["y"] = y;
        json["w"] = w;
        json["h"] = h;
        return json;
    }
};

template <>
struct glz::meta<DetectionMeta> {
    using T = DetectionMeta;
    static constexpr auto value = glz::object(
        "bdx", &DetectionMeta::bdx,
        "conf", &DetectionMeta::conf,
        "x", &DetectionMeta::x,
        "y", &DetectionMeta::y,
        "w", &DetectionMeta::w,
        "h", &DetectionMeta::h
    );
};

namespace cmn::gui {

using namespace dyn;
using namespace track;

struct ImageCache {
    std::unordered_map<uint32_t, Image::SPtr> cached_images;
};

struct LiveSegmentation::Data {
    ImageCache cache;
    LabelCache_t unassigned_labels;
    std::unordered_map<Idx_t, Label_t> labels;
    double dt = 0;
    Timer timer;
};

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

LiveSegmentation::~LiveSegmentation() {
    
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
    
    _data = std::make_unique<Data>();
    
    FindCoord::set_video(video_size);
    
    _next_frame.reset();
    _terminated = false;
    SETTING(detect_type) = track::detect::ObjectDetectionType_t{track::detect::ObjectDetectionType::sam3};
    SETTING(detect_model) = file::Path("/Users/tristan/Downloads/sam3.pt");
    SETTING(detect_resolution) = track::detect::DetectResolution{};
    
    namespace py = Python;
    py::schedule([](){
        track::SAM3::init();
    });
    static const Size2 tile_size(640,640);
    
    assert(not _fetch_thread);
    _fetch_thread = std::make_unique<std::thread>(
        [this](){
            std::unique_lock guard{_next_frame_mutex};
            Frame_t index = 0_f;
            VideoFrame frame;
            const auto length = _video->length();
            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
            bool disable_waiting = false;
            useMat_t buffer;
            std::optional<std::future<SegmentationData>> result;
            
            while(not _terminated.load()) {
                if(not _next_frame.has_value()
                   || not frame.index.valid())
                {
                    guard.unlock();
                    
                    try {
                        /// load a frame into cache
                        if(not frame.image) {
                            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
                        }
                        
                        _video->frame(index, *frame.image);
                        frame.index = index;
                        frame.image->get().copyTo(buffer);
                        frame.image->set_index(index.valid() ? index.get() : -1);
                        cv::resize(buffer, buffer, tile_size);
                        
                        TileImage tiled(buffer, Image::Make(*frame.image), tile_size, frame.image->dimensions());
                        tiled.callback = [](){
                            Print("Fun is done!");
                        };
                        
                        result = track::SAM3::apply(std::move(tiled));
                        
                        guard.lock();
                    } catch(...) {
                        guard.lock();
                        throw;
                    }
                }
                
                if(not disable_waiting)
                    _condition.wait(guard, [&](){
                        return not (_next_frame.has_value()
                                    && (not result.has_value()
                                        || result.value().wait_for(std::chrono::milliseconds(0)) == std::future_status::ready))
                                || _terminated.load()
                                || _requested_frame.has_value();
                    });
                disable_waiting = false;
                
                /// check if there was a request for a frame
                if(_requested_frame.has_value())
                {
                    if(not _requested_frame.value().valid()) {
                        _requested_frame.reset();
                        continue;
                    }
                    index = _requested_frame.value();
                    _requested_frame.reset();
                    
                    if(not index.valid())
                        index = 0_f;
                    else if(index >= length)
                        index = length - 1_f;
                    
                    if(frame.index != index) {
                        disable_waiting = true;
                        continue;
                    }
                }
                
                if(not _next_frame.has_value()) {
                    SegmentationData data;
                    guard.unlock();
                    
                    try {
                        /// load new frame
                        data = result.value().get();
                        guard.lock();
                        
                        _next_frame = std::move(frame);
                        _next_data = std::move(data);
                        
                        if(_previous_frame.has_value())
                            frame = std::move(_previous_frame.value());
                        else
                            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
                        
                    } catch(const std::exception& ex) {
                        FormatExcept("We caught a problem when processing the data: ", no_quotes(ex.what()));
                    } catch(...) {
                        FormatExcept("Weird exception caught here.");
                    }
                    
                    if(not guard.owns_lock()) {
                        guard.lock();
                    }
                    
                    frame.index.invalidate();
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
    
    {
        std::unique_lock guard(_next_frame_mutex);
        _next_data.reset();
        _current_data.reset();
        _next_frame.reset();
        _previous_frame.reset();
        _current_frame.image = nullptr;
        _current_frame.index.invalidate();
        _video = nullptr;
        _data = nullptr;
    }

    namespace py = Python;
    py::schedule([]() {
        track::SAM3::deinit();
    });
}

// Custom drawing implementation
void LiveSegmentation::_draw(DrawStructure& graph) {
    auto coord = FindCoord::get();
    if (not _bowl) {
        _bowl = std::make_unique<Bowl>(nullptr);
        _bowl->set_video_aspect_ratio(coord.video_size().width, coord.video_size().height);
        _bowl->fit_to_screen(coord.screen_size());
    }
    
    if(_data) {
        _data->dt = saturate(_data->timer.elapsed(), 0.001, 1.0);
        _data->timer.reset();
    }
    
    if(not *_gui) {
        *_gui = DynamicGUI{
            .gui = SceneManager::getInstance().gui_task_queue(),
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
                    VarFunc("size2hud", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Size2>(props.parameters.front());
                        return coords.convert(BowlRect(Vec2(), p)).size();
                    }),
                    VarFunc("2bowl", [](const VarProps& props) {
                        auto coords = FindCoord::get();
                        auto p = Meta::fromStr<Vec2>(props.parameters.front());
                        return coords.convert(HUDCoord(p));
                    }),
                    VarFunc("data", [this](const VarProps&) -> std::vector<glz::json_t> {
                        std::vector<glz::json_t> result;
                        if(_current_data) {
                            auto blobs = _current_data->frame.get_blobs();
                            size_t i = 0;
                            for(auto& blob : blobs)
                            {
                                auto bds = blob->bounds();
                                result.push_back(DetectionMeta{
                                    .bdx = blob->blob_id(),
                                    .conf = _current_data->frame.predictions().at(i).probability(),
                                    .x = narrow_cast<uint16_t>(bds.x),
                                    .y = narrow_cast<uint16_t>(bds.y),
                                    .w = narrow_cast<uint16_t>(bds.width),
                                    .h = narrow_cast<uint16_t>(bds.height)
                                }.to_json());
                                
                                ++i;
                            }
                        }
                        return result;
                    })
                };
                
                context.actions = {
                    ActionFunc("set_frame", [this](const Action& action){
                        REQUIRE_EXACTLY(1, action);
                        
                        std::unique_lock guard{_next_frame_mutex};
                        _requested_frame = Meta::fromStr<Frame_t>(action.parameters.front());
                        _next_frame.reset();
                    })
                };
                
                context.custom_elements["label"] = std::unique_ptr<CustomElement>(
                    new LabelElement(&_data->unassigned_labels, &_data->labels, &_data->dt)
                );
                context.custom_elements["image_generator"] = std::unique_ptr<CustomElement>(
                    new ImageDisplayElement(&ImageGeneratorRegistry::instance())
                );
                
                ImageGeneratorRegistry::Generator generator{
                    .generate = [this](const dyn::VarProps& props) -> Image::SPtr
                    {
                        REQUIRE_EXACTLY(1, props);
                        auto index = Meta::fromStr<uint32_t>(props.parameters.front());
                        
                        if(_data) {
                            auto it = _data->cache.cached_images.find(index);
                            if(it != _data->cache.cached_images.end())
                            {
                                return it->second;
                            }
                        }
                        
                        assert(SceneManager::is_gui_thread());
                        
                        if(_current_data) {
                            auto blob = _current_data->frame.blob_at(index);
                            Background bg(FindCoord::get().video_size(), READ_SETTING_WITH_DEFAULT(meta_encoding, meta_encoding_t::gray ));
                            
                            
                            Image::SPtr image = Image::Make();
                            blob->rgba_image(bg, 0, *image);
                            if(_data) {
                                _data->cache.cached_images[index] = image;
                            }
                            
                            return image;
                        }
                        
                        return nullptr;
                    },
                    .reset = [this](){
                        assert(SceneManager::is_gui_thread());
                        if(_data) {
                            _data->cache.cached_images.clear();
                        }
                    }
                };
                ImageGeneratorRegistry::instance().register_generator("blob_image", std::move(generator));
                return context;
            }(),
            .base = window()
        };
    }
    
    if(_playback.load())
    {
        std::unique_lock guard{_next_frame_mutex};
        if(_next_frame.has_value()) {
            _current_frame.image = _current_image->update_with(nullptr);
            _previous_frame = std::move(_current_frame);
            _current_frame = std::move(_next_frame.value());
            _current_image->set_source(std::move(_current_frame.image));
            _current_data = std::move(_next_data);
            _next_data.reset();
            _next_frame.reset();
            ImageGeneratorRegistry::instance().reset_generator("blob_image");
        }
        _condition.notify_one();
    }
    
    graph.wrap_object(*_current_image);
    
    if(_current_data.has_value()) {
        if(_current_data.value().frame.n() > 0) {
            //Print(_current_data.value().frame);
        }
    }
    
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
            Print("adding point at ", original, " => ", p);
            
            std::unique_lock g{_next_frame_mutex};
            _requested_frame = _current_frame.index;
            _next_frame.reset();
        } else
            Print("not adding point at ", original, " => ", p);
        
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
            case Codes::Space:
                _playback = not _playback.load();
                Print("Playback = ", _playback.load());
                break;
            case Codes::Left:
                // retrieve a frame that is the highest frame before the current frame
                if(_current_frame.index.valid()
                   && _current_frame.index > 0_f)
                {
                    std::unique_lock guard{_next_frame_mutex};
                    _requested_frame = _current_frame.index - 1_f;
                    _next_frame.reset();
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
                if(_current_frame.index.valid()
                   && _current_frame.index < video_length)
                { // Assuming MAX_FRAME_INDEX is the upper limit for frame index
                    std::unique_lock guard{_next_frame_mutex};
                    _requested_frame = _current_frame.index + 1_f;
                    _next_frame.reset();
                    _condition.notify_one();
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
                } else {
                    std::unique_lock guard{_next_frame_mutex};
                    _requested_frame = 0_f;
                    _next_frame.reset();
                    _condition.notify_one();
                }
                break;

                
            default:
                break;
        }
    }
    
    return false; // Return true if the event is handled
}

}
