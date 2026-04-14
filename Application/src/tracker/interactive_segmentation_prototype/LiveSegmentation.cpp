#include "LiveSegmentation.h"
#include <file/PathArray.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/Action.h>
#include <gui/DynamicGUI.h>
#include <ui/Bowl.h>
#include <gui/DrawStructure.h>
#include <video/VideoSource.h>
#include <processing/encoding.h>
#include <python/Detection.h>
#include <python/PythonWrapper.h>
#include <ui/ImageDisplayElement.h>
#include <ui/LabelElement.h>
#include <processing/Background.h>
#include <core/idx_t.h>
#include <core/GPURecognitionTypes.h>
#include <file/DataLocation.h>
#include <pv.h>
#include <core/TileBuffers.h>
#include <processing/ResizeImage.h>

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

namespace {

static thread_local useMat_t sam3_letterboxed_input_buffer;

/**
 * Build the single-image SAM3 tile package used by both live processing and
 * interactive replay.
 */
TileImage make_sam3_tiled_frame(Image::Ptr&& frame_image)
{
    const auto detect_resolution = READ_SETTING(detect_resolution, track::detect::DetectResolution);
    const Size2 tile_size(detect_resolution.width, detect_resolution.height);
    const auto geometry = resize_image_into(
        frame_image->get(),
        tile_size,
        sam3_letterboxed_input_buffer,
        ImageResizeMode::letterbox);

    TileImage tiled(
        sam3_letterboxed_input_buffer,
        Image::Make(*frame_image),
        tile_size,
        frame_image->dimensions());
    tiled._offsets = {geometry.offset};
    tiled.source_size = geometry.content_size;
    return tiled;
}

} // namespace

namespace cmn::gui {

using namespace dyn;
using namespace track;

struct ImageCache {
    std::unordered_map<uint32_t, Image::SPtr> cached_images;
};

struct CachedFrame {
    SegmentationData _data;
    uint64_t prompt_revision = 0;
};

struct LiveSegmentation::Data {
    ImageCache cache;
    LabelCache_t unassigned_labels;
    std::unordered_map<Idx_t, Label_t> labels;
    double dt = 0;
    Timer timer;
    cmn::CallbackFuture _callback;
    std::unique_ptr<pv::File> _output_file;
    std::optional<track::detect::Sam3Prompts> _last_prompt_repository;
    
    bool has_annotations(Frame_t index) const {
        auto detect_sam3_prompt = READ_SETTING_WITH_DEFAULT(detect_sam3_prompt, std::optional<track::detect::Sam3Prompts>{});
        if(not detect_sam3_prompt)
            return false;
        
        auto it = detect_sam3_prompt->find(index);
        if(it == detect_sam3_prompt->end())
            return false;
        
        return not it->second.empty();
    }
    
    static std::optional<Frame_t> earliest_affected_frame(const std::optional<track::detect::Sam3Prompts>& previous,
                                                          const std::optional<track::detect::Sam3Prompts>& current) {
        std::optional<Frame_t> earliest;
        auto consider = [&](Frame_t frame) {
            auto affected = frame.valid() ? frame : 0_f;
            if(not earliest || affected < earliest.value()) {
                earliest = affected;
            }
        };

        if(previous) {
            for(const auto& [frame, prompts] : *previous) {
                if(not current) {
                    consider(frame);
                } else {
                    auto it = current->find(frame);
                    if(it == current->end() || it->second != prompts) {
                        consider(frame);
                    }
                }
            }
        }

        if(current) {
            for(const auto& [frame, prompts] : *current) {
                if(not previous) {
                    consider(frame);
                } else {
                    auto it = previous->find(frame);
                    if(it == previous->end() || it->second != prompts) {
                        consider(frame);
                    }
                }
            }
        }

        return earliest;
    }

    void init(std::function<void(Frame_t)> on_prompt_change) {
        _last_prompt_repository = READ_SETTING_WITH_DEFAULT(detect_sam3_prompt, std::optional<track::detect::Sam3Prompts>{});
        _callback = GlobalSettings::register_callbacks({
            "detect_sam3_prompt"
        }, [this, on_prompt_change = std::move(on_prompt_change)](auto) {
            const auto prompt_repository = READ_SETTING_WITH_DEFAULT(detect_sam3_prompt, std::optional<track::detect::Sam3Prompts>{});
            const auto earliest = earliest_affected_frame(_last_prompt_repository, prompt_repository);
            _last_prompt_repository = prompt_repository;

            /// we're changing the prompt, so we need to
            /// evict the cache here
            cleanup();

            if(earliest) {
                (void)bump_prompt_revision(*earliest);
                on_prompt_change(*earliest);
            }
        });
    }
    
    uint64_t prompt_revision(Frame_t index) const {
        std::unique_lock g{_cache_mutex};
        auto it = _prompt_revisions.find(index);
        return it == _prompt_revisions.end() ? 0 : it->second;
    }

    uint64_t bump_prompt_revision(Frame_t index) {
        std::unique_lock g{_cache_mutex};
        auto& revision = _prompt_revisions[index];
        ++revision;
        _frame_cache.erase(index);
        return revision;
    }

    void store_frame_if_annotated(SegmentationData&& data, uint64_t prompt_revision) {
        auto index = data.original_index();
        if(not has_annotations(index)) {
            return; /// no need to store
        }
        
        std::unique_lock g{_cache_mutex};
        /// moving out of data into the cache
        _frame_cache[index] = CachedFrame{
            ._data = std::move(data),
            .prompt_revision = prompt_revision
        };
        
        Print("* currently cached ", extract_keys(_frame_cache), " for ", index);
    }
    
    std::optional<CachedFrame> get_up2date_stored_frame(Frame_t index) {
        std::unique_lock g{_cache_mutex};
        auto it = _frame_cache.find(index);
        if(it == _frame_cache.end())
            return std::nullopt;

        const auto current_revision = _prompt_revisions.contains(index) ? _prompt_revisions.at(index) : uint64_t(0);
        if(it->second.prompt_revision != current_revision) {
            _frame_cache.erase(it);
            return std::nullopt;
        }
        
        auto obj = std::move(it->second);
        Print("* removing cached ", it->first);
        _frame_cache.erase(it);
        return obj;
    }
    
    void cleanup() {
        std::unique_lock g{_cache_mutex};
        _frame_cache.clear();
    }
    
    mutable std::mutex _cache_mutex;
    std::unordered_map<Frame_t, CachedFrame> _frame_cache;
    std::unordered_map<Frame_t, uint64_t> _prompt_revisions;
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

void LiveSegmentation::request_frame(Frame_t index) {
    thread_print("locking _next_frame_mutex");
    std::unique_lock guard{_next_frame_mutex};
    if(not _next_frame
       || _next_frame->index != index)
    {
        _requested_frame = index;
        _next_frame.reset();
    }
    _condition.notify_one();
    _last_requested_frame = index;
    Print("* _last_requested_frame = ", index);
    thread_print("unlocking _next_frame_mutex");
}

void LiveSegmentation::activate() {
    set_thread_name("main");
    Scene::activate();
    // Logic to activate the scene, e.g., initializing framePreloader
    auto source = READ_SETTING(source, file::PathArray);
    Print("Loading source = ", utils::ShortenText(source.toStr(), 1000));
    
    _data = std::make_unique<Data>();
    _data->init([this](Frame_t first_invalid_frame) {
        if(_sam3_session) {
            _sam3_session->invalidate_from(first_invalid_frame);
        }

        if(_current_frame.index.valid() && _current_frame.index == first_invalid_frame) {
            request_frame(first_invalid_frame);
        } else {
            _condition.notify_one();
        }
    });
    
    _video = std::make_unique<VideoSource>(source);
    _video->set_colors(ImageMode::RGB);
    
    video_length = _video->length();
    video_size = _video->size();
    
    SETTING(meta_video_size) = Size2(video_size);
    SETTING(video_size) = Size2(video_size);
    request_frame(0_f);
    
    FindCoord::set_video(video_size);
    
    _terminated = false;
    SETTING(detect_type) = track::detect::ObjectDetectionType_t{track::detect::ObjectDetectionType::sam3};
    //SETTING(detect_resolution) = track::detect::DetectResolution{320};
    
    auto filename = file::DataLocation::parse("output", "sam_output.pv").absolute();
    _data->_output_file = pv::File::Make<pv::FileMode::WRITE>(filename, meta_encoding_t::rgb8);
    
    namespace py = Python;
    auto f = py::schedule([](){
        if(const auto* hooks = track::detect::ensure_backend(track::detect::ObjectDetectionType::sam3); hooks && hooks->init) {
            hooks->init();
        } else {
            throw U_EXCEPTION("SAM3 backend is unavailable.");
        }
    });
    _sam3_session = std::make_unique<track::Sam3InteractiveSession>(
        track::make_python_sam3_interactive_backend(),
        [this](Frame_t replay_frame) {
            auto image = Image::Make(_video->size().height, _video->size().width, 4);
            _video->frame(replay_frame, *image);
            image->set_index(replay_frame.valid() ? replay_frame.get() : -1);
            return make_sam3_tiled_frame(std::move(image));
        });
    
    assert(not _fetch_thread);
    _fetch_thread = std::make_unique<std::thread>(
        [this, f = std::move(f)]() mutable {
            set_thread_name("fetch_thread");
            
            //thread_print("locking _next_frame_mutex");
            std::unique_lock guard{_next_frame_mutex};
            Frame_t index = 0_f;
            VideoFrame frame;
            const auto length = _video->length();
            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
            bool disable_waiting = false;
            std::optional<std::tuple<Frame_t, uint64_t, std::future<track::Sam3ProcessedFrame>>> result;
            
            /// retrieving the ensure_backend / init procedure
            f.get();
            
            while(not _terminated.load()) {
                if(not result
                   && (not _next_frame.has_value()
                       || not frame.index.valid()))
                {
                    bool same_index = not _requested_frame.has_value() || _requested_frame.value() == index;
                    //thread_print("unlocking _next_frame_mutex");
                    guard.unlock();
                    
                    if(_data && same_index) {
                        auto data = _data->get_up2date_stored_frame(index);
                        if(data) {
                            /*std::promise<SegmentationData> prom;
                            result = std::make_tuple(index, prom.get_future());
                            frame = VideoFrame{
                                .image = Image::Make(*data->_data.image),
                                .index = index
                            };
                            prom.set_value(std::move(data->_data));*/
                            //disable_waiting = true;
                            //thread_print("locking _next_frame_mutex");
                            //guard.lock();
                            
                            
                            
                            //SegmentationData data = f.get();
                            //result.reset();
                            
                            //thread_print("locking _next_frame_mutex");
                            guard.lock();
                            
                            Print("* Moving ", frame.index, " (",index,") to _next_frame with ", data->_data.frame.n(), " objects");
                            
                            frame = VideoFrame{
                                .image = Image::Make(*data->_data.image),
                                .index = index
                            };
                            
                            _next_frame = std::move(frame);
                            _next_data = std::move(data->_data);
                            _next_data_revision = data->prompt_revision;
                            
                            if(_previous_frame.has_value())
                                frame = std::move(_previous_frame.value());
                            else
                                frame.image = Image::Make(_video->size().height, _video->size().width, 4);
                            
                            frame.index.invalidate();
                            ++index;
                            
                            if(index >= length)
                                index = 0_f;
                            
                            continue;
                        }
                    }
                    
                    try {
                        /// load a frame into cache
                        if(not frame.image) {
                            frame.image = Image::Make(_video->size().height, _video->size().width, 4);
                        }
                        
                        _video->frame(index, *frame.image);
                        frame.index = index;
                        frame.image->set_index(index.valid() ? index.get() : -1);
                        TileImage tiled = make_sam3_tiled_frame(Image::Make(*frame.image));
                        tiled.callback = [](){
                            Print("Fun is done!");
                        };
                        const auto prompt_revision = _data ? _data->prompt_revision(index) : uint64_t(0);
                        
                        Print("* Moving ", frame.image->index(), " to result");
                        result = std::make_tuple(
                            index,
                            prompt_revision,
                            std::async(std::launch::async,
                                [session = _sam3_session.get(), tiled = std::move(tiled), prompt_revision]() mutable {
                                    return session->process_frame(std::move(tiled), prompt_revision);
                                })
                        );
                        
                        //thread_print("locking _next_frame_mutex");
                        guard.lock();
                    } catch(...) {
                        //thread_print("locking _next_frame_mutex");
                        guard.lock();
                        throw;
                    }
                }
                
                if(not disable_waiting) {
                    //thread_print("waiting locking _next_frame_mutex");
                    _condition.wait(guard, [&](){
                        return _terminated.load()
                        || _requested_frame.has_value()
                        || not _next_frame.has_value()
                        || (result.has_value()
                            && std::get<2>(result.value()).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready);
                    });
                    _condition.wait_for(guard, std::chrono::milliseconds(1));
                } else {
                    //thread_print("short waiting locking _next_frame_mutex");
                    _condition.wait_for(guard, std::chrono::milliseconds(1));
                }
                
                //thread_print("got locking _next_frame_mutex");
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
                    
                    Print("* Requesting frame ", index," over ", frame.index);
                    
                    if(_next_frame.has_value())
                        _next_frame.reset();
                    if(result.has_value()
                       && std::get<0>(result.value()) != index)
                    {
                        Print("* Discarding ", std::get<0>(result.value()), " in loop to get to ", index);
                        
                        //thread_print("unlocking _next_frame_mutex");
                        guard.unlock();
                        try {
                            std::get<2>(result.value()).get(); /// discard
                        } catch(...) {
                            /// Any python errors are ignored here...
                        }
                        //thread_print("locking _next_frame_mutex");
                        guard.lock();
                        
                        result.reset();
                        
                        //disable_waiting = true;
                        continue;
                    }
                }
                
                if(not _next_frame.has_value()) {
                    //thread_print("unlocking _next_frame_mutex");
                    guard.unlock();
                    
                    try {
                        /// load new frame
                        auto [loaded, expected_revision, f] = std::move(result.value());
                        if(not f.valid()) {
                            _requested_frame = loaded;
                            result.reset();
                            disable_waiting = true;
                            //thread_print("locking _next_frame_mutex");
                            guard.lock();
                            Print("* Future invalid for ", loaded, " re-requesting frame");
                            continue;
                        }
                        assert(f.valid());
                        auto processed = f.get();
                        result.reset();
                        
                        //thread_print("locking _next_frame_mutex");
                        guard.lock();
                        const auto current_revision = _data ? _data->prompt_revision(loaded) : expected_revision;
                        if(current_revision != expected_revision) {
                            Print("* Dropping stale result for ", loaded, " revision=", expected_revision, " current=", current_revision);
                            if(_sam3_session) {
                                _sam3_session->invalidate_from(loaded);
                            }
                            frame.index.invalidate();
                            ++index;
                            if(index >= length)
                                index = 0_f;
                            continue;
                        }
                        if(_sam3_session
                           && not _sam3_session->commit_frame(std::move(processed)))
                        {
                            Print("* Dropping invalidated result for ", loaded, " after session generation changed");
                            frame.index.invalidate();
                            ++index;
                            if(index >= length)
                                index = 0_f;
                            continue;
                        }
                        
                        Print("* Moving ", frame.index, " (",loaded,") to _next_frame with ", processed.data.frame.n(), " objects");
                        
                        //buffers::TileBuffers::get().move_back(std::move(processed.data.image));

                        _next_frame = std::move(frame);
                        _next_data = std::move(processed.data);
                        _next_data_revision = expected_revision;
                        
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
                        //thread_print("locking _next_frame_mutex");
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
        _next_data_revision.reset();
        _current_data.reset();
        _current_data_revision.reset();
        _next_frame.reset();
        _previous_frame.reset();
        _current_frame.image = nullptr;
        _current_frame.index.invalidate();
        _video = nullptr;
        _data = nullptr;
        _sam3_session = nullptr;
    }

    namespace py = Python;
    py::schedule([]() {
        if(const auto* hooks = track::detect::ensure_backend(track::detect::ObjectDetectionType::sam3); hooks && hooks->deinit) {
            hooks->deinit();
        }
    }).get();
    
    /// clear out remaining tilebuffer images
    buffers::TileBuffers::clear();
}

// Custom drawing implementation
void LiveSegmentation::_draw(DrawStructure& graph) {
    if(not _data)
        return;
    
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
                    VarFunc("annotations", [this](const VarProps&) -> std::vector<glz::json_t> {
                        std::vector<glz::json_t> result;
                        auto detect_sam3_prompt = READ_SETTING_WITH_DEFAULT(detect_sam3_prompt, std::optional<track::detect::Sam3Prompts>{});
                        if(not detect_sam3_prompt)
                            return result;
                        
                        if(_current_frame.index.valid()) {
                            auto it = detect_sam3_prompt->find(_current_frame.index);
                            if(it != detect_sam3_prompt->end()) {
                                for(auto& prompt : it->second) {
                                    if(prompt.type() == track::detect::Sam3PromptType::boxes)
                                        result.emplace_back(prompt.to_json());
                                }
                            }
                        }
                        
                        return result;
                    }),
                    VarFunc("data", [this](const VarProps&) -> std::vector<glz::json_t> {
                        std::vector<glz::json_t> result;
                        if(std::unique_lock guard{_next_frame_mutex};
                           _current_data)
                        {
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
                        request_frame(Meta::fromStr<Frame_t>(action.parameters.front()));
                    }),
                    ActionFunc("set_playback", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _playback = Meta::fromStr<bool>(action.parameters.front());
                        Print("Playback = ", _playback.load());
                    })
                };
                
                context.custom_elements["label"] = std::unique_ptr<CustomElement>(
                    new LabelElement(&_data->unassigned_labels, &_data->labels, &_data->dt)
                );
                context.custom_elements["image_generator"] = std::unique_ptr<CustomElement>(
                    new ImageDisplayElement(&_image_generators)
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
                            auto meta_encoding = READ_SETTING_WITH_DEFAULT(meta_encoding, meta_encoding_t::gray);
                            Background bg(FindCoord::get().video_size(), meta_encoding);
                            
                            
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
                _image_generators.register_generator("blob_image", std::move(generator));
                return context;
            }(),
            .base = window()
        };
    }
    
    if(_playback.load()
       || _last_requested_frame)
    {
        //thread_print("locking _next_frame_mutex");
        std::unique_lock guard{_next_frame_mutex};
        if(_next_frame.has_value()) {
            if(_last_requested_frame
               && _next_frame->index != _last_requested_frame)
            {
                /// discard
                Print("* Discarding ", _next_frame->index, " as its not ", _last_requested_frame);
                _previous_frame = std::move(_next_frame);
                _next_data.reset();
                _next_data_revision.reset();
                _next_frame.reset();
            } else {
                Print("* Accepting ", _next_frame->index," as next visible frame");
                _current_frame.image = _current_image->update_with(nullptr);
                _previous_frame = std::move(_current_frame);
                _current_frame = std::move(_next_frame.value());
                _current_image->set_source(std::move(_current_frame.image));
                if(_current_data)
                    _data->store_frame_if_annotated(std::move(_current_data.value()), _current_data_revision.value_or(0));
                _current_data = std::move(_next_data);
                _current_data_revision = _next_data_revision;
                _next_data.reset();
                _next_data_revision.reset();
                _next_frame.reset();
                _image_generators.reset_generator("blob_image");
                _last_requested_frame.reset();
            }
        }
        _condition.notify_one();
        //thread_print("unlocking _next_frame_mutex");
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
        
        
        if(_drag_box) {
            graph.circle(Loc{_drag_box->pos()}, Radius{5}, FillClr{Red.alpha(125)});
            graph.wrap_object(*_drag_box);
        }
        
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
        auto coords = FindCoord::get();
        auto p = coords.convert(HUDCoord(event.move.x, event.move.y));
        
        if(not _drag_box)
            _drag_box = std::make_unique<Rect>(Loc{p});
        
        auto pos = _drag_box->pos();
        _drag_box->create(Size{p - pos + Vec2(1)}, FillClr{Red.alpha(50)});
        
        
        
    }

    if(event.type == EventType::MBUTTON
       && event.mbutton.button == 0
       && not event.mbutton.pressed)
    {
        if(_drag_box
           && _drag_box->size().max() >= 10)
        {
            Print("Creating polygon at ", _drag_box->bounds(), " for frame ", _current_frame.index);
            auto detect_sam3_prompt = READ_SETTING_WITH_DEFAULT(detect_sam3_prompt, std::optional<track::detect::Sam3Prompts>{});
            if(not detect_sam3_prompt)
                detect_sam3_prompt = track::detect::Sam3Prompts{};
            
            (*detect_sam3_prompt)[_current_frame.index].emplace_back(std::vector<Bounds>{_drag_box->bounds()});
            SETTING(detect_sam3_prompt) = detect_sam3_prompt;
            if(_data) {
                (void)_data->bump_prompt_revision(_current_frame.index);
            }
            if(_sam3_session) {
                _sam3_session->invalidate_from(_current_frame.index + 1_f);
            }
            _drag_box = nullptr;
            
            request_frame(_current_frame.index);
        }
        
        auto p = Vec2(event.mbutton.x, event.mbutton.y);
        auto coord = FindCoord::get();
        auto original = p;
        p = coord.convert(HUDCoord(p));
        
        if(p.x >= 0 && p.y >= 0 && p.x < coord.video_size().width && p.y < coord.video_size().height) {
            Print("adding point at ", original, " => ", p);
        } else
            Print("not adding point at ", original, " => ", p);
        
    }
    if(not graph->is_mouse_down(0)
       && _drag_box)
    {
        _drag_box = nullptr;
    }
    
    if(event.type == EventType::MBUTTON
              && event.mbutton.button == 1
              && event.mbutton.pressed)
    {
        //_drag_box = nullptr;
    }
    
    if(event.type == EventType::KEY
       && event.key.pressed
       && not graph->selected_object())
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
                    Print("Reverting to frame ", _current_frame.index.try_sub(1_f));
                    request_frame(_current_frame.index.try_sub(1_f));
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
                    request_frame(_current_frame.index + 1_f);
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
                    request_frame(0_f);
                }
                break;

                
            default:
                break;
        }
    }
    
    return false; // Return true if the event is handled
}

}
