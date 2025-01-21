#include "ConvertScene.h"
#include <gui/IMGUIBase.h>
#include <video/VideoSource.h>
#include <file/DataLocation.h>
#include <file/Path.h>
#include <grabber/misc/Camera.h>
#include <grabber/misc/Webcam.h>
#include <grabber/misc/PylonCamera.h>
#include <grabber/misc/default_config.h>
#include <tracking/IndividualManager.h>
#include <misc/ThreadManager.h>
#include <misc/RecentItems.h>
#include <file/PathArray.h>
#include <misc/Coordinates.h>
#include <python/GPURecognition.h>
#include <gui/Label.h>
#include <gui/ParseLayoutTypes.h>
#include <python/YOLO.h>
#include <misc/CommandLine.h>
#include <gui/dyn/Action.h>
#include <gui/dyn/ParseText.h>
#include <python/TileBuffers.h>
#include <misc/DetectionTypes.h>
#include <tracking/Individual.h>
#include <tracking/LockGuard.h>
#include <tracking/Segmenter.h>
#include <gui/ScreenRecorder.h>
#include <gui/DynamicGUI.h>
#include <misc/SettingsInitializer.h>
#include <misc/PythonWrapper.h>
#include <gui/WorkProgress.h>
#include <misc/CTCollection.h>
#include <misc/ObjectCache.h>
#include <gui/GuiSettings.h>
#include <gui/PreviewAdapterElement.h>
#include <tracking/FilterCache.h>

namespace cmn::gui {
using namespace dyn;
using Skeleton = blob::Pose::Skeleton;

class LabelWrapper;
using LabelCache_t = ObjectCache<Label, 100, std::shared_ptr>;

class LabelWrapper : public Layout {
    std::shared_ptr<Label> _label;
    LabelCache_t* _cache;
    
public:
    LabelWrapper(LabelCache_t& cache, std::shared_ptr<Label>&& label)
        : _label(std::move(label)), _cache(&cache)
    {
        set_children({Layout::Ptr(_label)});
    }
    
    LabelWrapper(LabelWrapper&) = delete;
    LabelWrapper(LabelWrapper&&) = default;
    LabelWrapper& operator=(LabelWrapper&) = delete;
    LabelWrapper& operator=(LabelWrapper&&) = default;
    
    Label* label() const { return _label.get(); }
    
    ~LabelWrapper() {
        _cache->returnObject(std::move(_label));
    }
};

uint64_t interleaveBits(const Vec2& pos) {
    uint32_t x(pos.x), y(pos.y);
    uint64_t z = 0;
    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; ++i) {
        z |= (x & (1ULL << i)) << i | (y & (1ULL << i)) << (i + 1);
    }
    return z;
}

struct ConvertScene::Data {
    Segmenter* _segmenter{nullptr};
    LabelCache_t _unassigned_labels;

    // External images for background and overlay
    std::shared_ptr<ExternalImage> _background_image = std::make_shared<ExternalImage>(),
                                   _overlay_image = std::make_shared<ExternalImage>();

    // Vectors for object blobs and GUI objects
    std::vector<pv::BlobPtr> _object_blobs, _tmp_store_blobs;
    SegmentationData _current_data;
    SegmentationData _tmp_store_data;
    PPFrame _tmp_store_frame, _current_frame;
    
    std::map<Idx_t, std::shared_ptr<constraints::FilterCache>> filter_cache;
    std::map<Idx_t, BdxAndPred> fish_selected_blobs;
    
    CallbackCollection callback;
    Skeleton skelet;
    bool closed_loop_enable{SETTING(closed_loop_enable)};
    file::Path closed_loop_path{SETTING(closed_loop_path).value<file::Path>().remove_extension()};
    
    std::mutex _current_json_mutex;
    glz::json_t _current_json;

    // Individual properties for each object
    sprite::Map _primary_selection;
    std::vector<Vec2> _zoom_targets;
    std::vector<std::shared_ptr<VarBase_t>> _untracked_gui, _tracked_gui;
    std::map<Idx_t, sprite::Map> _individual_properties;
    std::vector<sprite::Map> _untracked_properties;
    std::vector<sprite::Map*> _tracked_properties;
    std::unordered_map<pv::bid, Identity> _visible_bdx;
    std::vector<std::vector<Vertex>> _trajectories;
    std::vector<std::tuple<Color, std::vector<Vec2>>> _postures;
    
    std::vector<Idx_t> _inactive_ids;
    std::vector<Idx_t> _active_ids;

    std::unordered_map<Idx_t, std::shared_ptr<Label>> _labels;
    std::vector<std::unique_ptr<Skelett>> _skeletts;
    std::unordered_map<Idx_t, std::tuple<Frame_t, Bounds>> _last_bounds;
    
    ScreenRecorder _recorder;
    
    ind::ProgressBar bar{
        ind::option::BarWidth{50},
        ind::option::Start{"["},
/*#ifndef _WIN32
        ind::option::Fill{"█"},
        ind::option::Lead{"▂"},
        ind::option::Remainder{"▁"},
#else*/
        ind::option::Fill{"="},
        ind::option::Lead{">"},
        ind::option::Remainder{" "},
//#endif
        ind::option::End{"]"},
        ind::option::PostfixText{"Converting video..."},
        ind::option::ShowPercentage{true},
        ind::option::ForegroundColor{ind::Color::white},
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
    };

    ind::ProgressSpinner spinner{
        ind::option::PostfixText{""},
        ind::option::ForegroundColor{ind::Color::white},
#ifndef _WIN32
        ind::option::SpinnerStates{std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
#else
        ind::option::SpinnerStates{std::vector<std::string>{".","..","..."}},
#endif
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
    };
    
    double dt = 0;
    std::atomic<double> _time{0};
    std::unique_ptr<Bowl> _bowl;
    
    Size2 output_size;
    Size2 video_size;
    Vec2 _last_mouse;
    
    // Frame data
    Frame_t _actual_frame;
    Frame_t _video_frame;
    
    dyn::DynamicGUI dynGUI;
    std::future<std::optional<std::set<std::string_view>>> _retrieve_video_info;
    std::set<std::string_view> _recovered_error;
    
    Frame_t last_frame;
    Timer timer;
    
    Data() {
        reset_properties();
    }
    
    void reset_properties() {
        _primary_selection["color"] = Transparent;
        _primary_selection["bdx"] = pv::bid();
        _primary_selection["fdx"] = Idx_t();
        _primary_selection["visible"] = false;
        _primary_selection["tracked"] = false;
        _primary_selection["speed"] = 0.f;
        _primary_selection["has_neighbor"] = false;
        _primary_selection["p"] = 0.0f;
        _primary_selection["ps"] = std::vector<std::tuple<double, double, double>>{};
        
        for (auto& [id, map] : _individual_properties) {
            map["visible"] = false;
        }
    }
    
    void drawBlobs(DrawStructure&, Frame_t, const std::map<uint16_t, std::string_view>& detect_classes, const Vec2& scale, Vec2 offset, const std::unordered_map<pv::bid, Identity>& visible_bdx, bool dirty);
    // Helper function to draw outlines
    void drawOutlines(DrawStructure& graph, const Size2& scale, Vec2 offset);
    void paint_blob_prediction(DrawStructure& graph, const Color& tracked_color, const pv::Blob& blob) {
        if(blob.prediction().valid()
           && blob.prediction().outlines.has_holes())
        {
            for(auto &line : blob.prediction().outlines.lines) {
                std::vector<Vec2> v = line;
                graph.line(gui::Line::Points_t{ v }, gui::LineClr{ tracked_color.alpha(150) });
            }
        }
    }
    void check_video_info(bool wait, std::set<std::string_view>*);
    void retry_video_info();
    
    bool fetch_new_data();
    void update_background_image();
    
    void check_module() {
        if(not closed_loop_enable)
            return;
        
        Python::schedule([this](){
            ModuleProxy proxy(closed_loop_path.str(), [this](ModuleProxy& m) {
                m.set_function<std::function<glz::json_t()>>("frame_info", [this]() {
                    std::unique_lock guard{_current_json_mutex};
                    return _current_json;
                });
                m.run("init");
            }, false, [](ModuleProxy& m){
                m.unset_function("frame_info");
                m.run("deinit");
            });
        });
    }
    
    dyn::DynamicGUI init_gui(Base* window);
    void check_gui(DrawStructure& graph, Base* window) {
        if(not dynGUI) {
            dynGUI = init_gui(window);
            
            dynGUI.context.custom_elements["preview"] = std::unique_ptr<CustomElement>(new PreviewAdapterElement([this]() -> const track::PPFrame*
            {
                return &_current_frame;
                
            }, [this](Idx_t fdx) -> std::tuple<const constraints::FilterCache*, std::optional<BdxAndPred>>
            {
                const constraints::FilterCache* filters{nullptr};
                if(auto it = filter_cache.find(fdx);
                   it != filter_cache.end())
                {
                    filters = it->second.get();
                }
                
                if(auto it = fish_selected_blobs.find(fdx);
                   it != fish_selected_blobs.end())
                {
                    return {filters, it->second.clone()};
                }
                
                return {filters, std::nullopt};
            }));
            dynGUI.context.custom_elements["label"] = std::unique_ptr<CustomElement>(
              new CustomElement {
                "label",
                [this](LayoutContext& layout) -> Layout::Ptr {
                    std::shared_ptr<Label> ptr;
                    auto text = layout.get<std::string>("", "text");
                    auto center = layout.get<Vec2>(Vec2(), "center");
                    auto line_length = layout.get<float>(float(60), "length");
                    auto id = layout.get<Idx_t>(Idx_t(), "id");
                    auto color = layout.textClr;
                    auto line = layout.line;
                    auto fill = layout.fill;

                    if (id.valid()) {
                        auto it = _labels.find(id);
                        if (it != _labels.end()) {
                            ptr = it->second;
                        } else
                            _labels[id] = ptr = _unassigned_labels.getObject();
                        
                    } else {
                        ptr = _unassigned_labels.getObject();
                    }

                    if (not ptr)
                        throw RuntimeError("Apparently out of memory generating label ", text, ".");

                    ptr->set_line_length(line_length);
                    ptr->set_data(0_f, text, Bounds(layout.pos, layout.size), center);
                    auto font = parse_font(layout.obj, layout._defaults.font);
                    ptr->text()->set(font);
                    ptr->text()->set(color);
                    ptr->set(FillClr{ fill });
                    ptr->set_line_color(line);
                    
                    if(not id.valid())
                        ptr->set_uninitialized();
                    
                    return Layout::Ptr(std::make_shared<LabelWrapper>(_unassigned_labels, std::move(ptr)));
                },
                [this](Layout::Ptr& o, const Context& context, State& state, const auto& patterns) -> bool
                {
                    //Print("Updating label with patterns: ", patterns);
                    //Print("o = ", o.get());

                    Idx_t id;
                    if (patterns.contains("id"))
                        id = Meta::fromStr<Idx_t>(parse_text(patterns.at("id").original, context, state));
                    
                    if (id.valid()) {
                        if (auto it = _labels.find(id);
                            it != _labels.end())
                        {
                            if(it->second.get() != o.get())
                                o = Layout::Ptr(it->second);
                        }
                    }

                    Label* p;
                    if(o.is<LabelWrapper>()) {
                        p = o.to<LabelWrapper>()->label();
                    } else
                        p = o.to<Label>();
                    
                    if(not id.valid()) {
                        p->set_uninitialized();
                    }
                    
                    auto source = p->source();
                    using namespace cmn::ct;
                    
                    CTCollection map{
                        Key<"text", "pos", "size", "center", "line", "fill", "color">{},
                        p->text()->text(),
                        Vec2(source.pos()),
                        source.size(),
                        source.pos() + Vec2(source.width, source.height) * 0.5_F,
                        p->line_color(),
                        p->fill_color(),
                        TextClr{p->text()->text_color()}
                    };
                    
                    map.apply([&](std::string_view key, auto& value) {
                        value = parse_value_with_default(value, key, patterns, context, state);
                    });
                    
                    source = Bounds{ map.get<"pos">(), map.get<"size">() };
                    p->set_line_color(map.get<"line">());
                    p->set_fill_color(map.get<"fill">());
                    p->text()->set(map.get<"color">());
                    
                    p->set_data(0_f,
                                map.get<"text">(),
                                source,
                                map.get<"center">());
                    
                    p->update(FindCoord::get(), 1, 1, false, dt, Scale{1});
                    
                    return true;
                }
              });
        }
        
        dynGUI.update(graph, nullptr);
    }
    void draw(bool dirty, DrawStructure& graph, Base* window);
    bool retrieve_and_prepare_data();
    void draw_scene(DrawStructure& graph, const detect::yolo::names::map_t& detect_classes, bool dirty);
};

Segmenter& ConvertScene::segmenter() const {
    if(not _data || not _data->_segmenter)
        throw U_EXCEPTION("No segmenter exists.");
    return *_data->_segmenter;
}

sprite::Map ConvertScene::fish = []() {
    sprite::Map fish;
    fish.set_print_by_default(false);
    fish["name"] = std::string("fish0");
    fish["color"] = Red;
    fish["pos"] = Vec2(100, 150);
    return fish;
}();

sprite::Map ConvertScene::_video_info = []() {
    sprite::Map fish;
    fish.set_print_by_default(false);
    fish["frame"] = Frame_t();
    fish["resolution"] = Size2();
    return fish;
}();

//Size2 ConvertScene::output_size() const {
//    return segmenter().output_size();
//}

ConvertScene::ConvertScene(Base& window, std::function<void(ConvertScene&)> on_activate, std::function<void(ConvertScene&)> on_deactivate) : Scene(window, "convert-scene",
    [this](Scene&, DrawStructure& graph) {
        _draw(graph);
    }),
/*menu{
    ,
    [this](const std::string& name) {
        if (name == "gui_frame") {
            this->segmenter().reset(SETTING(gui_frame).value<Frame_t>());
        }
    }
},*/
_on_activate(on_activate),
_on_deactivate(on_deactivate)

{ }

ConvertScene::~ConvertScene() {
    if (_data && _data->_segmenter)
        deactivate();
    else if(_scene_active.valid()) {
        auto status = _scene_active.wait_for(std::chrono::seconds(0));
        if (status == std::future_status::deferred) {
            std::cout << "Task has been deferred.\n";
        } else if (status == std::future_status::ready) {
            std::cout << "Task has finished.\n";
        } else {
            std::cout << "Task is still running.\n";
            _scene_active.wait();
        }
    }
}

void ConvertScene::set_segmenter(Segmenter* seg) {
    if(not _data)
        _data = std::make_unique<Data>();
    
    assert(_data->_segmenter == nullptr);
    _data->_segmenter = seg;
    if(seg) {
        seg->set_progress_callback([this](float percent){
            if(std::isnan(percent)
               || std::isinf(percent))
            {
                _data->spinner.tick();
                static std::once_flag flag;
                std::call_once(flag, [](){
                    FormatWarning("Percent is infinity.");
                });
                return;
            }
            
            if(percent >= 0)
                _data->bar.set_progress(percent);
            else if(last_tick.elapsed() > 1) {
                _data->spinner.set_option(ind::option::PrefixText{"Recording ("+Meta::toStr(_data->_video_frame)+")"});
                _data->spinner.tick();
                last_tick.reset();
            }
        });
    }
}

void ConvertScene::deactivate() {
    try {
        if(_data)
            GlobalSettings::map().unregister_callbacks(std::move(_data->callback));

        if(_data && _data->_recorder.recording())
            _data->_recorder.stop_recording(window(), nullptr);

        _data->bar.set_progress(100);
        _data->bar.mark_as_completed();
        
        _data->spinner.set_option(ind::option::ForegroundColor{ind::Color::green});
        _data->spinner.set_option(ind::option::PrefixText{"✔"});
        _data->spinner.set_option(ind::option::ShowSpinner{false});
        _data->spinner.set_option(ind::option::PostfixText{"Done."});
        _data->spinner.set_progress(100);
        _data->spinner.mark_as_completed();
        
        if(_data->closed_loop_enable) {
            Python::schedule([this](){
                ModuleProxy proxy(_data->closed_loop_path.str(), [](auto&){});
                proxy.run("deinit");
            }).get();
        }
        
        if (_data) {
            _data->_object_blobs.clear();
            _data->_current_data = {};
            _data->dynGUI.clear();
        }
        /// save the last settings used
        RecentItems::open(SETTING(source).value<file::PathArray>(), GlobalSettings::current_defaults_with_config());
        
        if(_data && _data->_segmenter)
            segmenter().force_stop();
        if(_data)
            _data->check_video_info(true, nullptr);
        
        if(_on_deactivate)
            _on_deactivate(*this);
        
        _data = nullptr;
        _scene_promise.set_value();
        
    } catch(const std::exception& e){
        FormatExcept(e.what());
        _scene_promise.set_exception(std::current_exception());
    }
    
    WorkProgress::stop();
}

void ConvertScene::Data::check_video_info(bool wait, std::set<std::string_view>* result) {
    if(_retrieve_video_info.valid()
       && (wait || _retrieve_video_info.wait_for(std::chrono::milliseconds(0)) 
                        == std::future_status::ready))
    {
        try {
            // invalidate existing future and throw away
            if(result) {
                auto r = _retrieve_video_info.get();
                if(r)
                    *result = *r;
                //else
                //    result->clear();
            } else {
                (void)_retrieve_video_info.get();
            }
            
            if(not wait) {
                retry_video_info();
            }
            
        } catch(const std::exception& e) {
            FormatError("There was in error retrieving video info from the future: ", e.what());
        }
    }
}

void ConvertScene::open_video() {
    _data->bar.set_progress(0);
    _data->bar.set_option(ind::option::ShowPercentage{true});
    segmenter().open_video();
    
    _video_info["resolution"] = segmenter().size();
    _video_info["length"] = segmenter().video_length();
}

void ConvertScene::open_camera() {
    if(not track::detect::yolo::valid_model(SETTING(detect_model).value<file::Path>()))
    {
        SETTING(detect_model) = file::Path(track::detect::yolo::default_model());
    }
    
    if(not GlobalSettings::current_defaults_with_config().has("save_raw_movie"))
    {
        SETTING(save_raw_movie) = true;
    }
    
    _data->spinner.set_option(ind::option::PrefixText{"Recording"});
    _data->spinner.set_option(ind::option::ShowPercentage{false});
    
    segmenter().open_camera();
    
    _video_info["resolution"] = segmenter().size();
    _video_info["length"] = segmenter().video_length();
}

glz::json_t sprite_map_to_json(const sprite::Map& map) {
    glz::json_t json;
    for(auto& key : map.keys()) {
        auto &prop = map.at(key).get();
        json[key] = prop.to_json();
        
    }
    return json;
}

void ConvertScene::activate()  {
    _scene_promise = {};
    _scene_active = _scene_promise.get_future().share();

    if(_on_activate)
        _on_activate(*this);

    GlobalSettings::map().set_print_by_default(true);
    
    auto source = SETTING(source).value<file::PathArray>();
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path(settings::find_output_name(GlobalSettings::map()));
    }
    
    Print("Loading source = ", source);
    SETTING(meta_source_path) = source.source();
    try {

        if (source == file::PathArray("webcam"))
            open_camera();
        else
            open_video();
    }
    catch (const std::exception& ex) {
        FormatExcept(ex.what());
		_scene_promise.set_exception(std::current_exception());
        _scene_promise = {};
        segmenter().error_stop(ex.what());
        throw;
    }

    if(not _data)
        _data = std::make_unique<Data>();

    _data->skelet = SETTING(detect_skeleton).value<Skeleton>();
    _data->callback = GlobalSettings::map().register_callbacks({
        "detect_skeleton"
    }, [this](auto) {
        SceneManager::getInstance().enqueue([this](auto,auto&) {
            _data->skelet = SETTING(detect_skeleton).value<Skeleton>();
            _data->_skeletts.clear();
        });
    });

    _data->video_size = _video_info["resolution"].value<Size2>();
    if(_data->video_size.empty()) {
        _data->video_size = Size2(640,480);
        FormatError("Cannot determine size of the video input. Defaulting to ", _data->video_size, ".");
    }
    
    _data->output_size = _data->_segmenter->output_size();
    buffers::TileBuffers::get().set_image_size(detect::get_model_image_size());
    
    window()->set_title(window_title());
    _data->bar.set_progress(0);
    
    SceneManager::getInstance().enqueue([this](IMGUIBase*, DrawStructure& graph) {
        if(not _data || not _data->_segmenter)
            return;
        if(not _data->_segmenter->output_size().empty())
            graph.set_size(Size2(1024, _data->_segmenter->output_size().height / _data->_segmenter->output_size().width * 1024));
    });
    
    auto range = SETTING(video_conversion_range).value<Range<long_t>>();
    if (range.start == -1 && range.end == -1) {
        if(segmenter().is_finite())
            SETTING(video_conversion_range) = Range<long_t>(0, segmenter().video_length().get());
        else
            SETTING(video_conversion_range) = Range<long_t>(-1,-1);
    }
    else if(range.start >= 0) {
        SETTING(gui_frame) = Frame_t(range.start);
    }
    
    segmenter().start();
    RecentItems::open(source, GlobalSettings::current_defaults_with_config());
    
    if(auto path = _data->closed_loop_path.add_extension("py");
       _data->closed_loop_enable)
    {
        if(not path.is_regular())
            throw U_EXCEPTION("Cannot find module ", path, " as is specified in `closed_loop_path`.");
        
        Print("Loading closed_loop module at ", path.absolute().add_extension("py"));
        
        Python::schedule([this](){
            ModuleProxy proxy(_data->closed_loop_path.str(), [this](ModuleProxy& m) {
                m.set_function<std::function<glz::json_t()>>("frame_info", [this]() {
                    std::unique_lock guard{_data->_current_json_mutex};
                    return _data->_current_json;
                });
                m.run("init");
            });
        });
    }
}

bool ConvertScene::on_global_event(Event e) {
    if(e.type == EventType::KEY
        && e.key.pressed) 
    {
        switch(e.key.code) {
            case Keyboard::T:
                SETTING(gui_show_texts) = not SETTING(gui_show_texts);
                break;
                
            case Keyboard::R:
                if(not _data)
                    return true;
                
                if (not _data->_recorder.recording()) {
                    _data->_recorder.start_recording(window(), {});
                }
                else {
                    _data->_recorder.stop_recording(window(), nullptr);
                }
                return true;
            default:
                break; /// we dont need anything else
        }
    }
    return true;
}

bool ConvertScene::Data::fetch_new_data() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        set_thread_name("GUI");
    });
    
    check_video_info(false, &_recovered_error);
    
    bool dirty = false;
    auto&& [data, frame, obj] = _segmenter->grab();
    if(data.image) {
        if(_tmp_store_data.image)
            _segmenter->overlayed_video()->source()->move_back(std::move(_tmp_store_data.image));
        
        _tmp_store_data = std::move(data);
        _tmp_store_frame = std::move(frame);
        _tmp_store_blobs = std::move(obj);
        dirty = true;
        
        std::sort(_tmp_store_blobs.begin(), _tmp_store_blobs.end(), [](auto& A, auto& B) {
            return interleaveBits(A->bounds().center()) < interleaveBits(B->bounds().center());
        });
    }
    
    return dirty;
}

// Helper function to calculate window dimensions
Size2 ConvertScene::calculateWindowSize(const Size2& output_size, const Size2& window_size) {
    auto ratio = output_size.width / output_size.height;
    Size2 wdim;

    if (window_size.width * output_size.height < window_size.height * output_size.width) {
        wdim = Size2(window_size.width, window_size.width / ratio);
    }
    else {
        wdim = Size2(window_size.height * ratio, window_size.height);
    }

    return wdim;
}

// Helper function to draw outlines
void ConvertScene::Data::drawOutlines(DrawStructure&, const Size2&, Vec2) {
    /*if (not _current_data.outlines.empty()) {
        //graph.text(Str(Meta::toStr(_current_data.outlines.size()) + " lines"), attr::Loc(10, 50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));

        ColorWheel wheel;
        for (const auto& v : _current_data.outlines) {
            auto clr = wheel.next();
            graph.line(Line::Points_t{ v }, LineClr{ clr.alpha(150) });
        }
    }*/
}

void ConvertScene::Data::drawBlobs(
    DrawStructure& graph,
    Frame_t frameIndex,
    const detect::yolo::names::map_t& detect_classes,
    const Vec2&, Vec2,
    const std::unordered_map<pv::bid, Identity>& visible_bdx, 
    bool dirty) 
{
    //size_t i = 0;
    size_t untracked = 0;

    auto coords = FindCoord::get();
    std::set<Idx_t> tracked_ids;
    reset_properties();

    auto selected_ids = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
    const bool is_background_subtraction = track::detect::detection_type() == track::detect::ObjectDetectionType::background_subtraction;
    
    std::vector<glz::json_t> acc_json;
    _zoom_targets.clear();

    for (auto& blob : _object_blobs) {
        auto bds = blob->bounds();
        bds = coords.convert(BowlRect(bds));

        Idx_t tracked_id;
        Color tracked_color;

        if (contains(visible_bdx, blob->blob_id())) {
            auto id = visible_bdx.at(blob->blob_id());
            tracked_color = id.color();
            tracked_id = id.ID();
        }
        else if (blob->parent_id().valid() && contains(visible_bdx, blob->parent_id()))
        {
            auto id = visible_bdx.at(blob->parent_id());
            tracked_color = id.color();
            tracked_id = id.ID();
        }
        else {
            tracked_color = Gray;
        }

        SegmentationData::Assignment assign{
            .clid = size_t(-1)
        };
        Vec2 first_pose = bds.center();
        blob::Pose pose;

        if (_current_data.frame.index().valid()) {
            if (blob->prediction().valid()) {
                auto pred = blob->prediction();
                assign = {
                    .clid = pred.clid,
                    .p = static_cast<float>(pred.p) / 255.f
                };
                if (pred.pose.size() > 0) {
                    auto& pt = pred.pose.point(0);
                    if(pt.x > 0 || pt.y > 0) {
                        first_pose = coords.convert(BowlCoord(pt));
                    }
                    
                    pose = pred.pose;
                }
            }
            //else
            //    Print("[draw]4 blob ", blob->blob_id(), " prediction not found...");
        }
        
        auto cname = [&]() -> std::string {
            if(assign.clid == size_t(-1)) {
                return is_background_subtraction
                        ? detect_classes.contains(0)
                            ? (std::string)detect_classes.at(0)
                            : FAST_SETTING(individual_prefix)
                        : "<no prediction>";
            } else if(auto it = detect_classes.find(assign.clid);
                      it != detect_classes.end())
            {
                return (std::string)it->second;
                
            } else {
                return "<unknown:" + Meta::toStr(assign.clid) + ">";
            }
        }();

        sprite::Map* tmp = nullptr;

        if (tracked_id.valid()) {
            tmp = &_individual_properties[tracked_id];
            tracked_ids.insert(tracked_id);
        }
        else {
            if (untracked >= _untracked_properties.size()) {
                _untracked_properties.emplace_back();
            }

            if (untracked >= _untracked_gui.size())
                _untracked_gui.emplace_back(new Variable([&, i = untracked](const VarProps&) -> sprite::Map& {
                    //Print("for ", props, " returning value of ", i, " / ", _individual_properties.size());
                    return _untracked_properties.at(i);
                }));

            tmp = &_untracked_properties[untracked++];
        }
        
        bool selected = contains(selected_ids, tracked_id);
        (*tmp)["pose"] = pose;
        (*tmp)["box"] = blob->bounds();
        (*tmp)["pos"] = bds.pos();
        (*tmp)["selected"] = selected;
        (*tmp)["bdx"] = blob->blob_id();
        (*tmp)["center"] = first_pose;
        (*tmp)["tracked"] = tracked_id.valid() ? true : false;
        (*tmp)["color"] = tracked_color;
        (*tmp)["fdx"] = tracked_id;
		(*tmp)["visible"] = true;
        (*tmp)["size"] = Size2(bds.size());
        (*tmp)["radius"] = bds.size().length() * 0.5;
        (*tmp)["type"] = std::string(cname);
        (*tmp)["speed"] = 0.f;
        (*tmp)["has_neighbor"] = false;
        //if(Tracker::instance() && Tracker::background())
        //    (*tmp)["px"] = blob->recount(FAST_SETTING(track_threshold), *Tracker::background());
        //else
            (*tmp)["px"] = blob->recount(-1);

        (*tmp)["p"] = assign.p;
        (*tmp)["ps"] = std::vector<std::tuple<double, double, double>>{};
        
        if(not selected_ids.empty()
           && selected_ids.front() == tracked_id)
        {
            _primary_selection = *tmp;
        }
        
        /// save for closed loop
        if(closed_loop_enable)
            acc_json.push_back(sprite_map_to_json(*tmp));

        if (tracked_id.valid() 
            && selected) 
        {
            if (blob) {
                auto bds = blob->bounds();
                _zoom_targets.push_back(bds.pos());
                _zoom_targets.push_back(bds.pos() + bds.size());
                _zoom_targets.push_back(bds.pos() + bds.size().mul(0, 1));
                _zoom_targets.push_back(bds.pos() + bds.size().mul(1, 0));
                _last_bounds[tracked_id] = { frameIndex, bds };
                selected_ids.erase(std::find(selected_ids.begin(), selected_ids.end(), tracked_id));
            }
        }
        
        if(blob && dirty)
            paint_blob_prediction(graph, tracked_color, *blob);
    }

    size_t tracked = 0;
    for (auto& [id, map] : _individual_properties) {
        if(_tracked_properties.size() <= tracked)
            _tracked_properties.emplace_back(&_individual_properties[id]);
        else
            _tracked_properties[tracked] = &_individual_properties[id];

        if (tracked >= _tracked_gui.size())
            _tracked_gui.emplace_back(new Variable([&, i = tracked](const VarProps&) -> sprite::Map& {
                //Print("for ", props, " returning value of ", i, " / ", _individual_properties.size());
                return *_tracked_properties.at(i);
            }));

        tracked++;
    }

    if (untracked < _untracked_gui.size())
        _untracked_gui.resize(untracked);
    if (untracked < _untracked_properties.size())
        _untracked_properties.resize(untracked);
    
    if(tracked < _tracked_properties.size())
        _tracked_properties.resize(tracked);
    if (tracked < _tracked_gui.size())
        _tracked_gui.resize(tracked);

    for (auto s : selected_ids) {
        if (auto it = _last_bounds.find(s);
            it != _last_bounds.end())
        {
            auto& [frame, bds] = it->second;
            _zoom_targets.push_back(bds.pos());
            _zoom_targets.push_back(bds.pos() + bds.size());
            _zoom_targets.push_back(bds.pos() + bds.size().mul(0, 1));
            _zoom_targets.push_back(bds.pos() + bds.size().mul(1, 0));

            if(frameIndex.try_sub(frame) > 10_f) {
				_last_bounds.erase(it);
			}
        }
    }

    if (dirty) {
        _bowl->fit_to_screen(coords.screen_size());
        _bowl->set_target_focus(_zoom_targets);
    }
    
    if(closed_loop_enable && dirty) {
        std::unique_lock guard{_current_json_mutex};
        _current_json["objects"] = acc_json;
    }
}

void ConvertScene::Data::retry_video_info() {
    _retrieve_video_info = _segmenter->video_recovered_error();
}

dyn::DynamicGUI ConvertScene::Data::init_gui(Base* window) {
    dyn::Context context;
    check_video_info(true, nullptr);
    retry_video_info();
    
    context.actions = {
        ActionFunc("terminate", [this](auto) {
            SceneManager::getInstance().enqueue([&](IMGUIBase*, DrawStructure& graph){
                graph.dialog([&](Dialog::Result result){
                    if(result == Dialog::OKAY) {
                        if (_segmenter)
                            _segmenter->force_stop();
                        else
                            SceneManager::getInstance().set_active("starting-scene");
                    }
                }, "<b>Do you want to stop recording here?</b>\nThe already converted parts of the video will still be saved to <c><cyan>"+(_segmenter ? _segmenter->output_file_name().str() : SETTING(filename).value<file::Path>().str())+"</cyan></c>.", "End recording", "Yes, stop", "Cancel");
            });
            
        }),
        ActionFunc("set", [](const Action& action) {
            auto name = action.parameters.at(0);
            auto value = action.parameters.at(1);
            if (GlobalSettings::has(name)) {
                GlobalSettings::get(name).get().set_value_from_string(value);
            }
        }),
        ActionFunc("set_clipboard", [](const Action& action) {
            auto text = action.parameters.at(0);
            gui::set_clipboard(text);
            SceneManager::getInstance().enqueue([text](auto, DrawStructure& graph) {
                graph.dialog("Copied to clipboard:\n<c><str>"+text+"</str></c>");
            });
        }),
        
        ActionFunc("python", [](Action action){
            REQUIRE_EXACTLY(1, action);
            
            Python::schedule(Python::PackagedTask{
                ._network = nullptr,
                ._task = Python::PromisedTask(
                    [action](){
                        using py = PythonIntegration;
                        Print("Executing: ", no_quotes(util::unescape(action.first())));
                        py::execute(action.first());
                    }
                ),
                ._can_run_before_init = false
            });
        })
    };
    context.variables = {
        VarFunc("resizecvt", [this](const VarProps&) -> double {
            if (not _segmenter || not _segmenter->overlayed_video() || not _segmenter->overlayed_video()->source())
                return 0;//throw U_EXCEPTION("No source available.");
            return _segmenter->overlayed_video()->source()->resize_cvt().average_fps.load();
        }),
        VarFunc("sourceframe", [this](const VarProps&) -> double {
            if (not _segmenter || not _segmenter->overlayed_video() || not _segmenter->overlayed_video()->source())
                return 0;
            return _segmenter->overlayed_video()->source()->source_frame().average_fps.load();
        }),
        VarFunc("fps", [](const VarProps&) {
            auto fps = AbstractBaseVideoSource::_fps.load();
            auto samples = AbstractBaseVideoSource::_samples.load();
            return samples > 0 ? fps / samples : 0;
        }),
        VarFunc("net_fps", [this](const VarProps&) -> double {
            if (not _segmenter || not _segmenter->overlayed_video())
                return 0;//throw U_EXCEPTION("No source available.");
            return _segmenter->overlayed_video()->network_fps();
        }),
        VarFunc("track_fps", [this](const VarProps&) -> double {
            if (not _segmenter)
                return 0;//throw U_EXCEPTION("No source available.");
            return _segmenter->fps();
        }),
        VarFunc("write_fps", [this](const VarProps&) -> double {
            if (not _segmenter)
                return 0;//throw U_EXCEPTION("No source available.");
            return _segmenter->write_fps();
        }),
        VarFunc("time", [this](const VarProps&) -> float {
            return (_time.load() * 4);
        }),
        VarFunc("vid_fps", [](const VarProps&) {
            auto fps = AbstractBaseVideoSource::_video_fps.load();
            auto samples = AbstractBaseVideoSource::_video_samples.load();
            return samples > 0 ? fps / samples : 0;
        }),
        VarFunc("window_size", [](const VarProps&) -> Vec2 {
            return FindCoord::get().screen_size();
        }),
        VarFunc("mouse", [this](const VarProps&) -> Vec2 {
            return this->_last_mouse;
        }),
        VarFunc("output", [](const VarProps& props) -> file::Path {
            if(props.parameters.empty())
                return file::DataLocation::parse("output");
            return file::DataLocation::parse("output", props.first());
        }),
        VarFunc("output_name", [this](const VarProps& ) -> file::Path {
            if (_segmenter)
                return _segmenter->output_file_name();
            return SETTING(filename).value<file::Path>();
        }),
        VarFunc("output_base", [](const VarProps& ) -> file::Path {
            return SETTING(filename).value<file::Path>().filename();
        }),
        VarFunc("output_size", [this](const VarProps&) -> Vec2 {
            if(not _segmenter)
                return Size2{};
            return _segmenter->output_size();
        }),
        VarFunc("gpu_device", [](const VarProps&) -> std::string {
            using namespace default_config;
            auto gpu_torch_device = SETTING(gpu_torch_device).value<gpu_torch_device_t::Class>();
            if (gpu_torch_device == gpu_torch_device_t::cpu) {
                if (utils::contains(utils::lowercase(python_gpu_name()), "metal")
                    || utils::contains(utils::lowercase(python_gpu_name()), "cuda")
                    || utils::contains(utils::lowercase(python_gpu_name()), "nvidia"))
                {
                    return "CPU ("+python_gpu_name()+")";
                }
                return "CPU";
            }
            return python_gpu_name();
        }),
        VarFunc("detect_format", [](const VarProps&) {
            return SETTING(detect_format).value<detect::ObjectDetectionFormat_t>();
        }),
        VarFunc("fish", [](const VarProps&) -> sprite::Map& {
            return fish;
        }),
        VarFunc("average_is_generating", [this](const VarProps&) {
            return _segmenter->is_average_generating();
        }),
        VarFunc("average_percent", [this](const VarProps&) {
            return _segmenter->average_percent();
        }),
        VarFunc("actual_frame", [this](const VarProps&) {
            return _actual_frame;
        }),
        VarFunc("video", [](const VarProps&) -> sprite::Map& {
            return _video_info;
        }),
        VarFunc("num_tracked", [this](const VarProps&) -> size_t {
            size_t count{0u};
            for(auto &v : _tracked_properties) {
                if(not v || not v->has("visible") || not v->has("tracked"))
                    continue;
                if(v->at("tracked").value<bool>() && v->at("visible").value<bool>())
                    ++count;
            }
            return count;
        }),
        VarFunc("inactive_ids", [this](const VarProps&) {
            return _inactive_ids;
        }),
        VarFunc("active_ids", [this](const VarProps&) {
            return _active_ids;
        }),
        VarFunc("fishes", [this](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>&{
            return _tracked_gui;
        }),
        VarFunc("untracked", [this](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>&{
            return _untracked_gui;
        }),
        VarFunc("primary_selection", [this](const VarProps&) -> sprite::Map& {
            return _primary_selection;
        }),
        VarFunc("video_error", [this](const VarProps&) -> std::string {
            if(_recovered_error.empty())
                return "";
            else if(_recovered_error.size() == 1)
                return (std::string)*_recovered_error.begin();
            return Meta::toStr(_recovered_error);
        }),
        VarFunc("is_initializing", [](const VarProps&) {
            return Detection::is_initializing();
        })
    };

    return dyn::DynamicGUI{
        .gui = SceneManager::getInstance().gui_task_queue(),
        .path = "alter_layout.json",
        .context = std::move(context),
        .base = window
    };
}

// Main _draw function
void ConvertScene::_draw(DrawStructure& graph) {
    //bool dirty = fetch_new_data();
    //dirty = true;

    if(window()) {
        FindCoord::set_video(_data->video_size);
    }
    
    _data->draw(false, graph, window());
}

void ConvertScene::Data::draw(bool, DrawStructure& graph, Base* window) {
    fetch_new_data();
    
    if(_recorder.recording()) {
        dt = 1.0 / double(FAST_SETTING(frame_rate));
    } else {
        dt = saturate(timer.elapsed(), 0.001, 1.0);
    }
    _time = _time + dt;
    timer.reset();
    
    auto coords = FindCoord::get();
    if (not _bowl) {
        _bowl = std::make_unique<Bowl>(nullptr);
        _bowl->set_video_aspect_ratio(output_size.width, output_size.height);
        _bowl->fit_to_screen(coords.screen_size());
    }
    
    _last_mouse = graph.mouse_position();
    const auto detect_classes = detect::yolo::names::get_map();

    graph.wrap_object(*_bowl);
    _bowl->update_scaling(dt);

    bool dirty = retrieve_and_prepare_data();
    draw_scene(graph, detect_classes, dirty);
    _bowl->update(_current_data.frame.index(), graph, coords);
    
    check_gui(graph, window);
}

bool ConvertScene::Data::retrieve_and_prepare_data() {
    if(not _tmp_store_data)
        return false;
    
    assert(last_frame != _tmp_store_data.frame.index());
    //if(last_frame == _current_data.frame.index())
    //    return false;
    
    LockGuard lguard(w_t{}, "drawing", 10);
    if (not lguard.locked()) {
        return false;
    }
    
    /// commit the objects:
    if(_current_data.image)
        _segmenter->overlayed_video()->source()->move_back(std::move(_current_data.image));
    _current_data = std::move(_tmp_store_data);
    _object_blobs = std::move(_tmp_store_blobs);
    _current_frame = std::move(_tmp_store_frame);
    ///
    
    SETTING(gui_frame) = _current_data.frame.index();
    last_frame = _current_data.frame.index();
    
    _active_ids.clear();
    _inactive_ids.clear();
    
    auto a = IndividualManager::active_individuals(last_frame);
    if(a && a.value()) {
        for(auto fish : *a.value()) {
            _active_ids.emplace_back(fish->identity().ID());
        }
    }
    
    IndividualManager::transform_all([this](Idx_t id, auto) {
        if(not contains(_active_ids, id))
            _inactive_ids.emplace_back(id);
    });
    
    using namespace track;
    std::unordered_map<pv::bid, Identity> visible_bdx;
    std::vector<std::vector<Vertex>> lines;
    std::vector<std::tuple<Color, std::vector<Vec2>>> postures;
    const bool output_normalize_midline_data = SETTING(output_normalize_midline_data);

    IndividualManager::transform_all([&, frameIndex = _current_data.frame.index()](Idx_t, Individual* fish) {
        if (not fish->has(frameIndex))
            return;
        
        auto p = fish->iterator_for(_current_data.frame.index());
        auto tracklet = p->get();
        Range<Frame_t> tracklet_range;
        
        if(tracklet) {
            auto filters = constraints::local_midline_length(fish, tracklet->range);
            filter_cache[fish->identity().ID()] = std::move(filters);
            tracklet_range = tracklet->range;
        }

        auto [basic, posture] = fish->all_stuff(frameIndex);
        
        if(basic) {
            //active_ids.insert(fish->identity().ID());
            
            BdxAndPred blob{
                .bdx = basic->blob.blob_id(),
                .basic_stuff = *basic,
                .automatic_match = fish->is_automatic_match(frameIndex),
                .tracklet = tracklet_range
            };
            if(posture) {
                blob.posture_stuff = posture->clone();
                
                /// this could be optimized by using the posture stuff
                /// in the fixed midline function + SETTING()
                blob.midline = output_normalize_midline_data ? fish->fixed_midline(frameIndex) : fish->calculate_midline_for(*posture);
            }
            
            //blob_selected_fish[blob.bdx] = fish->identity().ID();
            //fish_last_bounds[fish->identity().ID()] = basic->blob.calculate_bounds();
            fish_selected_blobs[fish->identity().ID()] = std::move(blob);
        }
        
        if (basic->blob.parent_id.valid())
            visible_bdx.emplace(basic->blob.parent_id, fish->identity());
        visible_bdx.emplace(basic->blob.blob_id(), fish->identity());
        
        std::vector<Vertex> line;
        fish->iterate_frames(Range(_current_data.frame.index().try_sub(50_f), _current_data.frame.index()), [&](Frame_t, const std::shared_ptr<TrackletInformation>& ptr, const BasicStuff* basic, const PostureStuff*) -> bool
        {
            if (ptr.get() != tracklet) //&& (ptr)->end() != tracklet->start().try_sub(1_f))
                return true;
            auto p = basic->centroid.pos<Units::PX_AND_SECONDS>();//.mul(scale);
            line.push_back(Vertex(p.x, p.y, fish->identity().color()));
            return true;
        });

        lines.emplace_back(std::move(line));
        //graph.vertices(line);
    
        if(posture
           && posture->outline)
        {
            auto pts = posture->outline.uncompress();
            auto p = basic->blob.calculate_bounds().pos();
            for(auto &pt : pts)
                pt += p;
            
            postures.emplace_back(fish->identity().color(), std::move(pts));
            //graph.vertices(pts, fish->identity().color(), PrimitiveType::LineStrip);
        }
    });
    
    _visible_bdx = std::move(visible_bdx);
    _trajectories = std::move(lines);
    _postures = std::move(postures);
    
    update_background_image();
    
    return true;
}

void ConvertScene::Data::update_background_image() {
    if (_current_data.image) {
        if (_background_image->source()
            && _background_image->source()->rows == _current_data.image->rows
            && _background_image->source()->cols == _current_data.image->cols
            && _background_image->source()->dims == 4)
        {
            if (_current_data.image->dims == 3)
                cv::cvtColor(_current_data.image->get(), _background_image->unsafe_get_source().get(), cv::COLOR_BGR2BGRA);
            else
                _current_data.image->get().copyTo(_background_image->unsafe_get_source().get());

            _segmenter->overlayed_video()->source()->move_back(std::move(_current_data.image));
            //OverlayBuffers::put_back(std::move(_current_data.image));
            _background_image->updated_source();
        }
        else {
            auto rgba = Image::Make(_current_data.image->rows,
                _current_data.image->cols, 4);
            if (_current_data.image->dims == 3)
                cv::cvtColor(_current_data.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
            else
                _current_data.image->get().copyTo(rgba->get());
            _segmenter->overlayed_video()->source()->move_back(std::move(_current_data.image));
            //OverlayBuffers::put_back(std::move(_current_data.image));
            _background_image->set_source(std::move(rgba));
        }

        _current_data.image = nullptr;
        
        check_module();
    }
}

void ConvertScene::Data::draw_scene(DrawStructure& graph, const detect::yolo::names::map_t& detect_classes, bool dirty) {
    graph.section("video", [&](auto&, Section* section) {
        section->set_size(output_size);
        section->set_pos(_bowl->_current_pos);
        section->set_scale(_bowl->_current_scale);
        
        Transform transform;
        transform.combine(section->global_transform());
        
        FindCoord::set_bowl_transform(transform);
        
        if(not dirty) {
            section->reuse_objects();
            
            drawBlobs(graph, _current_data.frame.index(), detect_classes, _bowl->_current_scale, _bowl->_current_pos, _visible_bdx, dirty);
            return;
        }

        if (_background_image->source()) {
            if(_background_image->source() && _background_image->source()->rows > 0 && _background_image->source()->cols > 0) {
                graph.wrap_object(*_background_image);
            }
        }

        for (auto &box : _current_data.tiles)
            graph.rect(Box(box), attr::FillClr{Transparent}, attr::LineClr{Red.alpha(200)});
        
        ColorWheel wheel;
        size_t pose_index{ 0 };
        for (auto& keypoint : _current_data.keypoints) {
            auto pose = keypoint.toPose();
            if (pose_index >= _skeletts.size())
                _skeletts.push_back(std::make_unique<Skelett>(std::move(pose), skelet));
            else
                _skeletts[pose_index]->set_pose(std::move(pose));
            _skeletts[pose_index]->set_color(wheel.next());
            graph.wrap_object(*_skeletts[pose_index]);
            pose_index++;
        }
        if(pose_index < _skeletts.size())
            _skeletts.resize(pose_index);

        try {
            drawBlobs(graph, _current_data.frame.index(), detect_classes, _bowl->_current_scale, _bowl->_current_pos, _visible_bdx, dirty);
            
        } catch(const std::exception& e) {
            FormatWarning("Cannot draw blobs: ", e.what());
        }
        
        for(auto &traj : _trajectories) {
            graph.vertices(traj);
        }
        
        for(auto &[color, pts] : _postures) {
            graph.vertices(pts, color, PrimitiveType::LineStrip);
        }
    });
    
    graph.section("menus", [&](auto&, Section*) {
        _video_info["frame"] = _current_data.frame.index();
        _actual_frame = _current_data.frame.source_index();
        _video_frame = _current_data.frame.index();
    });
}


}
