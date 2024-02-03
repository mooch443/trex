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
#include <gui/Coordinates.h>
#include <python/GPURecognition.h>
#include <gui/Label.h>
#include <gui/ParseLayoutTypes.h>
#include <python/Yolo8.h>
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

namespace gui {
using namespace dyn;
using Skeleton = blob::Pose::Skeleton;

struct ConvertScene::Data {
    Segmenter* _segmenter{nullptr};

    // External images for background and overlay
    std::shared_ptr<ExternalImage> _background_image = std::make_shared<ExternalImage>(),
                                   _overlay_image = std::make_shared<ExternalImage>();

    // Vectors for object blobs and GUI objects
    std::vector<pv::BlobPtr> _object_blobs;
    SegmentationData _current_data;

    CallbackCollection callback;
    Skeleton skelet;

    // Individual properties for each object
    std::vector<std::shared_ptr<VarBase_t>> _untracked_gui, _tracked_gui, _joint;
    std::map<Idx_t, sprite::Map> _individual_properties;
    std::vector<sprite::Map> _untracked_properties;
    std::vector<sprite::Map*> _tracked_properties;
    std::unordered_map<pv::bid, Identity> _visible_bdx;
    std::vector<Idx_t> _inactive_ids;
    std::vector<Idx_t> _active_ids;

    std::unordered_map<Idx_t, std::shared_ptr<Label>> _labels;
    std::vector<std::unique_ptr<Skelett>> _skeletts;
    std::unordered_map<Idx_t, std::tuple<Frame_t, Bounds>> _last_bounds;
    
    ScreenRecorder _recorder;
    
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
    std::future<std::string> _retrieve_video_info;
    std::string _recovered_error;
    
    Frame_t last_frame;
    Timer timer;
    
    void drawBlobs(Frame_t, const std::vector<std::string>& meta_classes, const Vec2& scale, Vec2 offset, const std::unordered_map<pv::bid, Identity>& visible_bdx, bool dirty);
    // Helper function to draw outlines
    void drawOutlines(DrawStructure& graph, const Size2& scale, Vec2 offset);
    void check_video_info(bool wait, std::string*);
    
    bool fetch_new_data();
    
    dyn::DynamicGUI init_gui(Base* window);
    void check_gui(DrawStructure& graph, Base* window) {
        if(not dynGUI) {
            dynGUI = init_gui(window);
            dynGUI.graph = &graph;

            dynGUI.context.custom_elements["label"] = std::unique_ptr<CustomElement>(new CustomElement {
                .name = "label",
                .create = [this](LayoutContext& layout) -> Layout::Ptr {
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
                            _labels[id] = ptr = std::make_shared<Label>();
                    }

                    if (not ptr)
                        ptr = std::make_shared<Label>(text, Bounds(layout.pos, layout.size), Bounds(layout.pos, layout.size).center());

                    ptr->set_line_length(line_length);
                    ptr->set_data(0_f, text, Bounds(layout.pos, layout.size), center);//Bounds(layout.pos, layout.size).center());
                    auto font = parse_font(layout.obj, layout._defaults.font);
                    ptr->text()->set(font);
                    ptr->text()->set(color);
                    ptr->text()->set(FillClr{ fill });
                    ptr->set_line_color(line);
                    //print("Create new label with text = ", text);

                    return Layout::Ptr(ptr);
                },
                .update = [this](Layout::Ptr& o, const Context& context, State& state, const robin_hood::unordered_map<std::string, Pattern>& patterns) -> bool
                {
                    //print("Updating label with patterns: ", patterns);
                    //print("o = ", o.get());

                    Idx_t id;
                    if (patterns.contains("id"))
                        id = Meta::fromStr<Idx_t>(parse_text(patterns.at("id").original, context, state));
                    
                    if (id.valid()) {
                        auto it = _labels.find(id);
                        if (it != _labels.end()) {
                            if(it->second.get() != o.get())
                                o = Layout::Ptr(it->second);
                        }
                    }

                    auto p = o.to<Label>();
                    auto source = p->source();
                    auto pos = source.pos();
                    auto center = p->center();
                    auto text = p->text()->text();

                    if(patterns.contains("text"))
                        text = parse_text(patterns.at("text").original, context, state);
                    if (patterns.contains("pos")) {
                        pos = Meta::fromStr<Vec2>(parse_text(patterns.at("pos").original, context, state));
                    }
                    if (patterns.contains("size")) {
                        source = Bounds(pos, Meta::fromStr<Size2>(parse_text(patterns.at("size").original, context, state)));
                    }
                    if (patterns.contains("center")) {
                        center = Meta::fromStr<Vec2>(parse_text(patterns.at("center").original, context, state));
                    } else
                        center = source.pos()+ Vec2(source.width, source.height) * 0.5;

                    if(patterns.contains("line"))
                        p->set_line_color(Meta::fromStr<Color>(parse_text(patterns.at("line").original, context, state)));
                    if (patterns.contains("fill"))
                        p->set_fill_color(Meta::fromStr<Color>(parse_text(patterns.at("fill").original, context, state)));
                    if(patterns.contains("color"))
                        p->text()->set(TextClr{ Meta::fromStr<Color>(parse_text(patterns.at("color").original, context, state)) });

                    p->set_data(0_f, text, source, center);
                    p->update(FindCoord::get(), 1, 1, false, dt, Scale{1});
                    
                    return true;
                }
            });
        }
        
        dynGUI.update(nullptr);
    }
    void draw(bool dirty, DrawStructure& graph, Base* window);
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

std::string ConvertScene::window_title() const {
    auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + (not filename.empty() ? " (" + filename + ")" : "")
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
}

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

bar{
    ind::option::BarWidth{50},
        ind::option::Start{"["},
#ifndef _WIN32
        ind::option::Fill{"█"},
        ind::option::Lead{"▂"},
        ind::option::Remainder{"▁"},
#else
        ind::option::Fill{"="},
        ind::option::Lead{">"},
        ind::option::Remainder{" "},
#endif
        ind::option::End{"]"},
        ind::option::PostfixText{"Converting video..."},
        ind::option::ShowPercentage{true},
        ind::option::ForegroundColor{ind::Color::white},
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
},

spinner{
    ind::option::PostfixText{""},
        ind::option::ForegroundColor{ind::Color::white},
#ifndef _WIN32
        ind::option::SpinnerStates{std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
#else
        ind::option::SpinnerStates{std::vector<std::string>{".","..","..."}},
#endif
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
},

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
            if(percent >= 0)
                bar.set_progress(percent);
            else if(last_tick.elapsed() > 1) {
                spinner.set_option(ind::option::PrefixText{"Recording ("+Meta::toStr(_data->_video_frame)+")"});
                spinner.tick();
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

        bar.set_progress(100);
        bar.mark_as_completed();
        
        spinner.set_option(ind::option::ForegroundColor{ind::Color::green});
        spinner.set_option(ind::option::PrefixText{"✔"});
        spinner.set_option(ind::option::ShowSpinner{false});
        spinner.set_option(ind::option::PostfixText{"Done."});
        spinner.set_progress(100);
        spinner.mark_as_completed();
        
        _data->_object_blobs.clear();
        _data->_current_data = {};
        _data->dynGUI.clear();
        
        /// save the last settings used
        RecentItems::open(SETTING(source).value<file::PathArray>(), GlobalSettings::current_defaults_with_config());
        
        segmenter().force_stop();
        _data->check_video_info(true, nullptr);
        
        if(_on_deactivate)
            _on_deactivate(*this);
        
        _data = nullptr;
        _scene_promise.set_value();
        
    } catch(const std::exception& e){
        FormatExcept(e.what());
        _scene_promise.set_exception(std::current_exception());
    }
}

void ConvertScene::Data::check_video_info(bool wait, std::string* result) {
    if(_retrieve_video_info.valid()
       && (wait || _retrieve_video_info.wait_for(std::chrono::milliseconds(0)) 
                        == std::future_status::ready))
    {
        try {
            // invalidate existing future and throw away
            if(result)
                *result = _retrieve_video_info.get();
            else
                (void)_retrieve_video_info.get();
            
        } catch(const std::exception& e) {
            FormatError("There was in error retrieving video info from the future: ", e.what());
        }
    }
}

void ConvertScene::open_video() {
    bar.set_option(ind::option::ShowPercentage{true});
    segmenter().open_video();
    
    _video_info["resolution"] = segmenter().size();
    _video_info["length"] = segmenter().video_length();
}

void ConvertScene::open_camera() {
    if(SETTING(detect_model).value<file::Path>().empty()
       || (not SETTING(detect_model).value<file::Path>().exists() 
            && not Yolo8::is_default_model(SETTING(detect_model).value<file::Path>().str())))
    {
        SETTING(detect_model) = file::Path(Yolo8::default_model());
    }
    
    if(not GlobalSettings::current_defaults_with_config().has("save_raw_movie"))
    {
        SETTING(save_raw_movie) = true;
    }
    
    spinner.set_option(ind::option::PrefixText{"Recording"});
    spinner.set_option(ind::option::ShowPercentage{false});
    
    segmenter().open_camera();
    
    _video_info["resolution"] = segmenter().size();
    _video_info["length"] = segmenter().video_length();
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
    
    print("Loading source = ", source);
    SETTING(meta_source_path) = source.source();
    if (source == file::PathArray("webcam"))
        open_camera();
    else
        open_video();

    if(not _data)
        _data = std::make_unique<Data>();

    _data->skelet = SETTING(meta_skeleton).value<Skeleton>();
    _data->callback = GlobalSettings::map().register_callbacks({
        "meta_skeleton"
    }, [this](auto) {
        SceneManager::getInstance().enqueue([this](auto,auto&) {
            _data->skelet = SETTING(meta_skeleton).value<Skeleton>();
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
    
    auto work_area = ((const IMGUIBase*)window())->work_area();
    print("work_area = ", work_area);
    auto window_size = Size2(
        (work_area.width) * 0.75,
                             _data->output_size.height / _data->output_size.width * (work_area.width) * 0.75
    );
    print("prelim window size = ", window_size);
    if (window_size.height > work_area.height) {
        auto ratio = window_size.width / window_size.height;
        window_size = Size2(
            ratio * (work_area.height),
            work_area.height
        );
        print("Restricting window size to ", window_size, " based on ratio ", ratio);
    }
    if (window_size.width > work_area.width) {
        auto ratio = window_size.height / window_size.width;
        auto h = min(ratio * (work_area.width), window_size.height);
        window_size = Size2(
            h / ratio,
            h
        );
        print("Restricting window size to width ", window_size, " based on ratio ", ratio);
    }

    Bounds bounds(
        Vec2((work_area.width) / 2 - window_size.width / 2,
            work_area.height / 2 - window_size.height / 2),
        window_size);
    print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
    bounds.restrict_to(Bounds(work_area.size()));
    print("Restricting bounds to work area: ", work_area, " -> ", bounds);

    print("setting bounds = ", bounds);
    window()->set_window_bounds(bounds);
    window()->set_title(window_title());
    bar.set_progress(0);
    
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
}

bool ConvertScene::on_global_event(Event e) {
    if(e.type == EventType::KEY
        && e.key.pressed) 
    {
        switch(e.key.code) {
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
    auto&& [data, obj] = _segmenter->grab();
    if(data.image) {
        _current_data = std::move(data);
        _object_blobs = std::move(obj);
        dirty = true;
    }
    
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
void ConvertScene::Data::drawOutlines(DrawStructure& graph, const Size2& scale, Vec2) {
    if (not _current_data.outlines.empty()) {
        graph.text(Str(Meta::toStr(_current_data.outlines.size()) + " lines"), attr::Loc(10, 50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));

        ColorWheel wheel;
        for (const auto& v : _current_data.outlines) {
            auto clr = wheel.next();
            graph.line(Line::Points_t{ v }, LineClr{ clr.alpha(150) });
        }
    }
}

uint64_t interleaveBits(const Vec2& pos) {
    uint32_t x(pos.x), y(pos.y);
    uint64_t z = 0;
    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; ++i) {
        z |= (x & (1ULL << i)) << i | (y & (1ULL << i)) << (i + 1);
    }
    return z;
}

void ConvertScene::Data::drawBlobs(
    Frame_t frameIndex,
    const std::vector<std::string>& meta_classes, 
    const Vec2&, Vec2, 
    const std::unordered_map<pv::bid, Identity>& visible_bdx, 
    bool dirty) 
{
    //size_t i = 0;
    size_t untracked = 0;
    std::sort(_object_blobs.begin(), _object_blobs.end(), [](auto& A, auto& B) {
        return interleaveBits(A->bounds().center()) < interleaveBits(B->bounds().center());
    });

    auto coords = FindCoord::get();
    std::set<Idx_t> tracked_ids;
    for (auto& [id, map] : _individual_properties) {
		map["visible"] = false;
	}

    auto selected_ids = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
    std::vector<Vec2> targets;
    const bool is_background_subtraction = track::detect::detection_type() == track::detect::ObjectDetectionType::background_subtraction;

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

        if (_current_data.frame.index().valid()) {
            if (blob->prediction().valid()) {
                auto pred = blob->prediction();
                assign = {
                    .clid = pred.clid,
                    .p = static_cast<float>(pred.p) / 255.f
                };
                if (pred.pose.size() > 0) {
                    auto& pt = pred.pose.point(0);
                    if(pt.x > 0 || pt.y > 0)
                        first_pose = coords.convert(BowlCoord(pt));
                }
            }
            //else
            //    print("[draw]4 blob ", blob->blob_id(), " prediction not found...");
        }
        
        auto cname = meta_classes.size() > assign.clid
            ? meta_classes.at(assign.clid)
            : ((assign.clid == size_t(-1)
                   ? (is_background_subtraction 
                      ? (meta_classes.empty()
                           ? FAST_SETTING(individual_prefix)
                           : meta_classes.front())
                      : "<no prediction>")
                   : "<unknown:" + Meta::toStr(assign.clid) + ">"));

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
                    //print("for ", props, " returning value of ", i, " / ", _individual_properties.size());
                    return _untracked_properties.at(i);
                }));

            tmp = &_untracked_properties[untracked++];
        }
        
        bool selected = contains(selected_ids, tracked_id);
        (*tmp)["pos"] = bds.pos();
        (*tmp)["selected"] = selected;
        (*tmp)["bdx"] = blob->blob_id();
        (*tmp)["center"] = first_pose;
        (*tmp)["tracked"] = tracked_id.valid() ? true : false;
        (*tmp)["color"] = tracked_color;
        (*tmp)["id"] = tracked_id;
		(*tmp)["visible"] = true;
        (*tmp)["size"] = Size2(bds.size());
        (*tmp)["radius"] = bds.size().length() * 0.5;
        (*tmp)["type"] = std::string(cname);
        //if(Tracker::instance() && Tracker::background())
        //    (*tmp)["px"] = blob->recount(FAST_SETTING(track_threshold), *Tracker::background());
        //else
            (*tmp)["px"] = blob->recount(-1);

        (*tmp)["p"] = Meta::toStr(assign.p);

        if (tracked_id.valid() 
            && selected) 
        {
            if (blob) {
                auto bds = blob->bounds();
                targets.push_back(bds.pos());
                targets.push_back(bds.pos() + bds.size());
                targets.push_back(bds.pos() + bds.size().mul(0, 1));
                targets.push_back(bds.pos() + bds.size().mul(1, 0));
                _last_bounds[tracked_id] = { frameIndex, bds };
                selected_ids.erase(std::find(selected_ids.begin(), selected_ids.end(), tracked_id));
            }
        }
    }

    size_t tracked = 0;
    for (auto& [id, map] : _individual_properties) {
        if(_tracked_properties.size() <= tracked)
            _tracked_properties.emplace_back(&_individual_properties[id]);
        else
            _tracked_properties[tracked] = &_individual_properties[id];

        if (tracked >= _tracked_gui.size())
            _tracked_gui.emplace_back(new Variable([&, i = tracked](const VarProps&) -> sprite::Map& {
                //print("for ", props, " returning value of ", i, " / ", _individual_properties.size());
                return *_tracked_properties.at(i);
            }));

        tracked++;
    }

    if (untracked < _untracked_gui.size())
        _untracked_gui.resize(untracked);
    if (untracked < _untracked_properties.size())
        _untracked_properties.resize(untracked);

    if (tracked < _tracked_gui.size())
        _tracked_gui.resize(tracked);

    _joint = _tracked_gui;
    _joint.insert(_joint.end(), _untracked_gui.begin(), _untracked_gui.end());

    for (auto s : selected_ids) {
        if (auto it = _last_bounds.find(s);
            it != _last_bounds.end())
        {
            auto& [frame, bds] = it->second;
            targets.push_back(bds.pos());
            targets.push_back(bds.pos() + bds.size());
            targets.push_back(bds.pos() + bds.size().mul(0, 1));
            targets.push_back(bds.pos() + bds.size().mul(1, 0));

            if(frameIndex.try_sub(frame) > 10_f) {
				_last_bounds.erase(it);
			}
        }
    }

    if (dirty) {
        _bowl->fit_to_screen(coords.screen_size());
        _bowl->set_target_focus(targets);
    }
}

dyn::DynamicGUI ConvertScene::Data::init_gui(Base* window) {
    dyn::Context context;
    check_video_info(true, nullptr);
    _retrieve_video_info = std::async(std::launch::async, [this]()
        -> std::string
    {
        // we need to throw this away since this may be blocking
        auto e = _segmenter->video_recovered_error().get();
        if(not e.has_value()) {
            return "";
        }
        return (std::string)e.value();
    });
    
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
                }, "<b>Do you want to stop recording here?</b>\nThe already converted parts of the video will still be saved to <c>"+SETTING(filename).value<file::Path>().str()+"</c>.", "End recording", "Yes, stop", "Cancel");
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
        })
    };
    context.variables = {
        VarFunc("resizecvt", [this](const VarProps&) -> double {
            return _segmenter->overlayed_video()->source()->resize_cvt().average_fps.load();
        }),
        VarFunc("sourceframe", [this](const VarProps&) -> double {
            return _segmenter->overlayed_video()->source()->source_frame().average_fps.load();
        }),
        VarFunc("fps", [](const VarProps&) {
            auto fps = AbstractBaseVideoSource::_fps.load();
            auto samples = AbstractBaseVideoSource::_samples.load();
            return samples > 0 ? fps / samples : 0;
        }),
        VarFunc("net_fps", [this](const VarProps&) {
            return _segmenter->overlayed_video()->network_fps();
        }),
        VarFunc("track_fps", [this](const VarProps&) {
            return _segmenter->fps();
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
        VarFunc("output_name", [](const VarProps& ) -> file::Path {
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
        VarFunc("video_error", [this](const VarProps&) -> std::string {
            return _recovered_error;
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
    
    auto coords = FindCoord::get();
    if (not _bowl) {
        _bowl = std::make_unique<Bowl>(nullptr);
        _bowl->set_video_aspect_ratio(output_size.width, output_size.height);//coord.video_size().width, coord.video_size().height);
        _bowl->fit_to_screen(coords.screen_size());
    }
    
    _last_mouse = graph.mouse_position();
    const auto meta_classes = SETTING(meta_classes).value<std::vector<std::string>>();

    graph.wrap_object(*_bowl);
    _bowl->update_scaling();
    //_bowl_mouse = coord.convert(HUDCoord(graph.mouse_position())); //_data->_bowl->global_transform().getInverse().transformPoint(graph.mouse_position());

    graph.section("video", [&](auto&, Section* section) {
        section->set_size(output_size);
        section->set_pos(_bowl->_current_pos);
        section->set_scale(_bowl->_current_scale);
        
        Transform transform;
        //transform.scale(_screen_size.div(video_size).reciprocal());
       // transform.scale(graph.scale() / gui::interface_scale());
        transform.combine(section->global_transform());
        
        FindCoord::set_bowl_transform(transform);

        LockGuard lguard(w_t{}, "drawing", 10);
        if (not lguard.locked()) {
            section->reuse_objects();
            return;
        }

        SETTING(gui_frame) = _current_data.frame.index();

        if (_background_image->source()) {
            if(_background_image->source() && _background_image->source()->rows > 0 && _background_image->source()->cols > 0) {
                graph.wrap_object(*_background_image);
            }
        }

        for (auto &box : _current_data.tiles)
            graph.rect(Box(box), attr::FillClr{Transparent}, attr::LineClr{Red});
        ColorWheel wheel;
        //auto coord = FindCoord::get();
        //print(coord.bowl_scale());
        
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

        bool dirty{ false };
        if (last_frame != _current_data.frame.index()) {
            last_frame = _current_data.frame.index();
            //_gui_objects.clear();
            //_individual_properties.clear();
            dirty = true;
            
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
        }

        // Draw outlines
        drawOutlines(graph, _bowl->_current_scale, _bowl->_current_pos);

        using namespace track;
        std::unordered_map<pv::bid, Identity> visible_bdx;

        IndividualManager::transform_all([&](Idx_t, Individual* fish)
            {
                if (not fish->has(_current_data.frame.index()))
                    return;
                auto p = fish->iterator_for(_current_data.frame.index());
                auto segment = p->get();

                auto basic = fish->compressed_blob(_current_data.frame.index());
                //auto bds = basic->calculate_bounds();//.mul(scale);

                if (dirty) {
                    if (basic->parent_id.valid())
                        visible_bdx.emplace(basic->parent_id, fish->identity());
                    visible_bdx.emplace(basic->blob_id(), fish->identity());
                }

                std::vector<Vertex> line;
                fish->iterate_frames(Range(_current_data.frame.index().try_sub(50_f), _current_data.frame.index()), [&](Frame_t, const std::shared_ptr<SegmentInformation>& ptr, const BasicStuff* basic, const PostureStuff*) -> bool
                    {
                        if (ptr.get() != segment) //&& (ptr)->end() != segment->start().try_sub(1_f))
                            return true;
                        auto p = basic->centroid.pos<Units::PX_AND_SECONDS>();//.mul(scale);
                        line.push_back(Vertex(p.x, p.y, fish->identity().color()));
                        return true;
                    });

                graph.vertices(line);
            });
        
        if(dirty)
            _visible_bdx = std::move(visible_bdx);

        //! do not need to continue further if the view isnt dirty
        if (not dirty) {
            size_t untracked = 0;
            for (auto& blob : _object_blobs) {
                auto bds = blob->bounds();
                bds = coords.convert(BowlRect(bds));

                Idx_t tracked_id;
                Color tracked_color;

                if (contains(_visible_bdx, blob->blob_id())) {
                    auto id = _visible_bdx.at(blob->blob_id());
                    tracked_color = id.color();
                    tracked_id = id.ID();
                }
                else if (blob->parent_id().valid() && contains(_visible_bdx, blob->parent_id()))
                {
                    auto id = _visible_bdx.at(blob->parent_id());
                    tracked_color = id.color();
                    tracked_id = id.ID();
                }
                else {
                    tracked_color = Gray;
                }
                
                sprite::Map* tmp = nullptr;

                if (tracked_id.valid()) {
                    tmp = &_individual_properties[tracked_id];
                }
                else {
                    if(untracked < _untracked_properties.size())
                        tmp = &_untracked_properties[untracked++];
                }
                
                if(tmp) {
                    (*tmp)["pos"] = bds.pos();
                    (*tmp)["size"] = Size2(bds.size());
                    (*tmp)["radius"] = bds.size().length() * 0.5;
                }
            }
            return;
        }

        drawBlobs(_current_data.frame.index(), meta_classes, _bowl->_current_scale, _bowl->_current_pos, _visible_bdx, dirty);
    });
    
    graph.section("menus", [&](auto&, Section*) {
        //section->set_scale(graph.scale().reciprocal() * gui::interface_scale());
        
        _video_info["frame"] = _current_data.frame.index();
        _actual_frame = _current_data.frame.source_index();
        _video_frame = _current_data.frame.index();
        
        dt = saturate(timer.elapsed(), 0.001, 1.0);
        timer.reset();

        _time = _time + dt;
    });

    check_gui(graph, window);
    _bowl->update(_current_data.frame.index(), graph, coords);
}

}
