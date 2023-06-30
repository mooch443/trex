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

namespace gui {

sprite::Map ConvertScene::fish = []() {
    sprite::Map fish;
    fish.set_do_print(false);
    fish["name"] = std::string("fish0");
    fish["color"] = Red;
    fish["pos"] = Vec2(100, 150);
    return fish;
}();

sprite::Map ConvertScene::_video_info = []() {
    sprite::Map fish;
    fish.set_do_print(false);
    fish["frame"] = Frame_t();
    fish["resolution"] = Size2();
    return fish;
}();

constexpr int thread_id = 42;

file::Path average_name() {
    auto path = file::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

std::string window_title() {
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + " (" + (std::string)SETTING(filename).value<file::Path>().filename() + ")"
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
}

ConvertScene::ConvertScene(Base& window) : Scene(window, "converting", 
    [this](Scene&, DrawStructure& graph) {
        _draw(graph);
    }),
menu{
    dyn::Context{
        .actions = {
            {
                "QUIT", [](auto) {
                    auto& manager = SceneManager::getInstance();
                    manager.set_active("");
                }
            },
            {
                "FILTER", [](auto) {
                    static bool filter { false };
                    filter = not filter;
                    SETTING(do_filter) = filter;
                }
            },
            {
                "RESET", [this](auto) {
                    _overlayed_video->reset(1500_f);
                }
            }
        },

        .variables = {
            {
                "fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                    return AbstractBaseVideoSource::_fps.load() / AbstractBaseVideoSource::_samples.load();
                }))
            },
            {
                "net_fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                    return AbstractBaseVideoSource::_network_fps.load() / AbstractBaseVideoSource::_network_samples.load();
                }))
            },
            {
                "vid_fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                    return AbstractBaseVideoSource::_video_fps.load() / AbstractBaseVideoSource::_video_samples.load();
                }))
            },
            {
                "fish",
                std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return fish;
                }))
            },
            {
                "global",
                std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            },
            {
                "actual_frame", std::unique_ptr<VarBase_t>(new Variable([this](std::string) {
                    return _actual_frame;
                }))
            },
            {
                "video", std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return _video_info;
                }))
            }
        }
    },
    [&](const std::string& name) {
        if (name == "gui_frame") {
            _overlayed_video->reset(SETTING(gui_frame).value<Frame_t>());
        }
    }
},

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
    ind::option::PostfixText{"Recording..."},
        ind::option::ForegroundColor{ind::Color::white},
        ind::option::SpinnerStates{std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
}

{
    _video_info.set_do_print(false);
    fish.set_do_print(false);

    menu.context.variables.emplace("fishes", new Variable([this](std::string) -> std::vector<std::shared_ptr<VarBase_t>>&{
        return _gui_objects;
    }));
    
    ThreadManager::getInstance().registerGroup(thread_id, "ConvertScene");
    
    ThreadManager::getInstance().addThread(thread_id, "generator-thread", ManagedThread{
        [this](){ generator_thread(); }
    });
    
    ThreadManager::getInstance().registerGroup(thread_id+1, "ConvertSceneTracking");
    ThreadManager::getInstance().addThread(thread_id+1, "tracking-thread", ManagedThread{
        [this](){ tracking_thread(); }
    });
    
    ThreadManager::getInstance().addOnEndCallback(thread_id+1, OnEndMethod{
        [this](){
            if (std::unique_lock guard(_mutex_general);
                _output_file != nullptr)
            {
                _output_file->close();
            }
            
            try {
                Detection::manager().clean_up();
                Detection::deinit();
            } catch(const std::exception& e) {
                FormatExcept("Exception when joining detection thread: ", e.what());
            }
            
        }
    });
}

ConvertScene::~ConvertScene() {
    if (not _should_terminate)
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

void ConvertScene::deactivate() {
    _should_terminate = true;
    
    try {
        bar.set_progress(100);
        bar.mark_as_completed();
        
        spinner.set_option(ind::option::ForegroundColor{ind::Color::green});
        spinner.set_option(ind::option::PrefixText{"✔"});
        spinner.set_option(ind::option::ShowSpinner{false});
        spinner.set_option(ind::option::PostfixText{"Done."});
        spinner.set_progress(100);
        spinner.mark_as_completed();
        
        {
            std::unique_lock guard(_mutex_general);
            _cv_ready_for_tracking.notify_all();
            _cv_messages.notify_all();
        }
        
        ThreadManager::getInstance().terminateGroup(thread_id+1);
        ThreadManager::getInstance().terminateGroup(thread_id);
        
        std::unique_lock guard(_mutex_general);
        _overlayed_video = nullptr;
        _tracker = nullptr;
        
        if (_output_file) {
            pv::File test(_output_file->filename(), pv::FileMode::READ);
            test.print_info();
        }
        
        _output_file = nullptr;
        _next_frame_data = {};
        _progress_data = {};
        _current_data = {};
        _transferred_blobs.clear();
        _object_blobs.clear();
        _progress_blobs.clear();
        _transferred_current_data = {};
        
        _scene_promise.set_value();
        
    } catch(const std::exception& e){
        FormatExcept(e.what());
        _scene_promise.set_exception(std::current_exception());
    }
}

void ConvertScene::open_video() {
    bar.set_option(ind::option::ShowPercentage{true});

    VideoSource video_base(SETTING(source).value<std::string>());
    video_base.set_colors(ImageMode::RGB);

    SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);

    print("filename = ", SETTING(filename).value<file::Path>());
    print("video_base = ", video_base.base());
    if (SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path(file::Path(video_base.base()).filename());
    }

    setDefaultSettings();
    _output_size = (Size2(video_base.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(meta_video_size).value<Size2>() = video_base.size();
    SETTING(output_size) = _output_size;
    _video_info["resolution"] = _output_size;

    _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
        Detection{},
        std::move(video_base),
        [this]() {
            _cv_messages.notify_one();
        }
    );
    _video_info["length"] = _overlayed_video->source->length();
    SETTING(video_length) = uint64_t(_overlayed_video->source->length().get());
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(get_model_image_size().width * 10);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
    bg.setTo(255);

    VideoSource tmp(SETTING(source).value<std::string>());
    if (not average_name().exists()) {
        tmp.generate_average(bg, 0);
        cv::imwrite(average_name().str(), bg);
    }
    else {
        print("Loading from file...");
        bg = cv::imread(average_name().str());
        if (bg.cols == tmp.size().width && bg.rows == tmp.size().height)
            cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
        else {
            tmp.generate_average(bg, 0);
            cv::imwrite(average_name().str(), bg);
        }
    }

    _tracker = std::make_unique<Tracker>(Image::Make(bg), float(get_model_image_size().width * 10));
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);

    auto path = filename.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    _output_file = std::make_unique<pv::File>(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    _output_file->set_average(bg);

    
    ThreadManager::getInstance().startGroup(thread_id);
    ThreadManager::getInstance().startGroup(thread_id+1);
}

void ConvertScene::open_camera() {
    spinner.set_option(ind::option::PrefixText{"Recording..."});
    spinner.set_option(ind::option::ShowPercentage{false});

    using namespace grab;
    fg::Webcam camera;
    camera.set_color_mode(ImageMode::RGB);

    SETTING(frame_rate) = Settings::frame_rate_t(25);
    if (SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::Path("webcam");

    setDefaultSettings();
    _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(output_size) = _output_size;
    _video_info["resolution"] = _output_size;
    SETTING(meta_video_size).value<Size2>() = camera.size();

    _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
        Detection{},
        std::move(camera),
        [this]() {
            _cv_messages.notify_one();
        }
    );

    _overlayed_video->source->notify();

    _video_info["length"] = _overlayed_video->source->length();
    SETTING(video_length) = uint64_t(_overlayed_video->source->length().get());
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(get_model_image_size().width * 10);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
    bg.setTo(255);

    /*VideoSource tmp(SETTING(source).value<std::string>());
    if(not average_name().exists()) {
        tmp.generate_average(bg, 0);
        cv::imwrite(average_name().str(), bg);
    } else {
        print("Loading from file...");
        bg = cv::imread(average_name().str());
        cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
    }*/

    _tracker = std::make_unique<Tracker>(Image::Make(bg), float(get_model_image_size().width * 10));
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);

    auto path = filename.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    _output_file = std::make_unique<pv::File>(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    _output_file->set_average(bg);
    
    ThreadManager::getInstance().startGroup(thread_id);
    ThreadManager::getInstance().startGroup(thread_id+1);
}

void ConvertScene::activate()  {
    _scene_promise = {};
    _scene_active = _scene_promise.get_future().share();
    _should_terminate = false;

    try {
        print("Loading source = ", SETTING(source).value<std::string>());
        if (SETTING(source).value<std::string>() == "webcam")
            open_camera();
        else
            open_video();

        auto size = _overlayed_video->source->size();
        auto work_area = ((const IMGUIBase*)window())->work_area();
        print("work_area = ", work_area);
        auto window_size = Size2(
            (work_area.width - work_area.x) * 0.75,
            size.height / size.width * (work_area.width - work_area.x) * 0.75
        );
        print("prelim window size = ", window_size);
        if (window_size.height > work_area.height - work_area.y) {
            auto ratio = window_size.width / window_size.height;
            window_size = Size2(
                ratio * (work_area.height - work_area.y),
                work_area.height - work_area.y
            );
            print("Restricting window size to ", window_size, " based on ratio ", ratio);
        }
        if (window_size.width > work_area.width - work_area.x) {
            auto ratio = window_size.height / window_size.width;
            auto h = min(ratio * (work_area.width - work_area.x), window_size.height);
            window_size = Size2(
                h / ratio,
                h
            );
            print("Restricting window size to width ", window_size, " based on ratio ", ratio);
        }

        Bounds bounds(
            Vec2((work_area.width - work_area.x) / 2 - window_size.width / 2,
                work_area.height / 2 - window_size.height / 2 + work_area.y),
            window_size);
        print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
        bounds.restrict_to(work_area);
        print("Restricting bounds to work area: ", work_area, " -> ", bounds);


        print("setting bounds = ", bounds);
        window()->set_window_bounds(bounds);
        bar.set_progress(0);

    }
    catch (const std::exception& e) {
        FormatExcept("Exception when switching scenes: ", e.what());
        //_scene_promise.set_value();
        //deactivate();
        SceneManager::getInstance().set_active("starting-scene");
    }
}

void ConvertScene::setDefaultSettings() {
    SETTING(do_filter) = false;
    SETTING(filter_classes) = std::vector<uint8_t>{};
    SETTING(is_writing) = true;
}

void ConvertScene::printDebugInformation() {
    DebugHeader("Starting tracking of");
    print("average at: ", average_name());
    if (detection_type() != ObjectDetectionType::yolo8) {
        print("model: ", SETTING(model).value<file::Path>());
        print("segmentation model: ", SETTING(segmentation_path).value<file::Path>());
    }
    else
        print("model: ", SETTING(model).value<file::Path>() != "" ? SETTING(model).value<file::Path>() : SETTING(segmentation_path).value<file::Path>());
    print("region model: ", SETTING(region_model).value<file::Path>());
    print("video: ", SETTING(source).value<std::string>());
    print("model resolution: ", SETTING(detection_resolution).value<uint16_t>());
    print("output size: ", SETTING(output_size).value<Size2>());
    print("output path: ", SETTING(filename).value<file::Path>());
    print("color encoding: ", SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>());
}

void ConvertScene::fetch_new_data() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        set_thread_name("GUI");
        });

    {
        std::unique_lock guard(_mutex_current);
        if (_transferred_current_data.image) {
            _current_data = std::move(_transferred_current_data);
            _object_blobs = std::move(_transferred_blobs);
        }
    }

    if (_current_data.image) {
        if (_background_image->source()
            && _background_image->source()->rows == _current_data.image->rows
            && _background_image->source()->cols == _current_data.image->cols
            && _background_image->source()->dims == 4)
        {
            cv::cvtColor(_current_data.image->get(), _background_image->unsafe_get_source().get(), cv::COLOR_BGR2BGRA);
            OverlayBuffers::put_back(std::move(_current_data.image));
            _background_image->updated_source();
        }
        else {
            auto rgba = Image::Make(_current_data.image->rows,
                _current_data.image->cols, 4);
            cv::cvtColor(_current_data.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
            OverlayBuffers::put_back(std::move(_current_data.image));
            _background_image->set_source(std::move(rgba));
        }

        _current_data.image = nullptr;
    }
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
void ConvertScene::drawOutlines(DrawStructure& graph, const Size2& scale) {
    if (not _current_data.outlines.empty()) {
        graph.text(Meta::toStr(_current_data.outlines.size()) + " lines", attr::Loc(10, 50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));

        ColorWheel wheel;
        for (const auto& v : _current_data.outlines) {
            auto clr = wheel.next();
            graph.line(v, 1, clr.alpha(150));
        }
    }
}

void ConvertScene::drawBlobs(const std::vector<std::string>& meta_classes, const Vec2& scale, const std::unordered_map<pv::bid, Identity>& visible_bdx) {
    for (auto& blob : _object_blobs) {
        const auto bds = blob->bounds();
        //graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
        auto [pos, image] = blob->image();

        SegmentationData::Assignment assign{
            .clid = size_t(-1)
        };
        if (_current_data.frame.index().valid()) {
            if (blob->prediction().valid()) {
                auto pred = blob->prediction();
                assign = {
                    .clid = pred.clid,
                    .p = static_cast<float>(pred.p) / 255.f
                };
                /*if(auto it = _current_data.predictions.find(blob->parent_id());
                    it != _current_data.predictions.end())
                {
                    assign = it->second;

                } else if((it = _current_data.predictions.find(blob->blob_id())) != _current_data.predictions.end())
                {
                    assign = it->second;

                } else
                    print("[draw]3 blob ", blob->blob_id(), " not found...");*/

            }
            else
                print("[draw]4 blob ", blob->blob_id(), " prediction not found...");
        }

        auto cname = meta_classes.size() > assign.clid
            ? meta_classes.at(assign.clid)
            : "<unknown:" + Meta::toStr(assign.clid) + ">";

        sprite::Map tmp;
        tmp.set_do_print(false);
        tmp["pos"] = bds.pos().mul(scale);
        tmp["size"] = Size2(bds.size().mul(scale));
        tmp["type"] = std::string(cname);

        if (contains(visible_bdx, blob->blob_id())) {
            auto id = visible_bdx.at(blob->blob_id());
            tmp["color"] = id.color();
            tmp["id"] = id.ID();
            tmp["tracked"] = true;

        }
        else if (blob->parent_id().valid() && contains(visible_bdx, blob->parent_id()))
        {
            auto id = visible_bdx.at(blob->parent_id());
            tmp["color"] = id.color();
            tmp["id"] = id.ID();
            tmp["tracked"] = true;

        }
        else {
            tmp["tracked"] = false;
            tmp["color"] = Gray;
            tmp["id"] = Idx_t();
        }
        tmp["p"] = Meta::toStr(assign.p);
        _individual_properties.push_back(std::move(tmp));
        _gui_objects.emplace_back(new Variable([&, i = _individual_properties.size() - 1](std::string) -> sprite::Map& {
            return _individual_properties.at(i);
            }));
    }
}

// Main _draw function
void ConvertScene::_draw(DrawStructure& graph) {
    fetch_new_data();

    const auto meta_classes = SETTING(meta_classes).value<std::vector<std::string>>();
    graph.section("video", [&](auto&, Section* section) {
        auto output_size = SETTING(output_size).value<Size2>();
        auto window_size = window()->window_dimensions();

        // Calculate window dimensions
        Size2 wdim = calculateWindowSize(output_size, window_size);

        auto scale = wdim.div(output_size);
        section->set_scale(scale);

        LockGuard lguard(ro_t{}, "drawing", 10);
        if (not lguard.locked()) {
            section->reuse_objects();
            return;
        }

        SETTING(gui_frame) = _current_data.frame.index();

        if (_background_image->source()) {
            graph.wrap_object(*_background_image);
        }

        for (auto box : _current_data.tiles)
            graph.rect(box, attr::FillClr{Transparent}, attr::LineClr{Red});

        static Frame_t last_frame;
        bool dirty{ false };
        if (last_frame != _current_data.frame.index()) {
            last_frame = _current_data.frame.index();
            _gui_objects.clear();
            _individual_properties.clear();
            dirty = true;
        }

        // Draw outlines
        drawOutlines(graph, scale);

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
                        visible_bdx[basic->parent_id] = fish->identity();
                    visible_bdx[basic->blob_id()] = fish->identity();
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

        //! do not need to continue further if the view isnt dirty
        if (not dirty)
            return;

        drawBlobs(meta_classes, scale, visible_bdx);
        });

    graph.section("menus", [&](auto&, Section* section) {
        section->set_scale(graph.scale().reciprocal());
        _video_info["frame"] = _current_data.frame.index();
        _actual_frame = _current_data.frame.source_index();
        _video_frame = _current_data.frame.index();

        menu.draw(*(IMGUIBase*)window(), graph);
        });
}

void ConvertScene::generator_thread() {
    set_thread_name("GeneratorT");
    std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;

    std::unique_lock guard(_mutex_general);
    while (not _should_terminate) {
        try {
            if (not _next_frame_data and not items.empty()) {
                if (std::get<1>(items.front()).valid()
                    && std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
                {
                    auto data = std::get<1>(items.front()).get();
                    //thread_print("Got data for item ", data.frame.index());

                    _next_frame_data = std::move(data);
                    _cv_ready_for_tracking.notify_one();

                    items.erase(items.begin());

                }
                else if (not std::get<1>(items.front()).valid()) {
                    FormatExcept("Invalid future ", std::get<0>(items.front()));

                    items.erase(items.begin());
                }
            }

            auto result = _overlayed_video->generate();
            if (not result) {
                //_overlayed_video->reset(0_f);
            }
            else {
                items.push_back(std::move(result.value()));
            }

        }
        catch (...) {
            // pass
        }

        if (items.size() >= 10 && _next_frame_data) {
            //thread_print("Entering wait with ", items.size(), " items queued up.");
            _cv_messages.wait(guard, [&]() {
                return not _next_frame_data or _should_terminate;
                });
            //thread_print("Received notification: next(", (bool)next, ") and ", items.size()," items in queue");
        }
    }

    thread_print("ended.");
};

void ConvertScene::perform_tracking() {
    static Frame_t running_id = 0_f;
    auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
    _progress_data.frame.set_timestamp(uint64_t(fake));
    _progress_data.frame.set_index(running_id++);
    _progress_data.frame.set_source_index(Frame_t(_progress_data.image->index()));
    assert(_progress_data.frame.source_index() == Frame_t(_progress_data.image->index()));

    _progress_blobs.clear();
    for (size_t i = 0; i < _progress_data.frame.n(); ++i) {
        _progress_blobs.emplace_back(_progress_data.frame.blob_at(i));
    }

    if (SETTING(is_writing)) {
        if (not _output_file->is_open()) {
            _output_file->set_start_time(_start_time);
            _output_file->set_resolution(_output_size);
        }
        _output_file->add_individual(pv::Frame(_progress_data.frame));
    }

    {
        PPFrame pp;
        Tracker::preprocess_frame(pv::Frame(_progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, false);
        _tracker->add(pp);
        /*if (pp.index().get() % 100 == 0) {
            print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
        }*/
    }

    {
        std::unique_lock guard(_mutex_current);
        //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
        _transferred_current_data = std::move(_progress_data);
        _transferred_blobs = std::move(_progress_blobs);
    }

    static Timer last_add;
    static double average{ 0 }, samples{ 0 };
    auto c = last_add.elapsed();
    average += c;
    ++samples;


    static Timer frame_counter;
    static size_t num_frames{0};
    static std::mutex mFPS;
    static double FPS{ 0 };

    {
        std::unique_lock g(mFPS);
        num_frames++;

        if (frame_counter.elapsed() > 30) {
            FPS = num_frames / frame_counter.elapsed();
            num_frames = 0;
            AbstractBaseVideoSource::_fps = FPS;
            AbstractBaseVideoSource::_samples = 1;
            frame_counter.reset();
            print("FPS: ", FPS);
        }

    }

    if (samples > 1000) {
        print("Average time since last frame: ", average / samples * 1000.0, "ms (", c * 1000, "ms)");

        average /= samples;
        samples = 1;
    }
    last_add.reset();
};

void ConvertScene::tracking_thread() {
    set_thread_name("Tracking thread");
    std::unique_lock guard(_mutex_general);
    while (not _should_terminate) {
        if (_next_frame_data) {
            try {
                _progress_data = std::move(_next_frame_data);
                assert(not _next_frame_data);
                //thread_print("Got next: ", progress.frame.index());
            }
            catch (...) {
                FormatExcept("Exception while moving to progress");
                continue;
            }
            //guard.unlock();
            //try {
            if (_overlayed_video->source->is_finite()) {
                auto L = _overlayed_video->source->length();
                auto C = _progress_data.original_index();

                if (L.valid() && C.valid()) {
                    size_t percent = float(C.get()) / float(L.get()) * 100;
                    //print(C, " / ", L, " => ", percent);
                    static size_t last_progress = 0;
                    if (abs(float(percent) - float(last_progress)) >= 1)
                    {
                        bar.set_progress(percent);
                        last_progress = percent;
                    }
                }
            }
            else {
                spinner.tick();
            }

            perform_tracking();
            //guard.lock();
        //} catch(...) {
        //    FormatExcept("Exception while tracking");
        //    throw;
        //}
        }

        //thread_print("Waiting for next...");
        _cv_messages.notify_one();
        if (not _should_terminate)
            _cv_ready_for_tracking.wait(guard);
        //thread_print("Received notification: next(", (bool)next,")");
    }
    thread_print("Tracking ended.");
};

}
