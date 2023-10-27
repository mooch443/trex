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

Size2 ConvertScene::output_size() const {
    return segmenter().output_size();
}

std::string window_title() {
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
menu{
    dyn::Context{
        ActionFunc("QUIT", [](auto) {
            auto& manager = SceneManager::getInstance();
            manager.set_active("starting-scene");
        }),
        ActionFunc("FILTER", [](auto) {
            static bool filter { false };
            filter = not filter;
            SETTING(do_filter) = filter;
        }),
        VarFunc("fps", [](VarProps) {
            return AbstractBaseVideoSource::_fps.load() / AbstractBaseVideoSource::_samples.load();
        }),
        VarFunc("net_fps", [](VarProps) {
            return AbstractBaseVideoSource::_network_fps.load() / AbstractBaseVideoSource::_network_samples.load();
        }),
        VarFunc("vid_fps", [](VarProps) {
            return AbstractBaseVideoSource::_video_fps.load() / AbstractBaseVideoSource::_video_samples.load();
        }),
        VarFunc("fish", [](VarProps) -> sprite::Map& {
            return fish;
        }),
        VarFunc("average_is_generating", [this](VarProps) {
            return _segmenter->is_average_generating();
        }),
        VarFunc("actual_frame", [this](VarProps) {
            return _actual_frame;
        }),
        VarFunc("video", [](VarProps) -> sprite::Map& {
            return _video_info;
        })
    },
    [this](const std::string& name) {
        if (name == "gui_frame") {
            this->segmenter().reset(SETTING(gui_frame).value<Frame_t>());
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
},

_on_activate(on_activate),
_on_deactivate(on_deactivate)

{
    _video_info.set_do_print(false);
    fish.set_do_print(false);

    menu.dynGUI.context.variables.emplace("fishes", new Variable([this](dyn::VarProps) -> std::vector<std::shared_ptr<VarBase_t>>&{
        return _gui_objects;
    }));
}

ConvertScene::~ConvertScene() {
    if (_segmenter)
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
    assert(_segmenter == nullptr);
    _segmenter = seg;
    if(seg) {
        seg->set_progress_callback([this](float percent){
            if(percent >= 0)
                bar.set_progress(percent);
            else if(last_tick.elapsed() > 1) {
                spinner.set_option(ind::option::PostfixText{"Recording ("+Meta::toStr(video_frame())+")..."});
                spinner.tick();
                last_tick.reset();
            }
        });
    }
}

void ConvertScene::deactivate() {
    try {
        bar.set_progress(100);
        bar.mark_as_completed();
        
        spinner.set_option(ind::option::ForegroundColor{ind::Color::green});
        spinner.set_option(ind::option::PrefixText{"✔"});
        spinner.set_option(ind::option::ShowSpinner{false});
        spinner.set_option(ind::option::PostfixText{"Done."});
        spinner.set_progress(100);
        spinner.mark_as_completed();
        
        _segmenter = nullptr;
        _object_blobs.clear();
        _current_data = {};
        menu.dynGUI.clear();
        
        if(_on_deactivate)
            _on_deactivate(*this);
        
        _scene_promise.set_value();
        
    } catch(const std::exception& e){
        FormatExcept(e.what());
        _scene_promise.set_exception(std::current_exception());
    }
}

void ConvertScene::open_video() {
    bar.set_option(ind::option::ShowPercentage{true});
    segmenter().open_video();
    
    _video_info["resolution"] = segmenter().output_size();
    _video_info["length"] = segmenter().video_length();
}

void ConvertScene::open_camera() {
    spinner.set_option(ind::option::PrefixText{"Recording..."});
    spinner.set_option(ind::option::ShowPercentage{false});
    segmenter().open_camera();
    
    _video_info["resolution"] = segmenter().output_size();
    _video_info["length"] = segmenter().video_length();
}

void ConvertScene::activate()  {
    _scene_promise = {};
    _scene_active = _scene_promise.get_future().share();

    try {
        if(_on_activate)
            _on_activate(*this);
        
        print("Loading source = ", SETTING(source).value<file::PathArray>());
        SETTING(meta_source_path) = SETTING(source).value<file::PathArray>().source();
        if (SETTING(source).value<file::PathArray>() == file::PathArray({file::Path("webcam")}))
            open_camera();
        else
            open_video();

        RecentItems::open(SETTING(source).value<file::PathArray>(), GlobalSettings::map());

        auto size = segmenter().size();
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
        window()->set_title(window_title());
        bar.set_progress(0);

    }
    catch (const std::exception& e) {
        FormatExcept("Exception when switching scenes: ", e.what());
        //_scene_promise.set_value();
        //deactivate();
        SceneManager::set_switching_error(e.what());
        SceneManager::getInstance().set_active("starting-scene");
        
        if(SETTING(scene_crash_is_fatal)) {
            throw U_EXCEPTION("Aborting since an exception here is considered a fatal error.");
        }
    }
}

void ConvertScene::fetch_new_data() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        set_thread_name("GUI");
    });

    auto&& [data, obj] = segmenter().grab();
    if(data.image) {
        _current_data = std::move(data);
        _object_blobs = std::move(obj);
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
        graph.text(Str(Meta::toStr(_current_data.outlines.size()) + " lines"), attr::Loc(10, 50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));

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
        if(Tracker::instance() && Tracker::background())
            tmp["px"] = blob->recount(FAST_SETTING(track_threshold), *Tracker::background());
        else
            tmp["px"] = -1;

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
        _gui_objects.emplace_back(new Variable([&, i = _individual_properties.size() - 1](VarProps) -> sprite::Map& {
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

        LockGuard lguard(w_t{}, "drawing", 10);
        if (not lguard.locked()) {
            section->reuse_objects();
            return;
        }

        SETTING(gui_frame) = _current_data.frame.index();

        if (_background_image->source()) {
            graph.wrap_object(*_background_image);
        }

        for (auto box : _current_data.tiles)
            graph.rect(Box(box), attr::FillClr{Transparent}, attr::LineClr{Red});
        ColorWheel wheel;
        for (auto& keypoint : _current_data.keypoints) {
            auto clr = wheel.next();
            auto last = keypoint.bones.back();
            for(auto& bone : keypoint.bones) {
                graph.circle(Loc{bone.x, bone.y}, LineClr{clr}, Radius{10}, FillClr{clr.alpha(50)});
                graph.line(Vec2{last.x, last.y}, {bone.x, bone.y}, 5, LineClr{clr.exposureHSL(0.5)});
                last = bone;
            }
        }

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

}
