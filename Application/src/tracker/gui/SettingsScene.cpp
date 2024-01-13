#include "SettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <misc/RecentItems.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Checkbox.h>
#include <gui/dyn/Action.h>
#include <video/VideoSource.h>
#include <misc/AbstractVideoSource.h>
#include <misc/VideoVideoSource.h>
#include <misc/WebcamVideoSource.h>

namespace gui {

SettingsScene::SettingsScene(Base& window)
: Scene(window, "settings-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
_preview_image(std::make_shared<ExternalImage>())
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _preview_image->size());
    
    _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{});
    //_button_layout->set_pos(Vec2(1024 - 10, 550));
    //_button_layout->set_origin(Vec2(1, 0));
    
    
    _logo_title_layout->set_children({
        Layout::Ptr(_preview_image)
    });
    
    // Set the list and button layout to the main layout
    _main_layout.set_children({
        Layout::Ptr(_logo_title_layout),
        Layout::Ptr(_buttons_and_items)
    });
    //_main_layout.set_origin(Vec2(1, 0));
    
    group = ThreadManager::getInstance().registerGroup("VideoBackground");
    ThreadManager::getInstance().addThread(group, "preloader", ManagedThread{
        [this](auto& gid){
            bool expected{true};
            if(video_changed.compare_exchange_weak(expected, false)
               || not _source)
            {
                // video has changed! need to update
                auto path = SETTING(source).value<file::PathArray>();
                
                std::unique_ptr<AbstractBaseVideoSource> tmp;
                try {
                    if(path == file::PathArray{"webcam"}) {
                        try {
                            fg::Webcam cam;
                            cam.set_color_mode(ImageMode::RGB);
                            tmp = std::unique_ptr<AbstractBaseVideoSource>(new WebcamVideoSource{std::move(cam)});
                            
                            _last_image_timer.reset();
                        } catch(...) {
                            // webcam probably needs allowance
                            auto a = allowances.load();
                            if(a < 15) {
                                std::this_thread::sleep_for(std::chrono::seconds(1));
                                
                                if(allowances.compare_exchange_strong(a, a + 1)) {
                                    video_changed = true;
                                }
                                
                                ThreadManager::getInstance().notify(gid);
                            }
                        }
                        
                    } else {
                        VideoSource video(path);
                        tmp = std::unique_ptr<AbstractBaseVideoSource>(new VideoSourceVideoSource{ std::move(video) });
                    }
                    
                    if(tmp) {
                        _last_image_timer.reset();
                        _source = std::move(tmp);
                        last_frame.invalidate();
                    }
                } catch(...) {
                    // could not load
                }
            }
            
            if(not _source)
                return;
            
            if(not intermediate
               || _last_image_timer.elapsed() > 0.1
               || not _source->is_finite()) /// for webcams we need to keep grabbing
            {                               /// so we dont fill up the buffer queue
                auto e = _source->next();
                if(e.has_value()) {
                    auto &&[index, mat, image] = e.value();
                    if(intermediate)
                        _source->move_back(std::move(intermediate));
                    intermediate = std::move(mat);
                    _source->move_back(std::move(image));
                    _last_image_timer.reset();
                }
                else if(_source->is_finite()) {
                    _source->set_frame(0_f);
                    ThreadManager::getInstance().notify(gid);
                }
            }
            
            if(intermediate) {
                auto p = blur_percentage.load();
                
                //Size2 intermediate_size(mat->cols * 0.5, mat->rows * 0.5);
                auto size = Size2(intermediate->cols, intermediate->rows)
                                * (p <= 0
                                   ? 1.0
                                   : saturate(0.25 / p, 0.1, 1.0));
                
                auto mres = max_resolution.load();
                if(not mres.empty()) {
                    double ratio = max(1, min(size.width / mres.width,
                                              size.height / mres.height));
                    
                    size = Size2(size.width * ratio, size.height * ratio);
                    //print("Scaling to size ", size, " with ratio ", ratio);
                }
                
                if(std::unique_lock guard(image_mutex);
                   return_image)
                {
                    local_image = std::move(return_image);
                } else {
                    local_image = Image::Make();
                }
                
                local_image->create(size.height, size.width,
                                    intermediate->channels() == 3
                                    ? 4
                                    : intermediate->channels());
                
                if(intermediate->channels() == 3) {
                    useMat_t tmp;
                    cv::cvtColor(*intermediate, tmp, cv::COLOR_BGR2BGRA);
                    cv::resize(tmp, local_image->get(), size, 0, 0, cv::INTER_CUBIC);
                } else
                    cv::resize(*intermediate, local_image->get(), size, 0, 0, cv::INTER_CUBIC);
                
                uint8_t amount = saturate(0.02 * size.max(), 5, 25) * p;
                if(amount > 0) {
                    if(amount % 2 == 0)
                        amount++;
                    
                    cv::GaussianBlur(local_image->get(), local_image->get(), Size2(amount), 0);
                }
            
                if(local_image) {
                    std::unique_lock guard(image_mutex);
                    if(not transfer_image)
                        transfer_image = std::move(local_image);
                }
            }
            
            if(not _source->is_finite())
                ThreadManager::getInstance().notify(gid);
        }
    });
}

void SettingsScene::activate() {
    // Create a new HorizontalLayout for the buttons
    // Fill the recent items list
    /*auto items = RecentItems::read();
     items.show(*_recent_items);
     
     RecentItems::set_select_callback([](RecentItems::Item item){
     item._options.set_do_print(true);
     for (auto& key : item._options.keys())
     item._options[key].get().copy_to(&GlobalSettings::map());
     
     //RecentItems::open(item.operator DetailItem().detail(), GlobalSettings::map());
     //SceneManager::getInstance().set_active("convert-scene");
     SceneManager::getInstance().set_active("settings-menu");
     });*/
    //window()->set_window_size({1280,960});
    
    
    auto video_size = Size2(1200,920);
    auto work_area = ((const IMGUIBase*)window())->work_area();
    auto window_size = video_size;
    
    Bounds bounds(
        Vec2((work_area.width - work_area.x) / 2 - window_size.width / 2,
            work_area.height / 2 - window_size.height / 2 + work_area.y),
        window_size);
    
    print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
    bounds.restrict_to(work_area);
    print("Restricting bounds to work area: ", work_area, " -> ", bounds);

    print("setting bounds = ", bounds);
    //window()->set_window_size(window_size);
    window()->set_window_bounds(bounds);
    
    dyn::Modules::add(dyn::Modules::Module{
        ._name = "follow",
        ._apply = [](size_t index, dyn::State& state, const Layout::Ptr& o) {
            state.display_fns[index] = [o = o.get()](DrawStructure& g){
                o->set_pos(g.mouse_position() + Vec2(5));
            };
        }
    });
    ThreadManager::getInstance().startGroup(group);
    
    callback = GlobalSettings::map().register_callbacks({"source"}, [this](auto name) {
        if(name == "source") {
            // changed source, need to update background images
            video_changed = true;
            allowances = 0;
            ThreadManager::getInstance().notify(group);
        }
    });
    timer.reset();
}

void SettingsScene::deactivate() {
    if(callback)
        GlobalSettings::map().unregister_callbacks(std::move(callback));
    ThreadManager::getInstance().terminateGroup(group);
    _source = nullptr;
    
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    dynGUI.clear();
    dyn::Modules::remove("follow");
}

void SettingsScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = DynamicGUI{
            .path = "settings_layout.json",
            .graph = &graph,
            .context = {
                ActionFunc("set", [](Action action) {
                    if(action.parameters.size() != 2)
                        throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                    
                    auto parm = Meta::fromStr<std::string>(action.first());
                    if(not GlobalSettings::has(parm))
                        throw InvalidArgumentException("No parameter ",parm," in global settings.");
                    
                    auto value = action.last();
                    GlobalSettings::get(parm).get().set_value_from_string(value);
                }),
                ActionFunc("go-back", [](auto){
                    auto prev = SceneManager::getInstance().last_active();
                    if(prev)
                        SceneManager::getInstance().set_active(prev);
                    print("Going back");
                }),
                ActionFunc("convert", [](auto){
                    DebugHeader("Converting ", SETTING(source).value<file::PathArray>());
                    //SETTING(filename) = file::Path();
                    SceneManager::getInstance().set_active("convert-scene");
                }),
                ActionFunc("track", [](auto){
                    DebugHeader("Tracking ", SETTING(source).value<file::PathArray>());
                    //SETTING(filename) = file::Path();
                    SceneManager::getInstance().set_active("tracking-scene");
                }),
                ActionFunc("choose-source", [](auto){
                    print("choose-source");
                }),
                
                ActionFunc("change_scene", [](Action action) {
                    if(action.parameters.empty())
                        throw U_EXCEPTION("Invalid arguments for ", action, ".");

                    auto scene = Meta::fromStr<std::string>(action.first());
                    if(not SceneManager::getInstance().is_scene_registered(scene))
                        return false;
                    SceneManager::getInstance().set_active(scene);
                    return true;
                }),
                ActionFunc("choose-target", [](auto){
                    print("choose-target");
                }),
                ActionFunc("choose-model", [](auto){
                    print("choose-detection");
                }),
                ActionFunc("choose-region", [](auto){
                    print("choose-region");
                }),
                ActionFunc("choose-settings", [](auto){
                    print("choose-settings");
                }),
                ActionFunc("toggle-background-subtraction", [](auto){
                    SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
                }),
                VarFunc("settings_summary", [](const VarProps&) -> std::string {
                    return std::string(GlobalSettings::map().toStr());
                }),
                VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                    return window_size;
                })
            }
        };
    
    auto dt = saturate(animation_timer.elapsed(), 0.001, 1.0);
    //float target = graph.mouse_position().x / window_size.width;
    auto p = blur_percentage.load();
    p += (blur_target - p) * dt * 2.0;
    p = saturate(p, 0.0, 1.0);
    //print(p, " => ", target);
    blur_percentage = p;
    animation_timer.reset();
    
    double limit = 0.1;
    if(abs(blur_target - p) > 0.01)
        limit = 0.025;
    if(timer.elapsed() > limit) {
        std::unique_lock guard(image_mutex);
        if(transfer_image) {
            _preview_image->exchange_with(std::move(transfer_image));
            return_image = std::move(transfer_image);
            ThreadManager::getInstance().notify(group);
        }
        timer.reset();
    }
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width;
    //auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
    auto scale = Vec2(max(max_w / max(_preview_image->width(), 1), window()->window_dimensions().height / max(_preview_image->height(), 1)));
    
    auto w = Vec2(window()->window_dimensions().width, 
                  window()->window_dimensions().height);
    if(w != window_size) {
        window_size = w;
        max_resolution = w;
    }
    
    _preview_image->set_scale(scale);
    _preview_image->set_origin(Vec2(0.5));
    _preview_image->set_pos(window()->window_dimensions() * 0.5);
    _preview_image->set_color(White.alpha(100));
    
    graph.wrap_object(*_preview_image);
    //graph.wrap_object(_preview_image);
    dynGUI.update(nullptr/*, [this](auto &objs){
        objs.push_back(Layout::Ptr(_preview_image));
    }*/);
    
    _buttons_and_items->auto_size();
    _logo_title_layout->auto_size();
    _main_layout.auto_size();
}

bool SettingsScene::on_global_event(Event e) {
    if(e.type == EventType::KEY
       && not e.key.pressed) 
    {
        if(e.key.code == Keyboard::T) {
            if(blur_target < 1)
                blur_target = 1;
            else
                blur_target = 0;
            return true;
        }
    }
    return false;
}

}

