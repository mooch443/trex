#include "WorkProgress.h"
#include <gui/GuiTypes.h>
#include <gui/types/StaticText.h>
#include <gui/types/Entangled.h>
#include <gui/types/Button.h>
#ifdef WIN32
#include <ShObjIdl_core.h>
#endif
#include <gui/IMGUIBase.h>
#ifdef WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_EXPOSE_NATIVE_NSGL
#include <gui/MacProgressBar.h>
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3.h>
#if !defined(__EMSCRIPTEN__)
#include <GLFW/glfw3native.h>
#endif

namespace gui {

namespace work {
std::condition_variable _condition;
std::mutex _queue_lock;
std::queue<WorkItem> _queue;
std::atomic_bool _terminate_threads{false};

std::thread *_thread{nullptr};
std::thread::id _work_thread_id;

std::string _item;
std::atomic_bool _item_abortable{false}, _item_aborted{false}, _item_custom_triggered{false};
std::string _description;
std::string _custom_button_text;
std::queue<std::function<void(Entangled&)>> _additional_updates;
std::atomic<float> _percent{0.0f};
std::map<std::string, Image::Ptr> _images;
std::map<std::string, std::unique_ptr<ExternalImage>> _gui_images;
#if WIN32
ITaskbarList3* ptbl = NULL;
#endif

template <typename ... Args>
constexpr bool return_void(void(Args ...)) { return true; }

template <typename R, typename ... Args>
constexpr bool return_void(R(Args ...)) { return false; }

template<typename F>
    requires (!std::same_as<std::invoke_result_t<F>, void>)
auto check(F && fn) -> std::invoke_result_t<F> {
    using R = std::invoke_result_t<F>;
    std::promise<R> promise;
    auto f = promise.get_future();
    WorkProgress::instance();
    promise.set_value(fn());
    return f.get();
}

template<typename F>
    requires std::same_as<std::invoke_result_t<F>, void>
auto check(F&& fn) -> std::invoke_result_t<F> {
    using R = std::invoke_result_t<F>;
    std::promise<R> promise;
    auto f = promise.get_future();
    WorkProgress::instance();
    fn();
    promise.set_value();
}

}

using namespace work;

struct WorkProgress::WorkGUIObjects {
    Rect static_background{Box(0, 0, 0, 0), FillClr{Black.alpha(150)}};
    StaticText static_desc{Str("description"), Font(0.7, Align::Center)};
    //static StaticText static_additional("", Vec2(), Size2(-1), Font(0.7, Align::Center));
    Button static_button{Str{"abort"}, Box(0, 0, 100, 35)};
    Button custom_static_button{Str{"custom"}, Box(0, 0, 100, 35)};
    Entangled _additional;
    Entangled work_progress;
    VerticalLayout layout;
    
    WorkGUIObjects& operator=(const WorkGUIObjects&) = delete;
    WorkGUIObjects& operator=(WorkGUIObjects&&) = default;
};

WorkInstance::WorkInstance(const std::string& name)
    : _name(name), _previous(WorkProgress::item())
{
    print("Setting work item to ",_name);
    WorkProgress::set_item(name);
}

WorkInstance::~WorkInstance() {
    print("Resetting work item to ",_previous);
    WorkProgress::set_item(_previous);
}

WorkProgress& WorkProgress::instance() {
    static WorkProgress _instance;
    return _instance;
}

WorkProgress::WorkProgress() {
#ifdef __APPLE__
    MacProgressBar::set_visible(false);
#endif
    start();
}

WorkProgress::~WorkProgress() {
    stop();
}

void WorkProgress::start() {
    std::unique_lock guard(start_mutex);
    if(_thread)
        return;
    
    _terminate_threads = false;
    _thread = new std::thread([&]() {
        std::unique_lock lock(_queue_lock);
        set_thread_name("GUI::_work_thread");
        _work_thread_id = std::this_thread::get_id();
        
        while (!_terminate_threads) {
            _condition.wait_for(lock, std::chrono::seconds(1));
            
            while(!_queue.empty() && !_terminate_threads) {
                auto item =  _queue.front();
#if defined(__APPLE__)
                MacProgressBar::set_visible(true);
#endif
                set_percent(0);

                _item = item.name;
                _description = item.desc;
                while(!_additional_updates.empty())
                    _additional_updates.pop();
                _item_abortable = item.abortable;
                _custom_button_text = item.custom_button;
                _item_custom_triggered = false;
                _item_aborted = false;
                _queue.pop();
                
                lock.unlock();
                
                if(std::unique_lock guard(instance().gui_mutex);
                   instance().gui)
                {
                    auto stage = instance().gui->_additional.stage();
                    if(stage) {
                        auto guard = GUI_LOCK(stage->lock());
                        instance().gui->_additional.update([](auto&){});
                    }
                }
                item.fn();
                lock.lock();
                
                _images.clear();
                _gui_images.clear();
                
                _item = "";
                set_percent(0);
            }
        }
    });
}

void WorkProgress::stop() {
    // strong exchange, since we want to make sure that the thread is not running anymore
    if(!_terminate_threads.exchange(true)) {
        _condition.notify_all();
        
        if(std::unique_lock guard(instance().start_mutex);
           _thread)
        {
            _thread->join();
            delete _thread;
            _thread = nullptr;
        }
    }
    
    instance().gui = nullptr;
}

void WorkProgress::set_item(const std::string &item) {
    work::check([&](){
        _item = item;
    }); // thread + safety check
}

const std::string& WorkProgress::item() {
    return work::check([&]() -> const std::string& {
        return _item;
    });
}

bool WorkProgress::is_this_in_queue() {
    return std::this_thread::get_id() == _work_thread_id;
}

void WorkProgress::add_queue(const std::string& message, const std::function<void()>& fn, const std::string& descr, bool abortable)
{
    work::check([&](){
        {
            std::unique_lock<std::mutex> work(_queue_lock);
            _queue.push(WorkItem(fn, message, descr, abortable));
        }
        
        _condition.notify_all();
        
    });
}

void WorkProgress::abort_item() {
    work::check([&](){
        _item_aborted = true;
    });
}

bool WorkProgress::item_custom_triggered() {
    return work::check([&](){
        return _item_custom_triggered.load();
    });
}

void WorkProgress::custom_item() {
    work::check([&](){
        _item_custom_triggered = true;
    });
}

void WorkProgress::reset_custom_item() {
    work::check([&](){
        _item_custom_triggered = false;
    });
}

bool WorkProgress::has_custom_button() {
    return work::check([&](){
        return !_custom_button_text.empty();
    });
}

void WorkProgress::set_custom_button(const std::string& text) {
    work::check([&](){
        _custom_button_text = text;
    });
}

bool WorkProgress::item_aborted() {
    return work::check([&](){
        return _item_aborted.load();
    });
}

void WorkProgress::set_item_abortable(bool abortable) {
    work::check([&](){
        _item_abortable = abortable;
    });
}

void WorkProgress::update_taskbar(IMGUIBase* base) {
    if (not base)
        return;

    work::check([base]() {
        if (_percent <= 0 || _percent >= 1) {
#if defined(WIN32)
            if (ptbl) {
                HWND hwnd = glfwGetWin32Window(base->platform()->window_handle());
                ptbl->SetProgressState(hwnd, TBPF_NOPROGRESS);
            }
#elif defined(__APPLE__)
            MacProgressBar::set_visible(false);
#endif
#if !defined(WIN32)
            UNUSED(base);
#endif
            return;
        }

#if WIN32
        if (!ptbl) {
            // initialize the COM interface
            if (SUCCEEDED(CoInitialize(NULL))) {
                HRESULT hr = CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&ptbl));

                if (SUCCEEDED(hr))
                {
                    HRESULT hr2 = ptbl->HrInit();
                    if (!SUCCEEDED(hr2)) {
                        ptbl->Release();
                        ptbl = nullptr;
                    }

                }
                else {
                    FormatWarning("ITaskbarList3 could not be created.");
                }
            }
        }

#if !defined(__EMSCRIPTEN__)
        // only if it works... display Taskbar progress on Windows
        if (ptbl) {
            const ULONGLONG percent = (ULONGLONG)max(1.0, double(_percent) * 100.0);
            HWND hwnd = glfwGetWin32Window(base->platform()->window_handle());

            if (_percent > 0) {
                // show progress in green
                ptbl->SetProgressState(hwnd, TBPF_NORMAL);
                ptbl->SetProgressValue(hwnd, percent, 100ul);
            }
            else {
                // display "pause" color if no progress has been made
                ptbl->SetProgressState(hwnd, TBPF_PAUSED);
                ptbl->SetProgressValue(hwnd, 100ul, 100ul);
            }
#endif
        }
#elif defined(__APPLE__)
        MacProgressBar::set_percent(_percent);
#endif
    });
}

void WorkProgress::set_percent(float value) {
    work::check([&](){
        if(_percent == value)
            return;
        _percent = value;
    });
}

float WorkProgress::percent() {
    return work::check([&](){
        return _percent.load();
    });
}

void WorkProgress::set_description(const std::string& value) {
    work::check([&](){
        std::lock_guard<std::mutex> guard(_queue_lock);
        _description = value;
    });
}

void WorkProgress::set_image(const std::string& name, Image::Ptr&& image) {
    work::check([&](){
        std::lock_guard<std::mutex> guard(_queue_lock);
        _images[name] = std::move(image);
    });
}

std::string WorkProgress::description() {
    return work::check([&](){
        std::lock_guard<std::mutex> guard(_queue_lock);
        return _description;
    });
}

bool WorkProgress::has_additional() {
    return work::check([&](){
        if(not instance().gui)
            return false;
        
        auto stage = instance().gui->_additional.stage();
        if(stage) {
            auto guard = GUI_LOCK(stage->lock());
            return !instance().gui->_additional.children().empty();
        }
        return false;
    });
}

void WorkProgress::update_additional(std::function<void(Entangled&)> fn) {
    work::check([&](){
        std::lock_guard<std::mutex> guard(_queue_lock);
        _additional_updates.push(fn);
    });
}

void WorkProgress::set_progress(const std::string& title, float value, const std::string& desc) {
    work::check([&](){
        std::lock_guard<std::mutex> guard(_queue_lock);
        if(!title.empty()) {
            if(_item != title && !title.empty())
                print("[WORK] ", title.c_str());
            _item = title;
        }
        if(!desc.empty())
            _description = desc;
        if(value >= 0)
            set_percent(value);
    });
}


using namespace gui;
void WorkProgress::update(IMGUIBase* window, gui::DrawStructure &base, gui::Section *section, Size2 screen_dimensions) {
    work::check([&, &gui = instance().gui](){
        std::lock_guard<std::mutex> wlock(_queue_lock);
        if(_item.empty())
            return;
        
        std::unique_lock guard(instance().gui_mutex);
        if(not gui)
            gui = std::make_unique<WorkGUIObjects>();
        
        while(!_additional_updates.empty()) {
            auto && fn = _additional_updates.front();
            gui->_additional.auto_size(Margin{0,0});
            gui->_additional.update(fn);
            gui->_additional.auto_size(Margin{0,0});
            _additional_updates.pop();
        }
        
        static long_t abort_handler = -1, custom_handler = -1;
        section->set_scale(base.scale().reciprocal());
        
        const Vec2 bg_offset{0};
        gui->static_background.set_bounds(Bounds(-bg_offset, Size2(screen_dimensions.width, screen_dimensions.height).mul(section->scale().reciprocal())));
        gui->static_background.set_clickable(true);
        base.wrap_object(gui->static_background);
        
        Vec2 offset(0, 10);
        float width = 0;
        Vec2 center = (screen_dimensions * 0.5).mul(section->scale().reciprocal());
        base.wrap_object(gui->work_progress);
        
        screen_dimensions = screen_dimensions.mul(section->scale().reciprocal());
        
        //base.circle(center, 10, Red);
        //base.rect(Vec2(), screen_dimensions - Vec2(1), Transparent, Red);
        
        gui->work_progress.update([&](Entangled& base){
            const float margin = 5;
            
            auto text = base.add<Text>(Str(_item), Loc(offset), TextClr(0, 150, 225, 255), Font(0.8, Style::Bold), Origin(0.5, 0));
            offset.y += text->height() + margin;
            width = max(width, text->width());
            
            if(!_description.empty()) {
                gui->static_desc.set_txt(_description);
                gui->static_desc.set_pos(offset);
                gui->static_desc.set_origin(Vec2(0.5, 0));
                gui->static_desc.set_max_size(screen_dimensions * 0.66);
                gui->static_desc.set_background(Transparent, Transparent);
                
                base.advance_wrap(gui->static_desc);
                offset.y += gui->static_desc.height();
                width = max(width, gui->static_desc.width());
                //text = base.advance(new Text(_description, offset, Color(150, 150, 150, 255), Font(0.7, Align::Center), Vec2(1), Vec2(0.5, 0)));
                //offset.y += text->height() + margin;
                //width = max(width, text->width());
            }
            
            if(has_additional()) {
                base.advance_wrap(gui->_additional);
                
                gui->_additional.set_origin(Vec2(0.5, 0));
                gui->_additional.set_pos(offset);
                gui->_additional.set_background(Transparent, Transparent);
                gui->_additional.auto_size(Margin{0,0});
                
                offset.y += gui->_additional.height() + 10;
                width = max(width, gui->_additional.width());
            }
            
            if(_percent)
            {
                //Vec2 bar_offset(- bar_size.x * 0.5, size.y - bar_size.y - 10 - size.y * 0.5);
                Size2 bar_size(width, 30);
                
                base.add<Rect>(Box(Vec2(0, offset.y), bar_size), FillClr{White.alpha(100)}, LineClr{Black}, Origin(0.5, 0));
                auto bar = base.add<Rect>(Box(Vec2(1, 1 + offset.y), Size2(bar_size.width * saturate(_percent.load(), 0.f, 1.f) - 2, bar_size.height - 2)), FillClr{White.alpha(180)}, LineClr{White}, Origin(0.5, 0));
                offset += Vec2(0, bar->height() + margin);
                width = max(width, bar->width());
            }
            
            width += 30;
            offset.y += 10;
            
            if(_item_abortable && !_item_aborted) {
                gui->static_button.set_pos(Vec2(width * 0.5 - (has_custom_button() ? gui->static_button.width() * 0.5 + 5 : 0), offset.y));
                gui->static_button.set_origin(Vec2(0.5, 0));
                if(abort_handler == -1) {
                    abort_handler = 1;
                    gui->static_button.on_click([](auto){
                        WorkProgress::abort_item();
                    });
                }
                
                base.advance_wrap(gui->static_button);
                if(!has_custom_button()) // only go downwards once
                    offset.y += gui->static_button.height() + 10;
            }
            
            if(has_custom_button()) {
                gui->custom_static_button.set_txt(_custom_button_text);
                gui->custom_static_button.set_pos(Vec2(width * 0.5 + (_item_abortable && !_item_aborted ? gui->custom_static_button.width() * 0.5 + 5 : 0), offset.y));
                gui->custom_static_button.set_origin(Vec2(0.5, 0));
                if(custom_handler == -1) {
                    custom_handler = 1;
                    gui->custom_static_button.on_click([](auto){
                        print("Custom item triggered");
                        WorkProgress::custom_item();
                    });
                }
                
                base.advance_wrap(gui->custom_static_button);
                offset.y += gui->custom_static_button.height() + 10;
            }
        });
        
        if(gui->work_progress.content_changed()) {
            for(auto c : gui->work_progress.children()) {
                if(!c)
                    continue;
                
                if(is_in(c, &gui->custom_static_button , &gui->static_button)) {
                    gui->static_button.set_pos(Vec2(width * 0.5 - (has_custom_button() ? gui->static_button.width() * 0.5 + 5 : 0), c->pos().y));
                    gui->custom_static_button.set_pos(Vec2(width * 0.5 + (_item_abortable && !_item_aborted ? gui->custom_static_button.width() * 0.5 + 5 : 0), c->pos().y));
                } else
                    c->set_pos(Vec2(width * 0.5, c->pos().y));
            }

            update_taskbar(window);
        }
        
        gui->work_progress.set_origin(Vec2(0.5));
        gui->work_progress.set_background(Black.alpha(125));
        
        Vec2 screen_offset;
        
        auto &work_images = _images;
        if(!work_images.empty()) {
            gui->layout.set_policy(VerticalLayout::Policy::CENTER);
            std::vector<Layout::Ptr> objects;
            
            for(auto && [name, image] : work_images) {
                ExternalImage* ptr;
                
                auto it = _gui_images.find(name);
                if(it == _gui_images.end()) {
                    _gui_images[name] = std::make_unique<ExternalImage>();
                    ptr = _gui_images[name].get();
                } else
                    ptr = it->second.get();
                
                ptr->set_source(Image::Make(*image));
                
                float scale = float(screen_dimensions.width - 200) / float(image->cols);
                if(scale > 1)
                    scale = 1;
                const float max_height = (screen_dimensions.height - gui->work_progress.local_bounds().height - 100) / float(work_images.size());
                if(image->rows * scale > max_height) {
                    scale = max_height / float(image->rows);
                }
                
                ptr->set_scale(Vec2(scale));
                objects.push_back(ptr);
            }
            
            base.wrap_object(gui->layout);
            
            gui->layout.set_children(objects);
            center.y = max(gui->work_progress.local_bounds().height * 0.5 + 50,
                           center.y - ((gui->layout.local_bounds().height + gui->work_progress.local_bounds().height * 0.5) * 0.5 + 15));
            
            gui->layout.set_pos(Vec2(center.x, center.y + gui->work_progress.local_bounds().height * 0.5 + 15));
            gui->layout.set_origin(Vec2(0.5, 0));
            
            screen_offset.y += gui->layout.local_bounds().height + 15;
        }
        
        //float real_y = center.y - offset.y * 0.5 - 15;
        //if(screen_offset.y >= real_y)
        //    center.y = real_y + screen_offset.y - real_y;
        gui->work_progress.set_bounds(Bounds(center, Size2(width, offset.y)));
    });
}

}
