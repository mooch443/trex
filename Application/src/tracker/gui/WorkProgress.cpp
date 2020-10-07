#include "WorkProgress.h"
#include <gui/Timeline.h>
#include <gui/GuiTypes.h>
#include <gui/types/StaticText.h>
#include <gui/types/Entangled.h>
#include <gui/gui.h>

namespace gui {

WorkInstance::WorkInstance(const std::string& name)
    : _name(name), _previous(GUI::instance() ? GUI::work().item() : "")
{
    Debug("Setting work item to '%S'", &_name);
    if(GUI::instance())
        GUI::work().set_item(name);
}

WorkInstance::~WorkInstance() {
    Debug("Resetting work item to '%S'", &_previous);
    if(GUI::instance())
        GUI::work().set_item(_previous);
}

WorkProgress::WorkProgress()
    : _terminate_threads(false), _item_abortable(false), _item_aborted(false)
{
    _thread = new std::thread([&]() {
        std::unique_lock<std::mutex> lock(_queue_lock);
        set_thread_name("GUI::_work_thread");
        
        while (!_terminate_threads) {
            _condition.wait_for(lock, std::chrono::seconds(1));
            
            while(!_queue.empty()) {
                auto item =  _queue.front();
                //Debug("Starting item '%S'", &item.name);
                _percent = 0;
                _item = item.name;
                _description = item.desc;
                _additional.update([](auto&){});
                while(!_additional_updates.empty())
                    _additional_updates.pop();
                _item_abortable = item.abortable;
                _item_aborted = false;
                _queue.pop();
                
                lock.unlock();
                
                item.fn();
            
                //this->set_redraw();
                lock.lock();
                
                _images.clear();
                _gui_images.clear();
                
                //Debug("Finished item '%S'", &item.name);
                _item = "";
                //_condition.notify_one();
            }
        }
    });
}

WorkProgress::~WorkProgress() {
    _terminate_threads = true;
    _condition.notify_all();
    
    _thread->join();
    delete _thread;
}

void WorkProgress::add_queue(const std::string& message, const std::function<void()>& fn, const std::string& descr, bool abortable)
{
    if(!GUI::instance()) {
        Error("Cannot add work item '%S' to queue if GUI doesnt exist.", &message);
        return;
    }
    
    {
        std::unique_lock<std::mutex> work(_queue_lock);
        _queue.push(WorkItem(fn, message, descr, abortable));
    }
    
    _condition.notify_all();
}

void WorkProgress::abort_item() {
    _item_aborted = true;
}

bool WorkProgress::item_aborted() {
    return _item_aborted.load();
}

void WorkProgress::set_item_abortable(bool abortable) {
    _item_abortable = abortable;
}

void WorkProgress::set_percent(float value) {
    _percent = value;
}

std::atomic<float>& WorkProgress::percent() {
    return _percent;
}

float WorkProgress::percent() const {
    return _percent.load();
}

void WorkProgress::set_description(const std::string& value) {
    std::lock_guard<std::mutex> guard(_queue_lock);
    _description = value;
}

void WorkProgress::set_image(const std::string& name, const Image::Ptr& image) {
    std::lock_guard<std::mutex> guard(_queue_lock);
    _images[name] = image;
}

std::string WorkProgress::description() {
    std::lock_guard<std::mutex> guard(_queue_lock);
    return _description;
}

bool WorkProgress::has_additional() {
    auto& gui = GUI::instance()->gui();
    std::lock_guard<std::recursive_mutex> guard(gui.lock());
    return !_additional.children().empty();
}

void WorkProgress::update_additional(std::function<void(Entangled&)> fn) {
    std::lock_guard<std::mutex> guard(_queue_lock);
    _additional_updates.push(fn);
}

void WorkProgress::set_progress(const std::string& title, float value, const std::string& desc) {
    std::lock_guard<std::mutex> guard(_queue_lock);
    if(!title.empty()) {
        if(_item != title && !title.empty())
            Debug("[WORK] %S", &title);
        _item = title;
    }
    if(!desc.empty())
        _description = desc;
    if(value >= 0)
        _percent = value;
}

void WorkProgress::update(gui::DrawStructure &base, gui::Section *section) {
    std::lock_guard<std::mutex> wlock(_queue_lock);
    if(_item.empty())
        return;
    
    auto& gui = GUI::instance()->gui();
    auto window = GUI::instance()->base();
    
    while(!_additional_updates.empty()) {
        auto && fn = _additional_updates.front();
        _additional.auto_size(Margin{0,0});
        _additional.update(fn);
        _additional.auto_size(Margin{0,0});
        _additional_updates.pop();
    }
    
    auto && [bg_offset, max_w] = Timeline::timeline_offsets();
    static Rect static_background(Bounds(0, 0, max_w, GUI::background_image().rows), Black.alpha(150));
    static StaticText static_desc("description", Vec2(), Size2(-1), Font(0.7, Align::Center));
    //static StaticText static_additional("", Vec2(), Size2(-1), Font(0.7, Align::Center));
    static Button static_button("abort", Bounds(0, 0, 100, 35));
    static long_t abort_handler = -1;
    section->set_scale(base.scale().reciprocal());
    
    Size2 screen_dimensions = (window ? window->window_dimensions().div(gui.scale()) * gui::interface_scale() : GUI::background_image().dimensions());
    static_background.set_bounds(Bounds(-bg_offset, Size2(max_w / gui.scale().x, screen_dimensions.height).mul(section->scale().reciprocal())));
    static_background.set_clickable(true);
    base.wrap_object(static_background);
    
    static Entangled work_progress;
    static bool first = true;
    if(first) {
        GUI::static_pointers().push_back(&static_background);
        GUI::static_pointers().push_back(&static_desc);
        GUI::static_pointers().push_back(&static_button);
        GUI::static_pointers().push_back(&work_progress);
        //GUI::static_pointers().push_back(&static_additional);
        
        first = false;
    }
    
    Vec2 offset(0, 10);
    float width = 0;
    Vec2 center = (screen_dimensions * 0.5).mul(section->scale().reciprocal());
    base.wrap_object(work_progress);
    
    screen_dimensions = screen_dimensions.mul(section->scale().reciprocal());
    
    //base.circle(center, 10, Red);
    //base.rect(Vec2(), screen_dimensions - Vec2(1), Transparent, Red);
    
    work_progress.update([&](Entangled& base){
        const float margin = 5;
        
        auto text = base.advance(new Text(_item, offset, Color(0, 150, 225, 255), Font(0.8, Style::Bold), Vec2(1), Vec2(0.5, 0)));
        offset.y += text->height() + margin;
        width = max(width, text->width());
        
        if(!_description.empty()) {
            static_desc.set_txt(_description);
            static_desc.set_pos(offset);
            static_desc.set_origin(Vec2(0.5, 0));
            static_desc.set_background(Transparent, Transparent);
            
            base.advance_wrap(static_desc);
            offset.y += static_desc.height();
            width = max(width, static_desc.width());
            //text = base.advance(new Text(_description, offset, Color(150, 150, 150, 255), Font(0.7, Align::Center), Vec2(1), Vec2(0.5, 0)));
            //offset.y += text->height() + margin;
            //width = max(width, text->width());
        }
        
        if(has_additional()) {
            /*static_additional.set_txt(_additional);
            static_additional.set_pos(offset);
            static_additional.set_origin(Vec2(0.5, 0));
            static_additional.set_background(Transparent, Transparent);
            static_additional.set_base_text_color(Gray);
            
            base.advance_wrap(static_additional);*/
            
            base.advance_wrap(_additional);
            
            _additional.set_origin(Vec2(0.5, 0));
            _additional.set_pos(offset);
            _additional.set_background(Transparent, Transparent);
            _additional.auto_size(Margin{0,0});
            
            offset.y += _additional.height() + 10;
            width = max(width, _additional.width());
        }
        
        if(_percent)
        {
            //Vec2 bar_offset(- bar_size.x * 0.5, size.y - bar_size.y - 10 - size.y * 0.5);
            Size2 bar_size(width, 30);
            
            auto bar_bg = new Rect(Bounds(Vec2(0, offset.y), bar_size), Color(255, 255, 255, 100), Black.alpha(255));
            bar_bg->set_origin(Vec2(0.5, 0));
            base.advance(bar_bg);
            auto bar = base.advance(new Rect(Bounds(Vec2(1, 1 + offset.y), Size2(bar_size.width * _percent-2, bar_size.height-2)), Color(255, 255, 255, 180)));
            bar->set_origin(Vec2(0.5, 0));
            offset += Vec2(0, bar->height() + margin);
            width = max(width, bar->width());
        }
        
        width += 30;
        offset.y += 10;
        
        if(_item_abortable && !_item_aborted) {
            static_button.set_pos(Vec2(width * 0.5, offset.y));
            static_button.set_origin(Vec2(0.5, 0));
            if(abort_handler == -1) {
                abort_handler = 1;
                static_button.on_click([this](auto){
                    this->abort_item();
                });
            }
            
            base.advance_wrap(static_button);
            offset.y += static_button.height() + 10;
        }
    });
    
    if(work_progress.content_changed()) {
        for(auto c : work_progress.children())
            c->set_pos(Vec2(width * 0.5, c->pos().y));
    }
    
    work_progress.set_origin(Vec2(0.5));
    work_progress.set_background(Black.alpha(125));
    
    Vec2 screen_offset;
    
    auto work_images = _images;
    if(!work_images.empty()) {
        static VerticalLayout layout({});
        
        layout.set_policy(VerticalLayout::Policy::CENTER);
        std::vector<Layout::Ptr> objects;
        
        for(auto && [name, image] : work_images) {
            ExternalImage* ptr;
            
            auto it = _gui_images.find(name);
            if(it == _gui_images.end()) {
                _gui_images[name] = std::make_unique<ExternalImage>();
                ptr = _gui_images[name].get();
            } else
                ptr = it->second.get();
            
            ptr->set_source(std::make_unique<Image>(*image));
            
            float scale = float(screen_dimensions.width - 200) / float(image->cols);
            if(scale > 1)
                scale = 1;
            const float max_height = (screen_dimensions.height - work_progress.local_bounds().height - 100) / float(work_images.size());
            if(image->rows * scale > max_height) {
                scale = max_height / float(image->rows);
            }
            
            ptr->set_scale(Vec2(scale));
            objects.push_back(ptr);
        }
        
        base.wrap_object(layout);
        
        layout.set_children(objects);
        center.y = max(work_progress.local_bounds().height * 0.5 + 50,
                       center.y - ((layout.local_bounds().height + work_progress.local_bounds().height * 0.5) * 0.5 + 15));
        
        layout.set_pos(Vec2(center.x, center.y + work_progress.local_bounds().height * 0.5 + 15));
        layout.set_origin(Vec2(0.5, 0));
        
        screen_offset.y += layout.local_bounds().height + 15;
    }
    
    //float real_y = center.y - offset.y * 0.5 - 15;
    //if(screen_offset.y >= real_y)
    //    center.y = real_y + screen_offset.y - real_y;
    work_progress.set_bounds(Bounds(center, Size2(width, offset.y)));
}

}
