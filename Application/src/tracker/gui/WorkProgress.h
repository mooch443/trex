#pragma once

#include <types.h>
#include <gui/DrawStructure.h>
#include <gui/types/Drawable.h>

namespace gui {

struct WorkItem {
    std::function<void()> fn;
    std::string name, desc;
    bool abortable;
    std::string custom_button;
    
    WorkItem(std::function<void()> fn, const std::string& name, const std::string& desc, bool abortable = false, std::string custom_button = "")
        : fn(fn), name(name), desc(desc), abortable(abortable), custom_button(custom_button)
    {}
};

struct WorkInstance {
    std::string _name, _previous;
    
    WorkInstance(const std::string& name);
    ~WorkInstance();
};

class WorkProgress {
    std::condition_variable _condition;
    std::mutex _queue_lock;
    std::queue<WorkItem> _queue;
    std::atomic_bool _terminate_threads;
    
    std::thread *_thread;
    
    GETTER_SETTER(std::string, item)
    std::atomic_bool _item_abortable, _item_aborted, _item_custom_triggered;
    std::string _description;
    std::string _custom_button_text;
    gui::Entangled _additional;
    std::queue<std::function<void(Entangled&)>> _additional_updates;
    std::atomic<float> _percent;
    std::map<std::string, Image::Ptr> _images;
    std::map<std::string, std::unique_ptr<ExternalImage>> _gui_images;
    
public:
    WorkProgress();
    ~WorkProgress();
    
    bool item_aborted();
    bool item_custom_triggered();
    void reset_custom_item();
    bool has_custom_button() const;
    
    void add_queue(const std::string& message, const std::function<void()>& fn, const std::string& descr = "", bool abortable = false);
    void abort_item();
    void custom_item();
    
    void set_item_abortable(bool abortable);
    void set_custom_button(const std::string& text);
    void update(gui::DrawStructure &base, gui::Section *section);
    
    void set_progress(const std::string& title, float value, const std::string& description = "");
    
    std::atomic<float>& percent();
    float percent() const;
    void set_percent(float value);
    
    void set_image(const std::string& name, const Image::Ptr& image);
    
    std::string description();
    void set_description(const std::string& value);
    
    bool has_additional();
    void update_additional(std::function<void(Entangled&)> fn);
};

}
