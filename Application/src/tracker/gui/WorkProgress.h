#pragma once

#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/types/Drawable.h>
#if WIN32
#include <ShObjIdl_core.h>
#endif

namespace cmn::gui {
class IMGUIBase;

struct WorkItem {
    std::function<void()> fn;
    std::string name, desc;
    bool abortable;
    std::string custom_button;
    std::promise<void> promise;
    
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
private:
    WorkProgress();
    
    struct WorkGUIObjects;
    mutable std::mutex gui_mutex;
    std::unique_ptr<WorkGUIObjects> gui;
    std::mutex start_mutex;

protected:
    static std::unique_ptr<WorkProgress>& raw_instance();
public:
    static WorkProgress& instance();
    static void stop();
    void start();
    
public:
    ~WorkProgress();
    
    static const std::string& item();
    static void set_item(const std::string&);
    
    static bool item_aborted();
    static bool item_custom_triggered();
    static void reset_custom_item();
    static bool has_custom_button();
    
    static std::future<void> add_queue(const std::string& message, const std::function<void()>& fn, const std::string& descr = "", bool abortable = false);
    static void abort_item();
    static void custom_item();
    
    static void set_item_abortable(bool abortable);
    static void set_custom_button(const std::string& text);
    static void update(IMGUIBase*, gui::DrawStructure &base, gui::Section *section, Size2 screen_size);
    
    static void set_progress(const std::string& title, float value, const std::string& description = "");
    
    //std::atomic<float>& percent();
    static float percent();
    static void set_percent(float value);
    
    static void set_image(const std::string& name, Image::Ptr&& image);
    
    static std::string description();
    static void set_description(const std::string& value);
    
    static bool has_additional();
    
    //! check whether the calling thread is the work-queue-thread
    static bool is_this_in_queue();
    
    static void update_additional(std::function<void(Entangled&)> fn);
    static void update_taskbar(IMGUIBase*);
};

}
