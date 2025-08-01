#pragma once

#include <commons.pc.h>
#include <gui/GUITaskQueue.h>
#include <misc/derived_ptr.h>
#include <gui/Event.h>

namespace cmn::gui {

//class GUITaskQueue_t;
class DrawStructure;
class IMGUIBase;
class Base;
class Drawable;

class Scene {
    GETTER(std::string, name);
    std::vector<derived_ptr<Drawable>> _children;
    Base* _window{ nullptr };
    std::function<void(Scene&, DrawStructure& base)> _draw;

    static inline std::mutex _mutex;
    static inline std::unordered_map<const DrawStructure*, std::string> _active_scenes;

public:
    Scene(Base& window, const std::string& name, std::function<void(Scene&, DrawStructure& base)> draw);

    auto window() const { return _window; }
    virtual ~Scene();

    virtual void activate();
    virtual void deactivate();
    
    void draw(DrawStructure& base);
    
    virtual bool on_global_event(Event);
};

class SceneManager {
    Scene* fallback_scene{ nullptr };
    Scene* active_scene{ nullptr };
    Scene* last_active_scene{nullptr};
    std::map<std::string, Scene*> _scene_registry;
    std::queue<std::tuple<const Scene*, package::F<void()> >> _queue;
    std::unique_ptr<GUITaskQueue_t> _gui_queue;
    Size2 last_resolution;
    double last_dpi{0};
    mutable std::mutex _mutex;
    std::optional<std::thread::id> _gui_thread_id;

    // Private constructor to prevent external instantiation
    SceneManager();

    static read_once<std::string> _switching_error;
    static std::atomic<bool> _displaying_error;
    
public:
    struct AlwaysAsync {};
    
    static auto& switching_error() {
        return _switching_error;
    }
    
    static void set_switching_error(std::string str) {
        /// we are ignoring the return value here because
        /// we are only interested in the _first_ error.
        std::ignore = switching_error().set(str);
    }

public:
    // Deleted copy constructor and assignment operator
    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    static SceneManager& getInstance();

    void set_active(Scene* scene);
    bool is_scene_registered(std::string) const;

    void register_scene(Scene* scene);
    void unregister_scene(Scene* scene);

    void set_active(std::string name);
    Scene* last_active() const;

    void set_fallback(std::string name);

    ~SceneManager();

    void update(IMGUIBase*, DrawStructure& graph);

    void update_queue();
    bool on_global_event(Event);
    void clear();
    
    static bool is_gui_thread();
    
    template<typename F>
    static void enqueue(F&& task) {
        getInstance()._enqueue(std::move(task));
    }
    
    template<typename F>
    static void enqueue(AlwaysAsync, F&& task) {
        getInstance()._enqueue(AlwaysAsync{}, std::move(task));
    }
    
    GUITaskQueue_t* gui_task_queue() const;
    
private:
    template<typename F>
        requires (not std::is_invocable_v<F, IMGUIBase*, DrawStructure&>)
    void _enqueue(F&& task) {
        if(is_gui_thread()) {
            execute_task(std::move(task));
            return;
        }
        
        _enqueue(AlwaysAsync{}, std::move(task));
    }
    
    template<typename F>
        requires (not std::is_invocable_v<F, IMGUIBase*, DrawStructure&>)
    void _enqueue(AlwaysAsync, F&& task) {
        std::unique_lock guard(_mutex);
        _queue.emplace(active_scene, package::F<void()>(std::move(task)));
        /*_queue.emplace(active_scene, [task = std::move(task)]() mutable{
            task();
        });*/
    }
    
    template<typename F>
        requires (std::is_invocable_v<F, IMGUIBase*, DrawStructure&>)
    void _enqueue(F&& task) {
        std::unique_lock guard(_mutex);
        if(not _gui_queue)
            return;
        
        _gui_queue->enqueue([this, scene = active_scene, task = std::move(task)](IMGUIBase* gui, DrawStructure& base) mutable {
            if(scene && active_scene != scene) {
#ifndef NDEBUG
                FormatWarning("Will not execute task for scene ", scene->name(), " as it is no longer active.");
#endif
                return;
            }

            task(gui, base);
        });
    }
    
    template<typename F>
    void execute_task(F&& fn) {
        assert(is_gui_thread());

        try {
            fn();
        }
        catch (...) {
            // pass
        }
    }
};

}  // namespace gui
