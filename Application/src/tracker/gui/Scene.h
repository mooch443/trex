#pragma once

#include <commons.pc.h>
#include <gui/DrawBase.h>
#include <gui/DrawStructure.h>
#include <gui/types/Layout.h>

namespace gui {

class Scene {
    GETTER(std::string, name);
        std::vector<Layout::Ptr> _children;
    Base* _window{ nullptr };
    std::function<void(Scene&, DrawStructure& base)> _draw;

    static inline std::mutex _mutex;
    static inline std::unordered_map<const DrawStructure*, std::string> _active_scenes;

public:
    Scene(Base& window, const std::string& name, std::function<void(Scene&, DrawStructure& base)> draw)
        : _name(name), _window(&window), _draw(draw)
    {

    }

    auto window() const { return _window; }
    virtual ~Scene() {
        deactivate();
    }

    virtual void activate() {
        print("Activating scene ", _name);
    }
    virtual void deactivate() {
        print("Deactivating scene ", _name);
    }

    void draw(DrawStructure& base) {
        _draw(*this, base);
    }
    
    virtual bool on_global_event(Event) { return false; }
};

class SceneManager {
    Scene* fallback_scene{ nullptr };
    Scene* active_scene{ nullptr };
    Scene* last_active_scene{nullptr};
    std::map<std::string, Scene*> _scene_registry;
    std::queue<std::function<void()>> _queue;
    mutable std::mutex _mutex;

    // Private constructor to prevent external instantiation
    SceneManager() {}

    static read_once<std::string> _switching_error;
    
public:
    static auto& switching_error() {
        return _switching_error;
    }
    
    static void set_switching_error(const std::string& str) {
        switching_error().set(str);
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

    void update(DrawStructure& graph);

    void update_queue();
    bool on_global_event(Event);
    void clear();

private:
    void enqueue(auto&& task) {
        std::unique_lock guard(_mutex);
        _queue.push(std::move(task));
    }
};

}  // namespace gui
