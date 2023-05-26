#pragma once

#include <commons.pc.h>
#include <gui/DrawBase.h>
#include <gui/DrawStructure.h>

namespace gui {

class Scene {
    GETTER(std::string, name)
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
};

class SceneManager {
    Scene* active_scene{ nullptr };
    std::map<std::string, Scene*> _scene_registry;
    std::queue<std::function<void()>> _queue;
    std::mutex _mutex;

    // Private constructor to prevent external instantiation
    SceneManager() {}

public:
    // Deleted copy constructor and assignment operator
    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    static SceneManager& getInstance() {
        static SceneManager instance;  // Singleton instance
        return instance;
    }

    void set_active(Scene* scene) {
        auto fn = [this, scene]() {
            if (active_scene && active_scene != scene) {
                active_scene->deactivate();
            }
            active_scene = scene;
            if (scene)
                scene->activate();
        };
        enqueue(fn);
    }

    void register_scene(Scene* scene) {
        std::unique_lock guard{_mutex};
        _scene_registry[scene->name()] = scene;
    }

    void set_active(std::string name) {
        if (name.empty()) {
            set_active(nullptr);
            return;
        }

        Scene* ptr{ nullptr };

        if (std::unique_lock guard{_mutex};
            _scene_registry.contains(name))
        {
            ptr = _scene_registry.at(name);
        }

        if (ptr) {
            set_active(ptr);
        }
        else {
            throw std::invalid_argument("Cannot find the given Scene name.");
        }
    }

    ~SceneManager() {
        update_queue();
        if (active_scene)
            active_scene->deactivate();
    }

    void update(DrawStructure& graph) {
        update_queue();
        if (active_scene)
            active_scene->draw(graph);
    }

    void update_queue() {
        std::unique_lock guard{_mutex};
        while (not _queue.empty()) {
            auto f = std::move(_queue.front());
            _queue.pop();
            guard.unlock();
            try {
                f();
            }
            catch (...) {
                // pass
            }
            guard.lock();
        }
    }

private:
    void enqueue(auto&& task) {
        std::unique_lock guard(_mutex);
        _queue.push(std::move(task));
    }
};

}  // namespace gui