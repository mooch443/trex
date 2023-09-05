#include "Scene.h"


namespace gui {

IMPLEMENT(SceneManager::_switching_error);

SceneManager& SceneManager::getInstance() {
    static SceneManager instance;  // Singleton instance
    return instance;
}

void SceneManager::set_active(Scene* scene) {
    auto fn = [this, scene]() {
        if (active_scene && active_scene != scene) {
            active_scene->deactivate();
        }
        last_active_scene = active_scene;
        active_scene = scene;
        if (scene)
            scene->activate();
    };
    enqueue(fn);
}

Scene* SceneManager::last_active() const {
    std::unique_lock guard{_mutex};
    return last_active_scene;
}

void SceneManager::register_scene(Scene* scene) {
    std::unique_lock guard{_mutex};
    _scene_registry[scene->name()] = scene;
}

void SceneManager::unregister_scene(Scene* scene) {
    std::unique_lock guard{_mutex};
    for(auto&[name, s] : _scene_registry) {
        if(s == scene) {
            _scene_registry.erase(name);
            break;
        }
    }
}

void SceneManager::set_active(std::string name) {
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

SceneManager::~SceneManager() {
    update_queue();
    if (active_scene)
        active_scene->deactivate();
}

void SceneManager::update(DrawStructure& graph) {
    update_queue();
    if (active_scene)
        active_scene->draw(graph);
    
    auto str = switching_error().read();
    if(not str.empty()) {
        graph.dialog(str);
    }
}

void SceneManager::update_queue() {
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

}
