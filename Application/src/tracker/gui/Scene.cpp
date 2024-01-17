#include "Scene.h"
#include <misc/default_settings.h>

namespace gui {

IMPLEMENT(SceneManager::_switching_error);

SceneManager& SceneManager::getInstance() {
    static SceneManager* instance = new SceneManager;  // Singleton instance
    return *instance;
}

void SceneManager::set_active(Scene* scene) {
    auto fn = [this, scene]() {
        try {
            if(active_scene == scene) return;

            if (active_scene && active_scene != scene) {
                print("[SceneManager] Deactivating ", active_scene->name());
                active_scene->deactivate();
            }
            last_active_scene = active_scene;
            active_scene = scene;
            if (scene) {
                print("[SceneManager] Switching to ", scene->name());
                scene->activate();
            } else
                print("[SceneManager] Deactivating.");
            
        } catch(const std::exception& e) {
            SceneManager::set_switching_error(e.what());

            if (SceneManager::getInstance().fallback_scene) {
				SceneManager::getInstance().set_active(SceneManager::getInstance().fallback_scene);
			}
            else {
				print("[SceneManager] No fallback scene for error: ", e.what());
			}
        }
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
        if (std::unique_lock guard{ _mutex };
            active_scene) {
            set_active(nullptr);
        }
        return;
    }

    Scene* ptr{ nullptr };

    if (std::unique_lock guard{_mutex};
        _scene_registry.contains(name))
    {
        ptr = _scene_registry.at(name);
    }

    if (ptr) {
        if(ptr != active_scene)
            set_active(ptr);
    }
    else {
        throw std::invalid_argument("Cannot find the given Scene name.");
    }
}

void SceneManager::set_fallback(std::string name) {
    if (name.empty()) {
		fallback_scene = nullptr;
		return;
	}

	Scene* ptr{ nullptr };

    if (std::unique_lock guard{ _mutex };
        _scene_registry.contains(name))
    {
		ptr = _scene_registry.at(name);
	}

    if (ptr) {
		fallback_scene = ptr;
	}
    else {
		throw InvalidArgumentException("Cannot find the given Scene name (",name,").");
	}
}

bool SceneManager::is_scene_registered(std::string name) const {
    Scene* ptr{ nullptr };
    
    if (std::unique_lock guard{_mutex};
        _scene_registry.contains(name))
    {
        ptr = _scene_registry.at(name);
    }
    
    return ptr != nullptr;
}

SceneManager::~SceneManager() {
    clear();
}

void SceneManager::clear() {
    set_active(nullptr);
    update_queue();
    
    std::unique_lock guard{_mutex};
    active_scene = nullptr;
    fallback_scene = nullptr;
    last_active_scene = nullptr;
    _scene_registry.clear();
    
    while (not _queue.empty()) {
        _queue.pop();
    }
}

void SceneManager::update(DrawStructure& graph) {
    update_queue();

    try {
        if (active_scene)
            active_scene->draw(graph);
    }
    catch (const std::exception& e) {
        static const char* msg = nullptr;
        if (msg != e.what()) {
			msg = e.what();
            graph.dialog(settings::htmlify(e.what()), "Error");
			//print("[SceneManager] Error: ", msg);
		}
	}
    
    auto str = switching_error().read();
    if(not str.empty()) {
        graph.dialog(settings::htmlify(str), "Error");
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

bool SceneManager::on_global_event(Event event) {
    if(active_scene) {
        return active_scene->on_global_event(event);
    }
    return false;
}

}
