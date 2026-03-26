#include "BackendRegistry.h"

#if !COMMONS_NO_PYTHON
#include <python/PythonWrapper.h>
#endif

extern "C" {
namespace track::detect {

namespace {

auto& backend_registry() {
    static std::unordered_map<ObjectDetectionType::Class, BackendHooks> registry;
    return registry;
}

bool is_python_backend_type(ObjectDetectionType::Class type) {
    return type == ObjectDetectionType::yolo
        || type == ObjectDetectionType::sam3;
}

}

void register_backend(ObjectDetectionType::Class type, BackendHooks hooks) {
    backend_registry()[type] = std::move(hooks);
}

void unregister_backend(ObjectDetectionType::Class type) {
    backend_registry().erase(type);
}

const BackendHooks* backend(ObjectDetectionType::Class type) {
    auto it = backend_registry().find(type);
    if(it == backend_registry().end())
        return nullptr;
    return &it->second;
}

const BackendHooks* ensure_backend(ObjectDetectionType::Class type) {
    if(const auto* hooks = backend(type)) {
        return hooks;
    }

    if(is_python_backend_type(type)) {
#if !COMMONS_NO_PYTHON
        Python::ensure_python_impl_loaded();
#endif
        return backend(type);
    }

    return nullptr;
}

} // namespace track::detect
}
