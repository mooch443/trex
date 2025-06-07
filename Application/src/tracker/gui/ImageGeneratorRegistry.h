#pragma once

#include <commons.pc.h>
#include <gui/GuiTypes.h>   // for ExternalImage, Image::Ptr
#include <gui/dyn/ParseText.h>   // for parse_text
#include <gui/dyn/State.h>       // for dyn::State
#include <misc/idx_t.h>

namespace cmn::gui {

namespace dyn {
struct VarProps; // forward
}

/// Registry of image-generator lambdas.
/// Each generator produces an Image::Ptr given the current VarProps.
struct ImageGeneratorRegistry {
    struct Generator {
        std::function<Image::Ptr(const dyn::VarProps&)> generate;
        std::function<void()> reset;
    };

    static ImageGeneratorRegistry& instance() {
        static ImageGeneratorRegistry inst;
        return inst;
    }

    /// Register a generator under a name.
    void register_generator(std::string name, Generator gen) {
        _map.emplace(std::move(name), std::move(gen));
    }

    /// Lookup a generator by name. Throws if missing.
    Generator get_generator(const std::string& name) const {
        auto it = _map.find(name);
        if (it == _map.end())
            throw RuntimeError("No image generator named ", name,".");
        return it->second;
    }
    
    void reset_generator(const std::string& name) {
        auto it = _map.find(name);
        if (it == _map.end())
            throw RuntimeError("No image generator named ", name,".");
        if(it->second.reset)
            it->second.reset();
    }

private:
    std::unordered_map<std::string, Generator> _map;
};

} // namespace cmn::gui
