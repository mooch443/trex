#pragma once
#include <commons.pc.h>
#include <tracker/misc/default_config.h>
#include <gui/GUITaskQueue.h>
#include <misc/DetectionTypes.h>

namespace cmn::sprite {
    class Map;
}
namespace pv {
    class File;
}

namespace cmn::settings {
class ExtendableVector : public std::vector<std::string> {
public:
    // Inherit constructors
    using std::vector<std::string>::vector;

    // Overload the + operator
    ExtendableVector operator+(const ExtendableVector& other) const {
        ExtendableVector result(*this);  // Start with a copy of the current object
        result.insert(result.end(), other.begin(), other.end());
        return result;
    }
    
    // Templated + operator to handle StringLike types
    template<typename Container>
    requires utils::StringLike<typename Container::value_type>
    ExtendableVector operator+(const Container& other) const {
        ExtendableVector result(*this);
        for (const auto& element : other) {
            if constexpr (std::is_array_v<std::remove_cvref_t<typename Container::value_type>>) {
                // Handle C-style strings (char arrays) specifically
                result.emplace_back(element, std::size(element) - 1);  // -1 to ignore null terminator
            } else {
                // Handle other string-like types
                result.emplace_back(element);  // Convert to string and add
            }
        }
        return result;
    }
    
    // Method to get a const reference to the vector
    const std::vector<std::string>& toVector() const {
        return *this;
    }
    
    std::string toStr() const {
        auto set = std::set<std::string>{begin(), end()};
        return Meta::toStr(set);
    }
};

void load(file::PathArray source, 
          file::Path filename,
          default_config::TRexTask task,
          track::detect::ObjectDetectionType::Class type,
          ExtendableVector exclude_parameters,
          const cmn::sprite::Map&);

void write_config(bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");
float infer_cm_per_pixel(const cmn::sprite::Map* = nullptr);
float infer_meta_real_width_from(const pv::File &file, const sprite::Map* map = nullptr);

}
