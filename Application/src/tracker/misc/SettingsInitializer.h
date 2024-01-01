#pragma once
#include <commons.pc.h>
#include <tracker/misc/default_config.h>
#include <gui/GUITaskQueue.h>

namespace settings {
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
};

void load(file::PathArray source, file::Path filename, default_config::TRexTask task,
          ExtendableVector exclude_parameters, const cmn::sprite::Map&);

void write_config(bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");
}
