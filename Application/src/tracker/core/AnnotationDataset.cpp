#include "AnnotationDataset.h"

#include <algorithm>
#include <cctype>

namespace track::annotation_dataset {

using namespace cmn;

std::string clean_filename_suffix(std::string suffix) {
    for(auto& c : suffix) {
        if(c == '/' || c == '\\' || std::isspace(static_cast<unsigned char>(c)))
            c = '_';
    }
    while(!suffix.empty() && suffix.front() == '_')
        suffix.erase(suffix.begin());
    while(!suffix.empty() && suffix.back() == '_')
        suffix.pop_back();
    return suffix;
}

std::string normalize_source_name(std::string_view name) {
    auto value = file::Path(std::string(name)).filename();
    value = file::Path(value).remove_extension().filename();
    value = utils::lowercase(value);
    for(const auto& ext : {"mp4", "avi", "mov", "mkv"}) {
        for(const auto& separator : {"_", "-", "."}) {
            const auto suffix = std::string(separator) + ext;
            if(value.size() > suffix.size()
               && value.rfind(suffix) == value.size() - suffix.size())
            {
                value.erase(value.size() - suffix.size());
            }
        }
    }

    std::string out;
    out.reserve(value.size());
    for(char c : value) {
        if(std::isalnum(static_cast<unsigned char>(c)))
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::vector<std::string> source_prefix_candidates(std::string_view source_basename) {
    file::Path path{source_basename};
    const auto filename = path.filename();
    if(filename.empty())
        return {};

    const std::string stem{path.remove_extension().filename()};
    const std::string extension{path.extension()};
    std::vector<std::string> candidates;
    if(!stem.empty() && !extension.empty()) {
        candidates.push_back(utils::lowercase(stem + "_" + extension));
        candidates.push_back(utils::lowercase(stem + "-" + extension));
        candidates.push_back(utils::lowercase(stem + "." + extension));
    }
    candidates.push_back(utils::lowercase(filename));
    if(!stem.empty())
        candidates.push_back(utils::lowercase(stem));

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

std::string source_basename_from_paths(const file::PathArray& source) {
    return file::Path(file::find_basename(source)).filename();
}

std::optional<Format> format_from_dataset_file(const file::Path& path) {
    const auto extension = utils::lowercase(path.extension());
    if(extension == "yaml" || extension == "yml")
        return format_t::yolo;
    if(extension == "json")
        return format_t::coco;
    return std::nullopt;
}

}
