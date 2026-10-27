#include "DetectAnnotationDataset.h"

namespace track::detect::annotation_dataset {

using namespace cmn;

std::optional<Format> format_from_dataset_file(const file::Path& path) {
    const auto extension = utils::lowercase(path.extension());
    if(extension == "yaml" || extension == "yml")
        return format_t::yolo;
    if(extension == "json")
        return format_t::coco;
    return std::nullopt;
}

}
