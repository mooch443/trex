#include "annotation.h"

namespace track {
using namespace cmn;

glz::json_t AnnotationMap::to_json() const {
    if(not *this)
        return glz::json_t{};
    return cvt2json((Map_t)*this);
}

std::string AnnotationMap::toStr() const {
    if(not *this)
        return "null";
    return Meta::toStr<Map_t>((const Map_t&)*this);
}

}
