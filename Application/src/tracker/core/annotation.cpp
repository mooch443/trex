#include "annotation.h"

namespace track {
using namespace cmn;

glz::json_t AnnotationMap::to_json() const {
    if(not *this)
        return glz::json_t{};
    if(not _sources.empty())
        return cvt2json(_sources);
    return cvt2json((Map_t)*this);
}

std::string AnnotationMap::toStr() const {
    if(not *this)
        return "null";
    if(not _sources.empty())
        return Meta::toStr<SourceMap_t>(_sources);
    return Meta::toStr<Map_t>((const Map_t&)*this);
}

}
