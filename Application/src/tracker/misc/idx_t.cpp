#include "idx_t.h"
#include <misc/metastring.h>

namespace track {

/*Idx_t::operator std::string() const {
    return Meta::toStr(_identity);
}*/

Idx_t Idx_t::fromStr(const std::string& str) {
    return Idx_t(Meta::fromStr<uint32_t>(str));
}

}
