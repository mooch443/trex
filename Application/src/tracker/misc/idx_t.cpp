#include "idx_t.h"


namespace track {

/*Idx_t::operator std::string() const {
    return Meta::toStr(_identity);
}*/

Idx_t Idx_t::fromStr(const std::string& str) {
    return Idx_t(cmn::Meta::fromStr<uint32_t>(str));
}

glz::json_t Idx_t::to_json() const {
    return _identity;
}

}
