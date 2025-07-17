#include "idx_t.h"


namespace track {

/*Idx_t::operator std::string() const {
    return Meta::toStr(_identity);
}*/

glz::json_t Idx_t::to_json() const {
    return _identity;
}

}
