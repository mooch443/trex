#pragma once
#include <commons.pc.h>

namespace cmn::benchmark_previous::utils {
void fast_fromstr(std::string&, std::string_view);
void fast_fromstr_N(std::string&, std::string_view, size_t);
}

namespace cmn::benchmark {
void CompareStringDecoders();
}
