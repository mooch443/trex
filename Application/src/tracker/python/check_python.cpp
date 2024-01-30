#define PYBIND11_CPP17
#ifdef PYBIND11_CPP14
#undef PYBIND11_CPP14
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#endif
#include <pybind11/embed.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
namespace py = pybind11;

#include <cstdio>
#include <locale>

int main(int , char**) {
    const char* locale = "C";
    std::locale::global(std::locale(locale));
    
#ifndef NDEBUG
    printf("trex_check_python\n");
#endif
	py::scoped_interpreter guard;
    auto importlib = py::module::import("importlib");
#ifndef NDEBUG
    printf("loading keras...\n");
#endif
    importlib.attr("find_loader")("tensorflow.keras").attr("name");
#ifndef NDEBUG
    printf("success.\n");
#endif
	return 0;
}
