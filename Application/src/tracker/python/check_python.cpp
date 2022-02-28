#define PYBIND11_CPP17
#ifdef PYBIND11_CPP14
#undef PYBIND11_CPP14
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#include <pybind11/embed.h>
#pragma clang diagnostic pop
namespace py = pybind11;

#include <cstdio>

int main(int , char**) {
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
