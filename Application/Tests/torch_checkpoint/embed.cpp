#include <commons.pc.h>
#include <pybind11/embed.h>
#include <dlfcn.h>

namespace py = pybind11;

int main(int argc, char** argv) {
    if(argc < 3) {
        fprintf(stderr, "Usage: %s probe.py --load FIXTURE_DIR\n", argv[0]);
        return 2;
    }

    std::locale::global(std::locale::classic());
    if(not std::regex_match(std::string("trex"), std::regex("[a-z]+")))
        return 1;

    Dl_info locale_info{};
    const auto locale_address = reinterpret_cast<const void*>(&std::locale::classic);
    dladdr(locale_address, &locale_info);
    printf("Compiler: %s\n", __VERSION__);
#ifdef _GLIBCXX_USE_CXX11_ABI
    printf("libstdc++ C++11 ABI: %d\n", _GLIBCXX_USE_CXX11_ABI);
    const char* locale_symbol = "_ZNSt6locale7classicEv";
#else
    const char* locale_symbol = "_ZNSt3__16locale7classicEv";
#endif
    printf("std::locale::classic: %p in %s; exported lookup: %p\n",
           locale_address, locale_info.dli_fname ? locale_info.dli_fname : "unknown",
           dlsym(RTLD_DEFAULT, locale_symbol));
    fflush(stdout);

    int status = 1;
    // Python initialization, checkpoint loading, and shutdown share one worker thread.
    std::thread worker([&] {
        try {
            py::scoped_interpreter interpreter;
            try {
                auto sys = py::module_::import("sys");
                if(const char* executable = std::getenv("TREX_TEST_PYTHON_EXECUTABLE"))
                    sys.attr("executable") = executable;
                sys.attr("dont_write_bytecode") = true;
                py::list arguments;
                for(int i = 1; i < argc; ++i)
                    arguments.append(argv[i]);
                sys.attr("argv") = arguments;
                py::eval_file(argv[1], py::globals());
                status = 0;
            } catch(const py::error_already_set& error) {
                fprintf(stderr, "%s\n", error.what());
            }
        } catch(const std::exception& error) {
            fprintf(stderr, "%s\n", error.what());
        }
    });
    worker.join();
    if(status == 0)
        printf("PASS: embedded interpreter shut down cleanly.\n");
    return status;
}
