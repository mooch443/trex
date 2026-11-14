#include <commons.pc.h>
#include <opencv2/videoio/registry.hpp>
#include <pybind11/embed.h>
#include <dlfcn.h>

namespace py = pybind11;

#ifdef TREX_TEST_SHARED_CALL_ONCE
extern "C" int trex_checkpoint_once_count();
#endif

PYBIND11_EMBEDDED_MODULE(trex_checkpoint_probe, module) {
    module.def("identity", [](int value) { return value; });
}

int main(int argc, char** argv) {
    if(argc < 3) {
        fprintf(stderr, "Usage: %s probe.py --load FIXTURE_DIR\n", argv[0]);
        return 2;
    }

#ifdef __linux__
    if(std::getenv("TREX_TEST_TORCH_FIRST")) {
        for(const char* symbol : {"cblas_sgemm_batch", "cblas_dgemm_batch"}) {
            const auto address = dlsym(RTLD_DEFAULT, symbol);
            Dl_info info{};
            if(not address or not dladdr(address, &info) or not info.dli_fname
                    or not std::string_view(info.dli_fname).ends_with("/libtorch_cpu.so")) {
                fprintf(stderr, "FAIL: %s must resolve to Torch before Python starts (found %s).\n",
                        symbol, info.dli_fname ? info.dli_fname : "no library");
                return 1;
            }
            printf("PASS: %s resolves to %s before Python starts.\n", symbol, info.dli_fname);
        }
    }
#endif

#ifdef TREX_TEST_SHARED_CALL_ONCE
    if(trex_checkpoint_once_count() != 1) {
        fprintf(stderr, "FAIL: shared-library std::call_once initialization.\n");
        return 1;
    }
    std::once_flag flag;
    int calls = 0;
    std::call_once(flag, [&] { ++calls; });
    std::call_once(flag, [&] { ++calls; });
    if(calls != 1) {
        fprintf(stderr, "FAIL: executable std::call_once initialization.\n");
        return 1;
    }
    printf("PASS: shared-library and executable std::call_once initialization.\n");
#endif

    // Keep the native video backend dependencies loaded alongside Python.
    printf("OpenCV video backends: %zu\n", cv::videoio_registry::getBackends().size());

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
#ifdef _GLIBCXX_USE_CXX11_ABI
    for(const char* symbol : {"_ZNSs12_M_leak_hardEv", "_ZNSs9_M_mutateEmmm",
                             "_ZNSs4_Rep20_S_empty_rep_storageE", "__once_proxy",
                             "_ZNSt8__detail8_ScannerIcE10_M_advanceEv"}) {
        const auto address = dlsym(RTLD_DEFAULT, symbol);
        Dl_info info{};
        if(address)
            dladdr(address, &info);
        printf("%s: %p in %s\n", symbol, address,
               info.dli_fname ? info.dli_fname : "not exported");
    }
#endif
    fflush(stdout);

    int status = 1;
    // Python initialization, checkpoint loading, and shutdown share one worker thread.
    std::thread worker([&] {
        try {
            py::scoped_interpreter interpreter;
            try {
                if(py::module_::import("trex_checkpoint_probe").attr("identity")(42).cast<int>() != 42)
                    throw std::runtime_error("Embedded Python module callback failed.");
                printf("PASS: embedded Python module callback.\n");
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
