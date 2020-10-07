#pragma once
#include <cnpy.h>

namespace cmn {
    template <typename T>
    void npz_save(std::string zipname, std::string fname, const T *data,
                  const std::vector<size_t> &shape, std::string mode = "w") {
        try {
            cnpy::npz_save(zipname, fname, data, shape, mode);
        } catch(const std::runtime_error& e) {
            U_EXCEPTION("Exception while saving '%S': %s", &zipname, e.what());
        } catch(...) {
            U_EXCEPTION("Unknown exception while saving '%S'.", &zipname);
        }
    }
    
    template<typename T> void npz_save(std::string zipname, std::string fname, const std::vector<T> data, std::string mode = "w") {
        try {
            cnpy::npz_save(zipname, fname, data, mode);
        } catch(const std::runtime_error& e) {
            U_EXCEPTION("Exception while saving '%S': %s", &zipname, e.what());
        } catch(...) {
            U_EXCEPTION("Unknown exception while saving '%S'.", &zipname);
        }
    }
}
