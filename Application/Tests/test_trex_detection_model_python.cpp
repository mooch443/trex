#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#ifndef TREX_PYTHON_EXECUTABLE
#error "TREX_PYTHON_EXECUTABLE must be defined"
#endif

#ifndef TREX_DETECTION_MODEL_TEST_DIR
#error "TREX_DETECTION_MODEL_TEST_DIR must be defined"
#endif

namespace {

std::string quote_for_shell(const std::string& value) {
#ifdef _WIN32
    std::string quoted = "\"";
    for(char ch : value) {
        if(ch == '"') {
            quoted += "\\\"";
        } else {
            quoted += ch;
        }
    }
    quoted += "\"";
    return quoted;
#else
    std::string quoted = "'";
    for(char ch : value) {
        if(ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
#endif
}

} // namespace

TEST(TrexDetectionModelPythonTest, RunsEdgeCaseUnitTests) {
    const std::string python = TREX_PYTHON_EXECUTABLE;
    const std::string test_dir = TREX_DETECTION_MODEL_TEST_DIR;

#ifdef _WIN32
    const std::string command =
        "set PYTHONDONTWRITEBYTECODE=1 && " + quote_for_shell(python) +
        " -B -m unittest discover -s " + quote_for_shell(test_dir) +
        " -p test_trex_detection_model.py";
#else
    const std::string command =
        "PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE " + quote_for_shell(python) +
        " -B -m unittest discover -s " + quote_for_shell(test_dir) +
        " -p test_trex_detection_model.py";
#endif

    const int status = std::system(command.c_str());
    EXPECT_EQ(status, 0) << "Python unittest command failed: " << command;
}
