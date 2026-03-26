#include "gtest/gtest.h"
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>

int main(int argc, char** argv) {
    cmn::file::DataLocation::create();
    cmn::GlobalSettings::create();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
