#include "gtest/gtest.h"
#include <file/DataLocation.h>
#include <gui/Dispatcher.h>
#include <misc/GlobalSettings.h>

int main(int argc, char** argv) {
    cmn::file::DataLocation::create();
    cmn::GlobalSettings::create();
    static cmn::gui::attr::Dispatcher dispatcher;
    cmn::gui::attr::install_dispatcher_instance(&dispatcher);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
