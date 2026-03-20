import commons;
import commons.misc;

#include <gtest/gtest.h>

TEST(CommonsModules, SettingHelpersUseModuleFriendlyAPI) {
    EXPECT_FALSE(cmn::bool_setting("missing_setting"));
}
