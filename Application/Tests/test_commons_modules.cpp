#include <gtest/gtest.h>

import commons;
import commons.misc;

TEST(CommonsModules, SettingHelpersUseModuleFriendlyAPI) {
    EXPECT_FALSE(cmn::bool_setting("missing_setting"));
}
