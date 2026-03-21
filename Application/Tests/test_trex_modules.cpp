#include <gtest/gtest.h>
#include <type_traits>

import trex.core;
import trex.data;
import trex.detect;
import trex.tracking;
import trex.ml;
import trex.ui;

TEST(TRexModules, SemanticLayersExposeModuleImports) {
    track::Idx_t id{7u};
    EXPECT_TRUE(id.valid());
    EXPECT_EQ(id.get(), 7u);

    EXPECT_TRUE((std::is_class_v<Python::TrainingMode::Class>));
    EXPECT_TRUE((std::is_class_v<Python::VINetwork>));
    EXPECT_TRUE((std::is_class_v<cmn::gui::TrackingState>));
    EXPECT_TRUE((std::is_class_v<track::Results>));
}