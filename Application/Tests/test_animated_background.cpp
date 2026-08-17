#include <commons.pc.h>
#include <gtest/gtest.h>
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <ui/AnimatedBackground.h>
#include <pv.h>

using namespace cmn;
using cmn::Image;
using cmn::gui::AnimatedBackground;

namespace {

void reset_global_settings() {
    cmn::GlobalSettings::write([](cmn::Configuration& config) {
        config.values["video_conversion_range"] = cmn::Range<long_t>(-1, -1);
        config.values["meta_source_path"] = std::string{};
        config.values["gui_source_video_frame"] = cmn::Frame_t{0u};
        config.values["gui_show_video_background"] = true;

        // AnimatedBackground constructs a FramePreloader, whose cache also
        // initializes these two values even though it does not read them here.
        config.values["frame_rate"] = uint32_t{0};
        config.values["gui_playback_speed"] = 1.f;
    });
}

void register_data_locations_once() {
    static const bool registered = [] {
        auto passthrough = [](const cmn::sprite::Map&, cmn::file::Path path) {
            return path;
        };
        cmn::file::DataLocation::register_path("input", passthrough);
        cmn::file::DataLocation::register_path("output", passthrough);
        return true;
    }();
    (void)registered;
}

class AnimatedBackgroundTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset_global_settings();
        register_data_locations_once();
    }

    static Image::Ptr make_image(uint32_t rows = 2,
                                 uint32_t cols = 2,
                                 uint8_t channels = 4)
    {
        return Image::Make(rows, cols, channels);
    }
};

TEST_F(AnimatedBackgroundTest, AddBehaviorCoverageHere) {
    AnimatedBackground bg{
        make_image(240, 320, 4),
        nullptr,
        nullptr
    };
}

static_assert(not std::is_copy_constructible_v<AnimatedBackground>);
static_assert(not std::is_move_constructible_v<AnimatedBackground>);

}
