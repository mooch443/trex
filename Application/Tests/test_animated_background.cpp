#include <commons.pc.h>
#include <gtest/gtest.h>
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <ui/AnimatedBackground.h>
#include <pv.h>
#include <core/default_config.h>
#include <core/SettingsPaths.h>

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
        default_config::register_default_locations();
        return true;
    }();
    (void)registered;
}

pv::File make_artificial_pv_file(const file::Path& path,
                                 const file::Path& original_source,
                                 const std::string& output_prefix,
                                 const std::optional<file::Path>& output_dir,
                                 uint32_t conversion_start,
                                 uint32_t conversion_end)
{
    // Writing is opened lazily, so this only initializes the PV header.
    auto video = pv::File::Write<pv::FileMode::WRITE>(path, meta_encoding_t::gray);
    video.set_source(original_source.str());

    pv::Header::ConversionRange_t conversion_range;
    conversion_range.start = conversion_start;
    conversion_range.end = conversion_end;
    video.set_conversion_range(conversion_range);
    

    sprite::Map metadata;
    metadata["source"] = file::PathArray(original_source);
    metadata["meta_source_path"] = original_source.str();
    metadata["output_prefix"] = output_prefix;
    metadata["filename"] = file::Path{};//(output_dir ? *output_dir : original_source.remove_filename());
    metadata["output_dir"] = output_dir ? *output_dir : original_source.remove_filename();
    metadata["video_conversion_range"] = Range<long_t>(
        conversion_start,
        conversion_end);
    
    metadata["filename"] = settings::find_output_name(metadata, {}, {}, false);
    
    video.set_metadata(std::move(metadata));
    return video;
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

TEST_F(AnimatedBackgroundTest, BasicBehaviorMovedLocation) {
    const auto meta_source_path = file::Path("/Users/test/video.mp4").absolute();
    SETTING(meta_source_path) = meta_source_path.str();

    const auto source = file::Path("/Users/test/Downloads/converted_video.pv").absolute();
    SETTING(source) = file::PathArray{source};

    auto result = AnimatedBackground::configure_video_source(nullptr);

    std::set<std::pair<std::string, std::optional<int>>> expected = {
        { meta_source_path.str(), std::nullopt },
        { (source.remove_filename() / meta_source_path.filename()).str(), std::nullopt }
    };
    ASSERT_EQ(result.tests, expected);
}

TEST_F(AnimatedBackgroundTest, MovedEverythingSetManualSourcePath) {
    /// Video used to be at /Users/aalbi/video.mp4
    /// it was converted using video_conversion_range [10,1000]
    const auto meta_source_path = file::Path("/Users/aalbi/video.mp4").absolute();

    /// Downloaded video to my PC at /Users/test/Downloads/converted_video.pv
    const auto source = file::Path("/Users/test/Downloads/converted_video.pv").absolute();

    /// We set source path manually to /Users/test/share/videos/video.mp4
    const auto manual_source_path = file::Path("/Users/test/share/videos/video.mp4").absolute();

    auto video = make_artificial_pv_file(source, meta_source_path, "", std::nullopt, 10u, 1000u);

    SETTING(meta_source_path) = manual_source_path.str();
    SETTING(source) = file::PathArray{source};

    std::set<std::pair<std::string, std::optional<int>>> expected = {
        { manual_source_path.str(), 10 },
        { meta_source_path.str(), 10 },
        { (source.remove_filename() / meta_source_path.filename()).str(), 10 }
    };

    auto result = AnimatedBackground::configure_video_source(&video);
    ASSERT_EQ(result.tests, expected);
}

TEST_F(AnimatedBackgroundTest, MovedEverythingSetRelativePathPrefix) {
    /// Video used to be at /Users/aalbi/video.mp4
    /// it was converted using video_conversion_range [10,1000]
    /// with a prefix path inside the root folder that was zipped
    const auto meta_source_path = file::Path("/Users/aalbi/Videos/2026/video.mp4").absolute();

    /// Downloaded video to my PC at /Users/test/Downloads/2026/prefix/converted_video.pv
    const auto source = file::Path("/Users/test/Downloads/2026/prefix/converted_video.pv").absolute();

    auto video = make_artificial_pv_file(source, meta_source_path, "prefix", std::nullopt, 10u, 1000u);

    SETTING(meta_source_path) = std::string{};
    SETTING(source) = file::PathArray{source};
    
    auto original_video_name = meta_source_path.filename();

    std::set<std::pair<std::string, std::optional<int>>> expected = {
        { meta_source_path.str(), 10 },
        { (source.remove_filename() / original_video_name).str(), 10 },
        { (file::Path("/Users/test/Downloads/2026/").absolute() / original_video_name).str(), 10 }
    };

    auto result = AnimatedBackground::configure_video_source(&video);
    ASSERT_EQ(result.tests, expected);
}

TEST_F(AnimatedBackgroundTest, MovedEverythingSetRelativePathPrefixButPrefixIsNotStored) {
    /// Video used to be here on the pc where it was created
    /// it was converted using video_conversion_range [10,1000]
    const auto meta_source_path = file::Path("/Users/aalbi/Videos/2026/video.mp4").absolute();
    
    /// We directly set the output dir instead of the output prefix to create the video in
    const auto output_dir = file::Path("/Users/aalbi/Videos/2026/prefix").absolute();

    /// Downloaded video to my PC at `/Users/test/Downloads/2026/prefix/converted_video.pv`
    const auto source = file::Path("/Users/test/Downloads/2026/prefix/converted_video.pv").absolute();

    auto video = make_artificial_pv_file(source, meta_source_path, "", output_dir, 10u, 1000u);

    SETTING(meta_source_path) = std::string{};
    SETTING(source) = file::PathArray{source};

    const std::set<std::pair<std::string, std::optional<int>>> expected = {
        { meta_source_path.str(), 10 },
        { (source.remove_filename() / meta_source_path.filename()).str(), 10 },
        { (source.remove_filename().remove_filename() / meta_source_path.filename()).str(), 10 }
    };

    auto result = AnimatedBackground::configure_video_source(&video);
    ASSERT_EQ(result.tests, expected);

    /// now the same, but we dont even set the output dir. it should still find it:
    {
        auto result = AnimatedBackground::configure_video_source(&video);
        auto video = make_artificial_pv_file(source, meta_source_path, "", std::nullopt, 10u, 1000u);
        ASSERT_EQ(result.tests, expected);
    }
}

static_assert(not std::is_copy_constructible_v<AnimatedBackground>);
static_assert(not std::is_move_constructible_v<AnimatedBackground>);

}
