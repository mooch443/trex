#include <commons.pc.h>
#include <gtest/gtest.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <misc/GlobalSettings.h>
#include <ui/AnimatedBackground.h>
#include <pv.h>
#include <core/default_config.h>

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
    metadata["filename"] = file::Filename(path.remove_extension("pv").filename());
    metadata["output_dir"] = output_dir ? *output_dir : original_source.remove_filename();
    metadata["video_conversion_range"] = Range<long_t>(
        conversion_start,
        conversion_end);
    
    video.set_metadata(std::move(metadata));
    return video;
}

class AnimatedBackgroundTest : public ::testing::Test {
protected:
    std::filesystem::path root;

    void SetUp() override {
        reset_global_settings();
        register_data_locations_once();
    }

    void TearDown() override {
        if(not root.empty()) {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
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

TEST_F(AnimatedBackgroundTest, UnixSourceWithoutStoredFilenameDoesNotThrow) {
    const std::string original_source = "/media/some/weird/path.mp4";
    const auto source = file::Path("video.pv").absolute();
    SETTING(meta_source_path) = original_source;
    SETTING(source) = file::PathArray{source};
    SETTING(output_dir) = file::Path{};
    SETTING(output_prefix) = std::string{};

    // Writing is lazy; only the header is needed for the background source search.
    auto video = pv::File::Write<pv::FileMode::WRITE>(source, meta_encoding_t::gray);
    video.set_source(original_source);

    // PV metadata is a sparse delta and need not contain filename.
    sprite::Map metadata;
    metadata["meta_source_path"] = original_source;
    video.set_metadata(metadata);

    cmn::gui::BackgroundVideoConfig result;
    ASSERT_NO_THROW(result = AnimatedBackground::configure_video_source(&video));
    EXPECT_TRUE(result.tests.contains({original_source, std::nullopt}));
    EXPECT_TRUE(result.tests.contains({
        (source.remove_filename() / "path.mp4").str(), std::nullopt
    }));
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

TEST_F(AnimatedBackgroundTest, ReopenedPvFindsOriginalVideoAfterMovingTheFolder) {
    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() / ("trex-background-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(fs::create_directory(directory));
    root = directory;
    const file::Path workspace(fs::canonical(root).string());
    const auto original = workspace / "original";
    const auto relocated = workspace / "relocated";
    const auto recording = relocated / "converted" / "shoal.pv";
    ASSERT_TRUE(fs::create_directories(recording.remove_filename().str()));

    // The legacy filename is the only metadata field recording the original PV directory.
    auto video = pv::File::Write(recording, meta_encoding_t::gray);
    video.set_source((original / "shoal.mp4").str());
    video.set_conversion_range({10u, 11u});
    sprite::Map metadata;
    metadata["filename"] = original / "converted" / "shoal";
    video.set_metadata(metadata);
    video.set_resolution(Size2(2, 2));
    video.set_start_time(std::chrono::system_clock::now());
    pv::Frame frame;
    frame.set_index(0_f);
    frame.set_source_index(10_f);
    frame.set_timestamp(video.header().timestamp);
    video.add_individual(frame);
    video.close();

    SETTING(source) = file::PathArray(recording);
    SETTING(output_dir) = file::Path{};
    SETTING(output_prefix) = std::string{};
    auto reopened = pv::File::Read(recording);
    const auto result = AnimatedBackground::configure_video_source(&reopened);

    EXPECT_TRUE(result.tests.contains({(relocated / "shoal.mp4").str(), 10}));
}

static_assert(not std::is_copy_constructible_v<AnimatedBackground>);
static_assert(not std::is_move_constructible_v<AnimatedBackground>);

}
