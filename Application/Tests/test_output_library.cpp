#include <commons.pc.h>

#include "gtest/gtest.h"

#include <cnpy/cnpy.h>
#include <core/default_config.h>
#include <grabber/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <misc/Path.h>
#include <tracking/OutputLibrary.h>

using namespace cmn;

namespace {

namespace fs = std::filesystem;

std::string unique_suffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << now << "_" << std::this_thread::get_id();
    return ss.str();
}

std::vector<std::vector<std::string>> read_csv(const fs::path& path) {
    std::ifstream input(path);
    EXPECT_TRUE(input.good()) << path;

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while(std::getline(input, line)) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while(std::getline(stream, field, ','))
            fields.push_back(field);
        rows.push_back(std::move(fields));
    }
    return rows;
}

} // namespace

TEST(OutputLibraryExportTest, OutputFieldsBehaveAsAnOrderedSetAndFrameIsStructuralInCsv) {
    // https://github.com/mooch443/trex/issues/257 tracks the single structural CSV frame column.
    GlobalSettings::write([](Configuration& config) {
        grab::default_config::get(config);
        default_config::get(config);
    });

    Output::Library::Init();
    Output::Library::add("frame", [](
        Output::Library::LibInfo,
        Frame_t frame,
        const track::MotionRecord*,
        bool)
    {
        return double(frame.get());
    });

    const auto functions = Output::Library::functions();
    EXPECT_EQ(std::count(functions.begin(), functions.end(), std::string_view("frame")), 0);
    EXPECT_GT(std::count(functions.begin(), functions.end(), std::string_view("X")), 0);

    const Output::output_fields_t configured_fields{
        {"frame", {}},
        {"frame", {"RAW"}},
        {"X", {"RAW", "WCENTROID"}},
        {"X", {"WCENTROID", "RAW"}},
        {"X", {"RAW", "PCENTROID"}},
        {"X", {"SMOOTH", "WCENTROID"}},
        {"X", {"WCENTROID", "SMOOTH"}},
        {"X", {"RAW", "WCENTROID", "*2"}},
        {"X", {"RAW", "WCENTROID", "*2.0"}}
    };
    const auto parsed = Output::Library::parse_output_fields(configured_fields);

    ASSERT_TRUE(parsed.contains("frame"));
    EXPECT_EQ(parsed.at("frame").size(), 1u);
    ASSERT_TRUE(parsed.contains("X"));
    EXPECT_EQ(parsed.at("X").size(), 4u)
        << "Equivalent modifier order must deduplicate, while distinct sources, smoothing, and calculations remain.";
    const auto smooth_x = std::find_if(
        parsed.at("X").begin(), parsed.at("X").end(),
        [](const auto& instance) {
            return instance.first.is(Output::Modifiers::SMOOTH)
                && instance.first.is(Output::Modifiers::WEIGHTED_CENTROID);
        });
    ASSERT_NE(smooth_x, parsed.at("X").end());
    EXPECT_EQ(
        smooth_x->first.values(),
        (std::vector<Output::Modifiers::Class>{
            Output::Modifiers::SMOOTH,
            Output::Modifiers::WEIGHTED_CENTROID
        })) << "The first equivalent selection must be retained.";

    const fs::path root = fs::temp_directory_path() / ("trex-output-library-" + unique_suffix());
    fs::create_directories(root);
    struct WorkspaceCleanup {
        fs::path root;
        ~WorkspaceCleanup() {
            std::error_code error;
            fs::remove_all(root, error);
        }
    } cleanup{root};

    const Range<Frame_t> range{2_f, 4_f};
    const auto expect_structural_frame_csv = [&](const Output::cached_output_fields_t& fields,
                                                  const fs::path& destination)
    {
        Output::Library::save_csv(
            fields, range, nullptr, nullptr, file::Path(destination.string()));

        const auto rows = read_csv(destination);
        ASSERT_EQ(rows.size(), 4u);
        ASSERT_EQ(rows.front(), std::vector<std::string>{"frame"});
        for(size_t i = 1; i < rows.size(); ++i) {
            ASSERT_EQ(rows[i].size(), rows.front().size());
            EXPECT_EQ(std::stod(rows[i].front()), double(i + 1));
        }
    };

    expect_structural_frame_csv({}, root / "empty.csv");
    expect_structural_frame_csv(
        Output::cached_output_fields_t{{"frame", parsed.at("frame")}},
        root / "single-manual-frame.csv");

    Output::cached_output_fields_t repeated_frame;
    repeated_frame["frame"] = {
        {Output::Options_t{}, Output::Calculation{}},
        {Output::Options_t{}, Output::Calculation{}}
    };
    expect_structural_frame_csv(repeated_frame, root / "manual-frame.csv");

    Output::cached_output_fields_t npz_fields;
    npz_fields["frame"] = parsed.at("frame");
    const fs::path npz_path = root / "frame.npz";
    Output::Library::save_npz(
        npz_fields, range, nullptr, nullptr, file::Path(npz_path.string()), nullptr, true);

    const auto npz = cnpy::npz_load(npz_path.string());
    ASSERT_EQ(npz.size(), 1u);
    ASSERT_TRUE(npz.contains("frame"));
    const auto frame_values = npz.at("frame").as_vec<float>();
    ASSERT_EQ(frame_values.size(), 3u);
    EXPECT_EQ(frame_values, (std::vector<float>{2.f, 3.f, 4.f}));
}
