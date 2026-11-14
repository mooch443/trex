#include <commons.pc.h>
#include <gtest/gtest.h>
#include <misc/GlobalSettings.h>
#include <pv.h>

using namespace cmn;

namespace {

class PVMetadataTest : public ::testing::Test {
protected:
    std::filesystem::path root;
    sprite::Map schema, saved;

    void SetUp() override {
        root = std::filesystem::temp_directory_path()
            / ("trex-pv-metadata-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(root));

        schema["cm_per_pixel"] = Float2_t{};
        schema["meta_real_width"] = Float2_t{};
        schema["quiet"] = true;
        schema["pv_metadata_count"] = uint32_t(999);
        schema["pv_metadata_enabled"] = false;
        schema["pv_metadata_numbers"] = std::vector<int>{99};
        schema["pv_metadata_classes"] = std::map<uint16_t, std::string>{{99, "default"}};
        schema["pv_metadata_text"] = std::string("default");
        schema["pv_metadata_literal"] = std::string("default");
        schema["pv_metadata_empty"] = std::string("default");
        schema["pv_metadata_optional"] = std::optional<int>{99};
        schema["pv_metadata_omitted"] = int(999);
        GlobalSettings::write([&](Configuration& config) {
            for(const auto& key : schema.keys()) {
                if(config.values.has(key))
                    config.values.at(key).get().copy_to(saved);
                schema.at(key).get().copy_to(config.values);
            }
        });
    }

    void TearDown() override {
        GlobalSettings::write([&](Configuration& config) {
            for(const auto& key : schema.keys()) {
                config.values.erase(key);
                if(saved.has(key))
                    saved.at(key).get().copy_to(config.values);
            }
        });
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void check_metadata(const pv::Header& header) {
        ASSERT_TRUE(header.metadata.has_value());
        sprite::Map loaded;
        ASSERT_NO_THROW(sprite::parse_values(sprite::MapSource::CMD, loaded,
            *header.metadata, &schema, {}, {}));
        // The schema supplies types without pre-populating the loaded map.
        for(const auto& key : schema.keys()) {
            if(key != "pv_metadata_omitted" && key != "cm_per_pixel")
                ASSERT_TRUE(loaded.has(key)) << key;
        }
        EXPECT_FALSE(loaded.has("pv_metadata_omitted"));
        EXPECT_EQ(loaded.at("pv_metadata_count").value<uint32_t>(), 42u);
        EXPECT_TRUE(loaded.at("pv_metadata_enabled").value<bool>());
        EXPECT_EQ(loaded.at("pv_metadata_numbers").value<std::vector<int>>(), (std::vector<int>{1, 2, 3}));
        EXPECT_EQ((loaded.at("pv_metadata_classes").value<std::map<uint16_t, std::string>>()),
                  (std::map<uint16_t, std::string>{{1, "fish"}, {7, "ray, spotted"}}));
        EXPECT_EQ(loaded.at("pv_metadata_text").value<std::string>(), R"(C:\new\track.mp4 "quoted" 'single')");
        EXPECT_EQ(loaded.at("pv_metadata_literal").value<std::string>(), R"(\n\t\u0041)");
        EXPECT_EQ(loaded.at("pv_metadata_empty").value<std::string>(), "");
        EXPECT_EQ(loaded.at("pv_metadata_optional").value<std::optional<int>>(), std::nullopt);
        EXPECT_FLOAT_EQ(header.meta_real_width, 12.5f);
    }
};

TEST_F(PVMetadataTest, LoadsLegacyValueStringMetadata) {
    // Frozen output of the map<string, string> writer: each value is valueString(),
    // then quoted again by the outer map. Do not generate it with current serializers.
    const std::string metadata = R"json({
"meta_real_width":"12.5",
"pv_metadata_count":"42",
"pv_metadata_enabled":"true",
"pv_metadata_numbers":"[1,2,3]",
"pv_metadata_classes":"{1:\"fish\",7:\"ray, spotted\"}",
"pv_metadata_text":"\"C:\\\\new\\\\track.mp4 \\\"quoted\\\" 'single'\"",
"pv_metadata_literal":"\"\\\\n\\\\t\\\\u0041\"",
"pv_metadata_empty":"\"\"",
"pv_metadata_optional":"null"
})json";

    for(const auto version : {pv::V_14, pv::V_15}) {
        SCOPED_TRACE(static_cast<int>(version) + 1);
        const file::Path path((root / ("legacy-" + std::to_string(static_cast<int>(version)) + ".pv")).string());
        // A zero-frame PV14/PV15 header keeps the fixture independent of the current writer.
        DataFormat file(path);
        file.start_writing();
        file.write(std::string(version == pv::V_14 ? "PV14" : "PV15"));
        file.write(std::string("gray"));
        file.write(cv::Size(2, 2));
        file.write(CropOffsets{});
        if(version == pv::V_15) {
            file.write(int64_t(10));
            file.write(int64_t(11));
            file.write(std::string("original.mp4"));
        }
        file.write(uint8_t(sizeof(pv::Header::line_type)));
        file.write(uint32_t(0));
        const auto index_offset_position = file.write(uint64_t(0));
        file.write(timestamp_t{123000000u});
        file.write(std::string("legacy"));
        const std::array<char, 4> average{};
        file.write_data(average.size(), average.data());
        file.write(uint64_t(0));
        const auto metadata_offset = file.tell();
        file.write(metadata);
        file.write(metadata_offset, index_offset_position);
        file.close();

        auto video = pv::File::Read(path);
        ASSERT_EQ(video.header().version, version);
        ASSERT_EQ(video.length(), 0_f);
        check_metadata(video.header());
        sprite::Map loaded;
        ASSERT_TRUE(video.header().metadata.has_value());
        ASSERT_NO_THROW(sprite::parse_values(sprite::MapSource::CMD, loaded,
            *video.header().metadata, &schema, {}, {}));
        ASSERT_EQ(loaded.has("cm_per_pixel"), version == pv::V_14);
        if(version == pv::V_14) {
            EXPECT_FLOAT_EQ(loaded.at("cm_per_pixel").value<Float2_t>(), 6.25f);
            EXPECT_FALSE(video.header().source.has_value());
        } else {
            EXPECT_EQ(video.header().source, std::optional<std::string>{"original.mp4"});
            EXPECT_EQ(video.header().conversion_range.start, std::optional<uint32_t>{10});
            EXPECT_EQ(video.header().conversion_range.end, std::optional<uint32_t>{11});
        }
        video.close();

        const auto before_modify = path.read_file();
        {
            auto modified = pv::File::Write<pv::FileMode::MODIFY>(path, meta_encoding_t::gray);
            EXPECT_THROW((void)modified.header(), std::exception);
            EXPECT_FALSE(modified.is_open());
            EXPECT_THROW(modified.update_metadata(), std::exception);
            EXPECT_FALSE(modified.is_open());
        }
        EXPECT_EQ(path.read_file(), before_modify);
        auto reopened = pv::File::Read(path);
        EXPECT_EQ(reopened.header().version, version);
        check_metadata(reopened.header());
    }
}

TEST_F(PVMetadataTest, NativeJsonMetadataRoundTripsThroughFile) {
    const file::Path path((root / "current.pv").string());
    sprite::Map metadata;
    metadata["meta_real_width"] = Float2_t(12.5);
    metadata["pv_metadata_count"] = uint32_t(42);
    metadata["pv_metadata_enabled"] = true;
    metadata["pv_metadata_numbers"] = std::vector<int>{1, 2, 3};
    metadata["pv_metadata_classes"] = std::map<uint16_t, std::string>{{1, "fish"}, {7, "ray, spotted"}};
    metadata["pv_metadata_text"] = std::string(R"(C:\new\track.mp4 "quoted" 'single')");
    metadata["pv_metadata_literal"] = std::string(R"(\n\t\u0041)");
    metadata["pv_metadata_empty"] = std::string{};
    metadata["pv_metadata_optional"] = std::optional<int>{};
    uint64_t metadata_offset;
    {
        auto video = pv::File::Write(path, meta_encoding_t::gray);
        video.set_resolution(Size2(2, 2));
        video.set_start_time(std::chrono::system_clock::time_point(std::chrono::seconds(123)));
        video.set_source("original.mp4");
        video.set_conversion_range({10u, 11u});
        video.set_metadata(metadata);
        pv::Frame frame;
        frame.set_index(0_f);
        frame.set_source_index(10_f);
        frame.set_timestamp(video.header().timestamp);
        video.add_individual(frame);
        video.close();
        metadata_offset = video.header().meta_offset();
    }

    DataFormat file(path);
    file.start_reading();
    std::string version, raw;
    file.read(version);
    EXPECT_EQ(version, "PV16");
    file.read(raw, metadata_offset);
    glz::generic json;
    ASSERT_EQ(glz::read_json(json, raw), glz::error_code::none);
    ASSERT_TRUE(json.is_object());
    const auto& values = json.get_object();
    EXPECT_TRUE(values.at("pv_metadata_count").is_number());
    EXPECT_TRUE(values.at("pv_metadata_enabled").is_boolean());
    EXPECT_TRUE(values.at("pv_metadata_numbers").is_array());
    EXPECT_TRUE(values.at("pv_metadata_classes").is_object());
    EXPECT_TRUE(values.at("pv_metadata_optional").is_null());
    EXPECT_FALSE(values.contains("pv_metadata_omitted"));
    file.close();

    auto video = pv::File::Read(path);
    ASSERT_EQ(video.header().version, pv::Version::current);
    ASSERT_EQ(video.length(), 1_f);
    check_metadata(video.header());
    EXPECT_EQ(video.header().source, std::optional<std::string>{"original.mp4"});
    EXPECT_EQ(video.header().conversion_range.start, std::optional<uint32_t>{10});
    EXPECT_EQ(video.header().conversion_range.end, std::optional<uint32_t>{11});
    video.close();

    {
        auto modified = pv::File::Write<pv::FileMode::MODIFY>(path, meta_encoding_t::gray);
        ASSERT_NO_THROW((void)modified.header());
        modified.set_metadata(metadata);
        EXPECT_NO_THROW(modified.update_metadata());
        modified.close();
    }
    auto reopened = pv::File::Read(path);
    EXPECT_EQ(reopened.header().version, pv::Version::current);
    EXPECT_EQ(reopened.length(), 1_f);
    check_metadata(reopened.header());
}

TEST_F(PVMetadataTest, EmptyMetadataRemainsAnObject) {
    auto video = pv::File::Write(file::Path((root / "empty.pv").string()), meta_encoding_t::gray);
    video.set_metadata(sprite::Map{});
    ASSERT_TRUE(video.header().metadata.has_value());
    EXPECT_EQ(*video.header().metadata, "{}");
}

}
