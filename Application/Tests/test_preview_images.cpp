#include <commons.pc.h>
#include <gtest/gtest.h>
#include <core/TrackingSettings.h>
#include <processing/Background.h>
#include <processing/PVBlob.h>
#include <tracking/Tracker.h>
#include <ui/DrawPreviewImage.h>

using namespace cmn;

namespace {

namespace Normalization = default_config::individual_image_normalization_t;

struct PreviewSample {
    pv::BlobPtr blob;
    cv::Mat image;
    cv::Mat mask;
};

PreviewSample preview_sample(meta_encoding_t::Class encoding, int width, int height,
                             bool hollow = false, int offset_x = 71, int offset_y = 43) {
    const bool color = encoding == meta_encoding_t::rgb8 || encoding == meta_encoding_t::r3g3b2;
    cv::Mat image = cv::Mat::zeros(height, width, color ? CV_8UC3 : CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    auto lines = std::make_unique<std::vector<HorizontalLine>>();
    auto pixels = std::make_unique<PixelArray_t>();
    const std::array<uchar, 4> packed{7, 56, 192, 255};
    const std::array<cv::Vec3b, 4> decoded{
        cv::Vec3b(0, 0, 224), cv::Vec3b(0, 224, 0),
        cv::Vec3b(192, 0, 0), cv::Vec3b(192, 224, 224)
    };

    auto append_line = [&](int y, int first, int last) {
        lines->emplace_back(y + offset_y, first + offset_x, last + offset_x);
        for(int x = first; x <= last; ++x) {
            mask.at<uchar>(y, x) = 255;
            if(encoding == meta_encoding_t::binary) {
                image.at<uchar>(y, x) = 255;
            } else if(encoding == meta_encoding_t::gray) {
                const auto value = static_cast<uchar>(40 + (x * 17 + y * 31) % 160);
                pixels->push_back(value);
                image.at<uchar>(y, x) = value;
            } else if(encoding == meta_encoding_t::rgb8) {
                const cv::Vec3b value(30 + (x * 13) % 180, 40 + (y * 29) % 170,
                                      50 + (x * 7 + y * 11) % 160);
                pixels->push_back(value[0], value[1], value[2]);
                image.at<cv::Vec3b>(y, x) = value;
            } else {
                const auto index = (x + 2 * y) % packed.size();
                pixels->push_back(packed[index]);
                image.at<cv::Vec3b>(y, x) = decoded[index];
            }
        }
    };
    for(int y = 0; y < height; ++y) {
        if(hollow && y > 0 && y < height - 1) {
            append_line(y, 0, 0);
            append_line(y, width - 1, width - 1);
        } else {
            append_line(y, 0, width - 1);
        }
    }

    const uint8_t flags = encoding == meta_encoding_t::binary ? pv::Blob::flag(pv::Blob::Flags::is_binary)
        : encoding == meta_encoding_t::rgb8 ? pv::Blob::flag(pv::Blob::Flags::is_rgb)
        : encoding == meta_encoding_t::r3g3b2 ? pv::Blob::flag(pv::Blob::Flags::is_r3g3b2) : 0;
    if(encoding == meta_encoding_t::binary)
        pixels.reset();
    return {pv::Blob::Make(std::move(lines), std::move(pixels), flags, blob::Prediction{}),
            image, mask};
}

std::unique_ptr<Background> preview_background(meta_encoding_t::Class encoding) {
    if(encoding == meta_encoding_t::binary)
        return std::make_unique<Background>(Size2(256, 256), encoding);
    auto image = Image::Make(256, 256, encoding == meta_encoding_t::rgb8 ? 3 : 1);
    image->set_to(255);
    return std::make_unique<Background>(image->bounds(), std::move(image), encoding);
}

cv::Mat expected_source(const PreviewSample& sample, bool subtract) {
    auto expected = sample.image.clone();
    const auto encoding = sample.blob->encoding();
    if(subtract && encoding != meta_encoding_t::binary) {
        const cv::Scalar background = encoding == meta_encoding_t::r3g3b2
            ? cv::Scalar(192, 224, 224) : cv::Scalar::all(255);
        cv::subtract(background, sample.image, expected);
        expected.setTo(cv::Scalar::all(0), sample.mask == 0);
    }
    return expected;
}

cv::Mat expected_unrotated(const cv::Mat& source, cv::Size output_size, float scale) {
    cv::Mat scaled;
    cv::resize(source, scaled, cv::Size(), double(scale), double(scale), cv::INTER_NEAREST);
    const int horizontal = std::max(0, output_size.width - scaled.cols);
    const int vertical = std::max(0, output_size.height - scaled.rows);
    cv::Mat padded;
    cv::copyMakeBorder(scaled, padded, (vertical + 1) / 2, vertical / 2,
                      (horizontal + 1) / 2, horizontal / 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return padded(cv::Rect((padded.cols - output_size.width + 1) / 2,
                           (padded.rows - output_size.height + 1) / 2,
                           output_size.width, output_size.height)).clone();
}

cv::Mat expected_rotated(const cv::Mat& source, cv::Size output_size,
                         Normalization::Class mode, meta_encoding_t::Class encoding, float scale) {
    // Horizontal fixtures have moment angle zero; the midline points right too.
    // Posture anchors the head at 40% of its length; legacy uses half the length.
    const bool legacy = mode == Normalization::legacy;
    const bool moments = mode == Normalization::moments;
    const double cosine = scale * (legacy ? -1.0 : std::sqrt(0.5));
    const double sine = scale * (legacy ? 0.0 : std::sqrt(0.5));
    const double center_x = moments ? source.cols * 0.5 : source.cols / 2;
    const double center_y = moments ? source.rows * 0.5 : source.rows / 2;
    const double anchor_x = output_size.width * 0.5 + scale * (moments ? 0 : legacy ? -5 : 4);
    const double anchor_y = output_size.height * 0.5 + scale * (moments || legacy ? 0 : 4);
    const cv::Mat affine = (cv::Mat_<double>(2, 3) <<
        cosine, -sine, anchor_x - cosine * center_x + sine * center_y,
        sine, cosine, anchor_y - sine * center_x - cosine * center_y);
    cv::Mat expected;
    cv::warpAffine(source, expected, affine, output_size,
                   encoding == meta_encoding_t::r3g3b2 ? cv::INTER_NEAREST : cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return expected;
}

void expect_pixels(const cv::Mat& actual, const cv::Mat& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(cv::norm(actual, expected, cv::NORM_INF), 0);
}

class PreviewImageTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        static std::once_flag initialized;
        std::call_once(initialized, [] {
            GlobalSettings::write([](Configuration& config) {
                default_config::get(config);
                config.values["cm_per_pixel"] = Float2_t(1);
            });
            track::Settings::init();
        });
        // Tracker startup initializes the tracking library's settings cache too.
        tracker = track::Tracker::Make(Image::Make(256, 256, 1), meta_encoding_t::gray, Float2_t(256));
    }

    static void TearDownTestSuite() {
        tracker.reset();
    }

protected:
    inline static std::shared_ptr<track::Tracker> tracker;
    cv::Mat mask_buffer, image_buffer;
    Image raw_buffer;
    gui::ExternalImage display;

    void SetUp() override {
        GlobalSettings::write([](Configuration& config) {
            config.values["calculate_posture"] = true;
            config.values["track_threshold_is_absolute"] = true;
            config.values["individual_image_scale"] = float(1);
        });
    }

    void check_preview(const PreviewSample& sample, Normalization::Class mode,
                       cv::Size output_size, float scale, bool subtract) {
        const auto encoding = sample.blob->encoding();
        GlobalSettings::write([&](Configuration& config) {
            config.values["meta_encoding"] = encoding;
            config.values["track_background_subtraction"] = subtract;
            config.values["individual_image_normalization"] = mode;
            config.values["individual_image_size"] = Size2(output_size);
            config.values["individual_image_scale"] = scale;
        });
        ASSERT_EQ(FAST_SETTING(individual_image_size), Size2(output_size));
        ASSERT_EQ(FAST_SETTING(individual_image_scale), scale);
        ASSERT_EQ(Background::meta_encoding(), encoding);

        track::Midline midline;
        midline.segments() = {{1, 0, Vec2(0, 0)}, {1, 10, Vec2(10, 0)}};
        midline.len() = 10;
        midline.angle() = 0;
        midline.offset() = Vec2(sample.image.cols / 2, sample.image.rows / 2);
        midline.front() = Vec2(0, 0);
        track::constraints::FilterCache filters;
        filters.median_midline_length_px = 10;
        auto background = preview_background(encoding);

        auto source = expected_source(sample, subtract);
        const auto expected = mode == Normalization::none
            ? expected_unrotated(source, output_size, scale)
            : expected_rotated(source, output_size, mode, encoding, scale);
        cv::Mat rgba;
        cv::cvtColor(expected, rgba, expected.channels() == 1 ? cv::COLOR_GRAY2BGRA : cv::COLOR_BGR2BGRA);
        ASSERT_GT(cv::countNonZero(expected.reshape(1)), 0);

        auto [exact, exact_position] = gui::DrawPreviewImage::make_image(
            sample.blob.get(), &midline, &filters, background.get());
        ASSERT_TRUE(exact);
        ASSERT_EQ(exact->dimensions(), READ_SETTING(individual_image_size, Size2));
        ASSERT_NO_FATAL_FAILURE(expect_pixels(exact->get(), rgba));

        const auto position = gui::DrawPreviewImage::make_image_cached(
            sample.blob.get(), &midline, &filters, background.get(), raw_buffer,
            mask_buffer, image_buffer, display.unsafe_get_source());
        ASSERT_TRUE(position);
        EXPECT_EQ(*position, exact_position);
        ASSERT_EQ(raw_buffer.dimensions(), READ_SETTING(individual_image_size, Size2));
        ASSERT_EQ(display.source()->dimensions(), READ_SETTING(individual_image_size, Size2));
        ASSERT_NO_FATAL_FAILURE(expect_pixels(raw_buffer.get(), expected));
        ASSERT_NO_FATAL_FAILURE(expect_pixels(display.source()->get(), rgba));
        display.updated_source();
        EXPECT_EQ(display.size(), Size2(output_size));
    }
};

class EncodedPreviewImageTest : public PreviewImageTest,
    public ::testing::WithParamInterface<std::tuple<meta_encoding_t::Class, bool>> {};

class PreviewImageSettingsCacheTest : public PreviewImageTest {
    void TearDown() override {
        track::Settings::set<track::Settings::individual_image_size>(READ_SETTING(individual_image_size, Size2));
    }
};

TEST_F(PreviewImageSettingsCacheTest, ConfiguredCanvasOverridesEmptyOrStaleLocalCache) {
    auto sample = preview_sample(meta_encoding_t::gray, 9, 5);
    auto background = preview_background(meta_encoding_t::gray);
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(track_background_subtraction) = false;

    for(const auto output_size : {cv::Size(32, 24), cv::Size(48, 36)}) {
        SETTING(individual_image_size) = Size2(output_size);
        for(const auto mode : {Normalization::none, Normalization::moments}) {
            SETTING(individual_image_normalization) = mode;
            for(const auto cached_size : {Size2{}, Size2(12, 8)}) {
                SCOPED_TRACE(::testing::Message() << Meta::toStr(mode)
                    << " configured=" << Meta::toStr(Size2(output_size))
                    << " cached=" << Meta::toStr(cached_size));
                // A UI-local cache can differ from the initialized tracking DLL cache.
                track::Settings::set<track::Settings::individual_image_size>(cached_size);
                ASSERT_EQ(FAST_SETTING(individual_image_size), cached_size);
                ASSERT_EQ(READ_SETTING(individual_image_size, Size2), Size2(output_size));

                const auto expected = mode == Normalization::none
                    ? expected_unrotated(sample.image, output_size, 1.f)
                    : expected_rotated(sample.image, output_size, mode, meta_encoding_t::gray, 1.f);
                cv::Mat rgba;
                cv::cvtColor(expected, rgba, cv::COLOR_GRAY2BGRA);

                auto [exact, exact_position] = gui::DrawPreviewImage::make_image(
                    sample.blob.get(), nullptr, nullptr, background.get());
                ASSERT_TRUE(exact);
                ASSERT_EQ(exact->dimensions(), Size2(output_size));
                ASSERT_NO_FATAL_FAILURE(expect_pixels(exact->get(), rgba));

                const auto position = gui::DrawPreviewImage::make_image_cached(
                    sample.blob.get(), nullptr, nullptr, background.get(), raw_buffer,
                    mask_buffer, image_buffer, display.unsafe_get_source());
                ASSERT_TRUE(position);
                EXPECT_EQ(*position, exact_position);
                ASSERT_EQ(raw_buffer.dimensions(), Size2(output_size));
                ASSERT_EQ(display.source()->dimensions(), Size2(output_size));
                ASSERT_NO_FATAL_FAILURE(expect_pixels(raw_buffer.get(), expected));
                ASSERT_NO_FATAL_FAILURE(expect_pixels(display.source()->get(), rgba));
                display.updated_source();
                EXPECT_EQ(display.size(), Size2(output_size));
            }
        }
    }
}

TEST_P(EncodedPreviewImageTest, LinesPreserveSparsePixelsAndExplicitPadding) {
    const auto [encoding, subtract] = GetParam();
    SETTING(meta_encoding) = encoding;
    SETTING(track_background_subtraction) = subtract;
    auto background = preview_background(encoding);
    for(const auto size : {cv::Size(13, 7), cv::Size(15, 9), cv::Size(7, 3)}) {
        SCOPED_TRACE(::testing::Message() << size.width << "x" << size.height);
        auto sample = preview_sample(encoding, size.width, size.height, true);
        cv::Mat expected_mask, expected_image;
        cv::copyMakeBorder(sample.mask, expected_mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(expected_source(sample, subtract), expected_image,
                           2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat exact_mask, exact_image;
        for(const bool cached : {false, true}) {
            SCOPED_TRACE(cached ? "cached" : "exact");
            auto& mask = cached ? mask_buffer : exact_mask;
            auto& image = cached ? image_buffer : exact_image;
            const auto generate = cached ? imageFromLinesCached : imageFromLines;
            const auto [bounds, count] = generate(sample.blob->input_info(), sample.blob->hor_lines(),
                &mask, subtract ? nullptr : &image, subtract ? &image : nullptr,
                sample.blob->pixels().get(), 0, background.get(), 2);
            EXPECT_EQ(bounds, cv::Rect(69, 41, size.width + 4, size.height + 4));
            EXPECT_EQ(count, static_cast<size_t>(cv::countNonZero(sample.mask)));
            expect_pixels(mask, expected_mask);
            expect_pixels(image, expected_image);
        }
    }
}

TEST_P(EncodedPreviewImageTest, NonePreservesCenteredPixelsAtRequestedSizeAndScale) {
    const auto [encoding, subtract] = GetParam();
    auto sample = preview_sample(encoding, 9, 5, true);
    for(const auto size : {cv::Size(32, 24), cv::Size(17, 13), cv::Size(14, 12), cv::Size(6, 10), cv::Size(16, 3)}) {
        for(const float scale : {1.f, 0.5f, 1.1f, 2.f}) {
            SCOPED_TRACE(::testing::Message() << size.width << "x" << size.height << " scale=" << scale);
            ASSERT_NO_FATAL_FAILURE(check_preview(sample, Normalization::none, size, scale, subtract));
        }
    }
}

TEST_P(EncodedPreviewImageTest, RotationsPreserveExpectedContentsAndCanvas) {
    const auto [encoding, subtract] = GetParam();
    auto sample = preview_sample(encoding, 9, 5);
    for(const auto mode : {Normalization::moments, Normalization::posture, Normalization::legacy}) {
        for(const auto size : {cv::Size(64, 48), cv::Size(80, 80)}) {
            for(const float scale : {0.5f, 1.f, 2.f}) {
                SCOPED_TRACE(::testing::Message() << Meta::toStr(mode) << " " << size.width << "x" << size.height
                             << " scale=" << scale);
                ASSERT_NO_FATAL_FAILURE(check_preview(sample, mode, size, scale, subtract));
                const auto output = raw_buffer.get();
                EXPECT_EQ(cv::countNonZero(output.row(0).reshape(1)), 0);
                EXPECT_EQ(cv::countNonZero(output.row(output.rows - 1).reshape(1)), 0);
                EXPECT_EQ(cv::countNonZero(output.col(0).reshape(1)), 0);
                EXPECT_EQ(cv::countNonZero(output.col(output.cols - 1).reshape(1)), 0);
            }
        }
    }
}

TEST_P(EncodedPreviewImageTest, OversizedIndividualsAreClippedToConfiguredCanvas) {
    const auto [encoding, subtract] = GetParam();
    auto sample = preview_sample(encoding, 9, 5);
    const cv::Size output_size(16, 12);
    for(const auto mode : {Normalization::none, Normalization::moments, Normalization::posture, Normalization::legacy}) {
        for(const float scale : {2.f, 4.f}) {
            SCOPED_TRACE(::testing::Message() << Meta::toStr(mode) << " scale=" << scale);
            ASSERT_NO_FATAL_FAILURE(check_preview(sample, mode, output_size, scale, subtract));
            const auto output = raw_buffer.get();
            const auto border_pixels = cv::countNonZero(output.row(0).reshape(1))
                + cv::countNonZero(output.row(output.rows - 1).reshape(1))
                + cv::countNonZero(output.col(0).reshape(1))
                + cv::countNonZero(output.col(output.cols - 1).reshape(1));
            EXPECT_GT(border_pixels, 0);
        }
    }
}

TEST_F(PreviewImageTest, ReusedPreviewTracksEncodingBlobAndSettingsChanges) {
    const std::array encodings{meta_encoding_t::gray, meta_encoding_t::rgb8,
                              meta_encoding_t::binary, meta_encoding_t::r3g3b2, meta_encoding_t::gray};
    for(const auto encoding : encodings) {
        for(const auto mode : {Normalization::none, Normalization::moments, Normalization::posture, Normalization::legacy}) {
            for(const auto size : {cv::Size(40, 34), cv::Size(32, 24)}) {
                SCOPED_TRACE(::testing::Message() << Meta::toStr(encoding) << " " << Meta::toStr(mode)
                             << " " << size.width << "x" << size.height);
                auto sample = preview_sample(encoding, size.width == 40 ? 13 : 5, 3, false, 103, 89);
                ASSERT_NO_FATAL_FAILURE(check_preview(sample, mode, size, 1.f, size.width == 40));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AllEncodings, EncodedPreviewImageTest,
    ::testing::Combine(::testing::ValuesIn(meta_encoding_t::values), ::testing::Bool()),
    ([](const ::testing::TestParamInfo<EncodedPreviewImageTest::ParamType>& info) {
        const auto [encoding, subtract] = info.param;
        return std::string(encoding.name()) + (subtract ? "_difference" : "_color");
    }));

} // namespace
