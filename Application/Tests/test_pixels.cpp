#include <commons.pc.h>
#include <gtest/gtest.h>
#include <misc/Image.h>
#include <processing/Background.h>
#include <misc/PixelTree.h>
#include <misc/TrackingSettings.h>
#include <tracking/Posture.h>
#include <processing/LuminanceGrid.h>

using namespace cmn;
using namespace track;
using namespace blob;
using namespace pixel;

using ::testing::TestWithParam;
using ::testing::Values;

using cmn::gui::Color;

// Default‐constructed Color should be all zeros
TEST(ColorTest, DefaultConstructor) {
    Color c;
    EXPECT_EQ(c.r, 0);
    EXPECT_EQ(c.g, 0);
    EXPECT_EQ(c.b, 0);
    EXPECT_EQ(c.a, 0);
}

// RGBA constructor
TEST(ColorTest, RGBAConstructor) {
    Color c(10, 20, 30, 40);
    EXPECT_EQ(c.r, 10);
    EXPECT_EQ(c.g, 20);
    EXPECT_EQ(c.b, 30);
    EXPECT_EQ(c.a, 40);
}

// Gray constructor (grayscale + full alpha)
TEST(ColorTest, GrayConstructor) {
    Color c(uint8_t(100u));
    EXPECT_EQ(c.r, 100);
    EXPECT_EQ(c.g, 100);
    EXPECT_EQ(c.b, 100);
    EXPECT_EQ(c.a, 255);
}

// bgra() should swap red/blue channels
TEST(ColorTest, BGRAConversion) {
    Color c(5, 15, 25, 35);
    Color swapped = c.bgra();
    EXPECT_EQ(swapped.r, 25);
    EXPECT_EQ(swapped.g, 15);
    EXPECT_EQ(swapped.b, 5);
    EXPECT_EQ(swapped.a, 35);
}

// operator[] indexing
TEST(ColorTest, OperatorIndex) {
    Color c(7, 14, 21, 28);
    EXPECT_EQ(c[0], 7);
    EXPECT_EQ(c[1], 14);
    EXPECT_EQ(c[2], 21);
    EXPECT_EQ(c[3], 28);
}

// to_integer should pack channels as (r<<24)|(g<<16)|(b<<8)|a
TEST(ColorTest, ToInteger) {
    Color c(1, 2, 3, 4);
    uint32_t v = c.to_integer();
    uint32_t exp = (1u << 24) | (2u << 16) | (3u << 8) | 4u;
    EXPECT_EQ(v, exp);
}

// div255 edge‐cases
TEST(ColorTest, Div255EdgeCases) {
    EXPECT_EQ(Color::div255(0u), 0u);
    EXPECT_EQ(Color::div255(255u), 1u);
    EXPECT_EQ(Color::div255(255u * 255u), 255u);
}

constexpr Color simple_blend(const Color& A, const Color& B) {
    auto alphabg = A.a / 255.0;
    auto alphafg = B.a / 255.0;
    auto alpha = alphabg + alphafg * ( 1 - alphabg );
    return Color(
        (uint8_t)saturate((A.r * alphabg + B.r * alphafg * ( 1 - alphabg )) / alpha),
        (uint8_t)saturate((A.g * alphabg + B.g * alphafg * ( 1 - alphabg )) / alpha),
        (uint8_t)saturate((A.b * alphabg + B.b * alphafg * ( 1 - alphabg )) / alpha),
        (uint8_t)(alpha * 255.0)
    );
}

// Opaque blend: foreground fully opaque overwrites background
TEST(ColorTest, BlendOpaque) {
    Color bg(10, 20, 30, 255), fg(100, 110, 120, 255);
    Color out = Color::blend(bg, fg);
    Color ref = simple_blend(bg, fg);
    EXPECT_EQ(out.r, ref.r)
        << "BlendOpaque failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.g, ref.g)
        << "BlendOpaque failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.b, ref.b)
        << "BlendOpaque failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.a, ref.a)
        << "BlendOpaque failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
}

// Transparent blend: fully transparent foreground leaves background intact
TEST(ColorTest, BlendTransparent) {
    Color bg(50, 60, 70, 255), fg(80, 90, 100, 0);
    Color out = Color::blend(bg, fg);
    Color ref = simple_blend(bg, fg);
    EXPECT_EQ(out.r, ref.r)
        << "BlendTransparent failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.g, ref.g)
        << "BlendTransparent failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.b, ref.b)
        << "BlendTransparent failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.a, ref.a)
        << "BlendTransparent failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
}

// Semi-transparent blend: roughly a 50/50 mix
TEST(ColorTest, BlendHalfTransparent) {
    Color bg(0, 0, 255, 255), fg(255, 0, 0, 128);
    Color out = Color::blend(bg, fg);
    Color ref = simple_blend(bg, fg);
    EXPECT_EQ(out.a, ref.a)
        << "BlendHalfTransparent failed for bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_NEAR(out.r, ref.r, 1)
        << "BlendHalfTransparent failed for R channel; bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_EQ(out.g, ref.g)
        << "BlendHalfTransparent failed for G channel; bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
    EXPECT_NEAR(out.b, ref.b, 1)
        << "BlendHalfTransparent failed for B channel; bg=" << bg.toStr() << " fg=" << fg.toStr()
        << " => out=" << out.toStr() << " ref=" << ref.toStr();
}

// limit_alpha should cap the alpha channel
TEST(ColorTest, LimitAlpha) {
    Color c(1, 2, 3, 200);
    Color out = c.limit_alpha(100);
    EXPECT_EQ(out.a, 100);
}

// alpha setter should replace alpha
TEST(ColorTest, AlphaSetter) {
    Color c(1, 2, 3, 50);
    Color out = c.alpha(180);
    EXPECT_EQ(out.a, 180);
}

// float_multiply should scale each channel by other/255
TEST(ColorTest, FloatMultiply) {
    Color c1(100, 150, 200, 255);
    Color c2(128, 64, 32, 128);
    Color out = c1.float_multiply(c2);
    EXPECT_EQ(out.r, static_cast<uint8_t>(100 * (128.0f/255.0f)));
    EXPECT_EQ(out.g, static_cast<uint8_t>(150 * (64.0f/255.0f)));
    EXPECT_EQ(out.b, static_cast<uint8_t>(200 * (32.0f/255.0f)));
    EXPECT_EQ(out.a, static_cast<uint8_t>(255 * (128.0f/255.0f)));
}

// -----------------------------------------------------------------------------
// Compare Color::blend implementation against simple_blend reference
// -----------------------------------------------------------------------------
TEST(ColorBlendImplementation, MatchSimpleBlend) {
    struct Case { Color A, B; };
    std::vector<Case> cases = {
        // edge cases
        { Color(0,0,0,0),       Color(0,0,0,0) },
        { Color(255,255,255,255), Color(255,255,255,255) },
        { Color(0,0,0,255),     Color(255,0,0,0) },
        { Color(0,0,255,128),   Color(128,255,0,128) },
        { Color(50,100,150,75), Color(200,50,100,125) }
    };

    for (const auto& c : cases) {
        Color ref = simple_blend(c.A, c.B);
        Color act = Color::blend(c.A, c.B);
        EXPECT_NEAR(act.r, ref.r, 1) << "R channel differs for act=" << act.toStr() << " ref=" << ref.toStr() << " after blending " << c.A.toStr() << " with " << c.B.toStr();
        EXPECT_NEAR(act.g, ref.g, 1) << "G channel differs for act=" << act.toStr() << " ref=" << ref.toStr() << " after blending " << c.A.toStr() << " with " << c.B.toStr();
        EXPECT_NEAR(act.b, ref.b, 1) << "B channel differs for act=" << act.toStr() << " ref=" << ref.toStr() << " after blending " << c.A.toStr() << " with " << c.B.toStr();
        EXPECT_NEAR(act.a, ref.a, 1) << "A channel differs for act=" << act.toStr() << " ref=" << ref.toStr() << " after blending " << c.A.toStr() << " with " << c.B.toStr();
    }
}

// -----------------------------------------------------------------------------
// Tests for the simple float-based blend (USE_CUSTOM branch)
// -----------------------------------------------------------------------------

TEST(ColorTestSimpleBlend, OpaqueBackground) {
    // A fully opaque background should entirely mask the foreground.
    Color bg(10, 20, 30, 255);
    Color fg(200, 150, 100, 128);
    Color out = Color::blend(bg, fg);
    EXPECT_EQ(out.r, bg.r);
    EXPECT_EQ(out.g, bg.g);
    EXPECT_EQ(out.b, bg.b);
    EXPECT_EQ(out.a, bg.a);
}

TEST(ColorTestSimpleBlend, TransparentBackground) {
    // A fully transparent background should yield exactly the foreground.
    Color bg(10, 20, 30, 0);
    Color fg(200, 150, 100, 128);
    Color out = Color::blend(bg, fg);
    EXPECT_EQ(out.r, fg.r);
    EXPECT_EQ(out.g, fg.g);
    EXPECT_EQ(out.b, fg.b);
    EXPECT_EQ(out.a, fg.a);
}

TEST(ColorTestSimpleBlend, SemiTransparentBackground) {
    // Both BG and FG half-transparent – verify approximate composite.
    Color bg(100,  50,   0, 128);
    Color fg(200, 150, 100, 128);
    Color out = Color::blend(bg, fg);

    // Expected alpha ≈ 128/255 + (128/255)*(1-128/255) ≈ 0.502 + 0.502*0.498 ≈ 0.752 → 0.752*255 ≈ 192
    EXPECT_NEAR(out.a, 192, 1);

    // R channel ≈ (100*0.502 + 200*0.502*0.498) / 0.752 ≈ 133
    EXPECT_NEAR(out.r, 133, 2);
    // G channel ≈ ( 50*0.502 + 150*0.502*0.498) / 0.752 ≈  83
    EXPECT_NEAR(out.g,  83, 2);
    // B channel ≈ (  0*0.502 + 100*0.502*0.498) / 0.752 ≈  33
    EXPECT_NEAR(out.b,  33, 2);
}

// Test vec_to_r3g3b2 function
TEST(VecToR3G3B2Test, BasicAssertions) {
    std::array<unsigned char, 3> color = {255, 128, 64};
    uint8_t result = vec_to_r3g3b2(color);
    EXPECT_EQ(result, 0b11100010); // Manually calculated expected result
}

// Test r3g3b2_to_vec function for 3 channels
TEST(R3G3B2ToVecTest, BasicAssertions3Channels) {
    uint8_t r3g3b2 = 0b11100010;
    auto result = r3g3b2_to_vec<3>(r3g3b2);
    EXPECT_EQ(result[0], 192);
    EXPECT_EQ(result[1], 128);
    EXPECT_EQ(result[2], 64);
}

// Test r3g3b2_to_vec function for 4 channels
TEST(R3G3B2ToVecTest, BasicAssertions4Channels) {
    uint8_t r3g3b2 = 0b11100010;
    auto result = r3g3b2_to_vec<4>(r3g3b2, 255);
    EXPECT_EQ(result[0], 192);
    EXPECT_EQ(result[1], 128);
    EXPECT_EQ(result[2], 64);
    EXPECT_EQ(result[3], 255);
}

// Test convert_to_r3g3b2 function
TEST(ConvertToR3G3B2Test, BasicAssertions) {
    cv::Mat input(2, 2, CV_8UC3, cv::Scalar(255, 128, 64));
    cv::Mat output;
    convert_to_r3g3b2<3>(input, output);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            EXPECT_EQ(output.at<uchar>(y, x), 0b11100010);
        }
    }
}

// Test convert_from_r3g3b2 function for 3 channels
TEST(ConvertFromR3G3B2Test, BasicAssertions3Channels) {
    cv::Mat input(2, 2, CV_8UC1, cv::Scalar(0b11100010));
    cv::Mat output;
    convert_from_r3g3b2<3>(input, output);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            auto pixel = output.at<cv::Vec3b>(y, x);
            EXPECT_EQ(pixel[0], 192);
            EXPECT_EQ(pixel[1], 128);
            EXPECT_EQ(pixel[2], 64);
        }
    }
}

// Test convert_from_r3g3b2 function for 4 channels
TEST(ConvertFromR3G3B2Test, BasicAssertions4Channels) {
    cv::Mat input(2, 2, CV_8UC1, cv::Scalar(0b11100010));
    cv::Mat output;
    convert_from_r3g3b2<4>(input, output);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            auto pixel = output.at<cv::Vec4b>(y, x);
            EXPECT_EQ(pixel[0], 192);
            EXPECT_EQ(pixel[1], 128);
            EXPECT_EQ(pixel[2], 64);
            EXPECT_EQ(pixel[3], 255);
        }
    }
}

// Test convert_to_r3g3b2 function for 4 channels
TEST(ConvertToR3G3B2Test, BasicAssertions4Channels) {
    cv::Mat input(2, 2, CV_8UC4, cv::Scalar(255, 128, 64, 255));
    cv::Mat output;
    convert_to_r3g3b2<4>(input, output);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            EXPECT_EQ(output.at<uchar>(y, x), 0b11100010);
        }
    }
}

TEST(ConvertToR3G3B2Test, BasicConversion) {
    // Create a 3x1 BGR image
    cv::Mat input = (cv::Mat_<cv::Vec3b>(1, 3) << cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255));
    cv::Mat output;

    convert_to_r3g3b2<3>(input, output);

    // Check the output values
    EXPECT_EQ(output.at<uchar>(0, 0), 0b11000000); // Red
    EXPECT_EQ(output.at<uchar>(0, 1), 0b00111000); // Green
    EXPECT_EQ(output.at<uchar>(0, 2), 0b00000111); // Blue
}

TEST(VecToR3G3B2Test, BasicConversion) {
    // Define some RGB colors
    std::array<uchar, 3> red   = {255, 0, 0};
    std::array<uchar, 3> green = {0, 255, 0};
    std::array<uchar, 3> blue  = {0, 0, 255};
    std::array<uchar, 3> white = {255, 255, 255};
    std::array<uchar, 3> black = {0, 0, 0};

    // Convert to R3G3B2
    uint8_t red_r3g3b2   = vec_to_r3g3b2(red);
    uint8_t green_r3g3b2 = vec_to_r3g3b2(green);
    uint8_t blue_r3g3b2  = vec_to_r3g3b2(blue);
    uint8_t white_r3g3b2 = vec_to_r3g3b2(white);
    uint8_t black_r3g3b2 = vec_to_r3g3b2(black);

    // Check the R3G3B2 values
    EXPECT_EQ(red_r3g3b2,   0b11000000); // Red
    EXPECT_EQ(green_r3g3b2, 0b00111000); // Green
    EXPECT_EQ(blue_r3g3b2,  0b00000111); // Blue
    EXPECT_EQ(white_r3g3b2, 0b11111111); // White
    EXPECT_EQ(black_r3g3b2, 0b00000000); // Black
}

TEST(ImageConversionTest, ConvertToAndFromR3G3B2) {
    // Create a 2x2 BGR image
    cv::Mat input = (cv::Mat_<cv::Vec3b>(2, 2) << cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0),
                                                  cv::Vec3b(0, 0, 255), cv::Vec3b(255, 255, 255));
    cv::Mat r3g3b2_image;
    cv::Mat output;

    // Convert to R3G3B2
    convert_to_r3g3b2<3>(input, r3g3b2_image);

    // Convert back from R3G3B2
    convert_from_r3g3b2<3, 1>(r3g3b2_image, output);

    // Check the output dimensions
    EXPECT_EQ(output.rows, input.rows);
    EXPECT_EQ(output.cols, input.cols);

    // Check the output values
    EXPECT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(192, 0, 0));    // Red
    EXPECT_EQ(output.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 224, 0));    // Green
    EXPECT_EQ(output.at<cv::Vec3b>(1, 0), cv::Vec3b(0, 0, 224));    // Blue
    EXPECT_EQ(output.at<cv::Vec3b>(1, 1), cv::Vec3b(192, 224, 224)); // White
}

TEST(ConvertFromR3G3B2Test, BasicConversion) {
    // Create a 1x3 R3G3B2 image
    cv::Mat input = (cv::Mat_<uchar>(1, 3) << 0b11000000, 0b00111000, 0b00000111);
    cv::Mat output;

    convert_from_r3g3b2<3>(input, output);

    // Check the output values
    EXPECT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(192, 0, 0)); // Red
    EXPECT_EQ(output.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 224, 0)); // Green
    EXPECT_EQ(output.at<cv::Vec3b>(0, 2), cv::Vec3b(0, 0, 224)); // Blue
}

TEST(ConvertFromR3G3B2Test, AlphaChannel) {
    // Create a 1x1 R3G3B2 image with alpha
    cv::Mat input = (cv::Mat_<uchar>(1, 1) << 0b11000000);
    cv::Mat output;

    convert_from_r3g3b2<4, 1, true>(input, output);

    // Check the output values
    EXPECT_EQ(output.at<cv::Vec4b>(0, 0), cv::Vec4b(192, 0, 0, 192)); // Red with alpha
}

template<cmn::meta_encoding_t::Class InputEncoding, cmn::meta_encoding_t::Class OutputEncoding, DifferenceMethod DiffMethod>
struct LineWithoutGridParams {
    static constexpr InputInfo input_info = {
        .channels = (InputEncoding == meta_encoding_t::rgb8) ? 3u : 1u,
        .encoding = InputEncoding
    };
    static constexpr OutputInfo output_info = {
        .channels = (OutputEncoding == meta_encoding_t::rgb8) ? 3u : 1u,
        .encoding = OutputEncoding
    };
    static constexpr DifferenceMethod diff_method = DiffMethod;
    
    // For readability in test output
    std::string ToString() const {
        return format<FormatterType::UNIX>("Input:", input_info," Output:", output_info," DiffMethod: ", diff_method);
    }
};

template <typename T>
class LineWithoutGridTest2 : public ::testing::Test {
protected:
    void SetUp() override {
        // Image initialization is now done in the test function
    }

    Image::Ptr image;
    std::unique_ptr<Background> bg;
};

using LineWithoutGridTypes = ::testing::Types<
    LineWithoutGridParams<meta_encoding_t::gray, meta_encoding_t::gray, DifferenceMethod_t::absolute>,
    LineWithoutGridParams<meta_encoding_t::r3g3b2, meta_encoding_t::gray, DifferenceMethod_t::absolute>,
    LineWithoutGridParams<meta_encoding_t::rgb8, meta_encoding_t::gray, DifferenceMethod_t::absolute>,
    LineWithoutGridParams<meta_encoding_t::gray, meta_encoding_t::gray, DifferenceMethod_t::sign>,
    LineWithoutGridParams<meta_encoding_t::r3g3b2, meta_encoding_t::gray, DifferenceMethod_t::sign>,
    LineWithoutGridParams<meta_encoding_t::rgb8, meta_encoding_t::gray, DifferenceMethod_t::sign>,
    LineWithoutGridParams<meta_encoding_t::gray, meta_encoding_t::gray, DifferenceMethod_t::none>,
    LineWithoutGridParams<meta_encoding_t::r3g3b2, meta_encoding_t::gray, DifferenceMethod_t::none>,
    LineWithoutGridParams<meta_encoding_t::rgb8, meta_encoding_t::gray, DifferenceMethod_t::none>
>;

TYPED_TEST_SUITE(LineWithoutGridTest2, LineWithoutGridTypes);

template<InputInfo input>
void print_results(std::string_view name, const std::vector<HorizontalLine>& lines, const std::vector<uchar>& pixels)
{
    Print("------------ ",name,"  -----------");
    Print("Lines: ", lines);
    std::stringstream ss;
    for(const uchar* ptr = pixels.data(); ptr < pixels.data() + pixels.size(); ptr += input.channels)
    {
        auto value = diffable_pixel_value<input, OutputInfo{
            .channels = 1u,
            .encoding = meta_encoding_t::gray
        }>(ptr);
        
        if(ptr> pixels.data()) {
            ss << ",";
        }
        ss << format<FormatterType::UNIX>(value);
    }
    Print("Pixels: ", no_quotes(ss.str()));
    if constexpr(input.encoding != meta_encoding_t::gray)
        Print("\t(original) ", pixels);
}

TYPED_TEST(LineWithoutGridTest2, LineWithoutGridTest2) {
    constexpr auto params = TypeParam{};
    
    // Create appropriate background image
    if constexpr (params.input_info.encoding == meta_encoding_t::rgb8) {
        this->image = Image::Make(10, 10, 3);
        this->image->set_to(100);
    } else {
        this->image = Image::Make(10, 10, 1);
        this->image->set_to(100);
    }
    this->bg = std::make_unique<Background>(std::move(this->image), params.input_info.encoding == meta_encoding_t::rgb8 ? meta_encoding_t::rgb8 : meta_encoding_t::gray);

    /// HorizontalLine{ x0, x1, y }
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};
    std::vector<uchar> input_pixels(20 * params.input_info.channels);
    size_t i = 0;
    for(auto ptr = input_pixels.data(); ptr < input_pixels.data() + input_pixels.size(); ptr += params.input_info.channels, ++i) {
        for(size_t c = 0; c < params.input_info.channels; ++c) {
            if constexpr(params.input_info.encoding == meta_encoding_t::r3g3b2)
                *(ptr + c) = static_cast<uchar>(vec_to_r3g3b2(RGBArray{uint8_t(i * 10), uint8_t(i * 10), uint8_t(i * 10)}));
            else
                *(ptr + c) = static_cast<uchar>(i * 10);
        }
    }
    
    int threshold = 50;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;
    
    uchar* px = input_pixels.data();
    
    // Print debug information
    DebugHeader(no_quotes(params.ToString()));
    Print("Background: ", std::vector<uchar>{
        this->bg->image().data(),
        this->bg->image().data() + this->bg->image().size()
    });
    
    print_results<params.input_info>("input", input, input_pixels);
    
    // Define expected results based on the parameters
    std::vector<HorizontalLine> expected_lines;
    std::vector<uchar> expected_pixels;
    
    if constexpr (params.diff_method == DifferenceMethod_t::absolute) {
        if constexpr (params.input_info.encoding == meta_encoding_t::gray) {
            expected_lines = {{0, 0, 5}, {1, 5, 9}};
            expected_pixels = {0,10,20,30,40,50,150,160,170,180,190};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::r3g3b2) {
            expected_lines = {{0, 0, 6}, {1, 6, 9}};
            expected_pixels = {0,0,0,0,9,9,9,173,173,173,173};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::rgb8) {
            expected_lines = {{0, 0, 5}, {1, 5, 9}};
            expected_pixels = {0,0,0,10,10,10,20,20,20,30,30,30,40,40,40,50,50,50,150,150,150,160,160,160,170,170,170,180,180,180,190,190,190};
        }
    } else if constexpr (params.diff_method == DifferenceMethod_t::sign) {
        if constexpr (params.input_info.encoding == meta_encoding_t::gray) {
            expected_lines = {{0, 0, 5}};
            expected_pixels = {0,10,20,30,40,50};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::r3g3b2) {
            expected_lines = {{0, 0, 6}};
            expected_pixels = {0,0,0,0,9,9,9};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::rgb8) {
            expected_lines = {{0, 0, 5}};
            expected_pixels = {0,0,0,10,10,10,20,20,20,30,30,30,40,40,40,50,50,50};
        }
    } else if constexpr (params.diff_method == DifferenceMethod_t::none) {
        if constexpr (params.input_info.encoding == meta_encoding_t::gray) {
            expected_lines = {{0, 5, 9}, {1, 0, 9}};
            expected_pixels = {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::r3g3b2) {
            expected_lines = {{0, 7, 9}, {1, 0, 9}};
            expected_pixels = {82,82,82,91,91,91,164,164,164,173,173,173,173};
        } else if constexpr (params.input_info.encoding == meta_encoding_t::rgb8) {
            expected_lines = {{0, 5, 9}, {1, 0, 9}};
            expected_pixels = {50,50,50,60,60,60,70,70,70,80,80,80,90,90,90,100,100,100,110,110,110,120,120,120,130,130,130,140,140,140,150,150,150,160,160,160,170,170,170,180,180,180,190,190,190};
        }
    }
    
    // Call the function with the appropriate template parameters
    line_without_grid<params.input_info, params.output_info, params.diff_method>(
        this->bg.get(), input, px, threshold, lines, pixels);
    
    print_results<params.input_info>("output", lines, pixels);
    
    print_results<params.input_info>("expected", expected_lines, expected_pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

class LineWithoutGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize a simple image for testing purposes
        image = Image::Make(10, 10, 1);
        image->set_to(150);
        /*for (uint y = 0; y < 10; ++y) {
            for (uint x = 0; x < 10; ++x) {
                image->set_pixel(x, y, gui::Color(255, 255, 255, 255));
            }
        }*/

        // Initialize a LuminanceGrid
        grid = nullptr;

        // Initialize a Background object
        bg = std::make_unique<Background>(std::move(image), meta_encoding_t::gray);
        //cv::imshow("bg", bg->image().get());
        //cv::waitKey(0);
    }

    Image::Ptr image;
    LuminanceGrid* grid;
    std::unique_ptr<Background> bg;
};

TEST_F(LineWithoutGridTest, AbsoluteDifferenceMethod) {
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};
    std::array<uchar, 20> input_pixels;
    for (size_t i = 0; i < input_pixels.size(); ++i) {
        input_pixels[i] = static_cast<uchar>(i * 10);
    }
    
    int threshold = 50;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;
    
    Print("input: ", input);
    Print("pixels: ", input_pixels);

    Print("background:",
          std::vector<uchar>(bg->image().ptr(0, 0),
                             bg->image().ptr(0, 0) + bg->image().cols),
          std::vector<uchar>(bg->image().ptr(1, 0),
                             bg->image().ptr(1, 0) + bg->image().cols));
    Print(std::vector<uchar>(bg->image().ptr(0, 0),
                             bg->image().ptr(bg->image().rows-1, bg->image().cols-1)));
    
    constexpr InputInfo iinput{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    constexpr OutputInfo output{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };

    { auto v = bg->diff<output, DifferenceMethod_t::absolute>(0, 0, 200);  ASSERT_EQ(v, 50); }
    { auto v = bg->diff<output, DifferenceMethod_t::none>(0, 0, 200);  ASSERT_EQ(v, 200); }
    { auto v = bg->diff<output, DifferenceMethod_t::none>(0, 0, 55);  ASSERT_EQ(v, 55); }
    { auto v = bg->diff<output, DifferenceMethod_t::sign>(0, 0, 100);  ASSERT_EQ(v, 50); }
    { auto v = bg->diff<output, DifferenceMethod_t::sign>(0, 0, 200);  ASSERT_EQ(v, 0); }
    
    uchar* px = input_pixels.data();
    line_without_grid<iinput, output, DifferenceMethod_t::absolute>(bg.get(), input, px, threshold, lines, pixels);
    
    Print("result lines:", lines);
    Print("result pixels:", pixels);

    // Expected results
    std::vector<HorizontalLine> expected_lines = {{0, 0, 9}, {1, 0, 0}};
    std::vector<uchar> expected_pixels = {0,10,20,30,40,50,60,70,80,90,100};

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    
    line_without_grid<iinput, output, DifferenceMethod_t::none>(bg.get(), input, px, threshold, lines, pixels);
    
    Print("result lines:", lines);
    Print("result pixels:", pixels);
    
    expected_lines = std::vector<HorizontalLine>{{0, 5, 9}, {1, 0, 9}};
    expected_pixels = std::vector<uchar>{50,60,70,80,90,100,110,120,130,140,150,160,170,180,190};
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod_t::sign>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0, 0, 9}, {1, 0, 0}};
    expected_pixels = {0,10,20,30,40,50,60,70,80,90,100};
    
    Print("result lines:", lines);
    Print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    input = {{0, 0, 9}, {1, 0, 9}};
    for (size_t i = 0; i < input_pixels.size(); ++i) {
        if(i % 3 == 0)
            input_pixels[i] = static_cast<uchar>(125);
        else if(i < input_pixels.size() / 2)
            input_pixels[i] = static_cast<uchar>(100 - i * 5 - threshold);
        else
            input_pixels[i] = static_cast<uchar>(100 + i * 5 + threshold);
    }
    
    Print("Changed input to:", input_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod_t::absolute>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0,1,2},{0,4,5},{0,7,8},{1,0,1},{1,3,4},{1,6,7},{1,9,9}};
    expected_pixels = {45,40,30,25,15,10,200,205,215,220,230,235,245};
    
    Print("result lines:", lines);
    Print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod_t::sign>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0,1,2},{0,4,5},{0,7,8}};
    expected_pixels = {45,40,30,25,15,10};
    
    Print("result lines:", lines);
    Print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

namespace cmn {
    // gunit test printto method for printing std::vector<HorizontalLine>
    void PrintTo(const std::vector<HorizontalLine>& lines, std::ostream* os) {
        *os << "[";
        for (const auto& line : lines) {
            *os << line.toStr() << ", ";
        }
        *os << "]";
    }
}

// PrintTo method for std::vector<uchar>
namespace std {
    void PrintTo(const std::vector<uchar>& pixels, std::ostream* os) {
        *os << "[";
        for (const auto& pixel : pixels) {
            *os << static_cast<int>(pixel) << ", ";
        }
        *os << "]";
    }
}

TEST_F(LineWithoutGridTest, SignDifferenceMethod) {
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};
    std::vector<uchar> px;
    px.resize(20);
    for (int i = 0; i < 20; ++i) {
        px[i] = static_cast<uchar>((i % 2 == 0) ? i + 1 : 200);
    }
    int threshold = 50;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;
    
    constexpr InputInfo iinput{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    constexpr OutputInfo output{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    
    Print("background = ", px);
    
    uchar* ptr = px.data();
    line_without_grid<iinput, output, DifferenceMethod_t::sign>(bg.get(), input, ptr, threshold, lines, pixels);

    // Expected results
    using HL = HorizontalLine;
    std::vector<HorizontalLine> expected_lines = { HL(0,0,0), HL(0,2,2), HL(0,4,4), HL(0,6,6), HL(0,8,8), HL(1,0,0), HL(1,2,2), HL(1,4,4), HL(1,6,6), HL(1,8,8) };
    std::vector<uchar> expected_pixels = {1,3,5,7,9,11,13,15,17,19};

    
    print_results<iinput>("expected", expected_lines, expected_pixels);
    
    print_results<iinput>("result", lines, pixels);
    
    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

TEST_F(LineWithoutGridTest, NoneDifferenceMethod) {
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};

    std::array <uchar, 20> input_pixels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                            10,11,12,13,14,15,16,17,18,19 };
    int threshold = 5;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;

    auto start = input_pixels.data();
    constexpr InputInfo iinput{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    constexpr OutputInfo output{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    line_without_grid<iinput, output, DifferenceMethod_t::none>(bg.get(), input, start, threshold, lines, pixels);

    // Expected results
    std::vector<HorizontalLine> expected_lines = { {0, 5, 9}, {1, 0, 9}};

    // in the first row, the last 5 pixels are above the threshold
    // in the second row, all pixels are above the threshold, so all are added
    std::vector<uchar> expected_pixels = {                      5,  6,  7,  8,  9,
                                           10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

