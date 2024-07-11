#include <gtest/gtest.h>
#include <commons.pc.h>
#include <misc/Image.h>
#include <processing/Background.h>
#include <misc/PixelTree.h>
#include <misc/TrackingSettings.h>
#include <tracking/Posture.h>

using namespace cmn;
using namespace track;
using namespace blob;
using namespace pixel;

using ::testing::TestWithParam;
using ::testing::Values;

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
    this->bg = std::make_unique<Background>(std::move(this->image), nullptr);

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
        bg = std::make_unique<Background>(std::move(image), grid);
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

