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
        input_pixels[i] = static_cast<uchar>(i * 2);
    }
    
    int threshold = 50;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;
    
    print("input: ", input);
    print("pixels: ", input_pixels);

    print("background:",
          std::vector<uchar>(bg->image().ptr(0, 0),
                             bg->image().ptr(0, 0) + bg->image().cols),
          std::vector<uchar>(bg->image().ptr(1, 0),
                             bg->image().ptr(1, 0) + bg->image().cols));
    print(std::vector<uchar>(bg->image().ptr(0, 0),
                             bg->image().ptr(bg->image().rows-1, bg->image().cols-1)));
    
    constexpr InputInfo iinput{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    constexpr OutputInfo output{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    auto fn = [&]<DifferenceMethod method>(auto value) {
        return bg->diff<output, method>(0, 0, value);
    };

    ASSERT_EQ(fn.operator()<DifferenceMethod::absolute>(200), 50);
    ASSERT_EQ(fn.operator()<DifferenceMethod::none>(200), 200);
    ASSERT_EQ(fn.operator()<DifferenceMethod::none>(55), 55);
    ASSERT_EQ(fn.operator()<DifferenceMethod::sign>(100), 50);
    ASSERT_EQ(fn.operator()<DifferenceMethod::sign>(200), 0);
    
    uchar* px = input_pixels.data();
    line_without_grid<iinput, output, DifferenceMethod::absolute>(bg.get(), input, px, threshold, lines, pixels);
    
    print("result lines:", lines);
    print("result pixels:", pixels);

    // Expected results
    std::vector<HorizontalLine> expected_lines = {{0, 0, 9}, {1, 0, 9}};
    std::vector<uchar> expected_pixels = {150,148,146,144,142,140,138,136,134,132,130,128,126,124,122,120,118,116,114,112};

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    
    line_without_grid<iinput, output, DifferenceMethod::none>(bg.get(), input, px, threshold, lines, pixels);
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, std::vector<HorizontalLine>{});
    ASSERT_EQ(pixels, std::vector<uchar>{});
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0, 0, 9}, {1, 0, 9}};
    expected_pixels = {150,148,146,144,142,140,138,136,134,132,130,128,126,124,122,120,118,116,114,112};
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    input = {{0, 0, 9}, {1, 0, 9}};
    for (size_t i = 0; i < input_pixels.size(); ++i) {
        if(i < input_pixels.size() / 2)
            input_pixels[i] = static_cast<uchar>(150 - i - threshold);
        else
            input_pixels[i] = static_cast<uchar>(150 + i + threshold);
    }
    
    print("Changed input to:", input_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod::absolute>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0, 0, 9},{1, 0, 9}};
    expected_pixels = {50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69};
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<iinput, output, DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0,0,9}};
    expected_pixels = {50,51,52,53,54,55,56,57,58,59};
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
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
    uchar* px = new uchar[20];
    for (int i = 0; i < 20; ++i) {
        px[i] = static_cast<uchar>((i % 2 == 0) ? 0 : 255);
    }
    int threshold = 5;
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
    line_without_grid<iinput, output, DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);

    // Expected results
    using HL = HorizontalLine;
    std::vector<HorizontalLine> expected_lines = { HL(0,0,0), HL(0,2,2), HL(0,4,4), HL(0,6,6), HL(0,8,8), HL(1,0,0), HL(1,2,2), HL(1,4,4), HL(1,6,6), HL(1,8,8) };
    std::vector<uchar> expected_pixels = {150, 150, 150, 150, 150, 150, 150, 150, 150, 150 };

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
    line_without_grid<iinput, output, DifferenceMethod::none>(bg.get(), input, start, threshold, lines, pixels);

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

