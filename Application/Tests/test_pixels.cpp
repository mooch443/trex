#include <gtest/gtest.h>
#include <commons.pc.h>
#include <misc/Image.h>
#include <processing/Background.h>
#include <misc/PixelTree.h>

using namespace cmn;

template<DifferenceMethod method>
inline void line_without_grid (const Background* bg, const std::vector<HorizontalLine>& input, uchar*& px, int threshold, std::vector<HorizontalLine> &lines, std::vector<uchar> &pixels) {
    std::vector<uchar> output;
    output.reserve(pixels.capacity());
    for(const auto &line : input) {
        for (auto x=line.x0; x<=line.x1; ++x, ++px) {
            output.emplace_back(bg->diff<method>(x, line.y, *px));
        }
    }
    
    auto ox = output.data();
    for(const auto &line : input) {
        coord_t x0;
        uchar* start{nullptr};
        
        for (auto x=line.x0; x<=line.x1; ++x, ++ox) {
            if(*ox < threshold) {
                if(start) {
                    pixels.insert(pixels.end(), start, ox);
                    lines.emplace_back(line.y, x0, x - 1);
                    start = nullptr;
                }
                
            } else if(!start) {
                start = ox;
                x0 = x;
            }
        }
    
        if(start) {
            pixels.insert(pixels.end(), start, ox);
            lines.emplace_back(line.y, x0, line.x1);
        }
    }
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
    
    
    ASSERT_EQ(bg->diff<DifferenceMethod::absolute>(0, 0, 200), 50);
    ASSERT_EQ(bg->diff<DifferenceMethod::none>(0, 0, 200), 200);
    ASSERT_EQ(bg->diff<DifferenceMethod::none>(0, 0, 55), 55);
    ASSERT_EQ(bg->diff<DifferenceMethod::sign>(0, 0, 100), 50);
    ASSERT_EQ(bg->diff<DifferenceMethod::sign>(0, 0, 200), 0);
    
    uchar* px = input_pixels.data();
    line_without_grid<DifferenceMethod::absolute>(bg.get(), input, px, threshold, lines, pixels);
    
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
    line_without_grid<DifferenceMethod::none>(bg.get(), input, px, threshold, lines, pixels);
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, std::vector<HorizontalLine>{});
    ASSERT_EQ(pixels, std::vector<uchar>{});
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);
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
    line_without_grid<DifferenceMethod::absolute>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0, 0, 9},{1, 0, 9}};
    expected_pixels = {50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69};
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
    
    px = input_pixels.data();
    pixels.clear();
    lines.clear();
    line_without_grid<DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);
    expected_lines = {{0,0,9}};
    expected_pixels = {50,51,52,53,54,55,56,57,58,59};
    
    print("result lines:", lines);
    print("result pixels:", pixels);
    
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
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

    line_without_grid<DifferenceMethod::sign>(bg.get(), input, px, threshold, lines, pixels);

    // Expected results
    std::vector<HorizontalLine> expected_lines = {{0, 1, 2}, {1, 1, 2}};
    std::vector<uchar> expected_pixels = {255, 0, 255, 0};

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

TEST_F(LineWithoutGridTest, NoneDifferenceMethod) {
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};
    uchar* px = new uchar[20];
    for (int i = 0; i < 20; ++i) {
        px[i] = static_cast<uchar>(i);
    }
    int threshold = 5;
    std::vector<HorizontalLine> lines;
    std::vector<uchar> pixels;

    line_without_grid<DifferenceMethod::none>(bg.get(), input, px, threshold, lines, pixels);

    // Expected results
    std::vector<HorizontalLine> expected_lines = {{0, 0, 4}, {0, 6, 9}, {1, 0, 4}, {1, 6, 9}};
    std::vector<uchar> expected_pixels = {0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9};

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}

