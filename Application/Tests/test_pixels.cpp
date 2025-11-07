#include <commons.pc.h>
#include <gtest/gtest.h>
#include <misc/Image.h>
#include <processing/Background.h>
#include <misc/PVBlob.h>
#include <misc/PixelTree.h>
#include <misc/TrackingSettings.h>
#include <tracking/Posture.h>
#include <processing/LuminanceGrid.h>
#include <processing/HLine.h>

using namespace cmn;
using namespace track;
using namespace blob;
using namespace pixel;

using ::testing::TestWithParam;
using ::testing::Values;

using cmn::gui::Color;

// ---------------------------------------------------------------------------
// Additional unit tests for cmn::IllegalArray
// ---------------------------------------------------------------------------

namespace {

template <typename Container>
PixelArray_t make_rgb_pixel_array(const Container& values) {
    PixelArray_t pixels;
    pixels.reserve(values.size() * 3);
    for (const auto& v : values) {
        pixels.push_back(v[0], v[1], v[2]);
    }
    return pixels;
}

template <typename Container>
PixelArray_t make_gray_pixel_array_from_values(const Container& values) {
    PixelArray_t pixels;
    pixels.reserve(values.size());
    for (const auto& v : values) {
        pixels.push_back(cmn::bgr2gray(cmn::RGBArray{v[0], v[1], v[2]}));
    }
    return pixels;
}

PixelArray_t convert_rgb_pixels_to_gray(const PixelArray_t& rgb_pixels) {
    PixelArray_t result;
    result.reserve(rgb_pixels.size() / 3);
    for (std::size_t i = 0; i < rgb_pixels.size(); i += 3) {
        result.push_back(cmn::bgr2gray(cmn::RGBArray{
            rgb_pixels[i],
            rgb_pixels[i + 1],
            rgb_pixels[i + 2]
        }));
    }
    return result;
}

PixelArray_t convert_rgb_vector_to_gray(const std::vector<uchar>& rgb_pixels) {
    PixelArray_t result;
    result.reserve(rgb_pixels.size() / 3);
    for (std::size_t i = 0; i < rgb_pixels.size(); i += 3) {
        result.push_back(cmn::bgr2gray(cmn::RGBArray{
            rgb_pixels[i],
            rgb_pixels[i + 1],
            rgb_pixels[i + 2]
        }));
    }
    return result;
}

template <typename T>
bool mats_equal(const cv::Mat& lhs, const cv::Mat& rhs) {
    if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
        return false;
    }
    return std::equal(lhs.begin<T>(), lhs.end<T>(), rhs.begin<T>(), rhs.end<T>());
}

} // namespace

TEST(IllegalArrays, InitializerListConstructor) {
    cmn::IllegalArray<int> arr = {1, 2, 3, 4};
    EXPECT_EQ(arr.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(arr[i], static_cast<int>(i + 1));
}

TEST(IllegalArrays, InitializerListAssignment) {
    cmn::IllegalArray<int> arr = {1, 2, 3};
    arr = {10, 20, 30, 40};
    EXPECT_EQ(arr.size(), 4u);
    int ref[] = {10, 20, 30, 40};
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

TEST(IllegalArrays, CopyConstructorAndAssignment) {
    cmn::IllegalArray<int> src = {5, 6, 7, 8, 9};

    // Copy‑ctor
    cmn::IllegalArray<int> copy(src);
    EXPECT_EQ(copy, src);

    // Mutate copy and verify independence
    copy[0] = 42;
    EXPECT_NE(copy, src);

    // Copy‑assignment
    cmn::IllegalArray<int> assign;
    assign = src;
    EXPECT_EQ(assign, src);
}

TEST(IllegalArrays, MoveConstructorAndAssignment) {
    cmn::IllegalArray<int> src = {1, 2, 3, 4};
    auto old_ptr = src.data();

    // Move‑ctor
    cmn::IllegalArray<int> moved(std::move(src));
    EXPECT_TRUE(src.empty());
    EXPECT_EQ(moved.size(), 4u);
    EXPECT_EQ(moved.data(), old_ptr);

    // Move‑assignment
    cmn::IllegalArray<int> dest;
    dest.resize(2);
    dest = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(dest.size(), 4u);
    EXPECT_EQ(dest[3], 4);
}

TEST(IllegalArrays, SelfAssignment) {
    cmn::IllegalArray<int> arr = {1, 2, 3};
    arr = arr;               // copy self‑assign
    EXPECT_EQ(arr.size(), 3u);
    arr = std::move(arr);    // move self‑assign
    EXPECT_EQ(arr.size(), 3u);
    EXPECT_EQ(arr[1], 2);
}

TEST(IllegalArrays, PushBackVariadic) {
    cmn::IllegalArray<int> arr;
    arr.push_back(1, 2, 3, 4, 5);
    EXPECT_EQ(arr.size(), 5u);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(arr[i], i + 1);
}

TEST(IllegalArrays, InsertSingleValue) {
    cmn::IllegalArray<int> arr = {1, 2, 3};
    // Insert two copies of 99 before element '2'
    arr.insert(arr.data() + 1, 99, 2);
    int ref[] = {1, 99, 99, 2, 3};
    EXPECT_EQ(arr.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

TEST(IllegalArrays, InsertRange) {
    cmn::IllegalArray<int> arr = {1, 2, 3, 4};
    int extra[] = {7, 8, 9};
    // Insert range {7,8,9} before value '3'
    arr.insert(arr.data() + 2, extra, extra + 3);
    int ref[] = {1, 2, 7, 8, 9, 3, 4};
    EXPECT_EQ(arr.size(), 7u);
    for (std::size_t i = 0; i < 7; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}


TEST(IllegalArrays, InsertIntoEmpty) {
    cmn::IllegalArray<int> arr;
    arr.insert(nullptr, 42, 3);  // three copies of 42
    EXPECT_EQ(arr.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(arr[i], 42);
}

// ---------------------------------------------------------------------------
// Extra edge‑case coverage for IllegalArray::insert
// ---------------------------------------------------------------------------

// Prepend a single element at the very beginning
TEST(IllegalArrays, InsertSingleValuePrepend) {
    cmn::IllegalArray<int> arr = {2, 3, 4};
    arr.insert(arr.data(), 1, 1);          // insert one copy of '1' at front
    int ref[] = {1, 2, 3, 4};
    ASSERT_EQ(arr.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Append multiple copies at the end
TEST(IllegalArrays, InsertSingleValueAppendMultiple) {
    cmn::IllegalArray<int> arr = {1, 2, 3};
    arr.insert(arr.data() + arr.size(), 4, 3); // add 4,4,4 to tail
    int ref[] = {1, 2, 3, 4, 4, 4};
    ASSERT_EQ(arr.size(), 6u);
    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Insert with count = 0 should leave the container unchanged
TEST(IllegalArrays, InsertSingleValueZeroCount) {
    cmn::IllegalArray<int> arr = {5, 6, 7};
    auto it = arr.data() + 1;
    arr.insert(it, 99, 0);                 // no‑op
    ASSERT_EQ(arr.size(), 3u);
    int ref[] = {5, 6, 7};
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Prepend a range [first, last)
TEST(IllegalArrays, InsertRangePrepend) {
    cmn::IllegalArray<int> arr = {4, 5};
    int extra[] = {1, 2, 3};
    arr.insert(arr.data(), extra, extra + 3);
    int ref[] = {1, 2, 3, 4, 5};
    ASSERT_EQ(arr.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Append a range at the very end
TEST(IllegalArrays, InsertRangeAppend) {
    cmn::IllegalArray<int> arr = {1, 2};
    int extra[] = {3, 4, 5};
    arr.insert(arr.data() + arr.size(), extra, extra + 3);
    int ref[] = {1, 2, 3, 4, 5};
    ASSERT_EQ(arr.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Insert an empty range (first == last) — should be a no‑op
TEST(IllegalArrays, InsertRangeEmpty) {
    cmn::IllegalArray<int> arr = {7, 8, 9};
    arr.insert(arr.data() + 1, arr.data(), arr.data()); // zero elements
    int ref[] = {7, 8, 9};
    ASSERT_EQ(arr.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

// Ensure capacity growth works and data stays intact during insert
TEST(IllegalArrays, InsertTriggersCapacityGrowth) {
    cmn::IllegalArray<int> arr;
    arr.reserve(2);                   // force small initial capacity
    arr.push_back(1, 2);              // size == capacity == 2
    auto oldCap = arr.capacity();
    arr.insert(arr.data() + 1, 99, 1); // insert in the middle
    int ref[] = {1, 99, 2};
    ASSERT_EQ(arr.size(), 3u);
    EXPECT_GE(arr.capacity(), oldCap); // capacity must have grown
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(arr[i], ref[i]);
}

TEST(IllegalArrays, ComparisonOperators) {
    cmn::IllegalArray<int> a = {1, 2, 3};
    cmn::IllegalArray<int> b = {1, 2, 3};
    cmn::IllegalArray<int> c = {1, 2, 4};
    cmn::IllegalArray<int> d = {1, 2};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(d < a);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(c > a);
    EXPECT_TRUE(d <= a);
    EXPECT_TRUE(c >= b);
}

TEST(IllegalArrays, CapacityGrowth) {
    cmn::IllegalArray<int> arr;
    EXPECT_EQ(arr.capacity(), 0u);

    arr.reserve(10);
    EXPECT_GE(arr.capacity(), 10u);
    auto cap = arr.capacity();

    arr.reserve(5);  // should not shrink
    EXPECT_EQ(arr.capacity(), cap);

    arr.resize(20);
    EXPECT_EQ(arr.size(), 20u);
    EXPECT_GE(arr.capacity(), 20u);
}

TEST(IllegalArrays, Basic) {
    IllegalArray<int> arr;

    // 1. resize() should allocate memory and allow indexed writes/reads
    arr.resize(5);
    for (std::size_t i = 0; i < 5; ++i) {
        arr[i] = static_cast<int>(i * 10);
    }
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(arr.at(i), static_cast<int>(i * 10));
    }

    // 2. reserve() with a larger value must keep existing elements intact
    arr.reserve(20);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(arr.at(i), static_cast<int>(i * 10));
    }

    // 3. Growing again with resize() should enlarge the usable range
    arr.resize(10);
    for (std::size_t i = 5; i < 10; ++i) {
        arr[i] = static_cast<int>(i * 10);
    }
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(arr.at(i), static_cast<int>(i * 10));
    }

#ifndef NDEBUG
    // 4. at() past the current size should trigger an assert in debug builds
    EXPECT_DEATH_IF_SUPPORTED(arr.at(10), "");
#endif
}

TEST(IllegalArrays, RangeLoop) {
    IllegalArray<int> arr;
    arr.resize(5);
    for (std::size_t i = 0; i < 5; ++i) {
        arr[i] = static_cast<int>(i * 7);   // unique stride to catch ordering errors
    }

    // Iterate with a non‑const range‑based for loop
    std::size_t idx = 0;
    for (int v : arr) {
        EXPECT_EQ(v, static_cast<int>(idx * 7));
        ++idx;
    }
    EXPECT_EQ(idx, 5u);

    // Iterate with a const range‑based for loop
    const auto& cref = arr;
    idx = 0;
    for (int v : cref) {
        EXPECT_EQ(v, static_cast<int>(idx * 7));
        ++idx;
    }
    EXPECT_EQ(idx, 5u);

    // Use a standard algorithm to verify compatibility
    int sum = std::accumulate(arr.begin(), arr.end(), 0);
    EXPECT_EQ(sum, 7 * (0 + 1 + 2 + 3 + 4)); // 7 * 10 = 70
}

// New test inserted as requested
TEST(IllegalArrays, AssignFromEmptyArrayAndSelfAssignEmpty) {
    cmn::IllegalArray<int> arr1 = {1, 2, 3};
    cmn::IllegalArray<int> empty;
    
    // Assign from empty to filled
    arr1 = empty;
    EXPECT_TRUE(arr1.empty());
    
    // Assign from filled to empty
    empty = arr1;
    EXPECT_TRUE(empty.empty());
    
    // Copy assign empty to empty
    cmn::IllegalArray<int> arr2;
    arr2 = empty;
    EXPECT_TRUE(arr2.empty());
    
    // Move assign empty to filled
    arr1 = cmn::IllegalArray<int>{};
    EXPECT_TRUE(arr1.empty());
    
    // Move assign empty to empty
    arr2 = cmn::IllegalArray<int>{};
    EXPECT_TRUE(arr2.empty());
    
    // Self-assign when empty (copy)
    arr2 = arr2;
    EXPECT_TRUE(arr2.empty());
    
    // Self-assign when empty (move)
    arr2 = std::move(arr2);
    EXPECT_TRUE(arr2.empty());
    
    // Just to be sure: pushing afterwards
    arr2.push_back(42);
    EXPECT_EQ(arr2.size(), 1u);
    EXPECT_EQ(arr2[0], 42);
}

TEST(IllegalArrays, MoveAssignDestructSafety) {
    cmn::IllegalArray<int> a = {1,2,3,4};
    cmn::IllegalArray<int> b = {5,6};
    a = std::move(b);
    // Now destroy b (should not segfault)
    b.clear();
    EXPECT_TRUE(b.empty());
}

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

// -----------------------------------------------------------------------------
// Additional HLine32 performance benchmarks
// -----------------------------------------------------------------------------
TEST(PerformanceTest, HLine32ConstructorAndAccess) {
    const int iterations = 1000000;
    using HLine32 = cmn::CPULabeling::Line32;
    std::vector<HLine32, NoInitializeAllocator<HLine32>> lines;
    lines.reserve(iterations);
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i) {
        uint16_t x0 = i & 0x1FFF;
        uint16_t x1 = x0 + (i & 0x3F);
        uint16_t y  = i & 0x1FFF;
        HLine32 line(x0, x1, y);
        volatile auto vx0 = line.x0();
        volatile auto vy  = line.y();
        lines.push_back(line);
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Print("HLine32 constructor and x0/y access: ", ms, " ms");
    SUCCEED();
}

TEST(PerformanceTest, HLine32BatchConstructionAndDestruction) {
    const int batches = 1000;
    const int batchSize = 10000;
    using HLine32 = cmn::CPULabeling::Line32;
    auto start = std::chrono::steady_clock::now();
    for(int j = 0; j < batches; ++j) {
        std::vector<HLine32, NoInitializeAllocator<HLine32>> lines;
        lines.reserve(batchSize);
        for(int i = 0; i < batchSize; ++i) {
            lines.emplace_back(i & 0x1FFF, (i & 0x1FFF) + (i & 0x3F), i & 0x1FFF);
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Print("HLine32 batch construction and destruction: ", ms, " ms");
    SUCCEED();
}

TEST(PerformanceTest, HLine32MoveConstruction) {
    const int iterations = 100000;
    using HLine32 = cmn::CPULabeling::Line32;
    std::vector<HLine32, NoInitializeAllocator<HLine32>> src;
    src.reserve(iterations);
    for(int i = 0; i < iterations; ++i) {
        src.emplace_back(i & 0x1FFF, (i & 0x1FFF) + (i & 0x3F), i & 0x1FFF);
    }
    auto start = std::chrono::steady_clock::now();
    std::vector<HLine32, NoInitializeAllocator<HLine32>> dst = std::move(src);
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    Print("HLine32 move construction (microseconds): ", us, " us");
    SUCCEED();
}

// -----------------------------------------------------------------------------
// HorizontalLine performance benchmarks
// -----------------------------------------------------------------------------
TEST(PerformanceTest, HorizontalLineConstructorAndAccess) {
    const int iterations = 1000000;
    using HL = cmn::HorizontalLine;
    std::vector<HL, NoInitializeAllocator<HL>> lines;
    lines.reserve(iterations);
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i) {
        uint16_t x0 = i & 0x1FFF;
        uint16_t x1 = x0 + (i & 0x3F);
        uint16_t y  = i & 0x1FFF;
        HL line(y, x0, x1);
        volatile auto vx0 = line.x0;
        volatile auto vy  = line.y;
        lines.push_back(line);
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Print("HorizontalLine constructor and x0/y access: ", ms, " ms");
    SUCCEED();
}

TEST(PerformanceTest, HorizontalLineBatchConstructionAndDestruction) {
    const int batches = 1000;
    const int batchSize = 10000;
    using HL = cmn::HorizontalLine;
    auto start = std::chrono::steady_clock::now();
    for(int j = 0; j < batches; ++j) {
        std::vector<HL, NoInitializeAllocator<HL>> lines;
        lines.reserve(batchSize);
        for(int i = 0; i < batchSize; ++i) {
            uint16_t y  = i & 0x1FFF;
            uint16_t x0 = i & 0x1FFF;
            uint16_t x1 = x0 + (i & 0x3F);
            lines.emplace_back(y, x0, x1);
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Print("HorizontalLine batch construction and destruction: ", ms, " ms");
    SUCCEED();
}

TEST(PerformanceTest, HorizontalLineMoveConstruction) {
    const int iterations = 100000;
    using HL = cmn::HorizontalLine;
    std::vector<HL, NoInitializeAllocator<HL>> src;
    src.reserve(iterations);
    for(int i = 0; i < iterations; ++i) {
        uint16_t y  = i & 0x1FFF;
        uint16_t x0 = i & 0x1FFF;
        uint16_t x1 = x0 + (i & 0x3F);
        src.emplace_back(y, x0, x1);
    }
    auto start = std::chrono::steady_clock::now();
    std::vector<HL, NoInitializeAllocator<HL>> dst = std::move(src);
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    Print("HorizontalLine move construction (microseconds): ", us, " us");
    SUCCEED();
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
void print_results(std::string_view name, const std::vector<HorizontalLine>& lines, const PixelArray_t& pixels)
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
    PixelArray_t input_pixels(20 * params.input_info.channels);
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
    PixelArray_t pixels;
    
    uchar* px = input_pixels.data();
    
    // Print debug information
    DebugHeader(no_quotes(params.ToString()));
    Print("Background: ", PixelArray_t{
        this->bg->image().data(),
        this->bg->image().data() + this->bg->image().size()
    });
    
    print_results<params.input_info>("input", input, input_pixels);
    
    // Define expected results based on the parameters
    std::vector<HorizontalLine> expected_lines;
    PixelArray_t expected_pixels;
    
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

TEST(BackgroundThresholding, RGB8AbsoluteDifferenceSimulatedBlob) {
    constexpr int width = 4;
    constexpr int height = 2;

    auto background_image_rgb = Image::Make(height, width, 3);
    auto bg_mat_rgb = background_image_rgb->get();
    bg_mat_rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(30, 30, 30);
    bg_mat_rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(50, 50, 50);
    bg_mat_rgb.at<cv::Vec3b>(0, 2) = cv::Vec3b(70, 70, 70);
    bg_mat_rgb.at<cv::Vec3b>(0, 3) = cv::Vec3b(90, 90, 90);
    bg_mat_rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(40, 40, 40);
    bg_mat_rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(60, 60, 60);
    bg_mat_rgb.at<cv::Vec3b>(1, 2) = cv::Vec3b(80, 80, 80);
    bg_mat_rgb.at<cv::Vec3b>(1, 3) = cv::Vec3b(100, 100, 100);

    auto background_image_gray = Image::Make(height, width, 1);
    cv::cvtColor(bg_mat_rgb, background_image_gray->get(), cv::COLOR_BGR2GRAY);

    Background bg_rgb(std::move(background_image_rgb), meta_encoding_t::rgb8);
    Background bg_gray(std::move(background_image_gray), meta_encoding_t::gray);

    std::vector<HorizontalLine> input_lines = {
        HorizontalLine(0, 0, width - 1),
        HorizontalLine(1, 0, width - 1)
    };
    const std::array<std::array<uchar, 3>, width * height> blob_values{{
        {25, 25, 25},
        {110, 110, 110},
        {80, 80, 80},
        {10, 200, 10},
        {30, 30, 30},
        {95, 95, 95},
        {200, 200, 200},
        {100, 100, 100},
    }};

    PixelArray_t blob_pixels_rgb = make_rgb_pixel_array(blob_values);
    PixelArray_t blob_pixels_gray = make_gray_pixel_array_from_values(blob_values);

    constexpr InputInfo input_info{
        .channels = 3u,
        .encoding = meta_encoding_t::rgb8
    };
    constexpr OutputInfo output_info{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };

    constexpr int threshold = 25;

    std::vector<HorizontalLine> thresholded_lines;
    PixelArray_t thresholded_pixels;
    uchar* px = blob_pixels_rgb.data();

    line_without_grid<input_info, output_info, DifferenceMethod_t::absolute>(
        &bg_rgb, input_lines, px, threshold, thresholded_lines, thresholded_pixels);

    std::vector<HorizontalLine> expected_lines = {
        HorizontalLine(0, 1, 1),
        HorizontalLine(0, 3, 3),
        HorizontalLine(1, 1, 2)
    };
    PixelArray_t expected_pixels = {
        110, 110, 110,
        10,  200, 10,
        95,  95,  95,
        200, 200, 200
    };

    EXPECT_EQ(thresholded_lines, expected_lines);
    EXPECT_EQ(thresholded_pixels, expected_pixels);

    constexpr InputInfo gray_input{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };
    constexpr OutputInfo gray_output{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };

    std::vector<HorizontalLine> thresholded_lines_gray;
    PixelArray_t thresholded_pixels_gray;
    PixelArray_t expected_pixels_gray = convert_rgb_pixels_to_gray(expected_pixels);
    auto input_lines_gray = input_lines;
    auto blob_pixels_gray_copy = blob_pixels_gray;
    uchar* px_gray = blob_pixels_gray_copy.data();

    line_without_grid<gray_input, gray_output, DifferenceMethod_t::absolute>(
        &bg_gray, input_lines_gray, px_gray, threshold, thresholded_lines_gray, thresholded_pixels_gray);

    EXPECT_EQ(thresholded_lines_gray, expected_lines);
    EXPECT_EQ(thresholded_pixels_gray, expected_pixels_gray);
}

TEST(BackgroundThresholding, RGB8LuminanceAlphaImageSimulatedSimpleBlob) {
    constexpr int width = 4;
    constexpr int height = 2;
    constexpr int threshold = 25;

    auto background_image_rgb = Image::Make(height, width, 3);
    auto bg_mat_rgb = background_image_rgb->get();
    bg_mat_rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(30, 30, 30);
    bg_mat_rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(50, 50, 50);
    bg_mat_rgb.at<cv::Vec3b>(0, 2) = cv::Vec3b(70, 70, 70);
    bg_mat_rgb.at<cv::Vec3b>(0, 3) = cv::Vec3b(90, 90, 90);
    bg_mat_rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(40, 40, 40);
    bg_mat_rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(60, 60, 60);
    bg_mat_rgb.at<cv::Vec3b>(1, 2) = cv::Vec3b(80, 80, 80);
    bg_mat_rgb.at<cv::Vec3b>(1, 3) = cv::Vec3b(100, 100, 100);

    auto background_image_gray = Image::Make(height, width, 1);
    cv::cvtColor(bg_mat_rgb, background_image_gray->get(), cv::COLOR_BGR2GRAY);

    Background bg_rgb(std::move(background_image_rgb), meta_encoding_t::rgb8);
    Background bg_gray(std::move(background_image_gray), meta_encoding_t::gray);

    const std::vector<HorizontalLine> base_lines = {
        HorizontalLine(0, 0, width - 1),
        HorizontalLine(1, 0, width - 1)
    };
    const std::array<std::array<uchar, 3>, width * height> blob_values{{
        {25, 25, 25},
        {110, 110, 110},
        {80, 80, 80},
        {10, 200, 10},
        {30, 30, 30},
        {95, 95, 95},
        {200, 200, 200},
        {100, 100, 100},
    }};

    auto lines_rgb = std::make_unique<line_ptr_t::element_type>(base_lines);
    auto pixels_rgb = std::make_unique<pixel_ptr_t::element_type>();
    for (const auto& v : blob_values) {
        pixels_rgb->push_back(v[0], v[1], v[2]);
    }

    pv::Blob blob_rgb(std::move(lines_rgb), std::move(pixels_rgb), pv::Blob::flag(pv::Blob::Flags::is_rgb), {});

    Image image;
    const auto pos = blob_rgb.luminance_alpha_image(
        bg_rgb,
        threshold,
        image,
        /*padding=*/0,
        OutputInfo{
            .channels = 4u,
            .encoding = meta_encoding_t::rgb8
        });

    ASSERT_EQ(pos, Vec2(0));
    ASSERT_EQ(image.cols, static_cast<uint>(width));
    ASSERT_EQ(image.rows, static_cast<uint>(height));
    ASSERT_EQ(image.dims, 4u);
    ASSERT_EQ(image.size(), static_cast<size_t>(width * height * 4));

    const std::vector<uchar> expected = {
        // row 0
        0,   0,   0,   0,
        110, 110, 110, 210,
        0,   0,   0,   0,
        10,  200, 10,  118,
        // row 1
        0,   0,   0,   0,
        95,  95,  95,  130,
        200, 200, 200, 255,
        0,   0,   0,   0
    };

    const uchar* data_begin = image.data();
    ASSERT_NE(data_begin, nullptr);
    const uchar* data_end = data_begin + expected.size();

    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), data_begin, data_end));

    auto lines_gray = std::make_unique<line_ptr_t::element_type>(base_lines);
    auto pixels_gray = std::make_unique<pixel_ptr_t::element_type>();
    for (const auto& v : blob_values) {
        pixels_gray->push_back(cmn::bgr2gray(cmn::RGBArray{v[0], v[1], v[2]}));
    }

    pv::Blob blob_gray(std::move(lines_gray), std::move(pixels_gray), 0, {});

    Image image_gray;
    const auto pos_gray = blob_gray.luminance_alpha_image(
        bg_gray,
        threshold,
        image_gray,
        /*padding=*/0,
        OutputInfo{
            .channels = 4u,
            .encoding = meta_encoding_t::rgb8
        });

    ASSERT_EQ(pos_gray, pos);
    ASSERT_EQ(image_gray.cols, image.cols);
    ASSERT_EQ(image_gray.rows, image.rows);
    ASSERT_EQ(image_gray.dims, image.dims);
    ASSERT_EQ(image_gray.size(), image.size());

    cv::Mat color_mat = image.get();
    cv::Mat gray_mat = image_gray.get();

    cv::Mat color_alpha, gray_alpha;
    cv::extractChannel(color_mat, color_alpha, 3);
    cv::extractChannel(gray_mat, gray_alpha, 3);
    EXPECT_TRUE(mats_equal<uchar>(color_alpha, gray_alpha));

    cv::Mat color_gray_converted;
    cv::cvtColor(color_mat, color_gray_converted, cv::COLOR_BGRA2GRAY);
    cv::Mat gray_converted;
    cv::cvtColor(gray_mat, gray_converted, cv::COLOR_BGRA2GRAY);
    EXPECT_TRUE(mats_equal<uchar>(color_gray_converted, gray_converted));
}

TEST(BlobThresholding, RGB8AbsoluteDifferenceMultiRow) {
    constexpr int width = 4;
    constexpr int height = 2;
    constexpr int threshold = 25;

    auto background_image_rgb = Image::Make(height, width, 3);
    auto bg_mat_rgb = background_image_rgb->get();
    bg_mat_rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(30, 30, 30);
    bg_mat_rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(50, 50, 50);
    bg_mat_rgb.at<cv::Vec3b>(0, 2) = cv::Vec3b(70, 70, 70);
    bg_mat_rgb.at<cv::Vec3b>(0, 3) = cv::Vec3b(90, 90, 90);
    bg_mat_rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(40, 40, 40);
    bg_mat_rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(60, 60, 60);
    bg_mat_rgb.at<cv::Vec3b>(1, 2) = cv::Vec3b(80, 80, 80);
    bg_mat_rgb.at<cv::Vec3b>(1, 3) = cv::Vec3b(100, 100, 100);

    auto background_image_gray = Image::Make(height, width, 1);
    cv::cvtColor(bg_mat_rgb, background_image_gray->get(), cv::COLOR_BGR2GRAY);

    Background bg_rgb(std::move(background_image_rgb), meta_encoding_t::rgb8);
    Background bg_gray(std::move(background_image_gray), meta_encoding_t::gray);

    const std::array<std::array<uchar, 3>, width * height> blob_values{{
        {25, 25, 25},
        {110, 110, 110},
        {80, 80, 80},
        {10, 200, 10},
        {30, 30, 30},
        {95, 95, 95},
        {200, 200, 200},
        {90, 90, 90},
    }};

    auto lines_rgb = std::make_unique<line_ptr_t::element_type>();
    lines_rgb->emplace_back(0, 0, width - 1);
    lines_rgb->emplace_back(1, 0, width - 1);

    auto pixels_rgb = std::make_unique<pixel_ptr_t::element_type>();
    for (const auto& value : blob_values) {
        pixels_rgb->push_back(value[0], value[1], value[2]);
    }

    pv::Blob blob_rgb(std::move(lines_rgb), std::move(pixels_rgb), pv::Blob::flag(pv::Blob::Flags::is_rgb), {});

    auto thresholded = blob_rgb.threshold(threshold, bg_rgb);
    ASSERT_TRUE(thresholded);
    EXPECT_TRUE(thresholded->is_rgb());
    EXPECT_EQ(thresholded->channels(), 3u);

    ASSERT_TRUE(thresholded->lines());
    const auto& resulting_lines = *thresholded->lines();
    std::vector<HorizontalLine> expected_lines = {
        HorizontalLine(0, 1, 1),
        HorizontalLine(0, 3, 3),
        HorizontalLine(1, 1, 2)
    };
    EXPECT_EQ(resulting_lines, expected_lines);

    ASSERT_TRUE(thresholded->pixels());
    const auto& res_pixels = *thresholded->pixels();
    std::vector<uchar> actual(res_pixels.begin(), res_pixels.end());
    const std::vector<uchar> expected_pixels = {
        110, 110, 110,
        10,  200, 10,
        95,  95,  95,
        200, 200, 200
    };
    EXPECT_EQ(actual, expected_pixels);

    auto lines_gray = std::make_unique<line_ptr_t::element_type>();
    lines_gray->emplace_back(0, 0, width - 1);
    lines_gray->emplace_back(1, 0, width - 1);

    auto pixels_gray = std::make_unique<pixel_ptr_t::element_type>();
    for (const auto& value : blob_values) {
        pixels_gray->push_back(cmn::bgr2gray(cmn::RGBArray{value[0], value[1], value[2]}));
    }

    pv::Blob blob_gray(std::move(lines_gray), std::move(pixels_gray), 0, {});
    auto thresholded_gray = blob_gray.threshold(threshold, bg_gray);
    ASSERT_TRUE(thresholded_gray);
    EXPECT_FALSE(thresholded_gray->is_rgb());
    EXPECT_EQ(thresholded_gray->channels(), 1u);
    ASSERT_TRUE(thresholded_gray->lines());
    EXPECT_EQ(*thresholded_gray->lines(), expected_lines);

    ASSERT_TRUE(thresholded_gray->pixels());
    const auto& res_pixels_gray = *thresholded_gray->pixels();
    PixelArray_t actual_gray_from_rgb = convert_rgb_vector_to_gray(actual);
    EXPECT_EQ(res_pixels_gray, actual_gray_from_rgb);
}

TEST(ImageFromLines, RGB8AbsoluteThresholdWithBackground) {
    constexpr int width = 4;
    constexpr int height = 2;
    constexpr int threshold = 25;
    
    auto background_image_rgb = Image::Make(height, width, 3);
    auto bg_mat_rgb = background_image_rgb->get();
    bg_mat_rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(30, 30, 30);
    bg_mat_rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(50, 50, 50);
    bg_mat_rgb.at<cv::Vec3b>(0, 2) = cv::Vec3b(70, 70, 70);
    bg_mat_rgb.at<cv::Vec3b>(0, 3) = cv::Vec3b(90, 90, 90);
    bg_mat_rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(40, 40, 40);
    bg_mat_rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(60, 60, 60);
    bg_mat_rgb.at<cv::Vec3b>(1, 2) = cv::Vec3b(80, 80, 80);
    bg_mat_rgb.at<cv::Vec3b>(1, 3) = cv::Vec3b(100, 100, 100);

    auto background_image_gray = Image::Make(height, width, 1);
    cv::cvtColor(bg_mat_rgb, background_image_gray->get(), cv::COLOR_BGR2GRAY);

    Background bg_rgb(std::move(background_image_rgb), meta_encoding_t::rgb8);
    Background bg_gray(std::move(background_image_gray), meta_encoding_t::gray);

    std::vector<HorizontalLine> lines = {
        HorizontalLine(0, 0, width - 1),
        HorizontalLine(1, 0, width - 1)
    };

    const std::array<std::array<uchar, 3>, width * height> blob_values{{
        {25, 25, 25},
        {110, 110, 110},
        {80, 80, 80},
        {10, 200, 10},
        {30, 30, 30},
        {95, 95, 95},
        {200, 200, 200},
        {100, 100, 100},
    }};

    PixelArray_t pixels_rgb = make_rgb_pixel_array(blob_values);

    cv::Mat mask;
    cv::Mat image;
    cv::Mat differences;

    constexpr InputInfo input_info{
        .channels = 3u,
        .encoding = meta_encoding_t::rgb8
    };

    auto [rect, recount] = imageFromLines(
        input_info,
        lines,
        &mask,
        &image,
        &differences,
        &pixels_rgb,
        threshold,
        &bg_rgb,
        /*padding=*/0);

    EXPECT_EQ(rect, cv::Rect(0, 0, width, height));
    EXPECT_EQ(recount, size_t(4));

    ASSERT_FALSE(mask.empty());
    ASSERT_EQ(mask.rows, height);
    ASSERT_EQ(mask.cols, width);
    ASSERT_EQ(mask.type(), CV_8UC1);

    const cv::Mat expected_mask = (cv::Mat_<uchar>(height, width) <<
        0, 255,   0, 255,
        0, 255, 255,   0);
    EXPECT_TRUE(std::equal(mask.begin<uchar>(), mask.end<uchar>(), expected_mask.begin<uchar>(), expected_mask.end<uchar>()));

    ASSERT_FALSE(image.empty());
    ASSERT_EQ(image.rows, height);
    ASSERT_EQ(image.cols, width);
    ASSERT_EQ(image.type(), CV_8UC3);

    const cv::Mat expected_image = (cv::Mat_<cv::Vec3b>(height, width) <<
        cv::Vec3b(0,   0,   0),   cv::Vec3b(110, 110, 110), cv::Vec3b(0,   0,   0),   cv::Vec3b(10, 200,  10),
        cv::Vec3b(0,   0,   0),   cv::Vec3b(95,  95,  95),  cv::Vec3b(200, 200, 200), cv::Vec3b(0,   0,   0));
    EXPECT_TRUE(std::equal(image.begin<cv::Vec3b>(), image.end<cv::Vec3b>(), expected_image.begin<cv::Vec3b>(), expected_image.end<cv::Vec3b>()));

    ASSERT_FALSE(differences.empty());
    ASSERT_EQ(differences.rows, height);
    ASSERT_EQ(differences.cols, width);
    ASSERT_EQ(differences.type(), CV_8UC3);

    /// since we use threshold, some of these values will be
    /// zeroed out (10,10,10) -> (0,0,0):
    const cv::Mat expected_diff = (cv::Mat_<cv::Vec3b>(height, width) <<
        cv::Vec3b(0,   0,   0),   cv::Vec3b(60,  60,  60),  cv::Vec3b(0, 0, 0),   cv::Vec3b(80, 110, 80),
        cv::Vec3b(0, 0, 0),    cv::Vec3b(35, 35, 35),    cv::Vec3b(120, 120, 120), cv::Vec3b(0, 0, 0));
    
    Image diff_image(differences);
    Image ediff_image(expected_diff);
    
    EXPECT_EQ(std::vector<int>(diff_image.data(), diff_image.data()+diff_image.size()),
              std::vector<int>(ediff_image.data(), ediff_image.data()+ediff_image.size()));
    //EXPECT_TRUE(std::equal(differences.begin<cv::Vec3b>(), differences.end<cv::Vec3b>(), expected_diff.begin<cv::Vec3b>(), expected_diff.end<cv::Vec3b>()));

    PixelArray_t pixels_gray = make_gray_pixel_array_from_values(blob_values);
    cv::Mat mask_gray;
    cv::Mat image_gray;
    cv::Mat differences_gray;

    constexpr InputInfo gray_input{
        .channels = 1u,
        .encoding = meta_encoding_t::gray
    };

    auto [rect_gray, recount_gray] = imageFromLines(
        gray_input,
        lines,
        &mask_gray,
        &image_gray,
        &differences_gray,
        &pixels_gray,
        threshold,
        &bg_gray,
        /*padding=*/0);

    EXPECT_EQ(rect_gray, rect);
    EXPECT_EQ(recount_gray, recount);

    ASSERT_FALSE(mask_gray.empty());
    EXPECT_TRUE(mats_equal<uchar>(mask, mask_gray));

    ASSERT_FALSE(image_gray.empty());
    cv::Mat image_rgb_to_gray;
    cv::cvtColor(image, image_rgb_to_gray, cv::COLOR_BGR2GRAY);
    EXPECT_TRUE(mats_equal<uchar>(image_rgb_to_gray, image_gray));

    ASSERT_FALSE(differences_gray.empty());
    
    /// Unfortunately we cannot expect these two to be equivalent ever, because
    /// we perform non-linear functions such as abs() or saturate() within each.
    
    /*Image diff_rgb_to_gray;
    Image rgb_differences(differences);
    Image gray_thresholded_differences(differences_gray);
    
    std::vector<uchar> rgb_pixels(rgb_differences.data(), rgb_differences.data() + rgb_differences.size());
    auto gray_pixels = convert_rgb_vector_to_gray(rgb_pixels);
    //EXPECT_TRUE(mats_equal<uchar>(diff_rgb_to_gray, differences_gray));
    
    EXPECT_EQ(std::vector<int>(gray_thresholded_differences.data(), gray_thresholded_differences.data() + gray_thresholded_differences.size()),
              std::vector<int>(gray_pixels.data(), gray_pixels.data() + gray_pixels.size()));*/
}

TEST(ImageFromLines, RGB8BackgroundSubtractionUsesAllChannels) {
    constexpr int threshold = 20;
    constexpr auto encoding = meta_encoding_t::rgb8;

    const auto make_background = [encoding]() {
        auto image = Image::Make(1, 1, required_image_channels(encoding));
        image->set_to(10);
        return image;
    };

    const std::array<std::array<uchar, 3>, 4> blobs{{
        {200, 10, 10},   // R channel dominant
        {10, 200, 10},   // G channel dominant
        {10, 10, 200},   // B channel dominant
        {200, 200, 200}  // All channels
    }};

    size_t index = 0;
    for(const auto& blob : blobs) {
        PixelArray_t pixels;
        pixels.push_back(blob[0], blob[1], blob[2]);

        Background background(make_background(), encoding);

        std::vector<HorizontalLine> lines = { HorizontalLine(0, 0, 0) };

        cv::Mat mask;

        cv::Mat differences;
        InputInfo input_info{
            .channels = required_storage_channels(encoding),
            .encoding = encoding
        };

        auto [rect, recount] = imageFromLines(
            input_info,
            lines,
            &mask,
            nullptr,
            &differences,
            &pixels,
            threshold,
            &background,
            /*padding=*/0);

        EXPECT_EQ(rect, cv::Rect(0, 0, 1, 1));
        EXPECT_EQ(recount, size_t(1)) << "blob_index=" << index;
        ASSERT_FALSE(mask.empty());
        EXPECT_EQ(mask.at<uchar>(0, 0), 255) << "blob_index=" << index;

        ++index;
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
    PixelArray_t pixels;
    
    Print("input: ", input);
    Print("pixels: ", input_pixels);

    Print("background:",
          PixelArray_t(bg->image().ptr(0, 0),
                             bg->image().ptr(0, 0) + bg->image().cols),
          PixelArray_t(bg->image().ptr(1, 0),
                             bg->image().ptr(1, 0) + bg->image().cols));
    Print(PixelArray_t(bg->image().ptr(0, 0),
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
    PixelArray_t expected_pixels = {0,10,20,30,40,50,60,70,80,90,100};

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
    expected_pixels = PixelArray_t{50,60,70,80,90,100,110,120,130,140,150,160,170,180,190};
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
    void PrintTo(const PixelArray_t& pixels, std::ostream* os) {
        *os << "[";
        for (const auto& pixel : pixels) {
            *os << static_cast<int>(pixel) << ", ";
        }
        *os << "]";
    }
}

TEST_F(LineWithoutGridTest, SignDifferenceMethod) {
    std::vector<HorizontalLine> input = {{0, 0, 9}, {1, 0, 9}};
    PixelArray_t px;
    px.resize(20);
    for (int i = 0; i < 20; ++i) {
        px[i] = static_cast<uchar>((i % 2 == 0) ? i + 1 : 200);
    }
    int threshold = 50;
    std::vector<HorizontalLine> lines;
    PixelArray_t pixels;
    
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
    PixelArray_t expected_pixels = {1,3,5,7,9,11,13,15,17,19};

    
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
    PixelArray_t pixels;

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
    PixelArray_t expected_pixels = {                      5,  6,  7,  8,  9,
                                           10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };

    // Perform assertions to check if the results are as expected
    ASSERT_EQ(lines, expected_lines);
    ASSERT_EQ(pixels, expected_pixels);
}
