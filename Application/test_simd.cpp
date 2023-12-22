#include <vector>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <sstream>

#include <chrono>
#include <iostream>
#include <random>
#include <algorithm>
#include <mutex>     // For std::once_flag and std::call_once

// Detect architecture and include appropriate headers
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define USE_NEON
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
    #include <emmintrin.h>
    #include <smmintrin.h>
    #include <immintrin.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif
    #define USE_SSE
#endif

using uchar = uint8_t;
// Template function to call a lambda with a compile-time constant index
template<typename Lambda, auto Index>
constexpr void callLambdaWithIndex(Lambda&& lambda) {
    lambda.template operator()<Index>();
}

// Helper structure for compile-time recursion
template<int Start, int End>
struct LambdaCaller {
    template<typename Lambda>
    static constexpr void call(Lambda&& lambda) {
        callLambdaWithIndex<Lambda, Start>(lambda);
        LambdaCaller<Start + 1, End>::call(std::forward<Lambda>(lambda));
    }
};

// Base case for recursion
template<int End>
struct LambdaCaller<End, End> {
    template<typename Lambda>
    static constexpr void call(Lambda&& lambda) {}
};

namespace cmn {
struct HorizontalLine;

namespace blob {
using line_ptr_t = std::unique_ptr<std::vector<HorizontalLine>>;
using pixel_ptr_t = std::unique_ptr<std::vector<uchar>>;
}
}

namespace cmn {

/**
    * A structure that represents a horizontal line on an image.
    */
using coord_t = uint16_t;
using ptr_safe_t = uint64_t;
struct HorizontalLine {
    coord_t x0, x1;
    coord_t y;
    coord_t padding;
    
    constexpr HorizontalLine() noexcept = default;
    constexpr HorizontalLine(coord_t y_, coord_t x0_, coord_t x1_) noexcept
        : x0(x0_), x1(x1_), y(y_), padding(0)
    {
        //assert(x0 <= x1);
    }
    
    constexpr bool inside(coord_t x_, coord_t y_) const noexcept {
        return y_ == y && x_ >= x0 && x_ <= x1;
    }
    
    constexpr bool overlap_x(const HorizontalLine& other) const noexcept {
        return other.x1 >= x0-1 && other.x0 <= x1+1;
    }
    
    constexpr bool overlap(const HorizontalLine& other) const noexcept {
        return other.y == y && overlap_x(other);
    }
    
    constexpr bool operator==(const HorizontalLine& other) const noexcept {
        return other.x0 == x0 && other.y == y && other.x1 == x1;
    }
    
    constexpr bool operator<(const HorizontalLine& other) const noexcept {
        //! Compares two HorizontalLines and sorts them by y and x0 coordinates
        //  (top-bottom, left-right)
        return y < other.y || (y == other.y && x0 < other.x0);
    }
    
    constexpr HorizontalLine merge(const HorizontalLine& other) const noexcept {
        //if(other.y != y)
        //    throw U_EXCEPTION("Cannot merge lines from y=",y," and ",other.y,"");
        //assert(overlap(other));
        return HorizontalLine(y, std::min(x0, other.x0), std::max(x1, other.x1));
    }
    
    std::string str() const {
        std::stringstream ss;
        ss << "HL<" << y << ", " << x0 << " - " << x1 << ">";
        return ss.str();
    }
    
    static void repair_lines_array(std::vector<HorizontalLine>&, std::vector<uchar>&);
    static void repair_lines_array(std::vector<HorizontalLine>&);
    
    std::string toStr() const;
    static std::string class_name() {
        return "HorizontalLine";
    }
};

struct ShortHorizontalLine {
private:
    //! starting and end position on x
    //  the last bit of _x1 is a flag telling the program
    //  whether this line is the last line on the current y coordinate.
    //  the following lines are on current_y + 1.
    uint16_t _x0, _x1;
    
public:
    //! compresses an array of HorizontalLines to an array of ShortHorizontalLines
    static std::vector<ShortHorizontalLine> compress(const std::vector<cmn::HorizontalLine>& lines);
    //! uncompresses an array of ShortHorizontalLines back to HorizontalLines
#if defined(USE_NEON) || defined(USE_SSE)
    // if SSE is enabled and its not MSVC
    #if defined(USE_SSE) && !defined(_MSC_VER)
        __attribute__((target("avx512f"))) static cmn::blob::line_ptr_t uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed) noexcept;
        __attribute__((target("default"))) static cmn::blob::line_ptr_t uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed) noexcept;
    #else
        static cmn::blob::line_ptr_t uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed) noexcept;
        static cmn::blob::line_ptr_t uncompress_normal(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed) noexcept;
    #endif
#else
    static cmn::blob::line_ptr_t uncompress(uint16_t start_y, const std::vector<ShortHorizontalLine>& compressed) noexcept;
    using uncompress_normal = uncompress;
#endif
    
public:
    constexpr ShortHorizontalLine() : _x0(0), _x1(0) {}
    
    constexpr ShortHorizontalLine(uint16_t x0, uint16_t x1, bool eol = false)
        : _x0(x0), _x1((x1 & 0x7FFF) | uint16_t(eol << 15))
    {
        assert(x1 < 32768); // MAGIC NUMBERZ (uint16_t - 1 bit)
    }
    
    constexpr uint16_t x0() const { return _x0; }
    constexpr uint16_t x1() const { return _x1 & 0x7FFF; }
    
    //! returns true if this is the last element on the current y coordinate
    //  if true, the following lines are on current_y + 1.
    //  @note stored in the last bit of _x1
    constexpr bool eol() const { return (_x1 & 0x8000) != 0; }
    constexpr void eol(bool v) { _x1 = (_x1 & 0x7FFF) | uint16_t(v << 15); }
};

std::vector<ShortHorizontalLine>
    ShortHorizontalLine::compress(const std::vector<HorizontalLine>& lines)
{
    std::vector<ShortHorizontalLine> ret;
    ret.resize(lines.size());
    
    auto start = lines.data(), end = lines.data() + lines.size();
    auto rptr = ret.data();
    auto prev_y = ret.empty() ? 0 : lines.front().y;
    
    for(auto lptr = start; lptr != end; lptr++, rptr++) {
        *rptr = ShortHorizontalLine(lptr->x0, lptr->x1);
        
        if(prev_y != lptr->y)
            (rptr-1)->eol(true);
        prev_y = lptr->y;
    }
    
    return ret;
}

#ifdef USE_NEON
__attribute__((target("neon"))) // NEON is required for vld1q_u32
__attribute__((noinline))
blob::line_ptr_t ShortHorizontalLine::uncompress(
    uint16_t start_y,
    const std::vector<ShortHorizontalLine>& compressed) noexcept // Assuming compressed is a vector of uint32_t
{
    auto result = std::make_unique<std::vector<HorizontalLine>>();
    result->resize(compressed.size());

    auto y = start_y;
    auto uptr = result->data();
    auto cptr = reinterpret_cast<const uint32_t*>(compressed.data());
    auto end = cptr + compressed.size() - (compressed.size() % 4); // NEON processes 4 elements per loop with 128-bit vectors

    uint32x4_t x_mask = vdupq_n_u32(0x7FFFFFFF);
    uint32x4_t eol_mask = vdupq_n_u32(0x80000000);
    uint32x4_t zeros = vmovq_n_u32(0);

    static_assert(sizeof(ShortHorizontalLine) == 4, "ShortHorizontalLine is not 4 bytes");
    static_assert(sizeof(HorizontalLine) == sizeof(uint64_t), "HorizontalLine is not 8 bytes");

    for (; cptr < end; cptr += 4, uptr += 4) {

        uint32x4_t data_vec = vld1q_u32(cptr);
        //printf("Loaded data_vec: {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u32(data_vec, 0), vgetq_lane_u32(data_vec, 1), vgetq_lane_u32(data_vec, 2), vgetq_lane_u32(data_vec, 3));

        uint32x4_t X = vandq_u32(data_vec, x_mask);
        //printf("X coordinates: {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u32(X, 0), vgetq_lane_u32(X, 1), vgetq_lane_u32(X, 2), vgetq_lane_u32(X, 3));

        // Correcting the Low part (x0) extraction:
        // Narrow the 32-bit values in X to 16-bit values to extract the lower half (x0).
        // This should properly extract the lower 16 bits.
        uint16x4_t low_part = vmovn_u32(X);
        //printf("A: {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(low_part, 0), vget_lane_u16(low_part, 1), vget_lane_u16(low_part, 2), vget_lane_u16(low_part, 3));
        
        // Correcting the High part (x1) extraction:
        // First, shift the data right by 16 bits to position the x1 part in the lower 16 bits.
        // Then, apply a mask to ensure only the lower 15 bits are retained (assuming the 16th bit is not part of x1).
        // Finally, narrow the result to a 16-bit value.
        uint32x4_t high_bits_shifted = vshrq_n_u32(X, 16);
        uint32x4_t high_mask = vdupq_n_u32(0x7FFF); // 15-bit mask for x1.
        uint16x4_t high_part = vmovn_u32(vandq_u32(high_bits_shifted, high_mask));
        //printf("B: {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(high_part, 0), vget_lane_u16(high_part, 1), vget_lane_u16(high_part, 2), vget_lane_u16(high_part, 3));

        uint32x4_t eol_flags = vshrq_n_u32(vandq_u32(data_vec, eol_mask), 31);
        uint16x4_t y_values = vdup_n_u16(y); // Update y_values based on the new y
        coord_t y_increment = vaddvq_u32(eol_flags);// Prepare the y and padding values for all elements

        //printf("EOL flags: {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u32(eol_flags, 0), vgetq_lane_u32(eol_flags, 1), vgetq_lane_u32(eol_flags, 2), vgetq_lane_u32(eol_flags, 3));

        auto accum = [&]<int i>() {
            // Shift-right eol_flags by 1 bit and add the remaining bits to y_values
            // Shift elements to the left and introduce zeros from the right
            eol_flags = vextq_u32(zeros, eol_flags, 3);
            //printf("EOL update: {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u32(eol_flags, 0), vgetq_lane_u32(eol_flags, 1), vgetq_lane_u32(eol_flags, 2), vgetq_lane_u32(eol_flags, 3));

            y_values = vmovn_u32(vaddq_u32(vmovl_u16(y_values), eol_flags));
            //printf("Updated y: {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(y_values, 0), vget_lane_u16(y_values, 1), vget_lane_u16(y_values, 2), vget_lane_u16(y_values, 3));
        };

        LambdaCaller<0, 3>::call(accum);
        //printf("Updated y: {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(y_values, 0), vget_lane_u16(y_values, 1), vget_lane_u16(y_values, 2), vget_lane_u16(y_values, 3));

        uint16x4_t A = low_part;
        uint16x4_t B = high_part;
        uint16x4_t C = y_values;
        uint16x4_t D = vdup_n_u16(0);

        //printf("C: {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(C, 0), vget_lane_u16(C, 1), vget_lane_u16(C, 2), vget_lane_u16(C, 3));

        // Step 1: Combine A and B
        uint16x8_t tempAB = vcombine_u16(A, B); // Combine A and B into a single 8x16 vector
        uint16x4x2_t zippedAB = vzip_u16(vget_low_u16(tempAB), vget_high_u16(tempAB));
        // Now, zippedAB.val[0] contains A0, B0, A1, B1
        // And zippedAB.val[1] contains A2, B2, A3, B3
        //printf("zippedAB.val[0] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(zippedAB.val[0], 0), vget_lane_u16(zippedAB.val[0], 1), vget_lane_u16(zippedAB.val[0], 2), vget_lane_u16(zippedAB.val[0], 3));
        //printf("zippedAB.val[1] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(zippedAB.val[1], 0), vget_lane_u16(zippedAB.val[1], 1), vget_lane_u16(zippedAB.val[1], 2), vget_lane_u16(zippedAB.val[1], 3));

        // Step 2: Combine C and D
        uint16x8_t tempCD = vcombine_u16(C, D); // Combine C and D into a single 8x16 vector
        uint16x4x2_t zippedCD = vzip_u16(vget_low_u16(tempCD), vget_high_u16(tempCD));
        // Now, zippedCD.val[0] contains C0, D0, C1, D1
        // And zippedCD.val[1] contains C2, D2, C3, D3
        //printf("zippedCD.val[0] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(zippedCD.val[0], 0), vget_lane_u16(zippedCD.val[0], 1), vget_lane_u16(zippedCD.val[0], 2), vget_lane_u16(zippedCD.val[0], 3));
        //printf("zippedCD.val[1] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vget_lane_u16(zippedCD.val[1], 0), vget_lane_u16(zippedCD.val[1], 1), vget_lane_u16(zippedCD.val[1], 2), vget_lane_u16(zippedCD.val[1], 3));

        // Step 3: Create v0 and v1
        //uint64x2x2_t v0 = vzipq_u64(vreinterpretq_u64_u16(zippedAB.val[0]), vreinterpretq_u64_u16(zippedCD.val[0]));
        //uint64x2x2_t v1 = vzipq_u64(vreinterpretq_u64_u16(zippedAB.val[1]), vreinterpretq_u64_u16(zippedCD.val[1]));

        uint16x8_t combinedAB0 = vcombine_u16(zippedAB.val[0], zippedAB.val[1]);
        uint16x8_t combinedCD0 = vcombine_u16(zippedCD.val[0], zippedCD.val[1]);

        /*uint16_t combinedAB0_arr[8];
        vst1q_u16(combinedAB0_arr, combinedAB0);
        printf("combinedAB0 =");
        for (int i = 0; i < 8; ++i) {
            printf(" 0x%X", combinedAB0_arr[i]);
        }
        printf("\n");

        uint16_t combinedCD0_arr[8];
        vst1q_u16(combinedCD0_arr, combinedCD0);
        printf("combinedCD0 =");
        for (int i = 0; i < 8; ++i) {
            printf(" 0x%X", combinedCD0_arr[i]);
        }
        printf("\n");*/

        uint32x4x2_t interleaved1 = vzipq_u32(vreinterpretq_u32_u16(combinedAB0), vreinterpretq_u32_u16(combinedCD0));
        uint16x8_t interleaved2 = vreinterpretq_u16_u32(interleaved1.val[0]);
        uint16x8_t interleaved3 = vreinterpretq_u16_u32(interleaved1.val[1]);
        uint16x8x2_t _interleaved;
        _interleaved.val[0] = interleaved2;
        _interleaved.val[1] = interleaved3;
        /*printf("_interleaved[0][0:4] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u16(_interleaved.val[0], 0), vgetq_lane_u16(_interleaved.val[0], 1), vgetq_lane_u16(_interleaved.val[0], 2), vgetq_lane_u16(_interleaved.val[0], 3));
        printf("_interleaved[0][4:8] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u16(_interleaved.val[0], 4), vgetq_lane_u16(_interleaved.val[0], 5), vgetq_lane_u16(_interleaved.val[0], 6), vgetq_lane_u16(_interleaved.val[0], 7));
        printf("_interleaved[1][0:4] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u16(_interleaved.val[1], 0), vgetq_lane_u16(_interleaved.val[1], 1), vgetq_lane_u16(_interleaved.val[1], 2), vgetq_lane_u16(_interleaved.val[1], 3));
        printf("_interleaved[1][4:8] = {0x%X, 0x%X, 0x%X, 0x%X}\n", vgetq_lane_u16(_interleaved.val[1], 4), vgetq_lane_u16(_interleaved.val[1], 5), vgetq_lane_u16(_interleaved.val[1], 6), vgetq_lane_u16(_interleaved.val[1], 7));*/

        // Step 4: Store the interleaved results as 64-bit values in pairs of two
        uint64x2_t interleaved4 = vreinterpretq_u64_u16(_interleaved.val[0]);
        uint64x2_t interleaved5 = vreinterpretq_u64_u16(_interleaved.val[1]);

        vst1q_u64((uint64_t*)uptr, interleaved4);
        vst1q_u64((uint64_t*)(uptr + 2), interleaved5);



        // Store the interleaved results
        /*auto runC = [&]<int i>() {
            HorizontalLine* line = reinterpret_cast<HorizontalLine*>(reinterpret_cast<ptr_safe_t>(uptr) + i * sizeof(HorizontalLine));
            line->x0 = vget_lane_u16(low_part, i);
            line->x1 = vget_lane_u16(high_part, i);
            line->y = vget_lane_u16(y_values, i);
        };

        LambdaCaller<0, 4>::call(runC);*/

        y += y_increment;
        //printf("Updated y: %u\n", y);
    }

    // Process the remaining elements (if any)
    for (auto remaining_end = cptr + (compressed.size() % 4); cptr < remaining_end; ++cptr, ++uptr) {
        std::cout << "Processing remaining element y=" << y << "\n";
        uint32_t data = *cptr;
        uptr->y = y;
        uptr->x0 = uint16_t(data & 0xFFFF);
        uptr->x1 = uint16_t((data >> 16) & 0x7FFF);
        if (data & 0x80000000) y++;
    }

    return result;
}

    // NEON-specific implementation
#elif defined(USE_SSE)

#if defined(_MSC_VER)
void check_cpu_features(bool& use_avx512f) {
    int cpuInfo[4] = {};
    __cpuid(cpuInfo, 0);

    if (cpuInfo[0] >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        use_avx512f = (cpuInfo[1] & (1 << 16)) != 0;  // Check for AVX-512F support
    }
}
#endif

#if !defined(_MSC_VER)
__attribute__((target("avx512f"))) // AVX512F is required for _mm512_setr_epi64
#endif
blob::line_ptr_t ShortHorizontalLine::uncompress(
    uint16_t start_y,
    const std::vector<ShortHorizontalLine>& compressed) noexcept // Assuming compressed is a vector of uint32_t
{
#if defined(_MSC_VER)
    static bool use_avx512f = false;
    static std::once_flag cpu_check_flag;  // Global flag for std::call_once
    std::call_once(cpu_check_flag, check_cpu_features, std::ref(use_avx512f));
    if(not use_avx512f)
        return uncompress_normal(start_y, compressed);
#endif

    auto result = std::make_unique<std::vector<HorizontalLine>>();
    result->resize(compressed.size());

    auto y = start_y;
    auto uptr = result->data();
    auto cptr = reinterpret_cast<const __m256i*>(compressed.data());
    auto end = reinterpret_cast<const __m256i*>(compressed.data() + compressed.size() - compressed.size() % 8u); // Processing 4 elements per loop with 256-bit vectors

    const __m256i x_mask = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256i eol_mask = _mm256_set1_epi32(0x80000000);
    const __m512i zeros = _mm512_setzero_si512();

    static_assert(sizeof(ShortHorizontalLine) == 4, "ShortHorizontalLine is not 4 bytes");
    static_assert(sizeof(HorizontalLine) == sizeof(uint64_t), "HorizontalLine is not 8 bytes");

    for (; cptr < end; ++cptr, uptr += 8) {
        __m256i data_vec = _mm256_loadu_si256(cptr);

        // Extract x0 and x1, interleave with zeros to 512 bits
        __m256i X = _mm256_and_si256(data_vec, x_mask);
        __m512i X_512 = _mm512_zextsi256_si512(X);

        __m512i low_part = _mm512_unpacklo_epi32(X_512, zeros);
        __m512i high_part = _mm512_unpackhi_epi32(X_512, zeros);

        // Now interleave low_part and high_part correctly
        __m512i interleaved = _mm512_permutex2var_epi64(low_part, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), high_part);

        // Extract eol flags
        __m256i eol_flags = _mm256_srli_epi32(_mm256_and_si256(data_vec, eol_mask), 31);
        _mm512_storeu_si512(reinterpret_cast<void*>(uptr), interleaved);

        auto runC = [&]<int i>() {
            uptr[i].y = y;
            y += _mm256_extract_epi32(eol_flags, i % 8);
        };

        LambdaCaller<0, 8>::call(runC);
    }

    // Process the remaining elements (if any)
    auto remaining_ptr = reinterpret_cast<const uint32_t*>(cptr);
    for (auto remaining_end = remaining_ptr + (compressed.size() % 8); remaining_ptr < remaining_end; ++remaining_ptr, ++uptr) {
        uint32_t data = *remaining_ptr;
        uptr->y = y;
        uptr->x0 = uint16_t(data & 0xFFFF);
        uptr->x1 = uint16_t((data >> 16) & 0x7FFF);
        if (data & 0x80000000) y++;
    }

    return result;
}

#endif

#if defined(USE_NEON) || defined(_MSC_VER)
blob::line_ptr_t ShortHorizontalLine::uncompress_normal(uint16_t start_y,
    const std::vector<ShortHorizontalLine>& compressed) noexcept
#else
#if defined(USE_SSE)
__attribute__((target("default"))) 
#endif
blob::line_ptr_t ShortHorizontalLine::uncompress(uint16_t start_y,
    const std::vector<ShortHorizontalLine>& compressed) noexcept
#endif
{
    //std::cout << "fallback" << std::endl;
    auto uncompressed = std::make_unique<std::vector<HorizontalLine>>();
    uncompressed->resize(compressed.size());
    
    auto y = start_y;
    auto uptr = uncompressed->data();
    auto cptr = compressed.data(), end = compressed.data()+compressed.size();
    
    for(; cptr != end; cptr++, uptr++) {
        uptr->y = y;
        uptr->x0 = cptr->x0();
        uptr->x1 = cptr->x1();
        
        if(cptr->eol())
            y++;
    }
    
    return uncompressed;
}

}

using namespace cmn;
std::vector<ShortHorizontalLine> generateTestData(size_t numLines) {
    std::vector<ShortHorizontalLine> data;
    data.reserve(numLines);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> x_dist(0, 32767); // Example range

    for (size_t i = 0; i < numLines; ++i) {
        uint16_t x0 = x_dist(gen);
        uint16_t x1 = x_dist(gen);
        if (x1 < x0) std::swap(x0, x1); // Ensure x1 >= x0

        bool eol = (i % 5 == 0); // Example pattern for eol flag
        data.emplace_back(x0, x1, eol);
    }

    return data;
}

bool checkCorrectness(const std::vector<HorizontalLine>& resultNormal, const std::vector<HorizontalLine>& resultSIMD) {
    if (resultNormal.size() != resultSIMD.size()) {
        std::cerr << "Error: Size mismatch (" << resultNormal.size() << " != " << resultSIMD.size() << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < resultNormal.size(); ++i) {
        if (resultNormal[i].x0 != resultSIMD[i].x0) {
            std::cerr << "Error at index " << i << ": x0 mismatch (" << resultNormal[i].x0 << " != " << resultSIMD[i].x0 << ")" << std::endl;
            return false;
        }
        if (resultNormal[i].x1 != resultSIMD[i].x1) {
            std::cerr << "Error at index " << i << ": x1 mismatch (" << resultNormal[i].x1 << " != " << resultSIMD[i].x1 << ")" << std::endl;
            return false;
        }
        // Add more checks for other fields if necessary
    }
    return true;
}

void run_test() {
    std::vector<ShortHorizontalLine> compressed;

    // Adding ShortHorizontalLine instances with known x0, x1, and eol values
    compressed.emplace_back(7, 7, false);  // x0 = 7, x1 = 7, eol = false
    compressed.emplace_back(8, 9, true);   // x0 = 8, x1 = 9, eol = true
    compressed.emplace_back(10, 11, false); // x0 = 10, x1 = 11, eol = false
    compressed.emplace_back(12, 13, true);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(14, 15, true);  // x0 = 7, x1 = 7, eol = true
    compressed.emplace_back(16, 17, false);   // x0 = 8, x1 = 9, eol = false
    compressed.emplace_back(18, 19, false); // x0 = 10, x1 = 11, eol = false
    compressed.emplace_back(20, 21, true);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(55, 56, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(155, 156, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(255, 256, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(355, 356, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(455, 456, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(555, 556, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(655, 656, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(755, 756, false);  // x0 = 12, x1 = 13, eol = true
    compressed.emplace_back(855, 856, false);  // x0 = 12, x1 = 13, eol = true

    auto result = ShortHorizontalLine::uncompress(0, compressed);

    std::vector<HorizontalLine> expected = {
        {0, 7, 7},   // y = 0, x0 = 7, x1 = 7
        {0, 8, 9},   // y = 0, x0 = 8, x1 = 9
        {1, 10, 11}, // y = 1, x0 = 10, x1 = 11
        {1, 12, 13}, // y = 1, x0 = 12, x1 = 13
        {2, 14, 15},   // y = 2, x0 = 7, x1 = 7
        {3, 16, 17},   // y = 3, x0 = 8, x1 = 9
        {3, 18, 19}, // y = 3, x0 = 10, x1 = 11
        {3, 20, 21}, // y = 3, x0 = 12, x1 = 13
        {4, 55, 56}, // y = 3, x0 = 12, x1 = 13
        {4, 155, 156}, // y = 3, x0 = 12, x1 = 13
        {4, 255, 256}, // y = 3, x0 = 12, x1 = 13
        {4, 355, 356}, // y = 3, x0 = 12, x1 = 13
        {4, 455, 456}, // y = 3, x0 = 12, x1 = 13
        {4, 555, 556}, // y = 3, x0 = 12, x1 = 13
        {4, 655, 656}, // y = 3, x0 = 12, x1 = 13
        {4, 755, 756}, // y = 3, x0 = 12, x1 = 13
        {4, 855, 856}, // y = 3, x0 = 12, x1 = 13
    };

    assert(result->size() == expected.size());
    uint16_t y = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& inputLine = compressed.at(i);

        std::cout << "Comparing index " << i << ":\n";
        std::cout << "  Input - y: "<< y <<" x0: " << inputLine.x0() << ", x1: " << inputLine.x1() << ", eol: " << inputLine.eol() << "\n";
        std::cout << "  Result - y: " << result->at(i).y << ", x0: " << result->at(i).x0 << ", x1: " << result->at(i).x1 << "\n";
        std::cout << "  Expected - y: " << expected[i].y << ", x0: " << expected[i].x0 << ", x1: " << expected[i].x1 << "\n";

        if(result->at(i).y != expected[i].y) std::cerr << "Error at index " << i << ": y mismatch (" << result->at(i).y << " != " << expected[i].y << ")" << std::endl;
        if(result->at(i).x0 != expected[i].x0) std::cerr << "Error at index " << i << ": x0 mismatch (" << result->at(i).x0 << " != " << expected[i].x0 << ")" << std::endl;
        if(result->at(i).x1 != expected[i].x1) std::cerr << "Error at index " << i << ": x1 mismatch (" << result->at(i).x1 << " != " << expected[i].x1 << ")" << std::endl;

        if (inputLine.eol()) y++;
    }
}

void run_test2() {
    auto c = [](uint16_t x0, uint16_t x1) {
        return ShortHorizontalLine(x0, x1 & 0x7FFFFFFF, (x1 & 0x80000000) != 0);
    };


    std::vector<HorizontalLine> expected{
        {0, 953, 962},   // y = 0, x0 = 953, x1 = 962
        {1, 953, 962},   // y = 1, x0 = 953, x1 = 962
        {2, 953, 962},   // y = 2, x0 = 953, x1 = 962
        {3, 953, 961},   // y = 3, x0 = 953, x1 = 961
        {4, 953, 960},   // y = 4, x0 = 953, x1 = 960
        {5, 954, 959},   // y = 5, x0 = 954, x1 = 959
        {6, 954, 958},   // y = 6, x0 = 954, x1 = 958
        {7, 955, 957}    // y = 7, x0 = 955, x1 = 957
    };

    std::vector<ShortHorizontalLine> compressed = ShortHorizontalLine::compress(expected);

#if defined(USE_SSE)
    // Define the function type for uncompress
    using uncompress_func_type = cmn::blob::line_ptr_t (*)(uint16_t, const std::vector<ShortHorizontalLine>&);

    // Take the address of the default version of uncompress
    uncompress_func_type uncompress_default = static_cast<uncompress_func_type>(ShortHorizontalLine::uncompress);

    //auto result = ShortHorizontalLine::uncompress(0, compressed);  // Assuming this function exists
    auto result = uncompress_default(0, compressed);
#else
    auto result = ShortHorizontalLine::uncompress(0, compressed);
#endif


    assert(result->size() == expected.size());
    uint16_t y = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& inputLine = compressed.at(i);

        std::cout << "Comparing index " << i << ":\n";
        std::cout << "  Input - y: "<< y <<" x0: " << inputLine.x0() << ", x1: " << inputLine.x1() << ", eol: " << inputLine.eol() << "\n";
        std::cout << "  Result - y: " << result->at(i).y << ", x0: " << result->at(i).x0 << ", x1: " << result->at(i).x1 << "\n";
        std::cout << "  Expected - y: " << expected[i].y << ", x0: " << expected[i].x0 << ", x1: " << expected[i].x1 << "\n";

        if(result->at(i).y != expected[i].y) std::cerr << "Error at index " << i << ": y mismatch (" << result->at(i).y << " != " << expected[i].y << ")" << std::endl;
        if(result->at(i).x0 != expected[i].x0) std::cerr << "Error at index " << i << ": x0 mismatch (" << result->at(i).x0 << " != " << expected[i].x0 << ")" << std::endl;
        if(result->at(i).x1 != expected[i].x1) std::cerr << "Error at index " << i << ": x1 mismatch (" << result->at(i).x1 << " != " << expected[i].x1 << ")" << std::endl;

        if (inputLine.eol()) y++;
    }
}

int main() {
#ifdef USE_NEON
    std::cout << "Using NEON" << std::endl;
#elif defined(USE_SSE)
    std::cout << "Using SSE" << std::endl;
#else
    std::cout << "Using normal version" << std::endl;
#endif

    auto compressed = generateTestData(100000);

    const int numTests = 1000; // Number of times to repeat each test
    double totalTimeNormal = 0.0;
    double totalTimeSIMD = 0.0;
    std::vector<bool> testOrder(numTests * 2, true); // True for normal, false for SIMD
    std::fill(testOrder.begin() + numTests, testOrder.end(), false);

    // Randomly shuffle the test order
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(testOrder.begin(), testOrder.end(), g);

    for (bool isNormal : testOrder) {
        if (isNormal) {
            // Benchmark normal version
            auto start = std::chrono::high_resolution_clock::now();
            for(size_t i=0; i<100; ++i)
                auto result_normal = ShortHorizontalLine::uncompress_normal(0, compressed);
            auto end = std::chrono::high_resolution_clock::now();
            totalTimeNormal += std::chrono::duration<double>(end - start).count();
        } else {
            // Benchmark SIMD version
            auto start = std::chrono::high_resolution_clock::now();
            for(size_t i=0; i<100; ++i)
                auto result_simd = ShortHorizontalLine::uncompress(0, compressed);
            auto end = std::chrono::high_resolution_clock::now();
            totalTimeSIMD += std::chrono::duration<double>(end - start).count();
        }
    }

    // Calculate and display average timings
    std::cout << "Average time for normal version: " << (totalTimeNormal / numTests) << " seconds" << std::endl;
    std::cout << "Average time for SIMD version: " << (totalTimeSIMD / numTests) << " seconds" << std::endl;

#if defined(USE_SSE)
    // Define the function type for uncompress
    using uncompress_func_type = cmn::blob::line_ptr_t (*)(uint16_t, const std::vector<ShortHorizontalLine>&);

    // Take the address of the default version of uncompress
    uncompress_func_type uncompress_default = static_cast<uncompress_func_type>(ShortHorizontalLine::uncompress);

    // Now you can call the default version of uncompress using the function pointer
    auto result_normal = uncompress_default(0, compressed);
#else
    auto result_normal = ShortHorizontalLine::uncompress_normal(0, compressed);
#endif

    // Perform correctness test
    auto result_simd = ShortHorizontalLine::uncompress(0, compressed);

    if (checkCorrectness(*result_normal, *result_simd)) {
        std::cout << "Correctness test passed!" << std::endl;
    } else {
        std::cerr << "Correctness test failed!" << std::endl;
    }
    run_test();
    run_test2();

    return 0;
}
