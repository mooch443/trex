// Standalone pixel-buffer benchmark; no TRex/OpenCV dependencies or CMake wiring.
// Build from the repository root:
//   c++ -std=c++17 -O3 -DNDEBUG -fno-lto tools/benchmark_image_zero.cpp -o benchmark_image_zero.out
// Run the default size sweep, through 3840x2160 RGB, with at least 200 images/sample:
//   ./benchmark_image_zero.out
// Custom rows, columns, channels, optional iterations per sample and live buffers:
//   ./benchmark_image_zero.out 1080 1920 3 200 4
// Optionally select one footprint percentage and hole percentage instead of the sweep:
//   ./benchmark_image_zero.out 1080 1920 3 200 4 50 30
// Run again in a new process to observe allocator-history variability.
// Each run prints a final summary and replaces image_zero_benchmark.csv in the
// current directory. Plot it with:
//   conda run -n trex python tools/plot_image_zero_benchmark.py

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using Byte = unsigned char;
using Clock = std::chrono::steady_clock;
struct Free { void operator()(Byte* p) const { std::free(p); } };
using Buffer = std::unique_ptr<Byte, Free>;

// Volatile function pointers prevent malloc+fill from being folded into calloc.
// The observer keeps the complete buffer contents observable to the compiler.
void* (*volatile allocate_raw)(std::size_t) = &std::malloc;
void* (*volatile allocate_zero)(std::size_t, std::size_t) = &std::calloc;
void observe_buffer(const Byte*, std::size_t) {}
void (*volatile observe)(const Byte*, std::size_t) = &observe_buffer;
volatile std::uint64_t checksum_sink = 0;

enum Method { Old, Exact, WordCalloc, Calloc, Reuse, MethodCount };
constexpr std::array<const char*, MethodCount> method_names{
    "old malloc+fill", "exact malloc+fill", "word calloc", "exact calloc", "reuse+fill"
};
constexpr std::size_t rounds = 5;

struct Shape {
    std::size_t rows, cols, channels;
    std::size_t bytes() const {
        const auto limit = std::numeric_limits<std::size_t>::max() / 2;
        if (!rows || !cols || !channels || rows > limit / cols
            || rows * cols > limit / channels)
            throw std::runtime_error("Invalid or overflowing image dimensions");
        return rows * cols * channels;
    }
};

struct Result {
    Shape shape;
    std::size_t bytes, iterations, batch;
    unsigned blob_percent, hole_percent;
    std::size_t footprint_pixels, filled_pixels, spans;
    std::array<std::array<double, rounds>, MethodCount> timings;
};

struct Span {
    std::size_t offset, bytes;
};

struct BlobPattern {
    unsigned blob_percent, hole_percent;
    std::size_t footprint_pixels, filled_pixels = 0;
    std::vector<Span> spans;
};

// Percentages describe the rectangular footprint and holes within that footprint.
// Compact holes repeat in 8x8 cells; the outer pixel rim stays connected. Pixel
// rounding changes actual coverage, which is reported separately from targets.
BlobPattern make_blob(Shape shape, unsigned blob_percent, unsigned hole_percent) {
    const auto scale = std::sqrt(blob_percent / 100.0);
    const auto width = std::clamp(static_cast<std::size_t>(shape.cols * scale), std::size_t{1}, shape.cols);
    const auto height = std::clamp(static_cast<std::size_t>(shape.rows * scale), std::size_t{1}, shape.rows);
    const auto x0 = (shape.cols - width) / 2, y0 = (shape.rows - height) / 2;
    BlobPattern blob{blob_percent, hole_percent, width * height};

    std::array<int, 64> cells{};
    for (int i = 0; i < 64; ++i)
        cells[i] = i;
    const auto distance = [](int i) {
        const int x = 2 * (i % 8) - 7, y = 2 * (i / 8) - 7;
        return x * x + y * y;
    };
    std::sort(cells.begin(), cells.end(), [&](int a, int b) {
        return distance(a) != distance(b) ? distance(a) < distance(b) : a < b;
    });
    std::array<bool, 64> holes{};
    for (unsigned i = 0; i < (64 * hole_percent + 50) / 100; ++i)
        holes[cells[i]] = true;

    for (std::size_t y = 0; y < height; ++y) {
        auto start = width;
        for (std::size_t x = 0; x <= width; ++x) {
            const bool filled = x < width && (x == 0 || y == 0 || x + 1 == width || y + 1 == height
                || !holes[(y % 8) * 8 + x % 8]);
            if (filled) {
                if (start == width)
                    start = x;
                ++blob.filled_pixels;
            } else if (start != width) {
                blob.spans.push_back({((y0 + y) * shape.cols + x0 + start) * shape.channels,
                                      (x - start) * shape.channels});
                start = width;
            }
        }
    }
    return blob;
}

template<Method method>
Buffer allocate(std::size_t n) {
    Buffer result;
    if constexpr (method == Calloc)
        result.reset(static_cast<Byte*>(allocate_zero(n, 1)));
    else if constexpr (method == WordCalloc) {
        const auto words = n / sizeof(std::uint32_t) + (n % sizeof(std::uint32_t) != 0);
        result.reset(static_cast<Byte*>(allocate_zero(words, sizeof(std::uint32_t))));
    } else
        result.reset(static_cast<Byte*>(allocate_raw(n)));
    if (!result)
        throw std::bad_alloc();
    return result;
}

template<Method method>
std::size_t capacity_for(Shape shape, std::size_t n) {
    if constexpr (method == Old)
        if (shape.rows < 128 && shape.cols < 128)
            return 2 * n;
    return n;
}

template<Method method>
void initialize(Buffer& buffer, std::size_t n, std::size_t capacity) {
    if constexpr (method != Reuse)
        buffer = allocate<method>(capacity);
    if constexpr (method != Calloc && method != WordCalloc)
        std::fill_n(buffer.get(), n, Byte{0});
}

// Row spans are precomputed outside timing, like an existing blob's horizontal lines.
void paint_blob(Byte* data, const BlobPattern& blob) {
    for (const auto& span : blob.spans)
        std::fill_n(data + span.offset, span.bytes, Byte{255});
}

std::uint64_t checksum(const Byte* data, std::size_t n) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i)
        sum += data[i];
    return sum;
}

template<Method method>
void validate(Shape shape, std::size_t n, const BlobPattern& blob) {
    Buffer buffer;
    if constexpr (method == Reuse) {
        buffer = allocate<Exact>(n);
        std::fill_n(buffer.get(), n, Byte{165});
    }
    initialize<method>(buffer, n, capacity_for<method>(shape, n));
    if (!std::all_of(buffer.get(), buffer.get() + n, [](Byte b) { return b == 0; }))
        throw std::runtime_error("Zero initialization check failed");
    paint_blob(buffer.get(), blob);
    const auto expected = std::uint64_t(blob.filled_pixels) * shape.channels * 255;
    if (checksum(buffer.get(), n) != expected)
        throw std::runtime_error("Blob checksum check failed");
}

template<Method method>
double measure(Shape shape, std::size_t n, const BlobPattern& blob,
               std::size_t passes, std::size_t batch) {
    std::vector<Buffer> buffers(batch);
    const auto capacity = capacity_for<method>(shape, n);
    if constexpr (method == Reuse) {
        for (auto& buffer : buffers) {
            buffer = allocate<Exact>(n);
            std::fill_n(buffer.get(), n, Byte{165});
            observe(buffer.get(), n);
        }
    }

    const auto start = Clock::now();
    for (std::size_t pass = 0; pass < passes; ++pass) {
        for (auto& buffer : buffers) {
            initialize<method>(buffer, n, capacity);
            paint_blob(buffer.get(), blob);
            checksum_sink = checksum(buffer.get(), n);
            observe(buffer.get(), n);
        }
        if constexpr (method != Reuse)
            for (auto& buffer : buffers)
                buffer.reset();
    }
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count()
        / static_cast<double>(passes) / static_cast<double>(batch);
}

void benchmark(Shape shape, std::size_t requested_iterations, std::size_t batch,
               const std::vector<unsigned>& blob_percentages, const std::vector<unsigned>& hole_percentages,
               std::vector<Result>& results) {
    const auto n = shape.bytes();
    const auto iterations = requested_iterations ? requested_iterations
        : std::max({batch, std::size_t{200},
                    std::min(std::size_t{1000000}, std::size_t{256 * 1024 * 1024} / n)});
    const auto passes = 1 + (iterations - 1) / batch;
    if (passes > std::numeric_limits<std::size_t>::max() / batch)
        throw std::runtime_error("Iteration count overflow");
    std::cout << '\n' << shape.rows << 'x' << shape.cols << 'x' << shape.channels
              << " (" << n << " active bytes), " << passes * batch
              << " images/sample, " << batch << " live buffers\n";
    using ValidateFunction = void (*)(Shape, std::size_t, const BlobPattern&);
    using MeasureFunction = double (*)(Shape, std::size_t, const BlobPattern&, std::size_t, std::size_t);
    const std::array<ValidateFunction, MethodCount> validators{
        validate<Old>, validate<Exact>, validate<WordCalloc>, validate<Calloc>, validate<Reuse>
    };
    const std::array<MeasureFunction, MethodCount> measurements{
        measure<Old>, measure<Exact>, measure<WordCalloc>, measure<Calloc>, measure<Reuse>
    };

    for (const auto blob_percent : blob_percentages)
      for (const auto hole_percent : hole_percentages) {
        const auto blob = make_blob(shape, blob_percent, hole_percent);
        for (const auto validator : validators)
            validator(shape, n, blob);
        std::array<std::array<double, rounds>, MethodCount> timings{};
        for (int m = 0; m < MethodCount; ++m)
            measurements[m](shape, n, blob, 1, batch);
        for (std::size_t round = 0; round < rounds; ++round) {
            for (int offset = 0; offset < MethodCount; ++offset) {
                const auto m = (round + static_cast<std::size_t>(offset)) % MethodCount;
                timings[m][round] = measurements[m](shape, n, blob, passes, batch);
            }
        }
        for (auto& samples : timings)
            std::sort(samples.begin(), samples.end());
        results.push_back({shape, n, passes * batch, batch, blob_percent, hole_percent,
                           blob.footprint_pixels, blob.filled_pixels, blob.spans.size(), timings});
        std::cout << "  Footprint target " << blob_percent << "% of image, holes target " << hole_percent
                  << "% of footprint; actual filled " << std::fixed << std::setprecision(2)
                  << 100.0 * blob.filled_pixels / (shape.rows * shape.cols) << "% of image, holes "
                  << 100.0 * (blob.footprint_pixels - blob.filled_pixels) / blob.footprint_pixels
                  << "% of footprint, " << blob.spans.size() << " row spans\n"
                  << "    method                   median          min          max    old/median\n";
        for (int m = 0; m < MethodCount; ++m) {
            const auto median = timings[m][rounds / 2];
            std::cout << "    " << std::left << std::setw(22) << method_names[m]
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(11) << median << std::setw(13) << timings[m].front()
                      << std::setw(13) << timings[m].back() << std::setw(13)
                      << (median > 0 ? timings[Old][rounds / 2] / median : 0) << "x\n";
        }
      }
}

void summarize(const std::vector<Result>& results) {
    std::cout << "\nFINAL SUMMARY: byte fill, byte calloc and word calloc\n"
              << "Saved time > 0 means faster than old; exact/calloc > 1 favors calloc.\n"
              << "Byte/word > 1 favors word calloc over byte calloc.\n";
    std::cout << "Footprint and holes are targets; filled is the actual image percentage after holes.\n"
              << "    rows x cols x channels  footprint%  holes%  filled%      old us    calloc us     saved us   exact/calloc     word us     byte/word\n";
    for (const auto& result : results) {
        const auto old = result.timings[Old][rounds / 2];
        const auto exact = result.timings[Exact][rounds / 2];
        const auto zero = result.timings[Calloc][rounds / 2];
        const auto word = result.timings[WordCalloc][rounds / 2];
        const auto label = std::to_string(result.shape.rows) + "x"
            + std::to_string(result.shape.cols) + "x" + std::to_string(result.shape.channels);
        std::cout << "    " << std::left << std::setw(23) << label << std::right
                  << std::setw(10) << result.blob_percent << std::setw(8) << result.hole_percent
                  << std::fixed << std::setprecision(2) << std::setw(9)
                  << 100.0 * result.filled_pixels / (result.shape.rows * result.shape.cols)
                  << std::setprecision(3) << std::setw(12) << old
                  << std::setw(13) << zero << std::setw(13) << old - zero;
        if (zero > 0)
            std::cout << std::setw(14) << exact / zero << "x";
        else
            std::cout << std::setw(15) << "n/a";
        std::cout << std::setw(12) << word;
        if (word > 0)
            std::cout << std::setw(13) << zero / word << "x\n";
        else
            std::cout << std::setw(14) << "n/a\n";
    }
    std::cout << "\nAll times are median microseconds/image. Small differences may be noise;\n"
              << "inspect the min/max ranges and repeat the run before drawing conclusions.\n";
}

void save_results(const std::vector<Result>& results) {
    constexpr auto path = "image_zero_benchmark.csv";
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot open image_zero_benchmark.csv for writing");
    output << "rows,cols,channels,bytes,iterations,batch,samples,blob_percent,hole_percent,footprint_pixels,filled_pixels,spans,method,median_us,min_us,max_us\n"
           << std::setprecision(17);
    for (const auto& result : results) {
        for (int m = 0; m < MethodCount; ++m) {
            const auto& samples = result.timings[m];
            output << result.shape.rows << ',' << result.shape.cols << ',' << result.shape.channels
                   << ',' << result.bytes << ',' << result.iterations << ',' << result.batch
                   << ',' << rounds << ',' << result.blob_percent << ',' << result.hole_percent
                   << ',' << result.footprint_pixels << ',' << result.filled_pixels << ',' << result.spans
                   << ',' << method_names[m]
                   << ',' << samples[rounds / 2] << ',' << samples.front() << ',' << samples.back() << '\n';
        }
    }
    output.close();
    if (!output)
        throw std::runtime_error("Failed to write image_zero_benchmark.csv");
    std::cout << "\nSaved " << path << " (replaced on each run). Generate PNG and SVG charts with:\n"
              << "  conda run -n trex python tools/plot_image_zero_benchmark.py\n";
}

std::size_t number(const char* argument, std::size_t minimum = 1,
                   std::size_t maximum = std::numeric_limits<std::size_t>::max()) {
    const std::string text(argument);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Arguments must be nonnegative integers");
    const auto value = std::stoull(text);
    if (value < minimum || value > maximum)
        throw std::runtime_error("Argument outside its allowed range");
    return static_cast<std::size_t>(value);
}

int main(int argc, char** argv) {
    try {
        if (argc != 1 && (argc < 4 || argc > 8)) {
            std::cerr << "Usage: " << argv[0]
                      << " [rows cols channels [iterations [batch [blob_percent [hole_percent]]]]]\n";
            return 1;
        }
        std::cout << "Pixel storage only: excludes Image object, timestamps, OpenCV and GUI.\n"
                  << "Old: malloc(2*N) below 128 in both dimensions, otherwise malloc(N); fill N.\n"
                  << "Exact: malloc(N) + fill N. Calloc: calloc(N, 1). Reuse: fill existing N.\n"
                  << "Word calloc: calloc(ceil(N / sizeof(uint32_t)), sizeof(uint32_t)); rounds allocation up to a whole word.\n"
                  << "Allocation/free included except for reuse; opaque observer overhead included.\n"
                  << "Five warmed samples with rotating method order; this is not a cold-OS-page test.\n"
                  << "Default images/sample: 256 MiB / image bytes, at least 200, capped at 1000000, rounded up to a batch.\n"
                  << "Every measurement: initialize, draw blob row spans, checksum every byte, release.\n"
                  << "Checksum arithmetic is timed. Shape generation and method selection are outside timing.\n"
                  << "Default footprint targets: 5/25/50/100% of image; hole targets: 0/5/10/20/30/50/75/90% of footprint.\n"
                  << "Holes are compact gaps in repeating 8x8 cells, with the outer pixel rim retained.\n"
                  << "Pixel rounding changes coverage; actual filled and hole percentages are reported.\n"
                  << "This excludes actual blob thresholding, color conversion and other application work.\n"
                  << "Ratios > 1 mean faster than old. Inspect absolute times and min/max too.\n";
        std::vector<Result> results;
        std::vector<unsigned> blob_percentages{5, 25, 50, 100}, hole_percentages{0, 5, 10, 20, 30, 50, 75, 90};
        if (argc >= 7)
            blob_percentages = {static_cast<unsigned>(number(argv[6], 1, 100))};
        if (argc >= 8)
            hole_percentages = {static_cast<unsigned>(number(argv[7], 0, 100))};
        if (argc == 1) {
            for (Shape shape : std::array<Shape, 7>{{
                {64, 64, 1}, {127, 127, 3}, {128, 128, 3}, {480, 640, 1}, {1080, 1920, 3},
                {1440, 2560, 3}, {2160, 3840, 3}
            }})
                benchmark(shape, 0, 4, blob_percentages, hole_percentages, results);
        } else {
            benchmark({number(argv[1]), number(argv[2]), number(argv[3])},
                      argc >= 5 ? number(argv[4]) : 0, argc >= 6 ? number(argv[5]) : 4,
                      blob_percentages, hole_percentages, results);
        }
        std::cout << "\nChecksum sink: " << checksum_sink << '\n';
        summarize(results);
        save_results(results);
    } catch (const std::exception& error) {
        std::cerr << "Benchmark stopped: " << error.what() << '\n';
        return 1;
    }
}
