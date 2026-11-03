#include <commons.pc.h>
#include <gui/DrawBase.h>
#include <gui/types/StaticText.h>
#include <imgui.h>

// From the repository root, using the existing build configuration:
//   conda run -n trex cmake -S Application -B Application/build
//   conda run -n trex cmake --build Application/build --config Release --target benchmark_find_splitting_point
//   conda run -n trex ./Application/build/benchmarks/benchmark_find_splitting_point
// Optional arguments: calls per sample (1000), samples (5), CSV output path,
// operation (split or add_string; default split).
// Full wrapping comparison:
//   conda run -n trex ./Application/build/benchmarks/benchmark_find_splitting_point 1000 5 add_string_benchmark.csv add_string
// benchmark_find_splitting_point_local.cpp contains the old implementation;
// the linked commons library supplies the new implementation.
// Use Release for both the benchmark and commons. Font loading, input generation,
// width-limit preparation, and result comparisons are outside the timed loop.
// In add_string mode, every call measures and wraps the entire input string.

namespace benchmark_local {
size_t find_splitting_point(std::string_view, cmn::Float2_t, cmn::Float2_t,
                           cmn::gui::Drawable*, const cmn::gui::Font&);
void add_string(cmn::gui::Drawable*, const cmn::gui::StaticText::Settings&,
                cmn::gui::StaticText::RichString&&,
                std::vector<cmn::gui::StaticText::RichString>&, cmn::Vec2&);
}

namespace {
using namespace cmn;
using namespace cmn::gui;
using Clock = std::chrono::steady_clock;
using Split = decltype(&utils::find_splitting_point);

// Volatile dispatch and an observable checksum keep both call paths out of line.
Split volatile implementations[] = {
    &utils::find_splitting_point,
    &benchmark_local::find_splitting_point
};
using AddString = decltype(&StaticText::add_string);
AddString volatile add_implementations[] = {
    &StaticText::add_string,
    &benchmark_local::add_string
};
volatile uint64_t checksum_sink = 0;

struct Input {
    std::string text;
    Float2_t width, max_width;
};

struct Case {
    std::string name;
    size_t bytes;
    int width_percent;
    std::vector<Input> inputs;
};

size_t positive_number(const char* value) {
    const std::string text(value);
    if(text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a positive integer: " + text);
    size_t parsed = 0;
    const auto number = std::stoull(text, &parsed);
    if(!number || parsed != text.size() || number > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Invalid positive integer: " + text);
    return static_cast<size_t>(number);
}

std::vector<Case> make_cases(const Font& font) {
    std::vector<Case> cases;
    const std::array<std::pair<const char*, std::string_view>, 4> patterns{{
        {"words", "The quick brown fox jumps over the lazy dog with a longer description. "},
        {"paths", "C:\\datasets\\tracking/session_2026-10-19/camera_01/frame_000123.csv:"},
        {"punctuation", "alpha-beta_gamma.delta:epsilon,zeta/eta\\theta "},
        {"unbroken", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
    }};
    for(const auto& [name, pattern] : patterns) {
        for(const size_t bytes : {64u, 1000u, 2000u}) {
            for(const int percent : {10, 50, 95, 100}) {
                Case item{name, bytes, percent, {}};
                for(size_t variant = 0; variant < 8; ++variant) {
                    std::string text;
                    text.reserve(bytes);
                    for(size_t i = 0; i < bytes; ++i)
                        text += pattern[(i + variant * 7) % pattern.size()];
                    const auto width = utils::calculate_width(utils::calculate_bounds(text, nullptr, font));
                    item.inputs.push_back({std::move(text), width, width * (Float2_t(percent) / 100_F)});
                }
                cases.push_back(std::move(item));
            }
        }
    }

    for(const auto& [name, text, max_width] : std::vector<std::tuple<const char*, std::string, Float2_t>>{
            {"empty", "", 0_F},
            {"single", "W", 0_F},
            {"zero_width", "one two three", 0_F},
            {"negative_width", "one two three", -10_F},
            {"spaces", "                 ", 20_F},
            {"utf8", "Gr\xc3\xbc\xc3\x9f" "e aus K\xc3\xb6ln: caf\xc3\xa9 / na\xc3\xafve", 80_F}})
    {
        const auto width = utils::calculate_width(utils::calculate_bounds(text, nullptr, font));
        cases.push_back({name, text.size(), -1, {{text, width, max_width}}});
    }
    return cases;
}

double measure(size_t implementation, const Case& item, const Font& font, size_t calls) {
    uint64_t checksum = 0;
    size_t input_index = 0;
    const auto start = Clock::now();
    for(size_t i = 0; i < calls; ++i) {
        const auto& input = item.inputs[input_index];
        checksum += implementations[implementation](input.text, input.width, input.max_width, nullptr, font);
        if(++input_index == item.inputs.size())
            input_index = 0;
    }
    const auto end = Clock::now();
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(end - start).count() / double(calls);
}

double measure_add_string(size_t implementation, const Case& item, const Font& font, size_t calls) {
    uint64_t checksum = 0;
    size_t input_index = 0;
    StaticText::Settings settings;
    settings.margins = Margins{0, 0, 0, 0};
    const auto start = Clock::now();
    for(size_t i = 0; i < calls; ++i) {
        const auto& input = item.inputs[input_index];
        settings.max_size.x = input.max_width;
        std::vector<StaticText::RichString> strings;
        Vec2 offset(0, 0);
        add_implementations[implementation](nullptr, settings,
            StaticText::RichString(input.text, font, Vec2(), White), strings, offset);
        checksum += strings.size();
        if(not strings.empty())
            checksum += strings.back().str.size();
        if(++input_index == item.inputs.size())
            input_index = 0;
    }
    const auto end = Clock::now();
    checksum_sink = checksum;
    return std::chrono::duration<double, std::nano>(end - start).count() / double(calls);
}

int run(size_t calls, size_t samples, const char* output, bool whole_pipeline) {
    // The regular Quicksand font and CalcTextSizeA match IMGUIBase::text_bounds
    // at font.size=0.75 with an unscaled reference. No renderer/window is needed.
    const Font font(0.75_F);
    ImFontAtlas atlas;
    ImFontConfig config;
    config.OversampleH = 5;
    config.OversampleV = 5;
    if(!std::ifstream(TREX_BENCHMARK_FONT).good())
        throw std::runtime_error("Cannot read font: " TREX_BENCHMARK_FONT);
    auto* imfont = atlas.AddFontFromFileTTF(TREX_BENCHMARK_FONT, 24.f, &config);
    if(!imfont || !atlas.Build())
        throw std::runtime_error("Cannot initialize the benchmark font");
    Base::set_default_text_bounds([imfont](std::string_view text, Drawable*, const Font& current, Float2_t max_width, const char** remaining) {
        if(remaining)
            *remaining = text.data();
        if(text.empty())
            return Bounds();
        const auto size = imfont->CalcTextSizeA(32.f * current.size, max_width, -1.f, text.data(), text.data() + text.size(), remaining);
        return Bounds(0, 0, size.x, size.y);
    });
    // Clear the callback before the captured font is destroyed, including on errors.
    struct ResetMetrics {
        ~ResetMetrics() { Base::set_default_text_bounds({}); }
    } reset_metrics;

    const auto cases = make_cases(font);
    size_t mismatches = 0;
    size_t checked = 0;
    for(const auto& item : cases) {
        for(const auto& input : item.inputs) {
            if(whole_pipeline) {
                StaticText::Settings settings;
                settings.margins = Margins{0, 0, 0, 0};
                settings.max_size.x = input.max_width;
                std::array<std::vector<StaticText::RichString>, 2> results;
                for(size_t method = 0; method < 2; ++method) {
                    Vec2 offset(0, 0);
                    add_implementations[method](nullptr, settings,
                        StaticText::RichString(input.text, font, Vec2(), White), results[method], offset);
                }
                ++checked;
                if(not std::ranges::equal(results[0], results[1], [](const auto& a, const auto& b) {
                    return a.str == b.str;
                }))
                    ++mismatches;
                continue;
            }
            const auto library = implementations[0](input.text, input.width, input.max_width, nullptr, font);
            const auto local = implementations[1](input.text, input.width, input.max_width, nullptr, font);
            ++checked;
            if(library > input.text.size() || local > input.text.size()) {
                //if(mismatches < 10)
                //    std::fprintf(stderr, "Mismatch: %s, %zu bytes, max_width %.3f: new commons=%zu old local=%zu\n", item.name.c_str(), input.text.size(), double(input.max_width), library, local);
                ++mismatches;
            } else if(library != local) {
                auto loc = utils::trim(input.text.substr(0, local));
                auto lib = utils::trim(input.text.substr(0, library));
                if(loc != lib) {
                    //if(mismatches < 10)
                        prefixed_print(item.name.c_str(), "Mismatch for max=", input.max_width, " new commons=",(std::string)input.text.substr(0, library), "@", utils::calculate_bounds(input.text.substr(0, library), nullptr, font).width," vs. old local=", (std::string)input.text.substr(0, local),"@",utils::calculate_bounds(input.text.substr(0, local), nullptr, font).width, " original=",input.text);
                    ++mismatches;
                }
            }
        }
    }
    if(whole_pipeline)
        std::printf("Checked %zu inputs: %zu wrapping differences (split positions may differ).\n", checked, mismatches);
    else
        std::printf("Checked %zu inputs: %zu mismatches.\n", checked, mismatches);
    //if(mismatches)
    //    return 2;

    std::ofstream csv(output);
    if(!csv)
        throw std::runtime_error(std::string("Cannot write CSV: ") + output);
    csv << "case,bytes,width_percent,inputs,calls,sample,new_commons_ns,old_local_ns,speedup"
        << (whole_pipeline ? ",operation\n" : "\n");
    csv << std::fixed << std::setprecision(3);
    std::printf("Quicksand / ImGui metrics, warm cache, %zu calls per sample, %zu samples.\n", calls, samples);
    if(whole_pipeline) {
        std::printf("add_string: one call wraps the entire input, starting on an empty line.\n");
        std::printf("Timings include all width measurements, cache handling, splitting, trimming, output-vector allocation/cleanup, dispatch and checksum.\n");
    } else
        std::printf("Timings include splitting, internal text measurements, dispatch and checksum.\n");
    std::printf("%-16s %6s %7s %14s %14s %9s\n", "case", "bytes", "width %", "new commons ns", "old local ns", "speedup");

    const auto measure_method = whole_pipeline ? &measure_add_string : &measure;
    double library_total = 0, local_total = 0;
    for(const auto& item : cases) {
        // Small boundary inputs are checked above but excluded from timing.
        if(item.width_percent < 0)
            continue;
        for(size_t method = 0; method < 2; ++method)
            measure_method(method, item, font, 128);

        std::array<std::vector<double>, 2> times;
        for(size_t sample = 0; sample < samples; ++sample) {
            std::array<double, 2> pair;
            // Alternate which implementation runs first to reduce order bias.
            for(size_t turn = 0; turn < 2; ++turn) {
                const size_t method = (sample + turn) % 2;
                pair[method] = measure_method(method, item, font, calls);
                times[method].push_back(pair[method]);
            }
            csv << item.name << ',' << item.bytes << ',' << item.width_percent << ','
                << item.inputs.size() << ',' << calls << ',' << sample << ','
                << pair[0] << ',' << pair[1] << ',' << pair[1] / pair[0]
                << (whole_pipeline ? ",add_string\n" : "\n");
        }
        std::array<double, 2> medians;
        for(size_t method = 0; method < 2; ++method) {
            auto& sorted = times[method];
            std::sort(sorted.begin(), sorted.end());
            medians[method] = (sorted[(samples - 1) / 2] + sorted[samples / 2]) * 0.5;
        }
        library_total += medians[0];
        local_total += medians[1];
        std::printf("%-16s %6zu %7d %14.1f %14.1f %8.2fx\n", item.name.c_str(), item.bytes,
                    item.width_percent, medians[0], medians[1], medians[1] / medians[0]);
    }
    csv.close();
    if(!csv)
        throw std::runtime_error(std::string("Failed writing CSV: ") + output);
    std::printf("\nSummary (one call per case, sum of medians): new commons %.1f us, old local %.1f us, %.2fx.\n",
                library_total / 1000., local_total / 1000., local_total / library_total);
    std::printf("Speedup = old local/new commons; >1 means new commons is faster.\n");
    std::printf("Synthetic workloads; the summary weights each case equally. Raw samples: %s\n", output);
    std::printf("Checksum: %llu\n", static_cast<unsigned long long>(checksum_sink));
    return 0;
}
}

int main(int argc, char** argv) {
    if(argc > 5 || (argc > 1 && std::string_view(argv[1]) == "--help")) {
        std::printf("Usage: %s [calls_per_sample=1000] [samples=5] [output=find_splitting_point_benchmark.csv] [operation=split|add_string]\n", argv[0]);
        return argc > 5 ? 1 : 0;
    }
    try {
#ifndef NDEBUG
        std::fprintf(stderr, "Warning: benchmark compiled without NDEBUG; build both benchmark and commons in Release.\n");
#endif
        const std::string_view operation = argc > 4 ? argv[4] : "split";
        if(operation != "split" && operation != "add_string")
            throw std::runtime_error("Expected operation split or add_string");
        return run(argc > 1 ? positive_number(argv[1]) : 1000,
                   argc > 2 ? positive_number(argv[2]) : 5,
                   argc > 3 ? argv[3] : "find_splitting_point_benchmark.csv",
                   operation == "add_string");
    } catch(const std::exception& error) {
        std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
        return 1;
    }
}
