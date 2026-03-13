#include <commons.pc.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gui/DynamicGUI.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/UnresolvedStringPattern.h>

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

namespace {

struct NestedValue {
    int value;
};

struct BenchmarkObject {
    int x;
    std::string name;
    bool enabled;
    NestedValue value;
};

struct ParseTextTag {};
struct ResolveTag {};

template<typename Tag>
struct BenchmarkRunner;

template<>
struct BenchmarkRunner<ParseTextTag> {
    explicit BenchmarkRunner(std::string_view pattern)
        : pattern(pattern)
    {}

    std::string operator()(Context& ctx, State& st) const {
        return parse_text(pattern, ctx, st);
    }

    std::string pattern;
};

template<>
struct BenchmarkRunner<ResolveTag> {
    explicit BenchmarkRunner(std::string_view pattern)
        : prepared(cmn::pattern::UnresolvedStringPattern::prepare(pattern))
    {}

    std::string operator()(Context& ctx, State& st) {
        return prepared.realize(ctx, st);
    }

    cmn::pattern::UnresolvedStringPattern prepared;
};

glz::json_t to_json(const BenchmarkObject& object) {
    glz::json_t nested;
    nested["value"] = object.value.value;

    glz::json_t json;
    json["x"] = object.x;
    json["name"] = object.name;
    json["enabled"] = object.enabled;
    json["value"] = std::move(nested);
    return json;
}

sprite::Map to_map(const BenchmarkObject& object) {
    sprite::Map map;
    map["x"] = object.x;
    map["name"] = object.name;
    map["enabled"] = object.enabled;

    glz::json_t nested;
    nested["value"] = object.value.value;
    map["value"] = std::move(nested);
    return map;
}

struct PatternCase {
    std::string_view pattern;
    std::string_view expected;
};

template<typename Tag, typename Factory>
double benchmark_backend(std::string_view parser_name,
                         std::string_view backend_name,
                         const PatternCase& pattern_case,
                         Factory&& factory,
                         size_t warmup_iterations,
                         size_t timed_iterations,
                         volatile std::uint64_t& sink)
{
    auto [context, state] = factory();
    BenchmarkRunner<Tag> runner(pattern_case.pattern);

    auto actual = runner(context, state);
    if(actual != pattern_case.expected) {
        throw std::runtime_error(
            "Unexpected parser output for " + std::string(parser_name)
            + " / " + std::string(backend_name)
            + " / " + std::string(pattern_case.pattern)
            + ": got '" + actual + "', expected '" + std::string(pattern_case.expected) + "'.");
    }

    for(size_t i = 0; i < warmup_iterations; ++i) {
        auto value = runner(context, state);
        sink += static_cast<std::uint64_t>(value.size());
    }

    const auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < timed_iterations; ++i) {
        auto value = runner(context, state);
        sink += static_cast<std::uint64_t>(value.size());
    }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(total_ns) / static_cast<double>(timed_iterations);
}

template<typename Tag, typename Factory>
void run_backend_suite(std::string_view parser_name,
                       std::string_view backend_name,
                       const std::vector<PatternCase>& patterns,
                       Factory&& factory,
                       size_t warmup_iterations,
                       size_t timed_iterations,
                       volatile std::uint64_t& sink)
{
    for(const auto& pattern_case : patterns) {
        const auto ns_per_op = benchmark_backend<Tag>(
            parser_name,
            backend_name,
            pattern_case,
            factory,
            warmup_iterations,
            timed_iterations,
            sink);

        std::cout << std::left
                  << std::setw(12) << parser_name
                  << std::setw(12) << backend_name
                  << std::setw(24) << pattern_case.pattern
                  << std::right << std::fixed << std::setprecision(2)
                  << ns_per_op << '\n';
    }
}

} // namespace

int main() {
    try {
        const BenchmarkObject sample{
            .x = 42,
            .name = "trex",
            .enabled = true,
            .value = {.value = 123}
        };

        const glz::json_t json = to_json(sample);
        sprite::Map map = to_map(sample);

        const std::vector<PatternCase> patterns{
            {"{object.x}", "42"},
            {"{object.name}", "trex"},
            {"{object.enabled}", "true"},
            {"{object.value.value}", "123"}
        };

        constexpr size_t warmup_iterations = 5000;
        constexpr size_t timed_iterations = 200000;
        volatile std::uint64_t sink = 0;

        std::cout << std::left
                  << std::setw(12) << "parser"
                  << std::setw(12) << "backend"
                  << std::setw(24) << "pattern"
                  << "ns/op\n";

        const auto make_json_backend = [&json]() {
            State state;
            Context context{
                VarFunc("object", [&json](const VarProps&) -> glz::json_t { return json; })
            };
            return std::pair{std::move(context), std::move(state)};
        };

        const auto make_map_backend = [&map]() {
            State state;
            Context context{
                VarFunc("object", [&map](const VarProps&) -> sprite::Map& { return map; })
            };
            return std::pair{std::move(context), std::move(state)};
        };

        run_backend_suite<ParseTextTag>("parse_text", "json", patterns, make_json_backend, warmup_iterations, timed_iterations, sink);
        run_backend_suite<ParseTextTag>("parse_text", "map", patterns, make_map_backend, warmup_iterations, timed_iterations, sink);
        run_backend_suite<ResolveTag>("resolve", "json", patterns, make_json_backend, warmup_iterations, timed_iterations, sink);
        run_backend_suite<ResolveTag>("resolve", "map", patterns, make_map_backend, warmup_iterations, timed_iterations, sink);

        std::cerr << "sink=" << static_cast<std::uint64_t>(sink) << '\n';
        return 0;
    } catch(const std::exception& exception) {
        std::cerr << "Benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}
