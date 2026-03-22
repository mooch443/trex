#include <commons.pc.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
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

glz::json_t make_write_json_object() {
    glz::json_t nested;
    nested["field"] = 0;

    glz::json_t json;
    json["number"] = 0;
    json["obj"] = std::move(nested);
    return json;
}

sprite::Map make_write_map_object() {
    sprite::Map map;
    map["number"] = double(0);

    glz::json_t nested;
    nested["field"] = double(0);
    map["obj"] = std::move(nested);
    return map;
}

struct PatternCase {
    std::string_view pattern;
    std::string_view expected;
};

struct CreateInputs {
    double number;
    double nested;
    bool enabled;
    std::string_view name;
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

template<typename Object, typename Writer>
double benchmark_write_backend(std::string_view backend_name,
                               std::string_view field_name,
                               Object object,
                               Writer&& writer,
                               const std::vector<int>& values,
                               size_t warmup_iterations,
                               size_t timed_iterations,
                               volatile std::uint64_t& sink)
{
    for(size_t i = 0; i < warmup_iterations; ++i) {
        sink += static_cast<std::uint64_t>(writer(object, values[i]));
    }

    const auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < timed_iterations; ++i) {
        sink += static_cast<std::uint64_t>(writer(object, values[warmup_iterations + i]));
    }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const auto ns_per_op = static_cast<double>(total_ns) / static_cast<double>(timed_iterations);

    std::cout << std::left
              << std::setw(12) << "write"
              << std::setw(12) << backend_name
              << std::setw(24) << field_name
              << std::right << std::fixed << std::setprecision(2)
              << ns_per_op << '\n';

    return ns_per_op;
}

template<typename Object, typename Builder, typename Reader>
double benchmark_create_backend(std::string_view backend_name,
                                std::string_view object_name,
                                Builder&& builder,
                                Reader&& reader,
                                const std::vector<CreateInputs>& inputs,
                                size_t warmup_iterations,
                                size_t timed_iterations,
                                volatile std::uint64_t& sink)
{
    for(size_t i = 0; i < warmup_iterations; ++i) {
        auto created = builder(inputs[i]);
        Object copied = created;
        sink += static_cast<std::uint64_t>(reader(copied));
    }

    const auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < timed_iterations; ++i) {
        auto created = builder(inputs[warmup_iterations + i]);
        Object copied = created;
        sink += static_cast<std::uint64_t>(reader(copied));
    }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const auto ns_per_op = static_cast<double>(total_ns) / static_cast<double>(timed_iterations);

    std::cout << std::left
              << std::setw(12) << "create"
              << std::setw(12) << backend_name
              << std::setw(24) << object_name
              << std::right << std::fixed << std::setprecision(2)
              << ns_per_op << '\n';

    return ns_per_op;
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
        constexpr size_t timed_iterations = 500000;
        const size_t total_write_values = warmup_iterations + timed_iterations;
        volatile std::uint64_t sink = 0;

        std::vector<int> write_values;
        write_values.reserve(total_write_values);
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> dist(0, 1'000'000);
        for(size_t i = 0; i < total_write_values; ++i) {
            write_values.push_back(dist(rng));
        }

        constexpr std::string_view names[] = {
            "trex",
            "raptor",
            "dino",
            "tracker"
        };
        std::vector<CreateInputs> create_values;
        create_values.reserve(total_write_values);
        for(size_t i = 0; i < total_write_values; ++i) {
            create_values.push_back(CreateInputs{
                .number = static_cast<double>(write_values[i]),
                .nested = static_cast<double>(write_values[total_write_values - 1 - i]),
                .enabled = (i % 2) == 0,
                .name = names[i % std::size(names)]
            });
        }
        
        std::cout << '\n'
                  << std::left
                  << std::setw(12) << "mode"
                  << std::setw(12) << "backend"
                  << std::setw(24) << "field"
                  << "ns/op\n";

        benchmark_write_backend(
            "json",
            "[\"number\"]",
            make_write_json_object(),
            [](glz::json_t& object, double value) -> double {
                object["number"] = value;
                return object["number"].get_number();
            },
            write_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_write_backend(
            "json",
            "[\"obj\"][\"field\"]",
            make_write_json_object(),
            [](glz::json_t& object, double value) -> double {
                object["obj"]["field"] = value;
                return object["obj"]["field"].get_number();
            },
            write_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_write_backend(
            "map",
            "[\"number\"]",
            make_write_map_object(),
            [](sprite::Map& object, double value) -> double {
                object["number"] = value;
                return object["number"].value<double>();
            },
            write_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_write_backend(
            "map",
            "[\"obj\"][\"field\"]",
            make_write_map_object(),
            [](sprite::Map& object, double value) -> double {
                auto nested = object["obj"].value<glz::json_t>();
                nested["field"] = value;
                object["obj"] = nested;
                return nested["field"].get_number();
            },
            write_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_create_backend<glz::json_t>(
            "json",
            "cold_object",
            [](const CreateInputs& input) -> glz::json_t {
                glz::json_t nested;
                nested["field"] = input.nested;

                glz::json_t object;
                object["number"] = input.number;
                object["name"] = std::string(input.name);
                object["enabled"] = input.enabled;
                object["obj"] = std::move(nested);
                return object;
            },
            [](const glz::json_t& object) -> double {
                return object["number"].get_number()
                    + object["obj"]["field"].get_number()
                    + static_cast<double>(object["enabled"].get<bool>())
                    + static_cast<double>(object["name"].get<std::string>().size());
            },
            create_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_create_backend<sprite::Map>(
            "map",
            "cold_object",
            [](const CreateInputs& input) -> sprite::Map {
                sprite::Map object;
                object["number"] = input.number;
                object["name"] = std::string(input.name);
                object["enabled"] = input.enabled;

                glz::json_t nested;
                nested["field"] = input.nested;
                object["obj"] = std::move(nested);
                return object;
            },
            [](const sprite::Map& object) -> double {
                return object.at("number").value<double>()
                    + object.at("obj").value<glz::json_t>()["field"].get_number()
                    + static_cast<double>(object.at("enabled").value<bool>())
                    + static_cast<double>(object.at("name").value<std::string>().size());
            },
            create_values,
            warmup_iterations,
            timed_iterations,
            sink);

        std::cout << std::left
                  << std::setw(12) << "parser"
                  << std::setw(12) << "backend"
                  << std::setw(24) << "pattern"
                  << "ns/op\n";

        const auto make_json_backend = [&json]() {
            State state;
            Context context{
                VarFunc("object", [&json](const VarProps&) -> const glz::json_t& { return json; })
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
