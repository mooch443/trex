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
#include <core/idx_t.h>
#include "parser_benchmark/LegacyParsers.h"
#include "parser_benchmark/StringDecoding.h"

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
struct HeadParseTextTag { static constexpr auto parse = cmn::benchmark_head::dyn::parse_text_snapshot; };
struct OlderParseTextTag { static constexpr auto parse = cmn::benchmark_older::dyn::parse_text_snapshot; };
template<typename Prepared> struct HistoricalResolveTag {};
template<typename Prepared> struct PrepareAndResolveTag {};

template<typename Tag>
struct BenchmarkRunner {
    explicit BenchmarkRunner(std::string_view pattern) : pattern(pattern) {}
    std::string operator()(Context& ctx, State& st) const { return Tag::parse(pattern, ctx, st); }
    std::string pattern;
};

template<typename Prepared>
struct BenchmarkRunner<HistoricalResolveTag<Prepared>> {
    explicit BenchmarkRunner(std::string_view pattern) : prepared(Prepared::prepare(pattern)) {}
    std::string operator()(Context& ctx, State& st) { return prepared.realize(ctx, st); }
    Prepared prepared;
};

template<typename Prepared>
struct BenchmarkRunner<PrepareAndResolveTag<Prepared>> {
    using PreparedType = Prepared;
    explicit BenchmarkRunner(std::string_view pattern) : pattern(pattern) {}
    std::string operator()(Context& ctx, State& st) const {
        return Prepared::prepare(pattern).realize(ctx, st);
    }
    std::string pattern;
};

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

glz::generic to_json(const BenchmarkObject& object) {
    glz::generic nested;
    nested["value"] = object.value.value;

    glz::generic json;
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

    glz::generic nested;
    nested["value"] = object.value.value;
    map["value"] = std::move(nested);
    return map;
}

glz::generic make_write_json_object() {
    glz::generic nested;
    nested["field"] = 0;

    glz::generic json;
    json["number"] = 0;
    json["obj"] = std::move(nested);
    return json;
}

sprite::Map make_write_map_object() {
    sprite::Map map;
    map["number"] = double(0);

    glz::generic nested;
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

struct ParserBenchmarkOptions {
    bool current_only = false;
    std::vector<std::string> cases;
    std::vector<std::string> parsers;
    size_t iterations = 10000;
};

template<typename Tag>
void benchmark_parse_fn(const std::string& label,
                       std::string_view group,
                       const std::vector<std::tuple<std::string, std::string, std::string>>& cases,
                       Context& context, State& state, size_t samples = 10000,
                       std::string_view version = "current",
                       const ParserBenchmarkOptions& options = {})
{
    if(not options.parsers.empty()
       && std::find(options.parsers.begin(), options.parsers.end(), label) == options.parsers.end())
        return;
    const auto handler = state._current_object_handler.lock();
    for(const auto& [desc, pattern_str, expected] : cases) {
        if(not options.cases.empty()
           && std::find(options.cases.begin(), options.cases.end(), desc) == options.cases.end())
            continue;
        BenchmarkRunner<Tag> runner(pattern_str);
        handler->clear_variable_values();
        auto out = runner(context, state);
        if(out != expected) {
            throw std::runtime_error("Unexpected output for " + label + " / " + desc
                                     + ": got '" + out + "', expected '" + expected + "'.");
        }

        for(size_t i = 0; i < 100; ++i) {
            handler->clear_variable_values();
            out = runner(context, state);
        }

        std::uint64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for(size_t i = 0; i < samples; ++i) {
            handler->clear_variable_values();
            out = runner(context, state);
            checksum += out.size();
        }
        const auto end = std::chrono::steady_clock::now();
        const double ns_per_parse = std::chrono::duration<double, std::nano>(end - start).count()
            / static_cast<double>(samples);
        size_t prepared_parts = 0;
        if constexpr(requires { runner.prepared; })
            prepared_parts = runner.prepared.objects.size();
        else if constexpr(requires { typename BenchmarkRunner<Tag>::PreparedType; }) {
            using Prepared = typename BenchmarkRunner<Tag>::PreparedType;
            prepared_parts = Prepared::prepare(pattern_str).objects.size();
        }
        std::cout << version << ',' << label << ',' << group << ',' << desc << ',' << std::fixed << std::setprecision(2)
                  << ns_per_parse << ',' << samples << ',' << checksum << ',' << prepared_parts << '\n';
    }
}

void CompareParsers(const ParserBenchmarkOptions& options) {
    const std::string quoted_array = "[" + utils::repeat("'a,b',", 100) + "'c']";
    const std::string nested_array = "[['a,b','c'],[1,[2,[3,4]]]]";
    const std::vector<std::tuple<std::string, std::string, std::string>> cases = {
        {"plain", "hello world", "hello world"},
        {"single_char", "x", "x"},
        {"double_quotes_as_data", R"("hello")", R"("hello")"},
        {"var", "hello {foo}", "hello bar"},
        {"if", "{if:{cond}:'yes':'no'}", "yes"},
        {"nested", "hi {if:{cond}:nested {foo}:'no'}", "hi nested bar"},
        {"deep_nested", "start {if:{cond1}:{if:{cond2}:deep {foo}:'no'}:'fail'} end", "start no end"},
        {"many_if_5", "{if:{cond}:{if:{cond}:{if:{cond}:{if:{cond}:{if:{cond}:'deep':'no'}:'no'}:'no'}:'no'}:'no'}", "deep"},
        {"long_repeat", std::string(200, 'x') + "{foo}" + std::string(200, 'x') + "{foo}" + std::string(200, 'x'),
                        std::string(200, 'x') + "bar" + std::string(200, 'x') + "bar" + std::string(200, 'x')},
        {"long_flat", std::string(1000, 'x'), std::string(1000, 'x')},
        {"long_repeat_variable", std::string(200, 'x') + utils::repeat("{foo}x",100) + std::string(200, 'x') + "{foo}" + std::string(200, 'x'),
                                 std::string(200, 'x') + utils::repeat("barx",100) + std::string(200, 'x') + "bar" + std::string(200, 'x')},
        {"realistic", std::string("{if:{not:{has_pred}}:{name}:{if:{equal:{at:0:{max_pred}}:{id}}:<green>{name}</green>:<red>{name}</red> <i>loc</i>[<c><nr>{at:0:{max_pred}}</nr>:<nr>{int:{*:100:{at:1:{max_pred}}}}</nr><i>%</i></c>]}}{if:{tag}: <a>tag{colon}{tag.id} ({dec:2:{tag.p}})</a>:''}{if:{average_category}: <nr>{average_category}</nr>:''}{if:{&&:{category}:{not:{equal:{category}:{average_category}}}}: <b><i>{category}</i></b>:''}"),
                      "<green>fish0</green> <a>tag:0 (0)</a> <nr>male</nr>"},
        {"quoted_array", quoted_array, quoted_array},
        {"nested_array", nested_array, nested_array},
        {"for_array", "{for:i:[1,2,3]:{i}}", "[1,2,3]"}
    };
    const std::vector<std::tuple<std::string, std::string, std::string>> changed_cases = {
        {"quoted_expression", "prefix '{foo}' suffix",
            "prefix {foo} suffix"},
        {"quoted_if", "{if:{cond}:'prefix {foo}':no}",
            "prefix {foo}"},
        {"quoted_long", "'" + utils::repeat("{foo}x", 100) + "'",
            utils::repeat("{foo}x", 100)},
        {"quoted_collection_expression", "['{foo}','other']",
            "['{foo}','other']"},
        {"single_quoted_double_quotes", R"('say "hello"')", R"(say "hello")"},
        {"unquoted_backslash", R"(\q)", R"(\q)"},
        {"quoted_escaped_newline", R"('line\nnext')", "line\nnext"},
        {"quoted_escaped_apostrophe", R"('can\'t')", "can't"},
        {"quoted_unicode", R"('\u0041\u00e9\uD83D\uDE00')", "Aé😀"},
        {"quoted_dense_newlines", "'" + utils::repeat(R"(\n\t)", 256) + "'", utils::repeat("\n\t", 256)},
        {"mixed_escaped_literal", R"(prefix 'line\nnext' {foo} suffix)", "prefix line\nnext bar suffix"},
        {"quoted_single_char", "'x'", "x"},
        {"quoted_single_newline", R"('\n')", "\n"}
    };
    auto legacy_changed_cases = changed_cases;
    std::get<2>(legacy_changed_cases[0]) = "prefix 'bar' suffix";
    std::get<2>(legacy_changed_cases[1]) = "prefix bar";
    std::get<2>(legacy_changed_cases[2]) = "'" + utils::repeat("barx", 100) + "'";
    std::get<2>(legacy_changed_cases[3]) = "['bar','other']";
    std::get<2>(legacy_changed_cases[4]) = R"('say "hello"')";
    std::get<2>(legacy_changed_cases[5]) = "q";
    std::get<2>(legacy_changed_cases[6]) = "'linennext'";
    std::get<2>(legacy_changed_cases[7]) = "'can't'";
    std::get<2>(legacy_changed_cases[8]) = "'u0041u00e9uD83DuDE00'";
    std::get<2>(legacy_changed_cases[9]) = "'" + utils::repeat("nt", 256) + "'";
    std::get<2>(legacy_changed_cases[10]) = "prefix 'linennext' bar suffix";
    std::get<2>(legacy_changed_cases[11]) = "'x'";
    std::get<2>(legacy_changed_cases[12]) = "'n'";

    for(const auto& name : options.cases) {
        const auto matches = [&](const auto& entry) { return std::get<0>(entry) == name; };
        if(not std::any_of(cases.begin(), cases.end(), matches)
           && not std::any_of(changed_cases.begin(), changed_cases.end(), matches))
            throw std::invalid_argument("Unknown parser benchmark case: " + name);
    }
    for(const auto& name : options.parsers) {
        if(name != "parse_text" && name != "RealizeStringPattern" && name != "SingleUsePattern")
            throw std::invalid_argument("Unknown parser benchmark path: " + name);
    }

    bool condition = true;
    sprite::Map tag;
    tag["p"] = float();
    tag["id"] = uint32_t();
    Context context = {
        VarFunc("cond", [&condition](const VarProps&) -> bool { return condition; }),
        VarFunc("foo", [](const VarProps&) -> std::string { return "bar"; }),
        VarFunc("colon", [](const VarProps&) -> std::string { return ":"; }),
        VarFunc("cond1", [](const VarProps&) -> bool { return true; }),
        VarFunc("cond2", [](const VarProps&) -> bool { return false; }),
        VarFunc("has_pred", [](const VarProps&) -> bool { return true; }),
        VarFunc("max_pred", [](const VarProps&) -> std::pair<track::Idx_t, float> { return {track::Idx_t(0), 0.99}; }),
        VarFunc("category", [](const VarProps&) -> std::string { return "male"; }),
        VarFunc("tag", [&](const VarProps&) -> sprite::Map& { return tag; }),
        VarFunc("average_category", [](const VarProps&) -> std::string { return "male"; }),
        VarFunc("name", [](const VarProps&) -> std::string { return "fish0"; }),
        VarFunc("id", [](const VarProps&) -> track::Idx_t { return track::Idx_t(0); })
    };
    State state;
    auto object_handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = object_handler;

    auto run = [&]<typename Direct, typename Prepared>(std::string_view version, Context& ctx, const auto& changed) {
        using Pattern = std::remove_cvref_t<decltype(std::declval<BenchmarkRunner<Prepared>>().prepared)>;
        benchmark_parse_fn<Direct>("parse_text", "equivalent", cases, ctx, state, options.iterations, version, options);
        benchmark_parse_fn<Prepared>("RealizeStringPattern", "equivalent", cases, ctx, state, options.iterations, version, options);
        benchmark_parse_fn<PrepareAndResolveTag<Pattern>>("SingleUsePattern", "equivalent", cases, ctx, state, options.iterations, version, options);
        benchmark_parse_fn<Direct>("parse_text", "policy_change", changed, ctx, state, options.iterations, version, options);
        benchmark_parse_fn<Prepared>("RealizeStringPattern", "policy_change", changed, ctx, state, options.iterations, version, options);
        benchmark_parse_fn<PrepareAndResolveTag<Pattern>>("SingleUsePattern", "policy_change", changed, ctx, state, options.iterations, version, options);
    };
    std::cout << "version,parser,group,case,ns_per_parse,iterations,checksum,prepared_parts\n";
    if(options.current_only) {
        for(size_t round = 0; round < 6; ++round)
            run.template operator()<ParseTextTag, ResolveTag>("current", context, changed_cases);
        return;
    }

    Context head_context = context, older_context = context;
    cmn::benchmark_head::dyn::configure(head_context);
    cmn::benchmark_older::dyn::configure(older_context);
    std::array<std::function<void()>, 3> versions{
        [&] { run.template operator()<OlderParseTextTag, HistoricalResolveTag<cmn::benchmark_older::pattern::UnresolvedStringPattern>>("older", older_context, legacy_changed_cases); },
        [&] { run.template operator()<HeadParseTextTag, HistoricalResolveTag<cmn::benchmark_head::pattern::UnresolvedStringPattern>>("head", head_context, legacy_changed_cases); },
        [&] { run.template operator()<ParseTextTag, ResolveTag>("current", context, changed_cases); }
    };
    std::array<size_t, 3> order{0, 1, 2};
    do {
        for(auto index : order)
            versions[index]();
    } while(std::next_permutation(order.begin(), order.end()));
}

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

int main(int argc, char** argv) {
    try {
        bool parser_only = false;
        bool strings_only = false;
        ParserBenchmarkOptions options;
        const auto next_argument = [&](int& i) -> std::string {
            if(i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + std::string(argv[i]));
            return argv[++i];
        };
        for(int i = 1; i < argc; ++i) {
            if(std::string_view(argv[i]) == "--parser-only")
                parser_only = true;
            else if(std::string_view(argv[i]) == "--strings-only")
                strings_only = true;
            else if(std::string_view(argv[i]) == "--current-only") {
                options.current_only = true;
                parser_only = true;
            } else if(std::string_view(argv[i]) == "--case") {
                options.cases.push_back(next_argument(i));
                parser_only = true;
            } else if(std::string_view(argv[i]) == "--parser") {
                options.parsers.push_back(next_argument(i));
                parser_only = true;
            } else if(std::string_view(argv[i]) == "--iterations") {
                const auto value = next_argument(i);
                if(value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("--iterations requires a positive integer");
                const auto count = std::stoull(value);
                if(count == 0 || count > std::numeric_limits<size_t>::max())
                    throw std::invalid_argument("--iterations is out of range");
                options.iterations = static_cast<size_t>(count);
                parser_only = true;
            }
            else
                throw std::invalid_argument("Usage: bench_dyngui_parser [--parser-only|--strings-only] [--current-only] [--case NAME] [--parser NAME] [--iterations N]");
        }

        if(parser_only && strings_only)
            throw std::invalid_argument("Select only one benchmark mode");
        // Keep parser diagnostics out of the CSV stream.
        cmn::set_runtime_quiet(true);
        if(strings_only) {
            cmn::benchmark::CompareStringDecoders();
            return 0;
        }
        CompareParsers(options);
        if(parser_only)
            return 0;

        const BenchmarkObject sample{
            .x = 42,
            .name = "trex",
            .enabled = true,
            .value = {.value = 123}
        };

        const glz::generic json = to_json(sample);
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
            [](glz::generic& object, double value) -> double {
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
            [](glz::generic& object, double value) -> double {
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
                auto nested = object["obj"].value<glz::generic>();
                nested["field"] = value;
                object["obj"] = nested;
                return nested["field"].get_number();
            },
            write_values,
            warmup_iterations,
            timed_iterations,
            sink);

        benchmark_create_backend<glz::generic>(
            "json",
            "cold_object",
            [](const CreateInputs& input) -> glz::generic {
                glz::generic nested;
                nested["field"] = input.nested;

                glz::generic object;
                object["number"] = input.number;
                object["name"] = std::string(input.name);
                object["enabled"] = input.enabled;
                object["obj"] = std::move(nested);
                return object;
            },
            [](const glz::generic& object) -> double {
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

                glz::generic nested;
                nested["field"] = input.nested;
                object["obj"] = std::move(nested);
                return object;
            },
            [](const sprite::Map& object) -> double {
                return object.at("number").value<double>()
                    + object.at("obj").value<glz::generic>()["field"].get_number()
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
                VarFunc("object", [&json](const VarProps&) -> const glz::generic& { return json; })
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
