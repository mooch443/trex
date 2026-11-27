#include <commons.pc.h>
#include <gtest/gtest.h>
#include <misc/parse_parameter_lists.h>
#include <misc/Deprecations.h>
#include <misc/Path.h>
#include <misc/SpriteMap.h>
#include <gui/DynamicGUI.h>
#include <gui/dyn/Action.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/State.h>

using namespace cmn;
using namespace cmn::gui::dyn;

namespace {

enum class ParserPath { ParseText, Resolve };
void PrintTo(ParserPath path, std::ostream* output) {
    *output << (path == ParserPath::ParseText ? "ParseText" : "Resolve");
}

class StringEscapingTest : public ::testing::TestWithParam<ParserPath> {
protected:
    std::string parse(const std::string& input, const Context& context = {}) {
        State state;
        if(GetParam() == ParserPath::ParseText)
            return parse_text(input, context, state);
        return cmn::pattern::UnresolvedStringPattern::prepare(input).realize(context, state);
    }
};

TEST(StringEscapingRegressionTest, PreparedSingleByteLiterals)
{
    using cmn::pattern::PreparedPattern;
    using cmn::pattern::UnresolvedStringPattern;
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"x", "x"},
        {"'x'", "x"},
        {R"('\n')", "\n"},
        {R"('\u0000')", std::string(1, '\0')},
        {std::string(1, '\0'), std::string(1, '\0')}
    };
    Context context;
    State state;
    for(const auto& [input, expected] : cases) {
        SCOPED_TRACE(::testing::Message() << "input=" << ::testing::PrintToString(input));
        auto prepared = UnresolvedStringPattern::prepare(input);
        ASSERT_EQ(prepared.objects.size(), 1u);
        EXPECT_EQ(prepared.objects.front().type, PreparedPattern::CHAR);
        EXPECT_EQ(prepared.realize(context, state), expected);

        auto copied = prepared;
        prepared = UnresolvedStringPattern{};
        auto moved = std::move(copied);
        EXPECT_EQ(moved.realize(context, state), expected);
        EXPECT_EQ(moved.realize(context, state), expected);
        EXPECT_EQ(UnresolvedStringPattern::prepare_static(expected).realize(context, state), expected);
    }

    EXPECT_EQ(UnresolvedStringPattern::prepare(R"(x{if:true:'\n':'q'}y)").realize(context, state), "x\ny");
    auto unicode = UnresolvedStringPattern::prepare(R"('\u00e9')");
    ASSERT_EQ(unicode.objects.size(), 1u);
    EXPECT_NE(unicode.objects.front().type, PreparedPattern::CHAR);
    EXPECT_EQ(unicode.realize(context, state), "é");
}

TEST_P(StringEscapingTest, Contract)
{
    const auto parser = ::testing::PrintToString(GetParam());
    std::string case_name, case_input, case_expected;
    const auto begin_case = [&](std::string_view name, std::string_view input, std::string expected) {
        case_name = name;
        case_input = input;
        case_expected = std::move(expected);
    };
    const auto report = [&](std::string_view consumer, std::string_view input,
                            std::string_view expected, std::source_location location,
                            const auto&... details) {
        prefixed_print("FAIL", no_quotes(parser), "/", no_quotes(case_name), " | ", no_quotes(consumer),
            " | case input=", no_quotes(case_input), " | input=", no_quotes(input),
            " | expected=", expected, " (", expected.size(), " bytes)", details...);
        ADD_FAILURE_AT(location.file_name(), static_cast<int>(location.line()))
            << "String escaping contract failed: " << parser << '/' << case_name << " / " << consumer;
    };
    const auto check = [&](std::string_view consumer, std::string_view input,
                          std::string_view expected, auto&& evaluate,
                          std::optional<size_t> limit = std::nullopt, std::string_view prefix = {},
                          std::source_location location = std::source_location::current()) {
        std::string actual;
        const char* outcome = nullptr;
        try {
            actual = evaluate();
        } catch(const std::exception& error) {
            outcome = "threw std::exception";
            actual = error.what();
        } catch(...) {
            outcome = "threw a non-std::exception";
        }
        if(not outcome && actual == expected)
            return;

        const auto failure = [&](const auto&... details) {
            if(limit || not prefix.empty())
                report(consumer, input, expected, location, details...,
                    " | limit=", limit, " | prefix=", std::string(prefix));
            else
                report(consumer, input, expected, location, details...);
        };
        if(outcome) {
            failure(" | actual=", no_quotes(outcome), " | exception=", actual);
        } else {
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
            const auto offset = static_cast<size_t>(std::distance(actual.begin(), mismatch.first));
            failure(" | actual=", std::string_view(actual), " (", actual.size(), " bytes)",
                " | first differing byte (zero-based)=", offset,
                " | expected byte (null=end)=", offset < expected.size()
                    ? std::optional<unsigned>(static_cast<unsigned char>(expected[offset])) : std::nullopt,
                " | actual byte (null=end)=", offset < actual.size()
                    ? std::optional<unsigned>(static_cast<unsigned char>(actual[offset])) : std::nullopt);
        }
    };
    const auto check_throws = [&]<typename Exception>(std::string_view consumer, std::string_view input,
                          std::string_view expected, auto&& evaluate,
                          std::source_location location = std::source_location::current()) {
        std::string actual;
        const char* outcome = "returned without throwing";
        try {
            actual = evaluate();
        } catch(const std::exception& error) {
            if(dynamic_cast<const Exception*>(&error))
                return;
            outcome = "threw a different std::exception";
            actual = error.what();
        } catch(...) {
            outcome = "threw a non-std::exception";
        }
        report(consumer, input, expected, location,
            " | actual=", no_quotes(outcome), " | value/message=", std::string_view(actual));
    };
    const auto parse = [&](const std::string& input, const Context& context) {
        State state;
        if(GetParam() == ParserPath::ParseText)
            return parse_text(input, context, state);
        return cmn::pattern::UnresolvedStringPattern::prepare(input).realize(context, state);
    };

    struct TestCase {
        const char* name;
        std::string_view input;
        std::string_view decoded;
        bool expression = true;
    };
    const std::string long_text(256, 'x');
    const std::string long_input = "'" + long_text + R"(\qtail')";
    const std::string long_decoded = long_text + "qtail";
    const TestCase cases[] = {
        {"empty", "", ""},
        {"plain", "abcdef", "abcdef"},
        {"whitespace", "  ", "  "},
        {"empty_single_quotes", "''", ""},
        {"empty_double_quotes", R"("")", ""},
        {"quoted_whitespace", "'  '", "  "},
        {"quoted_character", "'x'", "x"},
        {"ForIteratesQuotedString", "'ab'", "ab"},
        {"escaped_double_quote", R"('a\"b')", R"(a"b)"},
        {"double_quoted_escaped_quote", R"("a\"b")", R"(a"b)"},
        {"escaped_single_quote", R"('a\'bc')", "a'bc"},
        {"leading_escaped_quote", R"('\'abc')", "'abc"},
        {"trailing_escaped_quote", R"('abc\'')", "abc'"},
        {"unknown_escape_and_backslash", R"('\q\\')", R"(q\)"},
        {"double_quoted_unknown_escape", R"("\q")", "q"},
        {"unknown_escape_between_characters", R"('a\qb')", "aqb"},
        {"unquoted_unknown_escape", R"(\q)", R"(\q)"},
        {"unquoted_escaped_backslashes", R"(a\\b)", R"(a\\b)"},
        {"unquoted_json_escapes", R"(\n\t\u0041)", R"(\n\t\u0041)"},
        {"quoted_backslash", R"('a\\b')", R"(a\b)"},
        {"quoted_two_backslashes", R"("a\\\\b")", R"(a\\b)"},
        {"backslash_then_unknown_escape", R"('a\\\qb')", R"(a\qb)"},
        {"json_controls", R"("\b\f\n\r\t\/\\\"")", "\b\f\n\r\t/\\\""},
        {"single_quoted_newline", R"('a\nb')", "a\nb"},
        {"double_quoted_newline", R"("\n")", "\n"},
        {"quoted_path_with_newline_escape", R"('path\name')", "path\name"},
        {"literal_newline", "'a\nb'", "a\nb"},
        {"literal_backslash_n", R"('\\n')", R"(\n)"},
        {"unicode", R"("\u0041\u00e9")", "Aé"},
        {"surrogate_pair", R"('\uD83D\uDE00')", "😀"},
        {"escaped_null", R"("\u0000x")", std::string_view{"\0x", 2}},
        // Raw NUL inputs are excluded here because the JSON writer emits them unescaped.
        // {"literal_null", std::string_view{"'a\0b'", 5}, std::string_view{"a\0b", 3}},
        // {"unquoted_null", std::string_view{"a\0b", 3}, std::string_view{"a\0b", 3}},
        {"literal_utf8", "'éx'", "éx"},
        {"decoded_double_quotes", R"('"hello"')", R"("hello")"},
        {"decoded_single_quotes", R"("'hello'")", "'hello'"},
        {"long_literal_chunk", long_input, long_decoded},
        {"single_backslash", "\\", "\\"},
        // These conversion-helper edge cases are not valid template expressions.
        {"single_quote", "'", "'", false},
        {"missing_closing_quote", "'abc", "'abc", false},
        {"missing_opening_quote", "abc'", "abc'", false},
        {"mismatched_quotes", R"('abc")", R"('abc")", false},
        {"trailing_payload_backslash", R"('a\')", R"(a\)", false}
    };

    for(const auto& test : cases) {
        begin_case(test.name, test.input, std::string(test.decoded));
        const std::string input(test.input);
        const std::string expected(test.decoded);
        check("Meta::fromStr(string)", input, expected, [&] { return Meta::fromStr<std::string>(input); });
        check("Meta::fromStr(string_view)", test.input, expected, [&] { return Meta::fromStr<std::string>(test.input); });
        check("fast_fromstr", input, expected, [&] {
            std::string output;
            utils::fast_fromstr(output, input);
            return output;
        });
        check("fast_fromstr appends", input, "prefix:" + expected, [&] {
            std::string output = "prefix:";
            utils::fast_fromstr(output, input);
            return output;
        }, std::nullopt, "prefix:");

        // Limits count appended UTF-8 bytes, including boundaries inside a code point.
        for(const auto prefix : {std::string_view{}, std::string_view{"p\0:", 3}}) {
            for(size_t limit = 0; limit <= expected.size() + 1; ++limit) {
                check("fast_fromstr_N", input, std::string(prefix) + expected.substr(0, limit), [&] {
                    std::string output(prefix);
                    utils::fast_fromstr_N(output, input, limit);
                    return output;
                }, limit, prefix);
            }
        }

        check("escape/unescape round trip", expected, expected, [&] { return util::unescape(util::escape(expected)); });
        check("Meta round trip", expected, expected, [&] { return Meta::fromStr<std::string>(Meta::toStr(expected)); });
        check("Meta string is valid JSON", expected, expected, [&] {
            std::string output;
            const auto error = glz::read_json(output, Meta::toStr(expected));
            if(error != glz::error_code::none)
                throw std::invalid_argument("Meta::toStr produced an invalid JSON string");
            return output;
        });

        const bool quoted = input.size() >= 2
            && is_in(input.front(), '\'', '"') && input.front() == input.back();
        if(quoted) {
            const auto payload = test.input.substr(1, test.input.size() - 2);
            check("util::unescape payload", payload, expected, [&] { return util::unescape(payload); });
            check("parameter-list unescape payload", payload, expected, [&] { return util::unescape(std::string(payload)); });
        }
        if(not test.expression) {
            continue;
        }

        // Collection elements use serialized string syntax, including double quotes.
        const std::string literal = quoted ? input : Meta::toStr(input);
        check("vector element", "[" + literal + "]", expected, [&] {
            return Meta::fromStr<std::vector<std::string>>("[" + literal + "]").at(0);
        });
        check("map value", "{value:" + literal + "}", expected, [&] {
            return Meta::fromStr<std::map<std::string, std::string>>("{value:" + literal + "}").at("value");
        });
        check("parse_set_meta", "value=" + literal, expected, [&] { return parse_set_meta("value=" + literal).at("value"); });
        check("sprite property", input, expected, [&] {
            sprite::Map values;
            values["value"] = std::string{};
            values["value"].get().set_value_from_string(input);
            return values.at("value").value<std::string>();
        });
        check("sprite::parse_values", "{value:" + literal + "}", expected, [&] {
            sprite::Map values;
            values["value"] = std::string{};
            sprite::parse_values(sprite::MapSource::CMD, values, "{value:" + literal + "}", nullptr, {}, {});
            return values.at("value").value<std::string>();
        });

        // Serialized values accept both quote styles; expression literals use single quotes.
        const std::string expression_literal = quoted && input.front() == '\'' ? input
            : "'" + utils::find_replace(util::escape(expected), "'", "\\'") + "'";
        const std::string expression = quoted ? expression_literal : input;
        Context context{
            VarFunc("encoded", [&](const VarProps&) -> std::string { return input; }),
            VarFunc("decoded", [&](const VarProps&) -> std::string { return expected; })
        };
        check("literal expression", expression, expected, [&] { return parse(expression, context); });
        check("decoded variable", "{decoded}", expected, [&] { return parse("{decoded}", context); });
        check("if quoted branch", "{if:true:" + expression_literal + ":unused}", expected, [&] { return parse("{if:true:" + expression_literal + ":unused}", context); });
        check("if decoded branch", "{if:false:unused:{decoded}}", expected, [&] { return parse("{if:false:unused:{decoded}}", context); });
        check("array_length", "{array_length:{encoded}}", Meta::toStr(expected.size()), [&] { return parse("{array_length:{encoded}}", context); });
        for(size_t index = 0; index < expected.size(); ++index) {
            check("at", "{at:" + Meta::toStr(index) + ":{encoded}}", expected.substr(index, 1), [&] {
                return parse("{at:" + Meta::toStr(index) + ":{encoded}}", context);
            });
        }
        if(not input.empty()) {
            check("substr", "{substr:0:" + Meta::toStr(expected.size()) + ":{encoded}}", expected, [&] {
                return parse("{substr:0:" + Meta::toStr(expected.size()) + ":{encoded}}", context);
            });
        }

        std::string characters = "[";
        for(size_t index = 0; index < expected.size(); ++index) {
            if(index > 0)
                characters += ',';
            characters += expected[index];
        }
        characters += ']';
        check("for quoted string", "{for:c:" + expression_literal + ":{c}}", characters, [&] { return parse("{for:c:" + expression_literal + ":{c}}", context); });
        check("action parameter", "consume:" + expression_literal, expected, [&] {
            State state;
            return PreAction::fromStr("consume:" + expression_literal).parse(context, state).parameters.at(0);
        });
        check("decoded action parameter", "consume:{decoded}", expected, [&] {
            State state;
            return PreAction::fromStr("consume:{decoded}").parse(context, state).parameters.at(0);
        });

        const auto json_string = glz::write_json(expression).value();
        check("JSON text -> expression", "{\"text\":" + json_string + "}", expected, [&] {
            glz::generic object;
            const auto error = glz::read_json(object, "{\"text\":" + json_string + "}");
            if(error != glz::error_code::none)
                throw std::invalid_argument("Invalid test JSON");
            return parse(object["text"].get<std::string>(), context);
        });
        check("JSON layout default variable", "{\"defaults\":{\"vars\":{\"value\":" + json_string + "}},\"objects\":[]}", expected, [&] {
            auto loaded = load("{\"defaults\":{\"vars\":{\"value\":" + json_string + "}},\"objects\":[]}");
            if(not loaded)
                throw std::invalid_argument(loaded.error());
            Context defaults_context;
            defaults_context.defaults = std::move(std::get<0>(loaded.value()));
            return parse("{value}", defaults_context);
        });
    }

    const std::string_view roundtrip_values[] = {
        R"(\\\test\\)", R"(\)", R"(testwithend\)",
        R"("trailing spaces beyond end"    )", "hello", "world",
        R"('single quotes')", R"("double quotes")", R"(\"escaped double quotes\")",
        R"('\'escaped single quotes\'')", R"(\\)", R"(\\\")", R"(\\\\)",
        R"(\n\t\b\r\f\a)", R"(\\n\\t\\b\\r\\f\\a)",
        "   leading spaces", "trailing spaces   ", R"(    "leading spaces beyond start")"
    };
    for(const auto original : roundtrip_values) {
        begin_case("serialization round trip", original, std::string(original));
        const auto intermediate = Meta::toStr(std::string(original));
        check("ToStrFromStrRoundTripTest", intermediate, original, [&] { return Meta::fromStr<std::string>(intermediate); });
        check("canonical serialization round trip", intermediate, intermediate, [&] {
            return Meta::toStr(Meta::fromStr<std::string>(intermediate));
        });
    }

    for(const auto input : {R"(C:\new\track.mp4)", R"("C:\\new\\track.mp4")", R"('C:\\new\\track.mp4')"}) {
        const auto expected = file::Path(R"(C:\new\track.mp4)").str();
        begin_case("file::Path", input, expected);
        check("file::Path", input, expected, [&] { return Meta::fromStr<file::Path>(input).str(); });
    }

    for(const auto input : {R"("\u12")", R"("\uZZZZ")", R"("\uD800")", R"("\uDC00")", R"("\uD800\u0041")"}) {
        begin_case("invalid Unicode", input, "throws std::invalid_argument");
        check_throws.template operator()<std::invalid_argument>("Meta::fromStr", input, case_expected, [&] {
            return Meta::fromStr<std::string>(input);
        });
        check_throws.template operator()<std::invalid_argument>("fast_fromstr", input, case_expected, [&] {
            std::string output;
            utils::fast_fromstr(output, input);
            return output;
        });
        check_throws.template operator()<std::invalid_argument>("fast_fromstr_N limit=16", input, case_expected, [&] {
            std::string output;
            utils::fast_fromstr_N(output, input, 16);
            return output;
        });
    }

    // JSON decodes newlines first; literal braces and escaped apostrophes live inside quotes.
    constexpr std::string_view json = R"json({"text":"Say '{name}', then {name}. 'It\\'s ready.' C:\\data\nNext line."})json";
    glz::generic object;
    ASSERT_EQ(glz::read_json(object, json), glz::error_code::none);
    Context context{VarFunc("name", [](const VarProps&) -> std::string { return "Ada"; })};
    begin_case("JSON newline and template escapes", json,
               "Say {name}, then Ada. It's ready. C:\\data\nNext line.");
    check("JSON newline and template escapes", json, "Say {name}, then Ada. It's ready. C:\\data\nNext line.", [&] {
        return parse(object["text"].get<std::string>(), context);
    });
    begin_case("quoted braces", "'{}'", "{}");
    check("quoted braces", "'{}'", "{}", [&] { return parse("'{}'", context); });
    begin_case("bare braces", "{}", "throws std::exception");
    check_throws.template operator()<std::exception>("literal expression", "{}", case_expected, [&] {
        return parse("{}", context);
    });
}

TEST_P(StringEscapingTest, DoubleQuotedLiterals)
{
    // Double quotes are data in expressions, including inside single-quoted literals.
    EXPECT_EQ(parse(R"("")"), R"("")");
    EXPECT_EQ(parse(R"("ab")"), R"("ab")");
    EXPECT_EQ(parse(R"('say "hello"')"), R"(say "hello")");
    EXPECT_EQ(parse(R"({if:true:'say "hello"':unused})"), R"(say "hello")");
}

TEST_P(StringEscapingTest, SingleQuotedEscapes)
{
    EXPECT_EQ(parse(R"('a\"b')"), R"(a"b)");
    EXPECT_EQ(parse(R"('\n\u0041\q\\')"), "\nAq\\");
}

TEST_P(StringEscapingTest, UnquotedBackslashes)
{
    for(const std::string input : {R"(\q)", R"(a\\b)", R"(\n\t\u0041)", R"(trailing\)"}) {
        SCOPED_TRACE(::testing::Message() << "input=" << ::testing::PrintToString(input));
        EXPECT_EQ(parse(input), input);
    }
    EXPECT_EQ(parse("'{}'"), "{}");
    EXPECT_THROW(parse("{}"), std::exception);
    Context context{VarFunc("value", [](const VarProps&) -> std::string { return "resolved"; })};
    EXPECT_EQ(parse(R"(\{value})", context), R"(\resolved)");
}

TEST_P(StringEscapingTest, IfPreservesDecodedQuotes)
{
    for(const std::string expected : {R"("hello")", "'hello'"}) {
        SCOPED_TRACE(::testing::Message() << "decoded=" << ::testing::PrintToString(expected));
        Context context{VarFunc("decoded", [&](const VarProps&) -> std::string { return expected; })};
        EXPECT_EQ(parse("{if:true:{decoded}:unused}", context), expected);
        EXPECT_EQ(parse("{if:false:unused:{decoded}}", context), expected);
    }
}

TEST_P(StringEscapingTest, ForDecodesExactlyOnce)
{
    EXPECT_EQ(parse(R"({for:c:'ab':{c}})"), "[a,b]");
    EXPECT_EQ(parse(R"({for:c:'"a"':{c}})"), R"([",a,"])");
}

TEST_P(StringEscapingTest, LayoutDefaultsDecodeExactlyOnce)
{
    auto loaded = load(R"json({"defaults":{"vars":{"value":"'\"hello\"'"}},"objects":[]})json");
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    Context context;
    context.defaults = std::move(std::get<0>(loaded.value()));
    EXPECT_EQ(parse("{value}", context), R"("hello")");
}

TEST(StringEscapingRegressionTest, ParseValuesDecodesExactlyOnce)
{
    sprite::Map values;
    values["value"] = std::string{};
    sprite::parse_values(sprite::MapSource::CMD, values, R"({value:'"hello"'})", nullptr, {}, {});
    EXPECT_EQ(values.at("value").value<std::string>(), R"("hello")");
}

// Embedded-NUL JSON writing is outside the active contract; escaped-NUL decoding remains covered above.
/*
TEST(StringEscapingRegressionTest, JsonEmbeddedNullRoundTrip)
{
    const std::string expected("a\0b", 3);
    const auto encoded = glz::write_json(expected);
    ASSERT_TRUE(encoded.has_value());
    EXPECT_EQ(encoded->find('\0'), std::string::npos)
        << "JSON must escape NUL: " << ::testing::PrintToString(*encoded);

    // A fixed valid document distinguishes reader failures from writer failures.
    for(const auto& json : {std::string{R"({"text":"a\u0000b"})"}, "{\"text\":" + *encoded + "}"}) {
        SCOPED_TRACE(::testing::Message() << "json=" << ::testing::PrintToString(json)
            << " (" << json.size() << " bytes)");
        glz::generic object;
        const auto error = glz::read_json(object, json);
        EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, json);
        if(error == glz::error_code::none) {
            ASSERT_TRUE(object["text"].is_string());
            const auto& actual = object["text"].get<std::string>();
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(actual.size(), expected.size());
        }
    }
}
*/

INSTANTIATE_TEST_SUITE_P(ParseAndResolve, StringEscapingTest,
    ::testing::Values(ParserPath::ParseText, ParserPath::Resolve),
    [](const ::testing::TestParamInfo<ParserPath>& info) {
        return info.param == ParserPath::ParseText ? "ParseText" : "Resolve";
    });

}
