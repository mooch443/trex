#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>
#include <gui/DynamicGUI.h>
#include <gui/DynamicVariable.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/ResolveVariable.h>
#include <gui/types/StaticText.h>
#include <gui/dyn/UnresolvedStringPattern.h>   // for ResolveStringPattern tests
#include <type_traits>

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

// ---------------------------------------------------------------------------
// Helper tags + overloads so failed typed‑tests show a *readable* type name
// (“ParseTextTag” or “ResolveTag”) instead of std::true_type / std::false_type.
// ---------------------------------------------------------------------------
struct ParseTextTag {};
struct ResolveTag   {};

template <typename Tag>
inline std::string run_parser(const std::string& pattern,
                              Context&            ctx,
                              State&              st);

// Parse‑time implementation
template <>
inline std::string run_parser<ParseTextTag>(const std::string& pattern,
                                            Context&            ctx,
                                            State&              st)
{
    return parse_text(pattern, ctx, st);
}

// Prepared‑pattern (“Resolve”) implementation
template <>
inline std::string run_parser<ResolveTag>(const std::string& pattern,
                                          Context&            ctx,
                                          State&              st)
{
    auto prepared = cmn::pattern::UnresolvedStringPattern::prepare(pattern);
    return prepared.realize(ctx, st);
}

// Register the two implementations for GoogleTest’s typed‑test suite
using ParserImpls = ::testing::Types<ParseTextTag, ResolveTag>;

template <typename T>
class ParseAndResolveTest : public ::testing::Test {};
TYPED_TEST_SUITE(ParseAndResolveTest, ParserImpls);

TEST(TestDerivedPtr, Construct) {
    derived_ptr<Drawable> ptr;
    ASSERT_EQ(ptr, nullptr);
    
    {
        auto text_ptr = Layout::Make<Text>();
        ptr = text_ptr;
        
        ASSERT_EQ(text_ptr, ptr);
        ASSERT_TRUE(text_ptr.get_smart());
        ASSERT_EQ(text_ptr.get_smart().use_count(), ptr.get_smart().use_count());
        ASSERT_EQ(text_ptr.get_smart().use_count(), 2);
    }
    
    {
        ASSERT_EQ(ptr.get_smart().use_count(), 1);
        auto smart = ptr.get_smart();
        ASSERT_EQ(ptr.get_smart().use_count(), 2);
        ptr = nullptr;
        ASSERT_EQ(smart.use_count(), 1);
    }
}

TEST(TestDerivedPtr, Convert) {
    auto button = Layout::Make<Button>();
    static_assert(std::same_as<decltype(button), derived_ptr<Drawable>>, "");
}

// Unit Tests
// ---------------------------------------------------------------------------
// The following typed tests are compiled twice: once with TypeParam::value
// == false (direct parse_text) and once with == true (ResolveStringPattern).
// This guarantees both implementations behave identically for each scenario.
// ---------------------------------------------------------------------------

TYPED_TEST(ParseAndResolveTest, BasicReplacement)
{
    State   state;
    Context ctx{
        VarFunc("variable", [](const VarProps&) -> std::string { return "mocked_value"; })
    };
    auto result = run_parser<TypeParam>("{variable}", ctx, state);
    ASSERT_EQ(result, "mocked_value");
}

TYPED_TEST(ParseAndResolveTest, IfReplacement)
{
    State   state;
    Context ctx{
        VarFunc("variable", [](const VarProps&) -> bool { return true; })
    };
    auto result = run_parser<TypeParam>("{if:{variable}:'correct':'wrong'}", ctx, state);
    ASSERT_EQ(result, "correct");
}

TYPED_TEST(ParseAndResolveTest, LazyEvalReplacement)
{
    State    state;
    bool     ran = false;
    Context  ctx{
        VarFunc("variable", [](const VarProps&) -> bool { return true; }),
        VarFunc("correct",  [](const VarProps&) -> std::string { return "c"; }),
        VarFunc("throws",   [&](const VarProps&) -> bool {
            ran = true;
            throw std::invalid_argument("Not supposed to run.");
        })
    };
    std::string result;
    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:'{correct}':'{throws}'}", ctx, state));
    ASSERT_EQ(result, "c");
    ASSERT_FALSE(ran);
}

TYPED_TEST(ParseAndResolveTest, NoReplacement)
{
    State   state;
    Context ctx;
    if constexpr(std::is_same_v<TypeParam, ParseTextTag>) {
        auto result = run_parser<TypeParam>("{missing_variable}", ctx, state);
        ASSERT_EQ(result, "null");
    } else {
        ASSERT_THROW(run_parser<TypeParam>("{missing_variable}", ctx, state), std::exception);
    }
}

TYPED_TEST(ParseAndResolveTest, EscapeCharacters)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("\\{variable\\}", ctx, state);
    ASSERT_EQ(result, "{variable}");
}

TYPED_TEST(ParseAndResolveTest, SpecialTypeSize2)
{
    State   state;
    Context ctx{
        VarFunc("size2_var", [](const VarProps&) -> Size2 { return Size2(10, 5); })
    };
    auto result = run_parser<TypeParam>("{size2_var.w}", ctx, state);
    ASSERT_EQ(result, "10");
}

TYPED_TEST(ParseAndResolveTest, SpecialTypeVec2)
{
    State   state;
    Context ctx{
        VarFunc("vec2_var", [](const VarProps&) -> Vec2 { return Vec2(10, 5); })
    };
    auto result = run_parser<TypeParam>("{vec2_var.x}", ctx, state);
    ASSERT_EQ(result, "10");
}

TYPED_TEST(ParseAndResolveTest, HtmlifySyntax)
{
    State   state;
    Context ctx{
        VarFunc("html_var", [](const VarProps&) -> std::string {
            return "classname::value<int>(parm)\n`https://address/`";
        })
    };
    auto result = run_parser<TypeParam>("{#html_var}", ctx, state);
    ASSERT_EQ(result, "classname::value&lt;<key>int</key>&gt;(parm)<br/><a>https://address/</a>");
}

TYPED_TEST(ParseAndResolveTest, ExceptionHandling)
{
    State   state;
    Context ctx{
        VarFunc("exception_var", [](const VarProps&) -> std::string {
            throw std::runtime_error("An exception"); // NOLINT
        })
    };
    auto result = run_parser<TypeParam>("{exception_var}", ctx, state);
    ASSERT_EQ(result, "null");
}

TYPED_TEST(ParseAndResolveTest, ArithmeticAddVector)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    auto result = run_parser<TypeParam>("{addVector:[{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}},10]:[10,0]}", ctx, state);
    ASSERT_EQ(result, "[17,10]");
}

TYPED_TEST(ParseAndResolveTest, ArithmeticNestedOperations)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    auto result = run_parser<TypeParam>("{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}}", ctx, state);
    ASSERT_EQ(result, "7");
}

// ---------------------------------------------------------------------------
// Additional scenarios ported from the legacy ParseText suite
// ---------------------------------------------------------------------------

// More deeply‑nested arithmetic expression: (frame + video_length) * (window_size.w / video_length)
// => (5 + 50) * (100 / 50) = 55 * 2 = 110
TYPED_TEST(ParseAndResolveTest, ArithmeticMultipleNestedOperations)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    auto result = run_parser<TypeParam>("{*: {+: {frame}:{video_length}}: {/: {window_size.w} : {video_length}}}", ctx, state);
    ASSERT_EQ(result, "110");
}

// Invalid variable inside a nested operation – ParseText returns "null",
// ResolveTag raises (same semantics as the NoReplacement test)
TYPED_TEST(ParseAndResolveTest, InvalidNestedOperation)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    if constexpr(std::is_same_v<TypeParam, ParseTextTag>) {
        auto result = run_parser<TypeParam>("{*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}", ctx, state);
        ASSERT_EQ(result, "null");
    } else {
        ASSERT_THROW(run_parser<TypeParam>("{*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}", ctx, state), std::exception);
    }
}

// Same invalid sub‑expression, but embedded in a literal string context
TYPED_TEST(ParseAndResolveTest, InvalidNestedString)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    constexpr const char* pattern = "This is a string: {*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}";
    if constexpr(std::is_same_v<TypeParam, ParseTextTag>) {
        auto result = run_parser<TypeParam>(pattern, ctx, state);
        ASSERT_EQ(result, "This is a string: null");
    } else {
        ASSERT_THROW(run_parser<TypeParam>(pattern, ctx, state), std::exception);
    }
}

// ---------------------------------------------------------------------------
// Legacy brace/escape‑error scenarios that were still missing
// ---------------------------------------------------------------------------

// "{variable_{inner"  → unmatched brace inside identifier
TYPED_TEST(ParseAndResolveTest, NestedMissingBraceThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{variable_{inner", ctx, state), std::runtime_error);
}

// "{{variable}" → double‑opening brace
TYPED_TEST(ParseAndResolveTest, DoubleOpeningBraceThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{{variable}", ctx, state), std::runtime_error);
}

// "{variable}}" → double‑closing brace
TYPED_TEST(ParseAndResolveTest, DoubleClosingBraceThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{variable}}", ctx, state), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Escape‑sequence handling
// ---------------------------------------------------------------------------

TYPED_TEST(ParseAndResolveTest, InvalidEscapeSequenceValidEscapes)
{
    State   state;
    Context ctx;
    std::string out;

    // \"{\\}"  ⇒ literally "{}"  (valid escaping)
    ASSERT_NO_THROW(out = run_parser<TypeParam>("\\{\\}", ctx, state));
    ASSERT_EQ(out, "{}");

    // "\"\\n\""  ⇒ payload contains \" and newline (valid)
    ASSERT_NO_THROW(out = run_parser<TypeParam>("\"\\n\"", ctx, state));
    ASSERT_EQ(out, "\"n\"");
}

TYPED_TEST(ParseAndResolveTest, InvalidEscapeSequenceThrows)
{
    State   state;
    Context ctx;

    // "{\\}"  ⇒ invalid backslash inside braces
    ASSERT_THROW(run_parser<TypeParam>("{\\}", ctx, state), std::runtime_error);

    // "\\{invalid\\_escape}" ⇒ unsupported \_escape
    ASSERT_THROW(run_parser<TypeParam>("\\{invalid\\_escape}", ctx, state), std::runtime_error);
}

// ---------------------------------------------------------------------------
// A trailing back‑slash at end of input must trigger an error
// ---------------------------------------------------------------------------
TYPED_TEST(ParseAndResolveTest, TrailingBackslashThrows)
{
    State   state;
    Context ctx{
        VarFunc("variable", [](const VarProps&) -> std::string { return "x"; })
    };
    ASSERT_THROW(run_parser<TypeParam>("{variable}\\", ctx, state), std::runtime_error);
}

// --- Error‑handling parity checks -------------------------------------------------

TYPED_TEST(ParseAndResolveTest, MissingClosingBraceThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{invalid_input", ctx, state), std::runtime_error);
}

TYPED_TEST(ParseAndResolveTest, MissingOpeningBraceThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("invalid_input}", ctx, state), std::runtime_error);
}

TYPED_TEST(ParseAndResolveTest, EmptyBracesThrows)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{}", ctx, state), std::runtime_error);
}


class StaticTextTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize any required resources
    }

    void TearDown() override {
        // Clean up any resources
    }

    // Utility function to check line breaks
    void checkLineBreaks(const std::vector<std::unique_ptr<cmn::gui::StaticText::RichString>>& strings, float max_width, cmn::gui::Drawable* reference) {
        for (const auto& str : strings) {
            Bounds bounds = cmn::utils::calculate_bounds(str->parsed, reference, str->font);
            float width = cmn::utils::calculate_width(bounds);
            
            //Print("** ", utils::ShortenText(str->parsed, 15)," w=", width, " max=",max_width, " font=",str->font);
            
            EXPECT_LE(width, max_width) << "Line exceeds max width: " << str->str;
        }
    }

    // Utility function to check that all characters from input are found in the correct order
    void checkCharactersInOrder(const std::string& input, const std::vector<std::unique_ptr<cmn::gui::StaticText::RichString>>& strings) {
        Vec2 prev(FLT_MAX);
        size_t index = 0;
        for (const auto& str : strings) {
            bool new_line = str->pos.y != prev.y;
            prev = str->pos;
            
            for(size_t i = 0; i < str->str.size(); ++i, ++index) {
                EXPECT_FALSE(index >= input.size());
                if(new_line && i == 0 && std::isspace(input[index]) && input[index] != str->str[i])
                    ++index;
                
                EXPECT_FALSE(index >= input.size());
                EXPECT_EQ(str->str[i], input[index]);
            }
        }
    }
};

TEST_F(StaticTextTest, NoMaxWidth) {
    StaticText::Settings settings;
    settings.default_font = Font(0.5);
    settings.max_size.x = 0;  // No max width
    
    std::vector<std::unique_ptr<StaticText::RichString>> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    auto richString = std::make_unique<StaticText::RichString>(input, cmn::gui::Font(), cmn::Vec2(), Red);
    
    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);
    
    // Verify that no lines are longer than the default width
    EXPECT_EQ(strings.size(), 1);

    // Verify that all characters are in the correct order
    checkCharactersInOrder(input, strings);
}

TEST_F(StaticTextTest, SmallMaxWidth) {
    StaticText::Settings settings;
    settings.default_font = Font(0.5);
    settings.max_size.x = 50;

    std::vector<std::unique_ptr<StaticText::RichString>> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    auto richString = std::make_unique<StaticText::RichString>(input, cmn::gui::Font(), cmn::Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    // Verify that lines do not exceed the specified max width
    checkLineBreaks(strings, settings.max_size.x, nullptr);

    // Additional verification to ensure multiple lines are created
    EXPECT_GT(strings.size(), 1);

    // Verify that all characters are in the correct order
    checkCharactersInOrder(input, strings);
}

TEST_F(StaticTextTest, MediumMaxWidth) {
    StaticText::Settings settings;
    settings.default_font = Font(0.5);
    settings.max_size.x = 100;  // Medium max width (arbitrary unit)

    std::vector<std::unique_ptr<StaticText::RichString>> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    auto richString = std::make_unique<StaticText::RichString>(input, cmn::gui::Font(), cmn::Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    // Verify that lines do not exceed the specified max width
    checkLineBreaks(strings, settings.max_size.x, nullptr);

    // Additional verification to ensure multiple lines are created
    EXPECT_GT(strings.size(), 1);

    // Verify that all characters are in the correct order
    checkCharactersInOrder(input, strings);
}

TEST_F(StaticTextTest, LargeMaxWidth) {
    StaticText::Settings settings;
    settings.default_font = Font(0.5);
    settings.max_size.x = 150;  // Large max width (arbitrary unit)

    std::vector<std::unique_ptr<StaticText::RichString>> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    auto richString = std::make_unique<StaticText::RichString>(input, cmn::gui::Font(), cmn::Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    // Verify that lines do not exceed the specified max width
    checkLineBreaks(strings, settings.max_size.x, nullptr);

    // Additional verification to ensure multiple lines are created
    EXPECT_GT(strings.size(), 1);

    // Verify that all characters are in the correct order
    checkCharactersInOrder(input, strings);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
