#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/format.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>
#include <gui/DynamicGUI.h>
#include <gui/DynamicVariable.h>

using namespace gui;
using namespace dyn;

// Unit Tests
TEST(ParseText, BasicReplacement) {
    Context context{
        VarFunc("variable", [](const VarProps&) -> std::string { return "mocked_value"; })
    };
    std::string result = parse_text("{variable}", context);
    ASSERT_EQ(result, "mocked_value");
}

TEST(ParseText, NoReplacement) {
    Context context;
    std::string result = parse_text("{missing_variable}", context);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, NestedReplacement) {
    Context context{
        VarFunc("variable_inner_variable", [](const VarProps&) -> std::string { return "mocked_value"; }),
        VarFunc("inner_variable", [](const VarProps&) -> std::string { return "inner"; }),
        VarFunc("variable_inner", [](const VarProps&) -> std::string { return "correct"; })
    };
    std::string result = parse_text("{variable_{inner_variable}}", context);
    ASSERT_EQ(result, "correct");
}

TEST(ParseText, EscapeCharacters) {
    Context context;
    std::string result = parse_text("\\{variable\\}", context);
    ASSERT_EQ(result, "{variable}");
}

TEST(ParseText, SpecialTypeSize2) {
    Context context{
        VarFunc("size2_var", [](const VarProps&) -> Size2 { return Size2(10, 5); })
    };
    std::string result = parse_text("{size2_var.w}", context);
    ASSERT_EQ(result, "10");
}

TEST(ParseText, SpecialTypeVec2) {
    Context context{
        VarFunc("vec2_var", [](const VarProps&) -> Vec2 { return Vec2(10, 5); })
    };
    std::string result = parse_text("{vec2_var.x}", context);
    ASSERT_EQ(result, "10");
}

TEST(ParseText, HtmlifySyntax) {
    Context context{
        VarFunc("html_var", [](const VarProps&) -> std::string {
            return "classname::value<int>(parm)\n`https://address/`";
        })
    };
    std::string result = parse_text("{#html_var}", context);
    ASSERT_EQ(result, "classname::value&lt;<key>int</key>&gt;(parm)<br/>\n<a>https://address/</a>");
}

TEST(ParseText, ExceptionHandling) {
    Context context{
        VarFunc("exception_var", [](const VarProps&) -> std::string {
            throw std::runtime_error("An exception");
            return "should not reach here";
        })
    };
    std::string result = parse_text("{exception_var}", context);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, PerformanceTest) {
    Context context{
        VarFunc("very_long_variable", [](const VarProps&) -> std::string {
            return "very long mocked value";
        })
    };
    auto start = std::chrono::high_resolution_clock::now();
    std::string result = parse_text("very_long_pattern", context);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ASSERT_TRUE(elapsed.count() < 500);
}

TEST(ParseText, MissingClosingBrace) {
    Context context;
    EXPECT_THROW(parse_text("{invalid_input", context), std::runtime_error);
}

TEST(ParseText, MissingOpeningBrace) {
    Context context;
    EXPECT_THROW(parse_text("invalid_input}", context), std::runtime_error);
}

TEST(ParseText, NestedMissingBrace) {
    Context context;
    EXPECT_THROW(parse_text("{variable_{inner", context), std::runtime_error);
}

TEST(ParseText, DoubleOpeningBrace) {
    Context context;
    EXPECT_THROW(parse_text("{{variable}", context), std::runtime_error);
}

TEST(ParseText, DoubleClosingBrace) {
    Context context;
    EXPECT_THROW(parse_text("{variable}}", context), std::runtime_error);
}

TEST(ParseText, EmptyBraces) {
    Context context;
    EXPECT_THROW(parse_text("{}", context), std::runtime_error);
}

TEST(ParseText, InvalidEscapeSequence) {
    Context context;
    EXPECT_THROW(parse_text("\\{invalid\\_escape}", context), std::runtime_error);
}

TEST(ParseText, TrailingBackslashes) {
    Context context;
    EXPECT_THROW(parse_text("{variable}\\", context), std::runtime_error);
}

TEST(ParseText, InvalidTypeUsage) {
    Context context{
        VarFunc("variable", [](const VarProps& props) -> std::string {
                if(not props.subs.empty())
                    throw InvalidArgumentException("Variable has no fields: ", props.subs);
                return "mocked_value";
        })
    };
    auto result = parse_text("{variable.wrong_field}", context);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, EmptyVariableName) {
    Context context;
    EXPECT_THROW(parse_text("{}", context), std::runtime_error);
}

TEST(ParseText, AddVectorTest) {
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{addVector:[{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}},10]:[10,0]}", context);
    ASSERT_EQ(result, "[17,10]");
}

TEST(ParseText, NestedOperations) {
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}}", context);
    ASSERT_EQ(result, "7"); // (5/50) * (100 - 30) = 0.1 * 70 = 7
}

TEST(ParseText, MultipleNestedOperations) {
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{*: {+: {frame}:{video_length}}: {/: {window_size.w} : {video_length}}}", context);
    ASSERT_EQ(result, "110"); // (5 + 50) * (100 / 50) = 55 * 2 = 110
}

TEST(ParseText, InvalidNestedOperation) {
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    auto str = parse_text("{*: {+: {invalid}:{video_length}}: {/: {window_size.x} : {video_length}}}", context);
    EXPECT_EQ(str, "null");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
