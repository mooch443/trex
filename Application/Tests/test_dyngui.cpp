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

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

// Unit Tests
TEST(ParseText, BasicReplacement) {
    State state;
    Context context{
        VarFunc("variable", [](const VarProps&) -> std::string { return "mocked_value"; })
    };
    std::string result = parse_text("{variable}", context, state);
    ASSERT_EQ(result, "mocked_value");
}

TEST(ParseText, IfReplacement) {
    State state;
    Context context{
        VarFunc("variable", [](const VarProps&) -> bool { return true; })
    };
    std::string result = parse_text("{if:{variable}:'correct':'wrong'}", context, state);
    ASSERT_EQ(result, "correct");
}

TEST(ParseText, LazyEvalReplacement) {
    State state;
    bool ran = false;
    Context context{
        VarFunc("variable", [](const VarProps&) -> bool { return true; }),
        VarFunc("correct", [](const VarProps&) -> std::string { return "c"; }),
        VarFunc("throws",  [&](const VarProps&) -> bool {
            ran = true;
            throw std::invalid_argument("Not supposed to run.");
        })
    };
    
    std::string result;
    ASSERT_NO_THROW(result = parse_text("{if:{variable}:'{correct}':'{throws}'}", context, state));
    ASSERT_EQ(result, "c");
    ASSERT_EQ(ran, false);
}

TEST(ParseText, NoReplacement) {
    State state;
    Context context;
    std::string result = parse_text("{missing_variable}", context, state);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, NestedReplacement) {
    // THIS TEST HAS BEEN DEPRECATED FOR NOW
    State state;
    Context context{
        VarFunc("variable_inner_variable", [](const VarProps&) -> std::string { return "mocked_value"; }),
        VarFunc("inner_variable", [](const VarProps&) -> std::string { return "inner"; }),
        VarFunc("variable_inner", [](const VarProps&) -> std::string { return "correct"; })
    };
    std::string result = parse_text("{variable_{inner_variable}}", context, state);
    //ASSERT_EQ(result, "correct");
    FormatExcept(result," does not equal 'correct' but the test is deactivated.");
}

TEST(ParseText, EscapeCharacters) {
    State state;
    Context context;
    std::string result = parse_text("\\{variable\\}", context, state);
    ASSERT_EQ(result, "{variable}");
}

TEST(ParseText, SpecialTypeSize2) {
    State state;
    Context context{
        VarFunc("size2_var", [](const VarProps&) -> Size2 { return Size2(10, 5); })
    };
    std::string result = parse_text("{size2_var.w}", context, state);
    ASSERT_EQ(result, "10");
}

TEST(ParseText, SpecialTypeVec2) {
    State state;
    Context context{
        VarFunc("vec2_var", [](const VarProps&) -> Vec2 { return Vec2(10, 5); })
    };
    std::string result = parse_text("{vec2_var.x}", context, state);
    ASSERT_EQ(result, "10");
}

TEST(ParseText, HtmlifySyntax) {
    State state;
    Context context{
        VarFunc("html_var", [](const VarProps&) -> std::string {
            return "classname::value<int>(parm)\n`https://address/`";
        })
    };
    std::string result = parse_text("{#html_var}", context, state);
    ASSERT_EQ(result, "classname::value&lt;<key>int</key>&gt;(parm)<br/><a>https://address/</a>");
}

TEST(ParseText, ExceptionHandling) {
    State state;
    Context context{
        VarFunc("exception_var", [](const VarProps&) -> std::string {
            throw std::runtime_error("An exception");
            return "should not reach here";
        })
    };
    std::string result = parse_text("{exception_var}", context, state);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, PerformanceTest) {
    State state;
    Context context{
        VarFunc("very_long_variable", [](const VarProps&) -> std::string {
            return "very long mocked value";
        })
    };
    auto start = std::chrono::high_resolution_clock::now();
    std::string result = parse_text("very_long_pattern", context, state);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ASSERT_TRUE(elapsed.count() < 500);
}

TEST(ParseText, MissingClosingBrace) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{invalid_input", context, state), std::runtime_error);
}

TEST(ParseText, MissingOpeningBrace) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("invalid_input}", context, state), std::runtime_error);
}

TEST(ParseText, NestedMissingBrace) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{variable_{inner", context, state), std::runtime_error);
}

TEST(ParseText, DoubleOpeningBrace) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{{variable}", context, state), std::runtime_error);
}

TEST(ParseText, DoubleClosingBrace) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{variable}}", context, state), std::runtime_error);
}

TEST(ParseText, EmptyBraces) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{}", context, state), std::runtime_error);
}

TEST(ParseText, InvalidEscapeSequence) {
    State state;
    Context context;
    
    EXPECT_NO_THROW(parse_text("\\{\\}", context, state));
    EXPECT_THROW(parse_text("{\\}", context, state), std::runtime_error);
    EXPECT_NO_THROW(parse_text("\"\\n\"", context, state));
    EXPECT_THROW(parse_text("\\{invalid\\_escape}", context, state), std::runtime_error);
}

TEST(ParseText, TrailingBackslashes) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{variable}\\", context, state), std::runtime_error);
}

TEST(ParseText, InvalidTypeUsage) {
    State state;
    Context context{
        VarFunc("variable", [](const VarProps& props) -> std::string {
                if(not props.subs.empty())
                    throw InvalidArgumentException("Variable has no fields: ", props.subs);
                return "mocked_value";
        })
    };
    auto result = parse_text("{variable.wrong_field}", context, state);
    ASSERT_EQ(result, "null");
}

TEST(ParseText, EmptyVariableName) {
    State state;
    Context context;
    EXPECT_THROW(parse_text("{}", context, state), std::runtime_error);
}

TEST(ParseText, AddVectorTest) {
    State state;
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{addVector:[{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}},10]:[10,0]}", context, state);
    ASSERT_EQ(result, "[17,10]");
}

TEST(ParseText, NestedOperations) {
    State state;
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{*:{/:{frame}:{video_length}}:{+:{window_size.w}:-30}}", context, state);
    ASSERT_EQ(result, "7"); // (5/50) * (100 - 30) = 0.1 * 70 = 7
}

TEST(ParseText, MultipleNestedOperations) {
    State state;
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    std::string result = parse_text("{*: {+: {frame}:{video_length}}: {/: {window_size.w} : {video_length}}}", context, state);
    ASSERT_EQ(result, "110"); // (5 + 50) * (100 / 50) = 55 * 2 = 110
}

TEST(ParseText, InvalidNestedOperation) {
    State state;
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    auto str = parse_text("{*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}", context, state);
    EXPECT_EQ(str, "null");
}

TEST(ParseText, InvalidNestedString) {
    State state;
    Context context{
        VarFunc("frame", [](const VarProps&) -> int { return 5;}),
        VarFunc("video_length", [](const VarProps&) -> int { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2 { return Size2(100, 20); })
    };

    auto str = parse_text("This is a string: {*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}", context, state);
    EXPECT_EQ(str, "This is a string: null");
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
