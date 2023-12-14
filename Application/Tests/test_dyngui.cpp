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

template<typename T>
    requires (std::convertible_to<T, std::string_view> || std::same_as<T, Pattern>)
std::string _parse_text(const T& _pattern, const Context& context, State& state) {
    std::string_view pattern;
    if constexpr(std::same_as<T, Pattern>) {
        pattern = std::string_view(_pattern.original);
    } else
        pattern = std::string_view(_pattern);
    
    std::stringstream output;
    std::stack<std::size_t> nesting_start_positions;
    bool comment_out = false;

    for(std::size_t i = 0; i < pattern.size(); ++i) {
        char ch = pattern[i];
        if(nesting_start_positions.empty()) {
            switch(ch) {
                case '\\':
                    if(not comment_out) {
                        comment_out = true;
                    } else {
                        comment_out = false;
                        output << '\\';
                    }
                    break;
                case '{':
                    if(!comment_out) nesting_start_positions.push(i + 1);
                    else output << ch;
                    comment_out = false;
                    break;
                case '}':
                    if(!comment_out) throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                    else output << ch;
                    comment_out = false;
                    break;
                default:
                    if(comment_out) {
                        throw InvalidSyntaxException("Invalid escape sequence at position ", i, ": ", ch, ". Only braces need to be escaped.");
                    }
                    output << ch;
            }
        } else {
            if(ch == '}') {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched closing brace at position ", i);
                }
                
                std::size_t start_pos = nesting_start_positions.top();
                nesting_start_positions.pop();

                if(nesting_start_positions.empty()) {
                    std::string_view current_word = pattern.substr(start_pos, i - start_pos);
                    if(current_word.empty()) {
                        throw InvalidSyntaxException("Empty braces at position ", i);
                    }
                    
                    //if(utils::contains(current_word, "segment"))
                    print("@", output.str(), " > parsing ", current_word);
                    
                    CTimer timer(current_word);
                    std::string resolved_word;
                    if(auto it = state._variable_values.find(current_word);
                       current_word != "hovered"
                       && current_word != "selected"
                       && it != state._variable_values.end())
                    {
                        resolved_word = it->second;
                        
                    } else {
                        resolved_word = resolve_variable(current_word, context, state, [](const VarBase_t& variable, const VarProps& modifiers) -> std::string {
                            try {
                                std::string ret;
                                if(modifiers.subs.empty())
                                    ret = variable.value_string(modifiers);
                                else if(variable.is<Size2>()) {
                                    if(modifiers.subs.front() == "w")
                                        ret = Meta::toStr(variable.value<Size2>(modifiers).width);
                                    else if(modifiers.subs.front() == "h")
                                        ret = Meta::toStr(variable.value<Size2>(modifiers).height);
                                    else
                                        throw InvalidArgumentException("Sub ",modifiers," of Size2 is not valid.");
                                    
                                } else if(variable.is<Vec2>()) {
                                    if(modifiers.subs.front() == "x")
                                        ret = Meta::toStr(variable.value<Vec2>(modifiers).x);
                                    else if(modifiers.subs.front() == "y")
                                        ret = Meta::toStr(variable.value<Vec2>(modifiers).y);
                                    else
                                        throw InvalidArgumentException("Sub ",modifiers," of Vec2 is not valid.");
                                    
                                } else if(variable.is<Range<Frame_t>>()) {
                                    if(modifiers.subs.front() == "start")
                                        ret = Meta::toStr(variable.value<Range<Frame_t>>(modifiers).start);
                                    else if(modifiers.subs.front() == "end")
                                        ret = Meta::toStr(variable.value<Range<Frame_t>>(modifiers).end);
                                    else
                                        throw InvalidArgumentException("Sub ",modifiers," of Range<Frame_t> is not valid.");
                                    
                                } else
                                    ret = variable.value_string(modifiers);
                                //throw InvalidArgumentException("Variable ", modifiers.name, " does not have arguments (requested ", modifiers.parameters,").");
                                //auto str = modifiers.toStr();
                                //print(str.c_str(), " resolves to ", ret);
                                //if(modifiers.html)
                                //    return settings::htmlify(ret);
                                return ret;
                            } catch(const std::exception& ex) {
                                if(not modifiers.optional)
                                    FormatExcept("Exception: ", ex.what(), " in variable: ", modifiers);
                                return modifiers.optional ? "" : "null";
                            }
                        }, [](bool optional) -> std::string {
                            return optional ? "" : "null";
                        });
                        
                        state._variable_values[std::string(current_word)] = resolved_word;
                    }
                    if(nesting_start_positions.empty()) {
                        output << resolved_word;
                    } else {
                        //nesting.top() += resolved_word;
                    }
                }
                
            } else if(ch == '{') {
                nesting_start_positions.push(i + 1);
            } else {
                if(nesting_start_positions.empty()) {
                    throw InvalidSyntaxException("Mismatched opening brace at position ", i);
                }
                //nesting.top() += ch;
            }
        }
    }

    if(not nesting_start_positions.empty()) {
        throw InvalidSyntaxException("Mismatched opening brace");
    }
    
    if(comment_out) {
        // Trailing backslash without a following character
        throw InvalidSyntaxException("Trailing backslash");
    }

    return output.str();
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
    ASSERT_NO_THROW(result = _parse_text("{if:{variable}:'{correct}':'{throws}'}", context, state));
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
    State state;
    Context context{
        VarFunc("variable_inner_variable", [](const VarProps&) -> std::string { return "mocked_value"; }),
        VarFunc("inner_variable", [](const VarProps&) -> std::string { return "inner"; }),
        VarFunc("variable_inner", [](const VarProps&) -> std::string { return "correct"; })
    };
    std::string result = parse_text("{variable_{inner_variable}}", context, state);
    ASSERT_EQ(result, "correct");
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
