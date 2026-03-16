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
#include <gui/types/ListItemTypes.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/StaticText.h>
#include <gui/dyn/UnresolvedStringPattern.h>   // for ResolveStringPattern tests
#include <type_traits>

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

struct JsonBackedSample {
    int x;
    file::Path name;
    bool enabled;

    glz::json_t to_json() const {
        glz::json_t json;
        json["x"] = x;
        json["name"] = cvt2json(name);
        json["enabled"] = enabled;
        return json;
    }
};

static void collect_static_text_strings(const Layout::Ptr& node, std::vector<std::string>& out) {
    if(not node) {
        return;
    }
    
    if(node.is<StaticText>()) {
        out.push_back(node.to<StaticText>()->text());
        return;
    }
    
    if(node.is<Layout>()) {
        for(const auto& child : node.to<Layout>()->objects()) {
            collect_static_text_strings(child, out);
        }
    }
}

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

TYPED_TEST(ParseAndResolveTest, SpriteMapFieldAccess)
{
    State state;
    sprite::Map map;
    glz::json_t test;
    test["value"] = 42;
    
    map["x"] = 42;
    map["name"] = std::string("trex");
    map["enabled"] = true;
    map["json"] = test;

    Context ctx{
        VarFunc("object", [&map](const VarProps&) -> sprite::Map& { return map; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.x}", ctx, state), "42");
    EXPECT_EQ(run_parser<TypeParam>("{object.name}", ctx, state), "trex");
    EXPECT_EQ(run_parser<TypeParam>("{object.enabled}", ctx, state), "true");
    EXPECT_EQ(run_parser<TypeParam>("{object.json}", ctx, state), "{\"value\":42}");
    EXPECT_EQ(run_parser<TypeParam>("{object.json.value}", ctx, state), "42");
}

TYPED_TEST(ParseAndResolveTest, JsonObjectSubfieldReplacement)
{
    State state;
    glz::json_t object;
    object["x"] = 42;
    object["name"] = std::string("trex");
    object["enabled"] = true;

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::json_t { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.x}", ctx, state), "42");
    EXPECT_EQ(run_parser<TypeParam>("{object.name}", ctx, state), "trex");
    EXPECT_EQ(run_parser<TypeParam>("{object.enabled}", ctx, state), "true");
}

TYPED_TEST(ParseAndResolveTest, JsonObjectNestedSubfieldReplacement)
{
    State state;
    glz::json_t nested;
    nested["value"] = 123;

    glz::json_t object;
    object["value"] = nested;

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::json_t { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.value.value}", ctx, state), "123");
}

TYPED_TEST(ParseAndResolveTest, NullableSpriteMaps)
{
    State state;
    Context ctx{
        VarFunc("object", [](const VarProps&) -> sprite::Map { return {}; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object}", ctx, state), "{}");
    EXPECT_EQ(run_parser<TypeParam>("{if:{object}:true:false}", ctx, state), "false");
}

TYPED_TEST(ParseAndResolveTest, NullableObjects)
{
    State state;
    Context ctx{
        VarFunc("object", [](const VarProps&) -> glz::json_t { return glz::json_t{}; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object}", ctx, state), "null");
    EXPECT_EQ(run_parser<TypeParam>("{if:{object}:true:false}", ctx, state), "false");
}

TYPED_TEST(ParseAndResolveTest, EmptyObjects)
{
    State state;
    Context ctx{
        VarFunc("object", [](const VarProps&) -> glz::json_t { return glz::json_t::object_t{}; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object}", ctx, state), "{}");
    EXPECT_EQ(run_parser<TypeParam>("{if:{object}:true:false}", ctx, state), "false");
}

TYPED_TEST(ParseAndResolveTest, CustomStructJsonSubfieldReplacement)
{
    State state;
    JsonBackedSample sample{
        .x = 7,
        .name = file::Path("/file/to/raptor"),
        .enabled = false
    };

    Context ctx{
        VarFunc("custom", [sample](const VarProps&) -> glz::json_t { return cvt2json(sample); })
    };

    EXPECT_EQ(run_parser<TypeParam>("{custom.x}", ctx, state), "7");
    EXPECT_EQ(run_parser<TypeParam>("{custom.name}", ctx, state), "/file/to/raptor");
    EXPECT_EQ(run_parser<TypeParam>("{custom.enabled}", ctx, state), "false");
}

TEST(ScopedVariableTest, DynamicScopedJsonVariableInvalidatesPreparedPatternCache)
{
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    Context ctx;
    auto pattern = cmn::pattern::UnresolvedStringPattern::prepare("{i.name}");

    {
        glz::json_t first;
        first["name"] = std::string("alpha");

        auto scope = handler->scope();
        scope.set("i", VarFunc("i", [first](const VarProps&) -> glz::json_t {
            return first;
        }).second);

        EXPECT_EQ(pattern.realize(ctx, state), "alpha");
    }

    {
        glz::json_t second;
        second["name"] = std::string("beta");

        auto scope = handler->scope();
        scope.set("i", VarFunc("i", [second](const VarProps&) -> glz::json_t {
            return second;
        }).second);

        EXPECT_EQ(pattern.realize(ctx, state), "beta");
    }
}

TEST(ScopedVariableTest, StringAndDynamicConflictsShadowGracefully)
{
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    Context ctx{
        VarFunc("item", [](const VarProps&) -> glz::json_t {
            glz::json_t value;
            value["name"] = std::string("global");
            return value;
        })
    };

    {
        auto outer = handler->scope();
        outer.set("item", VarFunc("item", [](const VarProps&) -> glz::json_t {
            glz::json_t value;
            value["name"] = std::string("outer");
            return value;
        }).second);

        EXPECT_EQ(parse_text("{item.name}", ctx, state), "outer");
        EXPECT_TRUE(ctx.has("item", state));

        auto inner = handler->scope();
        inner.set("item", "shadow");

        EXPECT_EQ(parse_text("{item}", ctx, state), "shadow");
        EXPECT_EQ(parse_text("{.item.name}", ctx, state), "");
        EXPECT_FALSE(ctx.has("item", state));

        inner.set("item", VarFunc("item", [](const VarProps&) -> glz::json_t {
            glz::json_t value;
            value["name"] = std::string("inner");
            return value;
        }).second);

        EXPECT_EQ(parse_text("{item.name}", ctx, state), "inner");
        EXPECT_TRUE(ctx.has("item", state));
    }

    EXPECT_EQ(parse_text("{item.name}", ctx, state), "global");
    EXPECT_TRUE(ctx.has("item", state));
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

TYPED_TEST(ParseAndResolveTest, NestedForKeepsOuterState)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{for:k:[10,20]:{for:j:[1,2]:[{k},{j}]}}", ctx, state);
    ASSERT_EQ(result, "[[[10,1],[10,2]],[[20,1],[20,2]]]");
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

TEST(EachElementTest, NestedEachRestoresOuterScope) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "outer",
  "do": {
    "type": "collection",
    "children": [
      {
        "type": "each",
        "var": "i.inner",
        "do": {
          "type": "stext",
          "text": "inner:{i}-{index}"
        }
      },
      {
        "type": "stext",
        "text": "outer:{i.label}-{index}"
      }
    ]
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<sprite::Map> outer_data(2);
    outer_data[0]["label"] = 10;
    outer_data[0]["inner"] = std::vector<int>{1, 2};
    outer_data[1]["label"] = 20;
    outer_data[1]["inner"] = std::vector<int>{3, 4};
    
    std::vector<std::shared_ptr<VarBase_t>> outer_entries;
    outer_entries.reserve(outer_data.size());
    for(size_t idx = 0; idx < outer_data.size(); ++idx) {
        outer_entries.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &outer_data](const VarProps&) -> sprite::Map& {
            return outer_data[idx];
        })));
    }
    
    Context context{
        VarFunc("outer", [&outer_entries](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return outer_entries;
        })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);

    ASSERT_THAT(texts, ::testing::ElementsAre(
        "inner:1-0",
        "inner:2-1",
        "outer:10-0",
        "inner:3-0",
        "inner:4-1",
        "outer:20-1"
    ));
}

TEST(EachElementTest, VectorOfJsonObjectsExpandsObjectFields) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "do": {
    "type": "collection",
    "children": [
      {
        "type": "stext",
        "text": "x:{i.x}"
      },
      {
        "type": "stext",
        "text": "name:{i.name}"
      },
      {
        "type": "stext",
        "text": "index:{index}"
      }
    ]
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<glz::json_t> items;
    {
        glz::json_t first;
        first["x"] = 42;
        first["name"] = std::string("trex");
        items.push_back(first);
    }
    {
        glz::json_t second;
        second["x"] = 7;
        second["name"] = std::string("raptor");
        items.push_back(second);
    }

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<glz::json_t> { return items; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);

    ASSERT_THAT(texts, ::testing::ElementsAre(
        "x:42",
        "name:trex",
        "index:0",
        "x:7",
        "name:raptor",
        "index:1"
    ));
}

TEST(ListElementTest, DynamicListTemplateRendersAndUpdatesItems) {
    constexpr std::string_view json = R"json(
{
  "type": "list",
  "var": "items",
  "template": {
    "text": "{i.name}",
    "detail": "{i.detail}",
    "tooltip": "{i.tooltip}",
    "disabled": "{i.disabled}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<sprite::Map> item_data(2);
    item_data[0]["name"] = std::string("alpha");
    item_data[0]["detail"] = std::string("first detail");
    item_data[0]["tooltip"] = std::string("first tooltip");
    item_data[0]["disabled"] = false;
    item_data[1]["name"] = std::string("beta");
    item_data[1]["detail"] = std::string("second detail");
    item_data[1]["tooltip"] = std::string("second tooltip");
    item_data[1]["disabled"] = true;

    std::vector<std::shared_ptr<VarBase_t>> items;
    items.reserve(item_data.size());
    for(size_t idx = 0; idx < item_data.size(); ++idx) {
        items.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &item_data](const VarProps&) -> sprite::Map& {
            return item_data[idx];
        })));
    }

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return items;
        })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<ScrollableList<DetailTooltipItem>>());

    auto list = root.to<ScrollableList<DetailTooltipItem>>();

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_EQ(list->items().size(), 2u);
    EXPECT_EQ(list->items().at(0).value().name(), "alpha");
    EXPECT_EQ(list->items().at(0).value().detail(), "first detail");
    EXPECT_EQ(list->items().at(0).value().tooltip(), "first tooltip");
    EXPECT_FALSE(list->items().at(0).value().disabled());
    EXPECT_EQ(list->items().at(1).value().name(), "beta");
    EXPECT_EQ(list->items().at(1).value().detail(), "second detail");
    EXPECT_EQ(list->items().at(1).value().tooltip(), "second tooltip");
    EXPECT_TRUE(list->items().at(1).value().disabled());

    item_data[0]["name"] = std::string("alpha-updated");
    item_data[1]["detail"] = std::string("second detail updated");
    item_data[1]["disabled"] = false;

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_EQ(list->items().size(), 2u);
    EXPECT_EQ(list->items().at(0).value().name(), "alpha-updated");
    EXPECT_EQ(list->items().at(0).value().detail(), "first detail");
    EXPECT_EQ(list->items().at(1).value().name(), "beta");
    EXPECT_EQ(list->items().at(1).value().detail(), "second detail updated");
    EXPECT_FALSE(list->items().at(1).value().disabled());
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
