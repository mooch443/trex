#include <commons.pc.h>
#include <gtest/gtest.h>
#include <gui/DynamicGUI.h>
#include <gui/ListAttributes.h>
#include <gui/dyn/Action.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/ResolveVariable.h>
#include <gui/dyn/UnresolvedStringPattern.h>
#include <misc/GlobalSettings.h>

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

struct JsonBackedSample {
    int x;
    file::Path name;
    bool enabled;

    glz::generic to_json() const {
        glz::generic json;
        json["x"] = x;
        json["name"] = cvt2json(name);
        json["enabled"] = enabled;
        return json;
    }
};

template<typename Fn>
static std::pair<std::string, std::string> capture_stdout(Fn&& fn) {
    testing::internal::CaptureStdout();
    try {
        auto result = fn();
        return {std::move(result), testing::internal::GetCapturedStdout()};
    } catch(...) {
        (void)testing::internal::GetCapturedStdout();
        throw;
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

TYPED_TEST(ParseAndResolveTest, NestedBooleanConditionsAndWarningsStayIndependent) {
    constexpr std::string_view json = R"json(
{
  "objects": [
    {
      "type": "condition",
      "var": "{&&:{valid_resource}:{input_available}:{not:{busy}}}",
      "then": { "type": "button", "text": "Build", "action": "build" },
      "else": {
        "type": "button",
        "text": "Unavailable",
        "tooltip": "{if:{||:{busy}:{input_available}}:'':'missing-input;'}{if:{&&:{not:{busy}}:{requires_resource}:{not:{valid_resource}}}:missing-resource {global.resource};:''}"
      }
    },
    {
      "type": "condition",
      "var": "{&&:{not:{busy}}:{has_result}}",
      "then": { "type": "button", "text": "Reuse", "action": "reuse" }
    }
  ]
}
)json";

    glz::generic layout;
    const auto parse_error = glz::read_json(layout, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    const auto& objects = layout.get_object().at("objects").get_array();
    const auto& build = objects.at(0).get_object().at("var").get_string();
    const auto& reuse = objects.at(1).get_object().at("var").get_string();
    const auto& tooltip = objects.at(0).get_object().at("else").get_object().at("tooltip").get_string();

    for(bool requires_resource : {false, true}) {
        for(bool resource_available : {false, true}) {
            for(bool input_available : {false, true}) {
                for(bool busy : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "requires_resource=" << requires_resource
                                 << " resource_available=" << resource_available
                                 << " input_available=" << input_available
                                 << " busy=" << busy);
                    State state;
                    sprite::Map global;
                    global["resource"] = file::Path("resource.dat");
                    Context context{
                        VarFunc("global", [&](const VarProps&) -> sprite::Map& { return global; }),
                        VarFunc("requires_resource", [&](const VarProps&) { return requires_resource; }),
                        VarFunc("valid_resource", [&](const VarProps&) { return not requires_resource || resource_available; }),
                        VarFunc("input_available", [&](const VarProps&) { return input_available; }),
                        VarFunc("has_result", [](const VarProps&) { return true; }),
                        VarFunc("busy", [&](const VarProps&) { return busy; })
                    };
                    EXPECT_EQ(run_parser<TypeParam>(build, context, state),
                              Meta::toStr(input_available && (not requires_resource || resource_available) && not busy));
                    EXPECT_EQ(run_parser<TypeParam>(reuse, context, state), Meta::toStr(not busy));
                    std::string expected_warning;
                    if(not input_available && not busy)
                        expected_warning += "missing-input;";
                    if(requires_resource && not resource_available && not busy)
                        expected_warning += "missing-resource resource.dat;";
                    EXPECT_EQ(run_parser<TypeParam>(tooltip, context, state), expected_warning);
                }
            }
        }
    }
}

TEST(ConvertToBoolTest, CStringPointerComparesContents) {
    const std::string value = "false";
    const char* pointer = value.c_str();

    EXPECT_FALSE(convert_to_bool(pointer));
}

TEST(ConvertToBoolTest, NullTerminatedArrayIgnoresUnusedCapacity) {
    char buffer[32] = "false";
    const char* pointer = buffer;

    EXPECT_FALSE(convert_to_bool(pointer));
    EXPECT_EQ(utils::string_like_view(buffer).size(), 5u);
    EXPECT_FALSE(convert_to_bool(buffer));
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

TYPED_TEST(ParseAndResolveTest, CollectionHelpersPreserveStringsAndRejectEmptyMaximum)
{
    State state;
    std::map<std::string, std::string> classes{{"fish", "before"}};
    const std::string label = R"(fish, juvenile: "spotted")";
    Context ctx{
        VarFunc("classes", [&](const VarProps&) { return classes; }),
        VarFunc("label", [&](const VarProps&) { return label; })
    };

    for(const auto* key : {"fish", "ray, spotted"}) {
        SCOPED_TRACE(key);
        const auto expression = std::string("{insert:") + key + ":{label}:{classes}}";
        const auto result = run_parser<TypeParam>(expression, ctx, state);
        classes = Meta::fromStr<decltype(classes)>(result);
        EXPECT_EQ(classes.at(key), label);
    }
    ASSERT_EQ(classes.size(), 2u);

    const auto erased = run_parser<TypeParam>("{erase:fish:{classes}}", ctx, state);
    classes = Meta::fromStr<decltype(classes)>(erased);
    ASSERT_EQ(classes.size(), 1u);
    EXPECT_EQ(classes.at("ray, spotted"), label);

    EXPECT_EQ(run_parser<TypeParam>("{max_element:[-3,2,1]}", ctx, state), "2");
    EXPECT_THROW(run_parser<TypeParam>("{max_element:[]}", ctx, state), std::exception);
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

TYPED_TEST(ParseAndResolveTest, UnknownFunctionInConditionalRaises)
{
    constexpr std::string_view json = R"json(
{
  "locals": {
    "tab": { "type": "int", "value": 0 }
  },
  "objects": []
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    (void)objects;
    Context context;
    context.apply_local_settings(defaults);
    State state;

    EXPECT_THROW(
        run_parser<TypeParam>(
            "{if:{equals:{local.tab}:0}:export:export-behavior}",
            context,
            state),
        std::exception);
}

TYPED_TEST(ParseAndResolveTest, InlineIfRaisesForMissingRequiredTerms)
{
    constexpr std::array cases{
        std::string_view{"{if:{not:{missing}}:yes:no}"},
        std::string_view{"{if:{||:{missing}:true}:yes:no}"},
        std::string_view{"{if:{&&:{missing}:true}:yes:no}"},
        std::string_view{"{if:{equal:{missing}:null}:yes:no}"}
    };

    for(const auto expression : cases) {
        SCOPED_TRACE(expression);
        State state;
        Context context;
        EXPECT_THROW(
            run_parser<TypeParam>(std::string(expression), context, state),
            std::exception);
    }
}

TYPED_TEST(ParseAndResolveTest, OptionalMissingInlineConditionIsSilent)
{
    State state;
    Context context;
    auto [result, diagnostics] = capture_stdout([&]() {
        return run_parser<TypeParam>(
            "{if:{.missing_optional}:yes:no}",
            context,
            state);
    });

    EXPECT_EQ(result, "no");
    EXPECT_EQ(diagnostics.find("missing_optional"), std::string::npos);
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
    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:{correct}:{throws}}", ctx, state));
    ASSERT_EQ(result, "c");
    ASSERT_FALSE(ran);
}

TYPED_TEST(ParseAndResolveTest, QuotedVariableNameHandling)
{
    bool variable_value = true;
    State    state;
    Context  ctx{
        VarFunc("variable", [&](const VarProps&) -> bool { return variable_value; })
    };
    std::string result;
    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:{variable}:'{variable}'}", ctx, state));
    ASSERT_EQ(result, "true");

    variable_value = false;

    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:{variable}:'{variable}'}", ctx, state));
    ASSERT_EQ(result, "{variable}");
}

TYPED_TEST(ParseAndResolveTest, QuotedExpressionPolicy)
{
    bool ran = false;
    State state;
    Context ctx{
        VarFunc("value", [](const VarProps&) -> std::string { return "a,b"; }),
        VarFunc("unreachable", [&](const VarProps&) -> std::string {
            ran = true;
            throw std::invalid_argument("Not supposed to run.");
        })
    };

    EXPECT_EQ(run_parser<TypeParam>("prefix '{value}' suffix", ctx, state),
              "prefix {value} suffix");
    EXPECT_EQ(run_parser<TypeParam>("['a,b','c']", ctx, state), "['a,b','c']");
    EXPECT_EQ(run_parser<TypeParam>("['{value}','other']", ctx, state),
              "['{value}','other']");
    EXPECT_EQ(run_parser<TypeParam>(R"('{\'key\':\'a,b\'}')", ctx, state), "{'key':'a,b'}");
    EXPECT_EQ(run_parser<TypeParam>("{at:0:['{value}','other']}", ctx, state),
              "'{value}'");
    EXPECT_EQ(run_parser<TypeParam>(R"({array_length:['can\'t','other']})", ctx, state), "2");
    EXPECT_EQ(run_parser<TypeParam>("{if:false:'{unreachable}':'ok'}", ctx, state), "ok");
    EXPECT_FALSE(ran);

    EXPECT_EQ(run_parser<TypeParam>("{if:true:'prefix: '{if:true:{value}:'unused'}:no}", ctx, state),
              "prefix: a,b");
}

TEST(DynamicGUIActions, QuotedExpressionPolicy)
{
    State state;
    Context ctx{
        VarFunc("value", [](const VarProps&) -> std::string { return "resolved"; })
    };

    const auto action = PreAction::fromStr("consume:'prefix: {value}'").parse(ctx, state);
    EXPECT_EQ(action.name, "consume");
    ASSERT_EQ(action.parameters.size(), 1u);
    EXPECT_EQ(action.parameters.front(),
              "prefix: {value}");

    EXPECT_EQ(PreAction::fromStr("consume:'prefix: '{value}").parse(ctx, state).parameters,
              (std::vector<std::string>{"prefix: resolved"}));

    const auto generated = PreAction::fromStr("{if:true:'consume:{value}':'unused'}").parse(ctx, state);
    EXPECT_EQ(generated.name, "consume");
    ASSERT_EQ(generated.parameters.size(), 1u);
    EXPECT_EQ(generated.parameters.front(), "{value}");
}

TYPED_TEST(ParseAndResolveTest, EscapesColons)
{
    State    state;
    Context  ctx{
        VarFunc("variable", [&](const VarProps&) -> int { return 1; })
    };
    std::string result;
    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:{variable}':':'0'}", ctx, state));
    ASSERT_EQ(result, "1:");

    ASSERT_NO_THROW(result = run_parser<TypeParam>("{if:{variable}:':'{variable}:'0'}", ctx, state));
    ASSERT_EQ(result, ":1");
}

TYPED_TEST(ParseAndResolveTest, ConditionalAndLogicalExpressionsEvaluateLazily)
{
    struct TestCase {
        std::string_view expression;
        std::string_view expected;
        size_t expected_probes;
    };

    constexpr std::array cases{
        TestCase{"{if:{probe:condition:true}:yes:{unreachable}}", "yes", 1},
        TestCase{"{if:{probe:condition:false}:{unreachable}:no}", "no", 1},
        TestCase{"{if:{probe:condition:true}:yes}", "yes", 1},
        TestCase{"{if:{probe:condition:false}:{unreachable}}", "", 1},
        TestCase{"{&&:{probe:first:false}:{unreachable}}", "false", 1},
        TestCase{"{&&:{probe:first:true}:{probe:second:true}}", "true", 2},
        TestCase{"{&&:{probe:first:true}:{probe:second:false}:{unreachable}}", "false", 2},
        TestCase{"{||:{probe:first:true}:{unreachable}}", "true", 1},
        TestCase{"{||:{probe:first:false}:{probe:second:false}}", "false", 2},
        TestCase{"{||:{probe:first:false}:{probe:second:true}:{unreachable}}", "true", 2},
        TestCase{"{&&:{not:{probe:first:true}}:{unreachable}}", "false", 1},
        TestCase{"{||:{not:{probe:first:false}}:{unreachable}}", "true", 1},
        TestCase{"{&&:{||:{probe:first:true}:{unreachable}}:{||:{probe:second:false}:{probe:third:true}}}", "true", 3},
        TestCase{"{||:{&&:{probe:first:false}:{unreachable}}:{&&:{probe:second:true}:{probe:third:true}}}", "true", 3}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.expression);

        State state;
        size_t probes = 0;
        bool unreachable_ran = false;
        Context ctx{
            VarFunc("probe", [&](const VarProps& props) -> bool {
                ++probes;
                return convert_to_bool(props.last());
            }),
            VarFunc("unreachable", [&](const VarProps&) -> bool {
                unreachable_ran = true;
                throw std::invalid_argument("Expression should have been short-circuited.");
            })
        };

        EXPECT_EQ(run_parser<TypeParam>(std::string(test.expression), ctx, state), std::string(test.expected));
        EXPECT_EQ(probes, test.expected_probes);
        EXPECT_FALSE(unreachable_ran);
    }
}

TYPED_TEST(ParseAndResolveTest, MissingRequiredVariableRaises)
{
    State   state;
    Context ctx;
    ASSERT_THROW(run_parser<TypeParam>("{missing_variable}", ctx, state), std::exception);

    EXPECT_EQ(run_parser<TypeParam>("{.missing_variable}", ctx, state), "");
}

TYPED_TEST(ParseAndResolveTest, EscapeCharacters)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("'{variable}'", ctx, state);
    ASSERT_EQ(result, "{variable}");
}

TYPED_TEST(ParseAndResolveTest, InsertedValuesAreNotReparsed)
{
    State state;
    Context ctx{
        VarFunc("name", [](const VarProps&) -> std::string { return "Ada"; }),
        VarFunc("label", [](const VarProps&) -> std::string { return "{name}"; })
    };
    EXPECT_EQ(run_parser<TypeParam>("Stored: {label}", ctx, state), "Stored: {name}");
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
    glz::generic test;
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

TYPED_TEST(ParseAndResolveTest, JsonSubArrayTest)
{
    State state;
    glz::generic object= cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::generic { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object}", ctx, state), "[1,2,3]");
    EXPECT_EQ(run_parser<TypeParam>("{object.0}", ctx, state), "1");
}

TYPED_TEST(ParseAndResolveTest, CanParseComplexString)
{
    State state;
    Context ctx{
        VarFunc("mouse_in_bowl", [](const VarProps&) -> Vec2 { return Vec2(123,456); }),
        VarFunc("video_size", [](const VarProps&) -> Vec2 { return Vec2(1024,2048); })
    };

    EXPECT_EQ(run_parser<TypeParam>("{if:{&&:{>=:{mouse_in_bowl.x}:0}:{<:{mouse_in_bowl.x}:{video_size.x}}:{>=:{mouse_in_bowl.y}:0}:{<:{mouse_in_bowl.y}:{video_size.y}}}:[255,255,255,255]:[200,120,80,100]}", ctx, state), "[255,255,255,255]");
}

TYPED_TEST(ParseAndResolveTest, CanParseComplexStringWithoutTypes)
{
    State state;
    Context ctx{
        VarFunc("mouse_in_bowl", [](const VarProps&) -> std::string { return Meta::toStr(Vec2(123,456)); }),
        VarFunc("video_size", [](const VarProps&) -> Vec2 { return Vec2(1024,2048); })
    };

    EXPECT_EQ(run_parser<TypeParam>("{if:{&&:{>=:{mouse_in_bowl.x}:0}:{<:{mouse_in_bowl.x}:{video_size.x}}:{>=:{mouse_in_bowl.y}:0}:{<:{mouse_in_bowl.y}:{video_size.y}}}:[255,255,255,255]:[200,120,80,100]}", ctx, state), "[255,255,255,255]");
}

TYPED_TEST(ParseAndResolveTest, CanParseComplexStringWithoutTypesInteger)
{
    State state;
    Context ctx{
        VarFunc("mouse_in_bowl", [](const VarProps&) -> std::string { return Meta::toStr(Vec2(123,456)); }),
        VarFunc("video_size", [](const VarProps&) -> Vec2 { return Vec2(1024,2048); })
    };

    EXPECT_EQ(run_parser<TypeParam>("{if:{&&:{>=:{mouse_in_bowl.0}:0}:{<:{mouse_in_bowl.0}:{video_size.x}}:{>=:{mouse_in_bowl.1}:0}:{<:{mouse_in_bowl.1}:{video_size.y}}}:[255,255,255,255]:[200,120,80,100]}", ctx, state), "[255,255,255,255]");
}

TYPED_TEST(ParseAndResolveTest, CanParseGlobalVariable)
{
    State state;
    sprite::Map map;
    map["test"] = Size2(1024,768);

    Context ctx{
        VarFunc("mouse_in_bowl", [](const VarProps&) -> std::string { return Meta::toStr(Vec2(123,456)); }),
        VarFunc("global", [&map](const VarProps&) -> const sprite::Map& { return map; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{global.test}", ctx, state), "[1024,768]");
    EXPECT_EQ(run_parser<TypeParam>("<h5><sym>🖰</sym></h5> <i>{round:{mouse_in_bowl.x}},{round:{mouse_in_bowl.y}}</i>", ctx, state), "<h5><sym>🖰</sym></h5> <i>123,456</i>");
}


TYPED_TEST(ParseAndResolveTest, JsonSubSubArrayTest)
{
    State state;
    glz::generic object;
    object["array"] = cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::generic { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.array}", ctx, state), "[1,2,3]");
    EXPECT_EQ(run_parser<TypeParam>("{object.array.0}", ctx, state), "1");
}

TYPED_TEST(ParseAndResolveTest, JsonDynamicSubSubArrayTest)
{
    State state;
    glz::generic object;
    object["array"] = cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("index", [](const VarProps&){ return 1; }),
        VarFunc("object", [&object](const VarProps&) -> glz::generic { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{index}", ctx, state), "1");
    EXPECT_EQ(run_parser<TypeParam>("{object.array}", ctx, state), "[1,2,3]");
    EXPECT_EQ(run_parser<TypeParam>("{object.array.{index}}", ctx, state), "2");
}

TYPED_TEST(ParseAndResolveTest, SpriteSubArrayTest)
{
    State state;
    sprite::Map map;
    map["object"] = std::vector<int>{1,2,3};

    Context ctx{
        VarFunc("map", [&map](const VarProps&) -> const sprite::Map& { return map; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{map.object}", ctx, state), "[1,2,3]");
    EXPECT_EQ(run_parser<TypeParam>("{map.object.0}", ctx, state), "1");
}

TYPED_TEST(ParseAndResolveTest, JsonObjectSubfieldReplacement)
{
    State state;
    glz::generic object;
    object["x"] = 42;
    object["name"] = std::string("trex");
    object["enabled"] = true;

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::generic { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.x}", ctx, state), "42");
    EXPECT_EQ(run_parser<TypeParam>("{object.name}", ctx, state), "trex");
    EXPECT_EQ(run_parser<TypeParam>("{object.enabled}", ctx, state), "true");
}

TYPED_TEST(ParseAndResolveTest, JsonObjectNestedSubfieldReplacement)
{
    State state;
    glz::generic nested;
    nested["value"] = 123;

    glz::generic object;
    object["value"] = nested;

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::generic { return object; })
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
        VarFunc("object", [](const VarProps&) -> glz::generic { return glz::generic{}; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object}", ctx, state), "null");
    EXPECT_EQ(run_parser<TypeParam>("{if:{object}:true:false}", ctx, state), "false");
}

TYPED_TEST(ParseAndResolveTest, EmptyObjects)
{
    State state;
    Context ctx{
        VarFunc("object", [](const VarProps&) -> glz::generic { return glz::generic::object_t{}; })
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
        VarFunc("custom", [sample](const VarProps&) -> glz::generic {
            return cvt2json(sample);
        })
    };

    EXPECT_EQ(run_parser<TypeParam>("{custom.x}", ctx, state), "7");
    EXPECT_EQ(run_parser<TypeParam>("{custom.name}", ctx, state), file::Path("/file/to/raptor").str());
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
        glz::generic first;
        first["name"] = std::string("alpha");

        auto scope = handler->scope();
        scope.set("i", VarFunc("i", [first](const VarProps&) -> glz::generic {
            return first;
        }).second);

        EXPECT_EQ(pattern.realize(ctx, state), "alpha");
    }

    {
        glz::generic second;
        second["name"] = std::string("beta");

        auto scope = handler->scope();
        scope.set("i", VarFunc("i", [second](const VarProps&) -> glz::generic {
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
        VarFunc("item", [](const VarProps&) -> glz::generic {
            glz::generic value;
            value["name"] = std::string("global");
            return value;
        })
    };

    {
        auto outer = handler->scope();
        outer.set("item", VarFunc("item", [](const VarProps&) -> glz::generic {
            glz::generic value;
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

        inner.set("item", VarFunc("item", [](const VarProps&) -> glz::generic {
            glz::generic value;
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

TYPED_TEST(ParseAndResolveTest, VariableExceptionPropagates)
{
    State   state;
    Context ctx{
        VarFunc("exception_var", [](const VarProps&) -> std::string {
            throw std::runtime_error("An exception"); // NOLINT
        })
    };
    ASSERT_THROW(run_parser<TypeParam>("{exception_var}", ctx, state), std::exception);
}

TYPED_TEST(ParseAndResolveTest, NestedForKeepsOuterState)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{for:k:[10,20]:{for:j:[1,2]:[{k},{j}]}}", ctx, state);
    ASSERT_EQ(result, "[[[10,1],[10,2]],[[20,1],[20,2]]]");
    EXPECT_EQ(run_parser<TypeParam>("{for:n:[1,2,3]:{*:2:{n}}}", ctx, state), "[2,4,6]");
}

TYPED_TEST(ParseAndResolveTest, RepeatedForKeepsArrayBrackets)
{
    State state;
    Context ctx;
    EXPECT_EQ(run_parser<TypeParam>("{for:i:[1,2]:{i}}|{for:i:[1,2]:{i}}", ctx, state), "[1,2]|[1,2]");
    EXPECT_EQ(run_parser<TypeParam>("{for:i:[]:{i}}|{for:i:[]:{i}}", ctx, state), "[]|[]");
}

TYPED_TEST(ParseAndResolveTest, AtIndexesArray)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:0:[alpha,beta,gamma]}", ctx, state);
    ASSERT_EQ(result, "alpha");
}

TYPED_TEST(ParseAndResolveTest, AtIndexesString)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:1:'abc'}", ctx, state);
    ASSERT_EQ(result, "b");
}

TYPED_TEST(ParseAndResolveTest, AtLooksUpObjectValueByKey)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:'{key:value,other:ignored}'}", ctx, state);
    ASSERT_EQ(result, "value");
}

TYPED_TEST(ParseAndResolveTest, AtLooksUpObjectValueByKeyQuoted)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:'{key:value,other:ignored}'}", ctx, state);
    ASSERT_EQ(result, "value");
}

TYPED_TEST(ParseAndResolveTest, FindsSingleLiteralBrace)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:0:'{test'}", ctx, state);
    ASSERT_EQ(result, "{");

}

TYPED_TEST(ParseAndResolveTest, FindsSingleLiteralBrace2)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:0:(test}", ctx, state);
    ASSERT_EQ(result, "(");

}

TYPED_TEST(ParseAndResolveTest, MaintainsEscapedBracePrefixes)
{
    State   state;
    Context ctx;

    EXPECT_NO_THROW({
        EXPECT_EQ(run_parser<TypeParam>("prefix'{y}'", ctx, state),
                  "prefix{y}");
    });

    EXPECT_NO_THROW({
        EXPECT_EQ(run_parser<TypeParam>("{if:true:'}':no}", ctx, state),
                  "}");
    });

    EXPECT_NO_THROW({
        EXPECT_EQ(run_parser<TypeParam>("{if:true:{if:true:'}':no}:no}", ctx, state),
                  "}");
    });

    EXPECT_NO_THROW({
        EXPECT_EQ(run_parser<TypeParam>("'can\\'t'", ctx, state),
                  "can't");
    });
    EXPECT_NO_THROW({
        EXPECT_EQ(run_parser<TypeParam>("{if:false:'can\\'t':no}", ctx, state),
                  "no");
    });
}

TYPED_TEST(ParseAndResolveTest, ArrayLengthTestQuotes)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{array_length:'abc'}", ctx, state);
    ASSERT_EQ(result, "3");
    result = run_parser<TypeParam>("{array_length:'  '}", ctx, state);
    ASSERT_EQ(result, "2");
}

TYPED_TEST(ParseAndResolveTest, ArrayLengthTestNoQuotes)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{array_length:abc}", ctx, state);
    ASSERT_EQ(result, "3");
}

TYPED_TEST(ParseAndResolveTest, ArrayLengthTestNumbers)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{array_length:[1,2,3]}", ctx, state);
    ASSERT_EQ(result, "3");
}

TYPED_TEST(ParseAndResolveTest, ArrayLengthTestQuotedNumbers)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{array_length:['1','2','3']}", ctx, state);
    ASSERT_EQ(result, "3");
}

TYPED_TEST(ParseAndResolveTest, QuotedStringEdgeCases)
{
    struct TestCase {
        const char* name;
        const char* pattern;
        const char* expected;
    };

    const TestCase cases[] = {
        {"quoted_value_length", "{array_length:{quoted_text}}", "3"},
        {"quoted_value_length_preserves_whitespace", "{array_length:{quoted_padded_text}}", "3"},
        {"quoted_array_item_with_comma", "{array_length:['a,b','c']}", "2"},
        {"bracket_inside_quoted_array_item", "{array_length:['[','a,b']}", "2"},
        {"quoted_map_item_with_comma", R"({at:key:'{\'key\':\'a,b\'}'})", "'a,b'"},
        {"quoted_for_item_with_comma", "{array_length:{for:i:['a,b']:1}}", "1"},
        {"quoted_false_condition", "{if:'false':yes:no}", "no"},
        {"quoted_false_logical_operand", "{&&:'false':true}", "false"},
        {"empty_value_equals_quoted_empty", "{equal:{empty_text}:''}", "true"},
        {"empty_value_not_unequal_to_quoted_empty", "{nequal:{empty_text}:''}", "false"},
        {"quoted_whitespace_index", "{at:0:' '}", " "},
        {"escaped_quote_in_skipped_branch", R"({if:false:'can\'t':no})", "no"},
        {"escape_before_non_quote_in_skipped_branch", R"({if:false:'a\b':no})", "no"}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.name);
        SCOPED_TRACE(test.pattern);
        State state;
        Context ctx{
            VarFunc("quoted_text", [](const VarProps&) -> std::string { return "'abc'"; }),
            VarFunc("quoted_padded_text", [](const VarProps&) -> std::string { return "' a '"; }),
            VarFunc("empty_text", [](const VarProps&) -> std::string { return {}; })
        };
        EXPECT_NO_THROW({
            EXPECT_EQ(run_parser<TypeParam>(test.pattern, ctx, state), test.expected);
        });
    }
}

TYPED_TEST(ParseAndResolveTest, AtLooksUpObjectValueWithWhitespace)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:'{ other: ignored, key: value }'}", ctx, state);
    EXPECT_EQ(result, "value");

    EXPECT_THROW({
        (void)run_parser<TypeParam>("{at:2:'ab'}", ctx, state);
    }, std::exception);
    EXPECT_THROW({
        (void)run_parser<TypeParam>("{at:0:''}", ctx, state);
    }, std::exception);
    EXPECT_THROW({
        (void)run_parser<TypeParam>("{at:0:}", ctx, state);
    }, std::exception);
}

TYPED_TEST(ParseAndResolveTest, AtRaisesForMissingObjectKey)
{
    State   state;
    Context ctx;
    ASSERT_THROW(
        run_parser<TypeParam>("{at:missing:'{key:value,other:ignored}'}", ctx, state),
        std::exception);
}

/*TYPED_TEST(ParseAndResolveTest, AtReturnsNullForMalformedObjectEntry)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:'{key:value,malformed}'}", ctx, state);
    ASSERT_EQ(result, "null");
}*/

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

TYPED_TEST(ParseAndResolveTest, InvalidNestedOperation)
{
    State   state;
    Context ctx{
        VarFunc("frame",       [](const VarProps&) -> int  { return 5; }),
        VarFunc("video_length",[](const VarProps&) -> int  { return 50; }),
        VarFunc("window_size", [](const VarProps&) -> Size2{ return Size2(100, 20); })
    };
    ASSERT_THROW(
        run_parser<TypeParam>("{*: {+: {invalid}:{video_length}}: {/: {window_size.w} : {video_length}}}", ctx, state),
        std::exception);
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
    ASSERT_THROW(run_parser<TypeParam>(pattern, ctx, state), std::exception);
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

    /// RESOLVE parses the syntax first before finding the variable is not present in a second step, so it throws a InvalidSyntaxException. `parse_text` evaluates it directly and finds the variable content is missing first => InvalidArgumentException.
    if constexpr(std::same_as<TypeParam, ParseTextTag>) {
        ASSERT_THROW(run_parser<TypeParam>("{variable}}", ctx, state), std::invalid_argument);
    } else {
        ASSERT_THROW(run_parser<TypeParam>("{variable}}", ctx, state), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// Escape‑sequence handling
// ---------------------------------------------------------------------------

TYPED_TEST(ParseAndResolveTest, InvalidEscapeSequenceThrows)
{
    State   state;
    Context ctx;

    // Backslashes are literal parts of these missing variable names.
    EXPECT_THROW(run_parser<TypeParam>("{\\}", ctx, state), std::runtime_error);

    EXPECT_THROW(run_parser<TypeParam>("\\{invalid\\_escape}", ctx, state), std::invalid_argument);
    EXPECT_THROW(run_parser<TypeParam>(R"('\uZZZZ')", ctx, state), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// An unquoted trailing backslash is data; an escaped closing quote leaves a literal unfinished.
// ---------------------------------------------------------------------------
TYPED_TEST(ParseAndResolveTest, TrailingBackslashThrows)
{
    State   state;
    Context ctx{
        VarFunc("variable", [](const VarProps&) -> std::string { return "x"; })
    };
    EXPECT_EQ(run_parser<TypeParam>("{variable}\\", ctx, state), "x\\");
    ASSERT_THROW(run_parser<TypeParam>(R"('text\')", ctx, state), std::exception);
}

TYPED_TEST(ParseAndResolveTest, EmptyIfClause)
{
    State   state;
    Context ctx{
        VarFunc("variable", [](const VarProps&) -> bool { return "x"; })
    };
    auto result = run_parser<TypeParam>("{if:{variable}:hi:}", ctx, state);
    EXPECT_EQ(result, "hi");

    result = run_parser<TypeParam>("{if:{variable}::hi}", ctx, state);
    EXPECT_EQ(result, "");

    result = run_parser<TypeParam>("{if:{not:{variable}}:hi:}", ctx, state);
    EXPECT_EQ(result, "");
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
    EXPECT_EQ(run_parser<TypeParam>("'{}'", ctx, state), "{}");
}

TEST(DefaultVariablesTest, LoadedDefaultExpressionThrowsForDirectSelfReference)
{
    constexpr std::string_view json = R"json(
{
  "defaults": {
    "vars": {
      "foo": "{foo}"
    }
  },
  "objects": []
}
)json";

    try {
        (void)load(std::string(json));
        FAIL() << "Expected recursive DynamicGUI default variable definition to throw.";
    } catch(const std::exception& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("recursive"), std::string::npos);
        EXPECT_NE(message.find("foo"), std::string::npos);
    }
}

TEST(DefaultVariablesTest, LoadedDefaultExpressionThrowsForIndirectCycle)
{
    constexpr std::string_view json = R"json(
{
  "defaults": {
    "vars": {
      "foo": "{bar}",
      "bar": "{foo}"
    }
  },
  "objects": []
}
)json";

    try {
        (void)load(std::string(json));
        FAIL() << "Expected recursive DynamicGUI default variable definition to throw.";
    } catch(const std::exception& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("recursive"), std::string::npos);
        EXPECT_NE(message.find("foo"), std::string::npos);
        EXPECT_NE(message.find("bar"), std::string::npos);
    }
}

namespace cmn::gui {

void PrintTo(const CornerFlags& flags, std::ostream* os) {
    *os << flags.radius << ", 0x" << hex(uint32_t(flags.mask)).toStr();
}

void PrintTo(const LabelCornerFlags& flags, std::ostream* os) {
    *os << flags.radius << ", 0x" << hex(uint32_t(flags.mask)).toStr();
}

}

namespace {
    struct LabelCornerFlagsRoundTripCase {
        const char* input;
        LabelCornerFlags expected;
        const char* canonical;
    };

void PrintTo(const LabelCornerFlagsRoundTripCase& c, std::ostream* os) {
    *os << "input=" << c.input << " expected=" << c.expected.radius << ",0x" << hex(uint32_t(c.expected.mask)).toStr() << " canonical=" << c.canonical;
}

    struct LabelCornerFlagsToStrCase {
        LabelCornerFlags flags;
        const char* canonical;
    };

    const CornerFlags& as_corner_flags(const LabelCornerFlags& flags) {
        return flags;
    }

}

class LabelCornerFlagsMetaRoundTrip
    : public ::testing::TestWithParam<LabelCornerFlagsRoundTripCase> {};

class LabelCornerFlagsMetaToStr
    : public ::testing::TestWithParam<LabelCornerFlagsToStrCase> {};

TEST_P(LabelCornerFlagsMetaRoundTrip, FromStrToStrFullCycle) {
    const auto& params = GetParam();

    const auto parsed = Meta::fromStr<LabelCornerFlags>(params.input);
    EXPECT_EQ(as_corner_flags(parsed), as_corner_flags(params.expected));

    const std::string serialized = Meta::toStr(parsed);
    EXPECT_EQ(serialized, params.canonical);

    const auto reparsed = Meta::fromStr<LabelCornerFlags>(serialized);
    EXPECT_EQ(as_corner_flags(reparsed), as_corner_flags(params.expected));
    EXPECT_EQ(Meta::toStr(reparsed), serialized);
}

TEST_P(LabelCornerFlagsMetaToStr, ConvertsObjectToExpectedString) {
    const auto& params = GetParam();

    EXPECT_EQ(Meta::toStr(params.flags), params.canonical);
}

INSTANTIATE_TEST_SUITE_P(
    VariousConfigs,
    LabelCornerFlagsMetaRoundTrip,
    ::testing::Values(
        LabelCornerFlagsRoundTripCase{"['none',0]", LabelCornerFlags::Square(), "['none']"},
        LabelCornerFlagsRoundTripCase{"8.5", LabelCornerFlags::Rounded(8.5f), "[8.5]"},
        LabelCornerFlagsRoundTripCase{"[3.25]", LabelCornerFlags::Rounded(3.25f), "[3.25]"},
        LabelCornerFlagsRoundTripCase{"['left',4]", LabelCornerFlags::Left(4.0f), "['left',4]"},
        LabelCornerFlagsRoundTripCase{"['RIGHT',7.25]", LabelCornerFlags::Right(7.25f), "['right',7.25]"},
        LabelCornerFlagsRoundTripCase{"['top',2]", LabelCornerFlags::Top(2.0f), "['top',2]"},
        LabelCornerFlagsRoundTripCase{"['bottom',6]", LabelCornerFlags::Bottom(6.0f), "['bottom',6]"},
        LabelCornerFlagsRoundTripCase{"['tl','br',1.5]", LabelCornerFlags(true, false, true, false, 1.5f), "['tl','br',1.5]"},
        LabelCornerFlagsRoundTripCase{"['tr','bl',9]", LabelCornerFlags(false, true, false, true, 9.0f), "['tr','bl',9]"},
        LabelCornerFlagsRoundTripCase{"['all',12]", LabelCornerFlags::Rounded(12.0f), "[12]"}));

INSTANTIATE_TEST_SUITE_P(
    VariousConfigs,
    LabelCornerFlagsMetaToStr,
    ::testing::Values(
        LabelCornerFlagsToStrCase{LabelCornerFlags::Rounded(8.5f), "[8.5]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags::Left(4.0f), "['left',4]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags::Right(7.25f), "['right',7.25]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags::Top(2.0f), "['top',2]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags::Bottom(6.0f), "['bottom',6]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags(true, false, true, false, 1.5f), "['tl','br',1.5]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags(false, true, false, true, 9.0f), "['tr','bl',9]"},
        LabelCornerFlagsToStrCase{LabelCornerFlags::Square(), "['none']"}));

TEST(PointerEventsTest, MetaSerializationRoundTripsEveryMask) {
    using pointer::Events;

    EXPECT_EQ(Events::None.toStr(), "none");
    EXPECT_EQ(Events::Hover.toStr(), "hover");
    EXPECT_EQ(Events::Click.toStr(), "click");
    EXPECT_EQ(Events::Drag.toStr(), "drag");
    EXPECT_EQ(Events::Scroll.toStr(), "scroll");
    EXPECT_EQ(Events::All.toStr(), "all");
    EXPECT_EQ((Events::Hover | Events::Drag).toStr(), "[hover,drag]");
    EXPECT_EQ((Events::Click | Events::Scroll).toStr(), "[click,scroll]");

    for(uint8_t mask = 0; mask <= 0x0F; ++mask) {
        const Events original{mask};
        const auto serialized = Meta::toStr(original);
        SCOPED_TRACE("mask=" + Meta::toStr(mask) + " serialized=" + serialized);
        EXPECT_EQ(Meta::fromStr<Events>(serialized), original);
    }

    EXPECT_EQ(Meta::fromStr<Events>("'CLICK'"), Events::Click);
    EXPECT_EQ(
        Meta::fromStr<Events>("['SCROLL', 'hover']"),
        Events::Scroll | Events::Hover);

    const attr::PointerEvents attribute{Events::Drag | Events::Hover};
    EXPECT_EQ(Meta::toStr(attribute), "[hover,drag]");
    EXPECT_EQ(
        static_cast<Events>(
            Meta::fromStr<attr::PointerEvents>(Meta::toStr(attribute))),
        static_cast<Events>(attribute));
}
