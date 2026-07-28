#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>
#include <gui/DynamicGUI.h>
#include <gui/DynamicVariable.h>
#include <gui/LabeledField.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/ResolveVariable.h>
#include <gui/types/ListItemTypes.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/ErrorElement.h>
#include <gui/types/Layout.h>
#include <gui/types/StaticText.h>
#include <gui/types/TagList.h>
#include <gui/dyn/UnresolvedStringPattern.h>   // for ResolveStringPattern tests
#include <type_traits>
#include <gui/dyn/Action.h>
#include <misc/GlobalSettings.h>

using namespace cmn;
using namespace cmn::gui;
using namespace dyn;

/// derived_ptr must never take ownership of a raw pointer implicitly, and
/// assignment must reject pointer types other than an exact match: a raw
/// Derived* (e.g. from to<Derived>()) is typically owned elsewhere already,
/// and accepting it here would create a second owner that double-deletes.
static_assert(not std::is_assignable_v<Layout::Ptr&, HorizontalLayout*>);
static_assert(std::is_assignable_v<derived_ptr<HorizontalLayout>&, HorizontalLayout*>);
static_assert(std::is_assignable_v<Layout::Ptr&, std::nullptr_t>);
static_assert(not std::is_convertible_v<Drawable*, Layout::Ptr>);

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

static void collect_rendered_text_strings(const Layout::Ptr& node, std::vector<std::string>& out) {
    if(not node) {
        return;
    }
    
    if(node.is<StaticText>()) {
        out.push_back(node.to<StaticText>()->text());
        return;
    }
    
    if(node.is<Text>()) {
        out.push_back(node.to<Text>()->txt());
        return;
    }
    
    if(node.is<Layout>()) {
        for(const auto& child : node.to<Layout>()->objects()) {
            collect_rendered_text_strings(child, out);
        }
    }
}

static Vec2 center_of(Drawable& drawable) {
    const auto bounds = drawable.global_bounds();
    return Vec2(bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5);
}

static std::shared_ptr<Drawable> update_until_named(
    DynamicGUI& gui,
    DrawStructure& graph,
    Layout& parent,
    std::string_view name)
{
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(1);
    do {
        gui.update(graph, &parent);
        if(auto object = gui.current_object_handler->retrieve_named(std::string(name)))
            return object;
        std::this_thread::yield();
    } while(std::chrono::steady_clock::now() < deadline);

    return nullptr;
}

static std::vector<FrameTag> frame_tags(
    std::initializer_list<std::string_view> values)
{
    std::vector<FrameTag> result;
    result.reserve(values.size());
    for(const auto value : values)
        result.emplace_back(FrameTag::fromStr(value));
    return result;
}

static std::vector<std::string> serialized_tags(
    const std::vector<FrameTag>& values)
{
    std::vector<std::string> result;
    result.reserve(values.size());
    for(const auto& value : values)
        result.emplace_back(value.toStr());
    return result;
}

static_assert(cmn::gui::detail::HasSet<FloatingLayout, attr::SizeLimit>);
static_assert(cmn::gui::detail::HasSet<FloatingLayout, FloatingLayout::Policy>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, attr::SizeLimit>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, TagList::AllowNew_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, TagList::MatchThreshold_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemFont_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemPadding_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemFillClr_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, std::optional<ItemFillClr_t>>);
static_assert(cmn::gui::dyn::takes_attribute<Drawable, attr::PointerEvents>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemLineColor_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemTextClr_t>);
static_assert(cmn::gui::dyn::takes_attribute<ScrollableList<DetailTooltipItem>, ItemFillClr_t>);
static_assert(cmn::gui::dyn::takes_attribute<ScrollableList<DetailTooltipItem>, ItemTextClr_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, CornerFlags_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, LabelDims_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, ListDims_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, Placeholder_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, LabelFont_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, LabelFillClr_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, LabelLineColor_t>);
static_assert(cmn::gui::dyn::takes_attribute<TagList, LabelColor_t>);
static_assert(!std::is_move_constructible_v<FloatingLayout>);
static_assert(!std::is_move_assignable_v<FloatingLayout>);
static_assert(!std::is_move_constructible_v<TagList>);
static_assert(!std::is_move_assignable_v<TagList>);

TEST(DynamicGUILocalSettings, ParsesPredefinedAliasesAndDefaults) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "enabled": { "type": "bool", "value": true },
    "count": { "type": "int", "value": 4 },
    "ratio": { "type": "double", "value": 0.5 },
    "label": { "type": "string", "value": "hello" },
    "dataset": { "type": "path", "value": "/tmp/data.yaml" },
    "inputs": { "type": "path_array" },
    "selected_ids": { "type": "int_array", "value": [1, 2, 3] },
    "origin": { "type": "vec2", "value": [2, 3] },
    "panel_size": { "type": "size", "value": [100, 40] },
    "accent": { "type": "color", "value": [10, 20, 30, 255] }
  },
  "objects": []
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    (void)objects;
    Context context;
    ASSERT_NO_THROW(context.apply_local_settings(defaults));

    EXPECT_TRUE(context.local_setting_ref("local.enabled").value<bool>());
    EXPECT_EQ(context.local_setting_ref("local.count").value<int>(), 4);
    EXPECT_DOUBLE_EQ(context.local_setting_ref("local.ratio").value<double>(), 0.5);
    EXPECT_EQ(context.local_setting_ref("local.label").value<std::string>(), "hello");
    EXPECT_EQ(context.local_setting_ref("local.dataset").value<file::Path>(), file::Path("/tmp/data.yaml"));
    EXPECT_THAT(context.local_setting_ref("local.selected_ids").value<std::vector<int>>(), ::testing::ElementsAre(1, 2, 3));
    EXPECT_EQ(context.local_setting_ref("local.origin").value<Vec2>(), Vec2(2, 3));
    EXPECT_EQ(context.local_setting_ref("local.panel_size").value<Size2>(), Size2(100, 40));
}

TEST(DynamicGUILocalSettings, SettingsWidgetsBindToLocalValuesWithoutGlobalSettings) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "local_string": { "type": "string", "value": "abc" },
    "local_bool": { "type": "bool", "value": true },
    "local_path": { "type": "path", "value": "/tmp/data.yaml" },
    "local_ids": { "type": "int_array", "value": [5, 6] }
  },
  "objects": [
    {
      "type": "collection",
      "children": [
        { "type": "settings", "var": "local.local_string" },
        { "type": "settings", "var": "local.local_bool" },
        { "type": "settings", "var": "local.local_path" },
        { "type": "settings", "var": "local.local_ids" }
      ]
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    (void)objects;
    Context context;
    context.apply_local_settings(defaults);
    context.defaults = std::move(defaults);

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    GUITaskQueue_t queue;
    DrawStructure graph(640, 480);
    auto root = parse_object(&queue, objects.get_array().front().get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(&queue, graph, root, context, state));

    EXPECT_FALSE(GlobalSettings::has_value("local_string"));
    EXPECT_FALSE(GlobalSettings::has_value("local_bool"));
    EXPECT_EQ(context.local_setting_ref("local.local_string").value<std::string>(), "abc");
    EXPECT_EQ(context.local_setting_ref("local.local_ids").value<std::vector<int>>(), std::vector<int>({5, 6}));
}

TEST(DynamicGUILocalSettings, LocalExpressionsResolveCurrentValues) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "title": { "type": "string", "value": "before" },
    "enabled": { "type": "bool", "value": true }
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
    context.defaults = std::move(defaults);
    State state;

    EXPECT_EQ(parse_text("{local.title}", context, state), "before");
    EXPECT_EQ(parse_text("{if:{local.enabled}:yes:no}", context, state), "yes");

    context.local_setting_ref("local.title").get().set_value_from_string("after");
    context.local_setting_ref("local.enabled").get() = false;
    EXPECT_EQ(parse_text("{local.title}", context, state), "after");
    EXPECT_EQ(parse_text("{if:{local.enabled}:yes:no}", context, state), "no");
}

TEST(DynamicGUILocalSettings, SetLocalSystemActionUpdatesDeclaredValue) {
    DynamicGUI gui{
        .path = file::Path(TREX_TEST_FOLDER) / "dyngui_set_local.json"
    };

    DrawStructure graph(640, 480);
    VerticalLayout parent;
    auto button = update_until_named(gui, graph, parent, "set-local-button");
    ASSERT_TRUE(button);
    graph.wrap_object(parent);

    EXPECT_EQ(gui.context.local_setting_ref("local.tab").value<int>(), 0);

    const auto button_center = center_of(*button);
    ASSERT_NO_THROW(graph.mouse_move(button_center.x, button_center.y));
    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_NO_THROW(graph.mouse_up(true));

    EXPECT_EQ(gui.context.local_setting_ref("local.tab").value<int>(), 1);
}

TEST(DynamicGUISystemVariables, RelativeMouseUsesCurrentOrNamedElementTransform) {
    DynamicGUI gui{
        .path = file::Path(TREX_TEST_FOLDER) / "dyngui_set_local.json"
    };

    DrawStructure graph(640, 480);
    VerticalLayout parent;
    auto target = update_until_named(gui, graph, parent, "mouse-relative-target");
    ASSERT_TRUE(target);
    graph.wrap_object(parent);

    const Vec2 expected{23, 17};
    const auto global_mouse = target->global_transform().transformPoint(expected);
    ASSERT_NO_THROW(graph.mouse_move(global_mouse.x, global_mouse.y));
    ASSERT_NO_THROW(gui.update(graph, &parent));
    graph.wrap_object(parent);

    EXPECT_EQ(
        parse_text("{mouse_relative:mouse-relative-target}", gui.context, gui.state),
        Meta::toStr(expected));

    const Vec2 event_expected{31, 19};
    const auto event_global_mouse = target->global_transform().transformPoint(event_expected);
    ASSERT_NO_THROW(graph.mouse_move(event_global_mouse.x, event_global_mouse.y));
    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_NO_THROW(graph.mouse_up(true));
    EXPECT_EQ(gui.context.local_setting_ref("local.tab").value<int>(), 31);

    EXPECT_EQ(
        parse_text("{mouse_relative}", gui.context, gui.state),
        Meta::toStr(event_expected));
}

TEST(DynamicGUIEventCapture, DragActionPreventsParentDragAction) {
    constexpr std::string_view json = R"json(
{
  "type": "collection",
  "name": "drag-parent",
  "size": [200, 100],
  "clickable": true,
  "drag": "parent-drag",
  "children": [
    {
      "type": "rect",
      "name": "drag-child",
      "size": [120, 60],
      "clickable": true,
      "drag": "child-drag"
    }
  ]
}
)json";

    glz::json_t object;
    const auto parse_error = glz::read_json(object, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    int parent_drags = 0;
    int child_drags = 0;
    Context context{
        ActionFunc("parent-drag", [&](const Action&) { ++parent_drags; }),
        ActionFunc("child-drag", [&](const Action&) { ++child_drags; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    graph.wrap_object(*root);

    auto parent = handler->retrieve_named("drag-parent");
    auto child = handler->retrieve_named("drag-child");
    ASSERT_TRUE(parent);
    ASSERT_TRUE(child);
    const auto start = child->global_transform().transformPoint(Vec2{20, 20});
    const auto moved = child->global_transform().transformPoint(Vec2{30, 20});

    ASSERT_NO_THROW(graph.mouse_move(start.x, start.y));
    ASSERT_NO_THROW(graph.mouse_down(true));
    EXPECT_TRUE(child->pressed());
    EXPECT_FALSE(parent->pressed());
    ASSERT_NO_THROW(graph.mouse_move(moved.x, moved.y));
    ASSERT_NO_THROW(graph.mouse_up(true));

    EXPECT_EQ(child_drags, 1);
    EXPECT_EQ(parent_drags, 0);
}

TEST(DynamicGUIPointerEvents, ParsesArraysSpecialValuesAndDefaults) {
    using pointer::Events;

    EXPECT_EQ(static_cast<Events>(Meta::fromStr<attr::PointerEvents>("all")), Events::All);
    EXPECT_EQ(static_cast<Events>(Meta::fromStr<attr::PointerEvents>("none")), Events::None);
    EXPECT_EQ(static_cast<Events>(Meta::fromStr<attr::PointerEvents>("['drag', 'hover']")), Events::Drag | Events::Hover);
    EXPECT_THROW(
        (void)Meta::fromStr<attr::PointerEvents>("unknown"),
        std::invalid_argument);

    constexpr std::string_view json = R"json(
{
  "defaults": {
    "pointer-events": ["click"]
  },
  "objects": [
    {
      "type": "rect",
      "name": "default-pointer-events",
      "size": [100, 50],
      "clickable": true
    },
    {
      "type": "rect",
      "name": "drag-overlay",
      "size": [100, 50],
      "clickable": true,
      "pointer-events": ["drag", "hover"]
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    auto [defaults, objects] = std::move(loaded.value());
    ASSERT_EQ(defaults.pointer_events, Events::Click);

    auto scalar_defaults = load(R"json(
{
  "defaults": {
    "pointer-events": "scroll"
  },
  "objects": []
}
)json");
    ASSERT_TRUE(scalar_defaults.has_value()) << scalar_defaults.error();
    EXPECT_EQ(
        std::get<0>(scalar_defaults.value()).pointer_events,
        Events::Scroll);

    Context context;
    context.defaults = defaults;
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    auto default_object = parse_object(
        nullptr,
        objects.get_array().at(0).get_object(),
        context,
        state,
        defaults);
    auto drag_overlay = parse_object(
        nullptr,
        objects.get_array().at(1).get_object(),
        context,
        state,
        defaults);

    ASSERT_TRUE(default_object);
    ASSERT_TRUE(drag_overlay);
    EXPECT_EQ(default_object->pointer_events(), Events::Click);
    EXPECT_EQ(
        drag_overlay->pointer_events(),
        Events::Drag | Events::Hover);
}

TEST(DynamicGUIPointerEvents, ResolvesDynamicExpressions) {
    using pointer::Events;

    constexpr std::string_view json = R"json(
{
  "type": "rect",
  "size": [100, 50],
  "clickable": true,
  "pointer-events": "{event_mask}"
}
)json";

    glz::json_t object;
    const auto parse_error = glz::read_json(object, json);
    ASSERT_EQ(parse_error, glz::error_code::none)
        << glz::format_error(parse_error, json);

    std::string event_mask = "['drag','hover']";
    Context context{
        VarFunc("event_mask", [&](const VarProps&) -> std::string {
            return event_mask;
        })
    };

    State state;
    auto object_handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = object_handler;
    auto drawable = parse_object(
        nullptr,
        object.get_object(),
        context,
        state,
        context.defaults);
    ASSERT_TRUE(drawable);
    EXPECT_EQ(drawable->pointer_events(), Events::Drag | Events::Hover);

    event_mask = "click";
    DrawStructure graph(640, 480);
    ASSERT_NO_THROW(
        (void)DynamicGUI::update_objects(
            nullptr,
            graph,
            drawable,
            context,
            state));
    EXPECT_EQ(drawable->pointer_events(), Events::Click);
}

TEST(DynamicGUILocalSettings, LocalValuesSurviveReloadWhenAliasMatches) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "title": { "type": "string", "value": "default" }
  },
  "objects": []
}
)json";

    auto first = load(std::string(json));
    ASSERT_TRUE(first.has_value()) << first.error();
    auto [defaults, objects] = std::move(first.value());
    (void)objects;

    Context context;
    context.apply_local_settings(defaults);
    context.local_setting_ref("local.title").get().set_value_from_string("edited");

    auto second = load(std::string(json));
    ASSERT_TRUE(second.has_value()) << second.error();
    auto [reloaded_defaults, reloaded_objects] = std::move(second.value());
    (void)reloaded_objects;
    context.apply_local_settings(reloaded_defaults);

    EXPECT_EQ(context.local_setting_ref("local.title").value<std::string>(), "edited");
}

TEST(DynamicGUILocalSettings, ChangedAliasReinitializesLocalValue) {
    constexpr std::string_view string_json = R"json(
{
  "locals": {
    "value": { "type": "string", "value": "default" }
  },
  "objects": []
}
)json";
    constexpr std::string_view int_json = R"json(
{
  "locals": {
    "value": { "type": "int", "value": 7 }
  },
  "objects": []
}
)json";

    auto first = load(std::string(string_json));
    ASSERT_TRUE(first.has_value()) << first.error();
    auto [defaults, objects] = std::move(first.value());
    (void)objects;

    Context context;
    context.apply_local_settings(defaults);
    context.local_setting_ref("local.value").get().set_value_from_string("edited");

    auto second = load(std::string(int_json));
    ASSERT_TRUE(second.has_value()) << second.error();
    auto [reloaded_defaults, reloaded_objects] = std::move(second.value());
    (void)reloaded_objects;
    context.apply_local_settings(reloaded_defaults);

    EXPECT_EQ(context.local_setting_ref("local.value").value<int>(), 7);
}

TEST(DynamicGUILocalSettings, InvalidAliasReportsAvailableAliases) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "bad": { "type": "not_a_type", "value": 1 }
  },
  "objects": []
}
)json";

    try {
        (void)load(std::string(json));
        FAIL() << "Expected invalid local setting alias to throw.";
    } catch(const std::exception& e) {
        const std::string message = e.what();
        EXPECT_THAT(message, ::testing::HasSubstr("not_a_type"));
        EXPECT_THAT(message, ::testing::HasSubstr("Available types"));
        EXPECT_THAT(message, ::testing::HasSubstr("bool"));
        EXPECT_THAT(message, ::testing::HasSubstr("string"));
    }
}

TEST(DynamicGUILocalSettings, ExistingGlobalSettingsStillCreateSettingsWidgets) {
    SETTING(dyngui_test_global_setting) = std::string("global");

    constexpr std::string_view json = R"json(
{
  "objects": [
    { "type": "settings", "var": "dyngui_test_global_setting" }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    Context context;
    context.defaults = std::move(defaults);

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    GUITaskQueue_t queue;
    auto root = parse_object(&queue, objects.get_array().front().get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    EXPECT_EQ(GlobalSettings::read_value<std::string>("dyngui_test_global_setting").value(), "global");
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
    ASSERT_FALSE(ptr != nullptr);
    
    {
        auto text_ptr = Layout::Make<Text>{}();
        ptr = text_ptr;
        
        ASSERT_EQ(text_ptr, ptr);
        ASSERT_FALSE(text_ptr != ptr);
        ASSERT_TRUE(ptr != nullptr);
        ASSERT_TRUE(text_ptr.get_smart());
        ASSERT_EQ(text_ptr.get_smart().use_count(), ptr.get_smart().use_count());
        ASSERT_EQ(text_ptr.get_smart().use_count(), 2);
    }
    
    {
        ASSERT_EQ(ptr.get_smart().use_count(), 1);
        auto smart = ptr.get_smart();
        ASSERT_EQ(ptr.get_smart().use_count(), 2);
        ptr = nullptr;
        ASSERT_FALSE(ptr != nullptr);
        ASSERT_EQ(smart.use_count(), 1);
    }
}

TEST(TestDerivedPtr, Convert) {
    auto button = Layout::Make<Button>{}();
    static_assert(std::same_as<decltype(button), derived_ptr<Button>>, "");

    derived_ptr<Button> typed_button = Layout::Make<Button>{};
    Layout::Ptr drawable_button = typed_button;
    ASSERT_EQ(typed_button, drawable_button);
    ASSERT_FALSE(typed_button != drawable_button);
    ASSERT_EQ(typed_button.get_smart().use_count(), drawable_button.get_smart().use_count());
    ASSERT_EQ(typed_button.get_smart().use_count(), 2);

    Layout::Ptr direct_drawable = button;
    ASSERT_TRUE(direct_drawable.is<Button>());
    ASSERT_EQ(button, direct_drawable);
    ASSERT_FALSE(button != direct_drawable);
    
    auto button2 = Layout::Make<Button>{}();
    ASSERT_TRUE(button != button2);
    ASSERT_TRUE(button2 != direct_drawable);
    ASSERT_FALSE(direct_drawable == button2);
}

TEST(LayoutOuterPadding, HorizontalCenterUsesInnerPaddedArea) {
    DrawStructure graph(640, 480);
    auto child = Layout::Make<Rect>{Box{0, 0, 10, 20}}();
    HorizontalLayout layout(
        std::vector<Layout::Ptr>{child},
        OuterPadding{0, 10, 0, 30},
        HorizontalLayout::Policy::CENTER);

    layout.set_stage(&graph);
    layout.set_content_changed(true);
    layout.update();

    EXPECT_FLOAT_EQ(layout.height(), 60);
    EXPECT_FLOAT_EQ(child->pos().y, 10);
}

TEST(LayoutOuterPadding, VerticalCenterUsesInnerPaddedArea) {
    DrawStructure graph(640, 480);
    auto child = Layout::Make<Rect>{Box{0, 0, 20, 10}}();
    VerticalLayout layout(
        std::vector<Layout::Ptr>{child},
        OuterPadding{10, 0, 30, 0},
        VerticalLayout::Policy::CENTER);

    layout.set_stage(&graph);
    layout.set_content_changed(true);
    layout.update();

    EXPECT_FLOAT_EQ(layout.width(), 60);
    EXPECT_FLOAT_EQ(child->pos().x, 10);
}

TEST(LayoutOuterPadding, PlainAutoSizeIncludesPaddingWithoutMovingChildren) {
    auto child = Layout::Make<Rect>{Box{7, 11, 13, 17}}();
    Layout layout(
        std::vector<Layout::Ptr>{child},
        OuterPadding{3, 5, 19, 23});

    layout.auto_size();

    EXPECT_FLOAT_EQ(layout.width(), 42);
    EXPECT_FLOAT_EQ(layout.height(), 56);
    EXPECT_FLOAT_EQ(child->pos().x, 7);
    EXPECT_FLOAT_EQ(child->pos().y, 11);
}

TEST(LayoutOuterPadding, EmptyGridSizesToPadding) {
    DrawStructure graph(640, 480);
    GridLayout grid(OuterPadding{3, 5, 7, 11});

    grid.set_stage(&graph);
    grid.set_content_changed(true);
    grid.update();

    EXPECT_FLOAT_EQ(grid.width(), 10);
    EXPECT_FLOAT_EQ(grid.height(), 16);
}

TEST(LayoutOuterPadding, SingleGridCellBoundsIncludeAllEdgePadding) {
    DrawStructure graph(640, 480);
    auto cell_child = Layout::Make<Rect>{Box{0, 0, 20, 10}}();
    auto cell = Layout::Make<HorizontalLayout>{std::vector<Layout::Ptr>{cell_child}, attr::Margins{0, 0, 0, 0}}();
    cell.to<Layout>()->set_stage(&graph);
    cell.to<Layout>()->set_content_changed(true);
    cell.to<Layout>()->update();
    auto row = Layout::Make<Layout>{std::vector<Layout::Ptr>{cell}, attr::Margins{0, 0, 0, 0}}();
    GridLayout grid(
        std::vector<Layout::Ptr>{row},
        attr::Margins{0, 0, 0, 0},
        OuterPadding{3, 5, 7, 11});

    grid.set_stage(&graph);
    grid.set_content_changed(true);
    grid.update();

    ASSERT_EQ(grid.grid_info().rowCount(), 1u);
    ASSERT_EQ(grid.grid_info().colCount(), 1u);

    const auto bounds = grid.grid_info().getCellBounds(0, 0);
    EXPECT_FLOAT_EQ(bounds.x, 0);
    EXPECT_FLOAT_EQ(bounds.y, 0);
    EXPECT_FLOAT_EQ(bounds.width, 30);
    EXPECT_FLOAT_EQ(bounds.height, 26);
}

TEST(LayoutOuterPadding, GridIgnoresEmptyCellsForEdgePaddingAndMargins) {
    DrawStructure graph(640, 480);

    /// fully empty
    auto zero_cell = Layout::Make<HorizontalLayout>{
        std::vector<Layout::Ptr>{},
        attr::Margins{0, 0, 0, 0}
    }();
    auto visible_child = Layout::Make<Rect>{Box{0, 0, 20, 10}}();
    auto visible_cell = Layout::Make<HorizontalLayout>{
        std::vector<Layout::Ptr>{visible_child},
        attr::Margins{0, 0, 0, 0}
    }();
    visible_cell.to<Layout>()->set_stage(&graph);
    visible_cell.to<Layout>()->set_content_changed(true);
    visible_cell.to<Layout>()->update();

    auto row = Layout::Make<Layout>{
        std::vector<Layout::Ptr>{zero_cell, visible_cell},
        attr::Margins{0, 0, 0, 0}
    }();
    GridLayout grid(
        std::vector<Layout::Ptr>{row},
        attr::Margins{2, 3, 4, 5},
        OuterPadding{7, 11, 13, 17});

    grid.set_stage(&graph);
    grid.set_content_changed(true);
    grid.update();

    ASSERT_EQ(grid.grid_info().rowCount(), 1u);
    ASSERT_EQ(grid.grid_info().colCount(), 2u);

    const auto zero_bounds = grid.grid_info().getCellBounds(0, 0);
    EXPECT_FLOAT_EQ(zero_bounds.x, 0);
    EXPECT_FLOAT_EQ(zero_bounds.width, 0);

    const auto visible_bounds = grid.grid_info().getCellBounds(0, 1);
    EXPECT_FLOAT_EQ(visible_bounds.x, 0);
    EXPECT_FLOAT_EQ(visible_bounds.width, 20 + 2 + 4 + 7 + 13);
    EXPECT_FLOAT_EQ(visible_cell->pos().x, 2);
    EXPECT_FLOAT_EQ(grid.width(), 20 + 2 + 4 + 7 + 13);
}

TEST(LayoutOuterPadding, GridIgnoresZeroWidthCellsForEdgePaddingAndMargins) {
    DrawStructure graph(640, 480);

    /// visibly empty, but actually contains a rect
    auto zero_cell = Layout::Make<HorizontalLayout>{
        std::vector<Layout::Ptr>{
            Layout::Make<Rect>{Box{0, 0, 0, 0}}(),
            Layout::Make<Rect>{Box{0, 0, 0, 0}}()
        },
        attr::Margins{5, 5, 5, 5}
    }();
    auto visible_child = Layout::Make<Rect>{Box{0, 0, 20, 10}}();
    auto visible_cell = Layout::Make<HorizontalLayout>{
        std::vector<Layout::Ptr>{visible_child},
        attr::Margins{0, 0, 0, 0}
    }();
    visible_cell.to<Layout>()->set_stage(&graph);
    visible_cell.to<Layout>()->set_content_changed(true);
    visible_cell.to<Layout>()->update();

    auto row = Layout::Make<Layout>{
        std::vector<Layout::Ptr>{zero_cell, visible_cell},
        attr::Margins{0, 0, 0, 0}
    }();
    GridLayout grid(
        std::vector<Layout::Ptr>{row},
        attr::Margins{2, 3, 4, 5},
        OuterPadding{7, 11, 13, 17});

    grid.set_stage(&graph);
    grid.set_content_changed(true);
    grid.update();

    ASSERT_EQ(grid.grid_info().rowCount(), 1u);
    ASSERT_EQ(grid.grid_info().colCount(), 2u);

    const auto zero_bounds = grid.grid_info().getCellBounds(0, 0);
    EXPECT_FLOAT_EQ(zero_bounds.x, 0);
    EXPECT_FLOAT_EQ(zero_bounds.width, 0);

    const auto visible_bounds = grid.grid_info().getCellBounds(0, 1);
    EXPECT_FLOAT_EQ(visible_bounds.x, 0);
    EXPECT_FLOAT_EQ(visible_bounds.width, 20 + 2 + 4 + 7 + 13);
    EXPECT_FLOAT_EQ(visible_cell->pos().x, 2);
    EXPECT_FLOAT_EQ(grid.width(), 20 + 2 + 4 + 7 + 13);
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

TYPED_TEST(ParseAndResolveTest, JsonSubArrayTest)
{
    State state;
    glz::json_t object= cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::json_t { return object; })
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
    glz::json_t object;
    object["array"] = cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("object", [&object](const VarProps&) -> glz::json_t { return object; })
    };

    EXPECT_EQ(run_parser<TypeParam>("{object.array}", ctx, state), "[1,2,3]");
    EXPECT_EQ(run_parser<TypeParam>("{object.array.0}", ctx, state), "1");
}

TYPED_TEST(ParseAndResolveTest, JsonDynamicSubSubArrayTest)
{
    State state;
    glz::json_t object;
    object["array"] = cvt2json(std::vector<int>{1,2,3});

    Context ctx{
        VarFunc("index", [](const VarProps&){ return 1; }),
        VarFunc("object", [&object](const VarProps&) -> glz::json_t { return object; })
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
    auto result = run_parser<TypeParam>("{at:key:\\{key:value,other:ignored\\}}", ctx, state);
    ASSERT_EQ(result, "value");
}

TYPED_TEST(ParseAndResolveTest, AtLooksUpObjectValueWithWhitespace)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:\\{ other: ignored, key: value \\}}", ctx, state);
    ASSERT_EQ(result, "value");
}

TYPED_TEST(ParseAndResolveTest, AtReturnsNullForMissingObjectKey)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:missing:\\{key:value,other:ignored\\}}", ctx, state);
    ASSERT_EQ(result, "null");
}

/*TYPED_TEST(ParseAndResolveTest, AtReturnsNullForMalformedObjectEntry)
{
    State   state;
    Context ctx;
    auto result = run_parser<TypeParam>("{at:key:\\{key:value,malformed\\}}", ctx, state);
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
        EXPECT_THAT(message, ::testing::HasSubstr("recursive"));
        EXPECT_THAT(message, ::testing::HasSubstr("foo"));
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
        EXPECT_THAT(message, ::testing::HasSubstr("recursive"));
        EXPECT_THAT(message, ::testing::HasSubstr("foo"));
        EXPECT_THAT(message, ::testing::HasSubstr("bar"));
    }
}

TEST(DefaultVariablesTest, LoadedDefaultExpressionSupportsVec2Subfields)
{
    constexpr std::string_view json = R"json(
{
  "defaults": {
    "vars": {
      "mouse_in_bowl": "{2bowl:{mouse}}"
    }
  },
  "objects": [
    {
      "type": "stext",
      "text": "<h5><sym>🖰</sym></h5> <i>{round:{mouse_in_bowl.x}},{round:{mouse_in_bowl.y}}</i>"
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    ASSERT_TRUE(objects.is_array());
    ASSERT_EQ(objects.get_array().size(), 1u);
    ASSERT_TRUE(objects.get_array().front().is_object());

    Context context{
        VarFunc("mouse", [](const VarProps&) -> Vec2 { return Vec2(123,456); }),
        VarFunc("2bowl", [](const VarProps& props) -> Vec2 {
            return Meta::fromStr<Vec2>(props.parameters.front());
        })
    };
    context.defaults = std::move(defaults);

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, objects.get_array().front().get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "<h5><sym>🖰</sym></h5> <i>123,456</i>"
    ));
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

TEST(EachElementTest, ConditionThenBranchNestedEachUsesCurrentOuterItem) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "{annotations}",
  "do": {
    "type": "collection",
    "children": [
      {
        "type": "condition",
        "var": "{equal:{i.type}:1}",
        "then": {
          "type": "each",
          "var": "{i.pts}",
          "do": {
            "type": "text",
            "text": "pt:{i}-{index}",
            "pos": [10, 10],
            "origin": [0.5, 0.5],
            "color": [255, 0, 255]
          }
        },
        "else": {
          "type": "text",
          "text": "not {i.type}: {i.pts}",
          "pos": [10, 20],
          "origin": [0.5, 1],
          "color": [255, 0, 255]
        }
      }
    ]
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    struct TestAnnotation {
        uint8_t uid{};
        uint8_t type{};
        std::vector<blob::Pose::Point> points{};
    };

    std::vector<TestAnnotation> source{
        TestAnnotation{.uid = 1, .type = 1, .points = {blob::Pose::Point{1, 2}, blob::Pose::Point{3, 4}}},
        TestAnnotation{.uid = 2, .type = 2, .points = {blob::Pose::Point{5, 6}}}
    };

    Context context{
        VarFunc("annotations", [&source](const VarProps&) -> std::vector<glz::json_t> {
            std::vector<glz::json_t> result;
            result.reserve(source.size());

            for(auto& object : source) {
                Bounds bds(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX);
                for(auto& pt : object.points) {
                    if(not pt.valid()) {
                        continue;
                    }

                    if(pt.x >= bds.width) {
                        bds.width = pt.x;
                    }
                    if(pt.x < bds.x) {
                        bds.x = pt.x;
                    }
                    if(pt.y >= bds.height) {
                        bds.height = pt.y;
                    }
                    if(pt.y < bds.y) {
                        bds.y = pt.y;
                    }
                }
                if(bds.x == FLT_MAX) {
                    continue;
                }

                result.push_back(glz::json_t::object_t{
                    {"id", object.uid},
                    {"seed_frame", glz::json_t{0}},
                    {"type", object.type},
                    {"x", bds.x},
                    {"y", bds.y},
                    {"w", bds.width - bds.x},
                    {"h", bds.height - bds.y},
                    {"pts", cvt2json(object.points)}
                });
            }

            return result;
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

    std::vector<std::string> texts;
    collect_rendered_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "pt:[1,2]-0",
        "pt:[3,4]-1",
        "not 2: [[5,6]]"
    ));

    source[0].points = {blob::Pose::Point{10, 11}};
    source[1].type = 1;
    source[1].points = {blob::Pose::Point{20, 21}, blob::Pose::Point{30, 31}};

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    texts.clear();
    collect_rendered_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "pt:[10,11]-0",
        "pt:[20,21]-0",
        "pt:[30,31]-1"
    ));
}

TEST(EachElementTest, GenericVectorLoopUpdatesStringifiedValues) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "do": {
    "type": "stext",
    "text": "item:{i}-{index}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<int> items{10, 20};

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<int> { return items; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "item:10-0",
        "item:20-1"
    ));

    items = {30, 40, 50};

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    texts.clear();
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "item:30-0",
        "item:40-1",
        "item:50-2"
    ));
}

TEST(EachElementTest, CustomLoopVariableNameWorksForGenericVectorLoops) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "as": "item",
  "do": {
    "type": "stext",
    "text": "item:{item}-{index}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<int> items{10, 20};

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<int> { return items; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "item:10-0",
        "item:20-1"
    ));
}

TEST(EachElementTest, SpriteMapSubfieldArrayLoopUpdatesValues) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items.values",
  "do": {
    "type": "stext",
    "text": "value:{i}-{index}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    sprite::Map items;
    items["values"] = std::vector<int>{1, 2};

    Context context{
        VarFunc("items", [&items](const VarProps&) -> sprite::Map& { return items; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "value:1-0",
        "value:2-1"
    ));

    items["values"] = std::vector<int>{7, 8, 9};

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    texts.clear();
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "value:7-0",
        "value:8-1",
        "value:9-2"
    ));
}

TEST(EachElementTest, CustomLoopVariableNamesWorkInNestedObjectLoops) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "outer",
  "as": "group",
  "do": {
    "type": "collection",
    "children": [
      {
        "type": "stext",
        "text": "outer:{group.label}-{index}"
      },
      {
        "type": "each",
        "var": "group.inner",
        "as": "point",
        "do": {
          "type": "stext",
          "text": "inner:{group.label}:{point}-{index}"
        }
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
    outer_data[1]["inner"] = std::vector<int>{3};

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
        "outer:10-0",
        "inner:10:1-0",
        "inner:10:2-1",
        "outer:20-1",
        "inner:20:3-0"
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

TEST(EachElementTest, CustomLoopVariableNameWorksForJsonObjectLoops) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "as": "item",
  "do": {
    "type": "stext",
    "text": "name:{item.name}-{index}"
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
        first["name"] = std::string("trex");
        items.push_back(first);
    }
    {
        glz::json_t second;
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

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "name:trex-0",
        "name:raptor-1"
    ));
}

TEST(EachElementTest, CustomLoopVariableNameWorksForStringArrayLoops) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "as": "item",
  "do": {
    "type": "stext",
    "text": "item:{item}-{index}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::string items = "[alpha,beta]";

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::string { return items; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    ASSERT_THAT(texts, ::testing::ElementsAre(
        "item:alpha-0",
        "item:beta-1"
    ));
}

TEST(EachElementTest, ConditionBranchNewlyParsedKeepsOuterScopedVariables) {
        constexpr std::string_view json = R"json(
{
    "type": "each",
    "var": "outer",
    "do": {
        "type": "condition",
        "var": "{i.enabled}",
        "then": {
            "type": "collection",
            "children": [
                {
                    "type": "stext",
                    "text": "outer:{i.label}-{index}"
                },
                {
                    "type": "each",
                    "var": "i.inner",
                    "do": {
                        "type": "stext",
                        "text": "inner:{i}-{index}"
                    }
                }
            ]
        },
        "else": {
            "type": "stext",
            "text": "disabled:{i.label}-{index}"
        }
    }
}
)json";

        glz::json_t obj;
        auto parse_error = glz::read_json(obj, json);
        ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
        ASSERT_TRUE(obj.is_object());

        std::vector<sprite::Map> outer_data(2);
        outer_data[0]["label"] = 10;
        outer_data[0]["enabled"] = false;
        outer_data[0]["inner"] = std::vector<int>{7};
        outer_data[1]["label"] = 20;
        outer_data[1]["enabled"] = false;
        outer_data[1]["inner"] = std::vector<int>{8, 9};

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

        std::vector<std::string> texts;
        collect_static_text_strings(root, texts);
        ASSERT_THAT(texts, ::testing::ElementsAre(
                "disabled:10-0",
                "disabled:20-1"
        ));

        outer_data[0]["enabled"] = true;
        outer_data[1]["enabled"] = true;

        ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

        texts.clear();
        collect_static_text_strings(root, texts);
        ASSERT_THAT(texts, ::testing::ElementsAre(
                "outer:10-0",
                "inner:7-0",
                "outer:20-1",
                "inner:8-0",
                "inner:9-1"
        ));
}

TEST(EachElementTest, ConditionBranchNestedEachUpdatesWithoutOuterLoopCacheMiss) {
        constexpr std::string_view json = R"json(
{
    "type": "each",
    "var": "outer",
    "do": {
        "type": "condition",
        "var": "{i.enabled}",
        "then": {
            "type": "collection",
            "children": [
                {
                    "type": "stext",
                    "text": "outer:{i.label}-{index}"
                },
                {
                    "type": "each",
                    "var": "i.inner",
                    "do": {
                        "type": "stext",
                        "text": "inner:{i}-{index}"
                    }
                }
            ]
        },
        "else": {
            "type": "stext",
            "text": "disabled:{i.label}-{index}"
        }
    }
}
)json";

        glz::json_t obj;
        auto parse_error = glz::read_json(obj, json);
        ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
        ASSERT_TRUE(obj.is_object());

        std::vector<sprite::Map> outer_data(2);
        outer_data[0]["label"] = 10;
        outer_data[0]["enabled"] = true;
        outer_data[0]["inner"] = std::vector<int>{1, 2};
        outer_data[1]["label"] = 20;
        outer_data[1]["enabled"] = false;
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

        std::vector<std::string> texts;
        collect_static_text_strings(root, texts);
        ASSERT_THAT(texts, ::testing::ElementsAre(
                "outer:10-0",
                "inner:1-0",
                "inner:2-1",
                "disabled:20-1"
        ));

        outer_data[0]["inner"] = std::vector<int>{7};

        ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

        texts.clear();
        collect_static_text_strings(root, texts);
        ASSERT_THAT(texts, ::testing::ElementsAre(
                "outer:10-0",
                "inner:7-0",
                "disabled:20-1"
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

TEST(ListElementTest, DynamicListTemplateDispatchesActionsOnSelection) {
    glz::json_t obj = glz::json_t::object_t{
        {"type", "list"},
        {"var", "items"},
        {"template", glz::json_t::object_t{
            {"text", "{i.name}"},
            {"action", "select:{i.name}:{index}"},
            {"disabled", "{i.disabled}"}
        }}
    };

    std::vector<sprite::Map> item_data(2);
    item_data[0]["name"] = std::string("alpha");
    item_data[0]["disabled"] = false;
    item_data[1]["name"] = std::string("beta");
    item_data[1]["disabled"] = true;

    std::vector<std::shared_ptr<VarBase_t>> items;
    items.reserve(item_data.size());
    for(size_t idx = 0; idx < item_data.size(); ++idx) {
        items.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &item_data](const VarProps&) -> sprite::Map& {
            return item_data[idx];
        })));
    }

    std::vector<Action> received_actions;
    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return items;
        })
    };
    context.actions["select"] = [&received_actions](Action action) {
        received_actions.push_back(std::move(action));
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

    ASSERT_NO_THROW(list->select_item(0));
    ASSERT_EQ(received_actions.size(), 1u);
    EXPECT_EQ(received_actions.at(0).name, "select");
    ASSERT_EQ(received_actions.at(0).parameters.size(), 2u);
    EXPECT_EQ(received_actions.at(0).parameters.at(0), "alpha");
    EXPECT_EQ(received_actions.at(0).parameters.at(1), "0");

    ASSERT_NO_THROW(list->select_item(1));
    ASSERT_EQ(received_actions.size(), 1u);

    item_data[1]["disabled"] = false;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    ASSERT_NO_THROW(list->select_item(1));
    ASSERT_EQ(received_actions.size(), 2u);
    EXPECT_EQ(received_actions.at(1).name, "select");
    ASSERT_EQ(received_actions.at(1).parameters.size(), 2u);
    EXPECT_EQ(received_actions.at(1).parameters.at(0), "beta");
    EXPECT_EQ(received_actions.at(1).parameters.at(1), "1");
}

TEST(EventBindingTest, ClickActionsOnlyFireOnMouseButtonWithScopedContext) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "do": {
    "type": "rect",
    "name": "item-{i.name}",
    "pos": "[{*:40:{index}},0]",
    "size": [30, 30],
    "origin": [0, 0],
    "clickable": true,
    "click": "select:{i.name}:{index}"
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<sprite::Map> item_data(2);
    item_data[0]["name"] = std::string("alpha");
    item_data[1]["name"] = std::string("beta");

    std::vector<std::shared_ptr<VarBase_t>> items;
    items.reserve(item_data.size());
    for(size_t idx = 0; idx < item_data.size(); ++idx) {
        items.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &item_data](const VarProps&) -> sprite::Map& {
            return item_data[idx];
        })));
    }

    std::vector<Action> received_actions;
    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return items;
        })
    };
    context.actions["select"] = [&received_actions](Action action) {
        received_actions.push_back(std::move(action));
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    graph.wrap_object(*root);

    auto* target = graph.find("item-beta");
    ASSERT_NE(target, (Drawable*)NULL);

    const auto target_center = center_of(*target);

    ASSERT_NO_THROW(graph.mouse_move(target_center.x, target_center.y));
    ASSERT_EQ(received_actions.size(), 0u);

    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_EQ(received_actions.size(), 1u);
    EXPECT_EQ(received_actions.at(0).name, "select");
    ASSERT_EQ(received_actions.at(0).parameters.size(), 2u);
    EXPECT_EQ(received_actions.at(0).parameters.at(0), "beta");
    EXPECT_EQ(received_actions.at(0).parameters.at(1), "1");

    ASSERT_NO_THROW(graph.mouse_up(true));
    ASSERT_EQ(received_actions.size(), 1u);
}

TEST(LineElementTest, ParsesLineFromEndpoints)
{
    constexpr std::string_view json = R"json(
{
  "type": "line",
  "from": [10, 20],
  "to": [30, 50],
  "color": [1, 2, 3, 4],
  "thickness": 3
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    Context context;
    State state;
    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Line>());

    auto line = root.to<Line>();
    EXPECT_EQ(line->line_clr(), Color(1, 2, 3, 4));
    EXPECT_EQ(line->thickness(), 3);
    EXPECT_EQ(line->bounds().pos(), Vec2(10, 20));
    EXPECT_EQ(line->bounds().size(), Size2(20, 30));
}

TEST(LineElementTest, UpdatesLinePatterns)
{
    constexpr std::string_view json = R"json(
{
  "type": "line",
  "from": "{from}",
  "to": "{to}",
  "color": "{line_color}",
  "thickness": "{line_thickness}"
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    Vec2 from{10, 20};
    Vec2 to{30, 50};
    Color line_color{10, 20, 30, 255};
    Float2_t line_thickness{2};
    Context context{
        VarFunc("from", [&from](const VarProps&) -> Vec2 { return from; }),
        VarFunc("to", [&to](const VarProps&) -> Vec2 { return to; }),
        VarFunc("line_color", [&line_color](const VarProps&) -> Color { return line_color; }),
        VarFunc("line_thickness", [&line_thickness](const VarProps&) -> Float2_t { return line_thickness; })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Line>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    auto line = root.to<Line>();
    EXPECT_EQ(line->line_clr(), line_color);
    EXPECT_EQ(line->thickness(), line_thickness);
    EXPECT_EQ(line->bounds().pos(), Vec2(10, 20));
    EXPECT_EQ(line->bounds().size(), Size2(20, 30));

    to = Vec2{40, 70};

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    EXPECT_EQ(line->bounds().pos(), Vec2(10, 20));
    EXPECT_EQ(line->bounds().size(), Size2(30, 50));

    from = Vec2{5, 7};
    to = Vec2{15, 22};
    line_color = Color{30, 40, 50, 200};
    line_thickness = 5;

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    EXPECT_EQ(line->line_clr(), line_color);
    EXPECT_EQ(line->thickness(), line_thickness);
    EXPECT_EQ(line->bounds().pos(), Vec2(5, 7));
    EXPECT_EQ(line->bounds().size(), Size2(10, 15));
}

TEST(LineElementTest, UpdatesParameterizedLineEndpointPatterns)
{
    constexpr std::string_view json = R"json(
{
  "type": "line",
  "from": "{2hud:[50,50]}",
  "to": "{2hud:[100,50]}",
  "color": [255, 255, 255, 255]
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    Float2_t conversion_factor{1};
    Context context{
        VarFunc("2hud", [&conversion_factor](const VarProps& props) -> Vec2 {
            auto point = Meta::fromStr<Vec2>(props.parameters.front());
            return Vec2(point.x * conversion_factor, point.y * conversion_factor);
        })
    };

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<Line>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    auto line = root.to<Line>();
    EXPECT_EQ(line->bounds().pos(), Vec2(50, 50));
    EXPECT_EQ(line->bounds().size(), Size2(50, 0));

    conversion_factor = 2;

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    EXPECT_EQ(line->bounds().pos(), Vec2(100, 100));
    EXPECT_EQ(line->bounds().size(), Size2(100, 0));
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

TEST(ConditionElementTest, IfInsideEachKeepsScopedVariables)
{
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "items",
  "do": {
    "type": "condition",
    "var": "{i.enabled}",
    "then": {
      "type": "stext",
      "text": "enabled:{i.name}-{index}"
    },
    "else": {
      "type": "stext",
      "text": "disabled:{i.name}-{index}"
    }
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<sprite::Map> item_data(2);
    item_data[0]["name"] = std::string("alpha");
    item_data[0]["enabled"] = true;
    item_data[1]["name"] = std::string("beta");
    item_data[1]["enabled"] = false;

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
    ASSERT_TRUE(root.is<Layout>());

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);

    ASSERT_THAT(texts, ::testing::ElementsAre(
        "enabled:alpha-0",
        "disabled:beta-1"
    ));
}

TEST(TagListTest, KeepsStableChipColors) {
    TagList tags;
    tags.set_tags(frame_tags({"beta", "alpha", "Alpha"}));
    tags.update();

    ASSERT_THAT(serialized_tags(tags.tags()),
                ::testing::ElementsAre("beta", "alpha", "Alpha"));
    ASSERT_EQ(tags.flow().objects().size(), 3u);
    const auto beta_color = (Color)tags.flow().objects()[0].to<HorizontalLayout>()->bg_fill_color();
    const auto alpha_color = (Color)tags.flow().objects()[1].to<HorizontalLayout>()->bg_fill_color();

    tags.set_tags(frame_tags({"alpha", "beta"}));
    tags.update();
    EXPECT_EQ((Color)tags.flow().objects()[0].to<HorizontalLayout>()->bg_fill_color(), alpha_color);
    EXPECT_EQ((Color)tags.flow().objects()[1].to<HorizontalLayout>()->bg_fill_color(), beta_color);
}

TEST(DynamicGUIListTest, KeepsItemFillAndTextColorAttributesDistinct) {
    constexpr std::string_view json = R"json(
{
  "type": "list",
  "items": [{"text": "One"}],
  "item": {
    "fill": "{item_fill}",
    "color": "{item_text}"
  }
}
)json";
    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    Color item_fill{20, 30, 40, 210};
    Color item_text{220, 230, 240, 255};
    Context context{
        VarFunc("item_fill", [&item_fill](const VarProps&) -> Color { return item_fill; }),
        VarFunc("item_text", [&item_text](const VarProps&) -> Color { return item_text; })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<ScrollableList<DetailTooltipItem>>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    auto list = root.to<ScrollableList<DetailTooltipItem>>();
    EXPECT_EQ(list->item_color(), item_fill);
    EXPECT_EQ(list->text_color(), item_text);

    item_fill = Color{50, 60, 70, 220};
    item_text = Color{180, 190, 200, 255};
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    EXPECT_EQ(list->item_color(), item_fill);
    EXPECT_EQ(list->text_color(), item_text);
}

TEST(ScrollableListTest, ClearingKeyboardHighlightClearsSelectionState) {
    ScrollableList<std::string> list{Box{0, 0, 200, 100}};
    list.set_items({"Alpha", "Beta"});
    DrawStructure graph(320, 200);
    graph.wrap_object(list);

    list.highlight_item(0);
    ASSERT_EQ(list.currently_highlighted_item().value(), 0u);
    ASSERT_EQ(list.last_hovered_item().value(), 0u);
    ASSERT_EQ(list.highlighted_item().value(), 0u);

    list.highlight_item(-1);
    EXPECT_FALSE(list.currently_highlighted_item().has_value());
    EXPECT_FALSE(list.last_hovered_item().has_value());
    EXPECT_FALSE(list.highlighted_item().has_value());
}

TEST(TagListTest, StandardAttributesConfigureItemsAndInputThroughSetOverloads) {
    TagList tags{
        ItemFont_t{Font{0.7f, Style::Bold}},
        ItemPadding_t{7, 3},
        ItemFillClr_t{Color{12, 34, 56, 220}},
        ItemLineColor_t{Color{1, 2, 3, 255}},
        ItemTextClr_t{Color{250, 251, 252, 255}},
        CornerFlags_t{CornerFlags::Rounded(6)},
        LabelDims_t{130, 26},
        ListDims_t{200, 140},
        Placeholder_t{"Search tags"},
        LabelFont_t{Font{0.55f, Style::Italic}},
        LabelFillClr_t{Color{240, 241, 242, 255}},
        LabelLineColor_t{Color{20, 21, 22, 255}},
        LabelColor_t{Color{3, 4, 5, 255}}
    };
    tags.set_tags(frame_tags({"alpha"}));
    tags.on_add([](const FrameTag&) {});
    tags.on_remove([](size_t, const FrameTag&) {});
    tags.update();

    auto chip = tags.flow().objects()[0].to<HorizontalLayout>();
    ASSERT_EQ(chip->objects().size(), 2u);
    auto label = chip->objects()[0].to<StaticText>();
    auto cross = chip->objects()[1].to<StaticText>();
    EXPECT_EQ((Color)chip->bg_fill_color(), Color(12, 34, 56, 220));
    EXPECT_EQ((Color)chip->bg_line_color(), Color(1, 2, 3, 255));
    EXPECT_EQ(chip->corner_flags(), CornerFlags::Rounded(6));
    EXPECT_EQ((Bounds)label->margins(), Bounds(7, 3, 1, 3));
    EXPECT_EQ((Bounds)cross->margins(), Bounds(0, 3, 7, 3));
    EXPECT_FLOAT_EQ(label->font().size, 0.7f);
    EXPECT_EQ(label->font().style, Style::Bold);
    EXPECT_EQ(label->text_color(), Color(250, 251, 252, 255));
    EXPECT_EQ(tags.input().size(), Size2(130, 26));
    EXPECT_EQ(tags.input().list().size(), Size2(200, 140));
    EXPECT_FLOAT_EQ(tags.input().textfield()->font().size, 0.55f);
    EXPECT_EQ(tags.input().textfield()->font().style, Style::Italic);
    EXPECT_EQ(tags.input().textfield()->fill_color(), Color(240, 241, 242, 255));
    EXPECT_EQ(tags.input().textfield()->line_color(), Color(20, 21, 22, 255));
    EXPECT_EQ(tags.input().textfield()->text_color(), Color(3, 4, 5, 255));
}

TEST(TagListTest, StandardAttributesAreAppliedByDynamicAttributeDelegation) {
    auto object = Layout::Make<TagList>{}();

    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        TagList::AllowNew_t{false}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        TagList::MatchThreshold_t{0.75f}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        ItemFont_t{Font{0.6f, Style::Bold}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(ItemPadding_t{6, 2}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        ItemFillClr_t{Color{40, 50, 60, 255}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        ItemLineColor_t{Color{10, 20, 30, 255}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        ItemTextClr_t{Color{220, 221, 222, 255}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        CornerFlags_t{CornerFlags::Rounded(5)}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(LabelDims_t{125, 24}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(ListDims_t{190, 135}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        Placeholder_t{"Find a tag"}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        LabelFont_t{Font{0.5f, Style::Italic}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        LabelFillClr_t{Color{230, 231, 232, 255}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        LabelLineColor_t{Color{70, 71, 72, 255}}, object));
    EXPECT_TRUE(LabeledField::delegate_to_proper_type(
        LabelColor_t{Color{4, 5, 6, 255}}, object));

    auto tags = object.to<TagList>();
    EXPECT_FALSE(tags->allow_new());
    EXPECT_FLOAT_EQ(tags->match_threshold(), 0.75f);
    tags->set_tags(frame_tags({"alpha"}));
    tags->on_add([](const FrameTag&) {});
    tags->update();

    auto chip = tags->flow().objects()[0].to<HorizontalLayout>();
    ASSERT_EQ(chip->objects().size(), 1u);
    auto label = chip->objects()[0].to<StaticText>();
    EXPECT_EQ((Bounds)label->margins(), Bounds(6, 2, 6, 2));
    EXPECT_FLOAT_EQ(label->font().size, 0.6f);
    EXPECT_EQ(label->font().style, Style::Bold);
    EXPECT_EQ(label->text_color(), Color(220, 221, 222, 255));
    EXPECT_EQ((Color)chip->bg_fill_color(), Color(40, 50, 60, 255));
    EXPECT_EQ((Color)chip->bg_line_color(), Color(10, 20, 30, 255));
    EXPECT_EQ(chip->corner_flags(), CornerFlags::Rounded(5));
    EXPECT_EQ(tags->input().size(), Size2(125, 24));
    EXPECT_EQ(tags->input().list().size(), Size2(190, 135));
    EXPECT_FLOAT_EQ(tags->input().textfield()->font().size, 0.5f);
    EXPECT_EQ(tags->input().textfield()->font().style, Style::Italic);
    EXPECT_EQ(tags->input().textfield()->fill_color(), Color(230, 231, 232, 255));
    EXPECT_EQ(tags->input().textfield()->line_color(), Color(70, 71, 72, 255));
    EXPECT_EQ(tags->input().textfield()->text_color(), Color(4, 5, 6, 255));
}

TEST(TagListTest, CallbacksAreControlledAndOnlyExposeTheirControls) {
    TagList tags;
    tags.set_tags(frame_tags({"alpha", "beta"}));
    tags.update();
    ASSERT_EQ(tags.flow().objects().size(), 2u);
    ASSERT_EQ(tags.children().size(), 1u);
    EXPECT_EQ(tags.children()[0], &tags.flow());
    EXPECT_EQ(tags.flow().objects()[0].to<HorizontalLayout>()->objects().size(), 1u);

    std::vector<std::pair<size_t, std::string>> removals;
    tags.on_remove([&](size_t index, const FrameTag& tag) {
        removals.emplace_back(index, tag.toStr());
    });
    tags.update();
    EXPECT_EQ(tags.flow().objects()[0].to<HorizontalLayout>()->objects().size(), 2u);
    tags.request_remove(1);
    ASSERT_THAT(removals, ::testing::ElementsAre(std::pair<size_t, std::string>{1, "beta"}));
    EXPECT_THAT(serialized_tags(tags.tags()), ::testing::ElementsAre("alpha", "beta"));

    std::vector<std::string> additions;
    tags.set_catalog({"alpha", "Gamma"});
    tags.on_add([&](const FrameTag& tag) { additions.push_back(tag.toStr()); });
    tags.update();
    EXPECT_EQ(tags.flow().objects().size(), 2u);
    ASSERT_THAT(tags.input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Gamma"}));

    tags.input().textfield()->set_text("draft");
    tags.set_catalog({"alpha", "Gamma", "Delta"});
    EXPECT_EQ(tags.input().textfield()->text(), "draft");
    ASSERT_EQ(tags.children().size(), 2u);
    EXPECT_EQ(tags.children()[0], &tags.flow());
    EXPECT_EQ(tags.children()[1], &tags.input());
    EXPECT_EQ(tags.input().parent(), &tags);
    EXPECT_NE(tags.input().parent(), &tags.flow());
    tags.input().textfield()->set_text(" gamma ");
    tags.input().textfield()->enter();
    EXPECT_THAT(additions, ::testing::ElementsAre("Gamma"));
    EXPECT_THAT(serialized_tags(tags.tags()), ::testing::ElementsAre("alpha", "beta"));

    tags.on_add({});
    tags.on_remove({});
    tags.update();
    ASSERT_EQ(tags.flow().objects().size(), 2u);
    ASSERT_EQ(tags.children().size(), 1u);
    EXPECT_EQ(tags.flow().objects()[0].to<HorizontalLayout>()->objects().size(), 1u);
}

TEST(TagListTest, OnlyTheTrailingCrossRequestsRemoval) {
    TagList tags;
    tags.set_tags(frame_tags({"alpha"}));
    std::vector<std::pair<size_t, std::string>> removals;
    tags.on_remove([&](size_t index, const FrameTag& tag) {
        removals.emplace_back(index, tag.toStr());
    });
    tags.update();

    DrawStructure graph(320, 120);
    graph.wrap_object(tags);
    auto chip = tags.flow().objects()[0].to<HorizontalLayout>();
    ASSERT_EQ(chip->objects().size(), 2u);
    auto label = chip->objects()[0].to<StaticText>();
    auto cross = chip->objects()[1].to<StaticText>();

    const auto label_center = center_of(*label);
    ASSERT_NE(graph.mouse_move(label_center.x, label_center.y), nullptr);
    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_NO_THROW(graph.mouse_up(true));
    EXPECT_TRUE(removals.empty());

    const auto cross_center = center_of(*cross);
    ASSERT_EQ(graph.mouse_move(cross_center.x, cross_center.y), cross);
    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_NO_THROW(graph.mouse_up(true));
    EXPECT_THAT(removals,
                ::testing::ElementsAre(std::pair<size_t, std::string>{0, "alpha"}));
    EXPECT_THAT(serialized_tags(tags.tags()), ::testing::ElementsAre("alpha"));
}

TEST(TagListTest, EntryHonorsThresholdCatalogOnlyAndExplicitSelection) {
    TagList tags;
    std::vector<std::string> additions;
    tags.on_add([&](const FrameTag& tag) { additions.push_back(tag.toStr()); });
    tags.set_catalog({"Alpha", "Beta"});

    tags.input().textfield()->set_text("Alph");
    tags.input().textfield()->enter();
    ASSERT_THAT(additions, ::testing::ElementsAre("Alph"));

    tags.set_match_threshold(0.75f);
    tags.input().textfield()->set_text("Alph");
    tags.input().textfield()->enter();
    ASSERT_THAT(additions, ::testing::ElementsAre("Alph", "Alpha"));

    tags.set_allow_new(false);
    tags.set_match_threshold(1.f);
    tags.input().textfield()->set_text("Alph");
    tags.input().textfield()->enter();
    EXPECT_THAT(additions, ::testing::ElementsAre("Alph", "Alpha"));
    EXPECT_EQ(tags.input().textfield()->text(), "Alph");

    ASSERT_GE(tags.input().items().size(), 2u);
    // Unfiltered Dropdown mouse selections currently carry an invalid raw index;
    // the canonical item still has to be accepted.
    tags.input().on_select()(Dropdown::RawIndex{}, tags.input().items()[1]);
    EXPECT_THAT(additions, ::testing::ElementsAre("Alph", "Alpha", "Beta"));
    EXPECT_TRUE(tags.input().textfield()->text().empty());
}

TEST(TagListTest, SelectedCanonicalValuesNeverDispatchDuplicates) {
    TagList tags;
    std::vector<std::string> additions;
    tags.on_add([&](const FrameTag& tag) { additions.push_back(tag.toStr()); });
    tags.set_tags(frame_tags({"Alpha"}));
    tags.set_catalog({"Alpha", "Beta"});

    tags.input().textfield()->set_text(" alpha ");
    tags.input().textfield()->enter();
    EXPECT_TRUE(additions.empty());
    EXPECT_TRUE(tags.input().textfield()->text().empty());
    ASSERT_THAT(tags.input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Beta"}));

    tags.input().textfield()->set_text("   ");
    tags.input().textfield()->enter();
    EXPECT_TRUE(additions.empty());
    EXPECT_TRUE(tags.input().textfield()->text().empty());
}

TEST(TagListTest, LocalizedTagsMatchCatalogAndTypedNames) {
    TagList tags;
    tags.set_tags(std::vector<FrameTag>{FrameTag{
        .name = SpatialTag{Bounds{1, 2, 3, 4}, std::string{"Alpha"}}
    }});
    tags.set_catalog({"Alpha", "Beta"});

    std::vector<FrameTag> additions;
    tags.on_add([&](const FrameTag& tag) { additions.push_back(tag); });

    ASSERT_THAT(tags.input().items(),
                ::testing::ElementsAre(Dropdown::TextItem{"Beta"}));

    tags.input().textfield()->set_text(" alpha ");
    tags.input().textfield()->enter();
    EXPECT_TRUE(additions.empty());
}

TEST(TagListTest, DisplayFilterAppliesToChipsAndDropdownItems) {
    TagList tags;
    tags.set(TagList::DisplayFilter{[](const std::string& value) {
        return "display:" + value;
    }});
    tags.set_tags(frame_tags({"Alpha"}));
    tags.set_catalog({"Alpha", "Beta"});
    tags.update();

    auto chip = tags.flow().objects()[0].to<HorizontalLayout>();
    ASSERT_FALSE(chip->objects().empty());
    EXPECT_EQ(chip->objects()[0].to<StaticText>()->text(), "display:Alpha");

    ASSERT_EQ(tags.input().items().size(), 1u);
    EXPECT_EQ(tags.input().items()[0].name(), "Beta");
    EXPECT_EQ(tags.input().items()[0].display_name(),
              std::optional<std::string>{"display:Beta"});
}

TEST(TagListTest, EnterCommitsAnArrowHighlightedCanonicalSuggestion) {
    TagList tags;
    std::vector<std::string> additions;
    tags.on_add([&](const FrameTag& tag) { additions.push_back(tag.toStr()); });
    tags.set_catalog({"Alpha", "Beta"});

    // Dropdown's Up/Down handler uses select_item; exercising the resulting
    // highlight here keeps this test independent of platform key-code routing.
    tags.input().textfield()->set_text("be");
    tags.input().before_draw();
    tags.input().select_item(Dropdown::RawIndex{1});
    tags.input().textfield()->enter();

    EXPECT_THAT(additions, ::testing::ElementsAre("Beta"));
    EXPECT_TRUE(tags.input().textfield()->text().empty());
}

TEST(TagListTest, ControlledRefreshPreservesDraftAndRecomputesSuggestions) {
    TagList tags;
    tags.on_add([](const FrameTag&) {});
    tags.set(attr::SizeLimit{Size2{100, 40}});
    tags.set_catalog({"Alpha", "Beta", "Gamma"});
    tags.set_tags(frame_tags({"Alpha"}));
    tags.input().textfield()->set_text("unfinished");
    tags.update();
    ASSERT_THAT(tags.input().items(), ::testing::ElementsAre(
        Dropdown::TextItem{"Beta"}, Dropdown::TextItem{"Gamma"}));
    ASSERT_TRUE(tags.flow().scroll_enabled());
    ASSERT_GT(tags.flow().scroll_limit_y().end, 0);
    const auto retained_offset = tags.flow().scroll_limit_y().end * 0.5f;
    tags.flow().set_scroll_offset(Vec2(0, retained_offset));

    tags.set_tags(frame_tags({"Beta"}));
    EXPECT_THAT(tags.input().items(), ::testing::ElementsAre(
        Dropdown::TextItem{"Alpha"}, Dropdown::TextItem{"Gamma"}));
    tags.update();
    EXPECT_EQ(tags.input().textfield()->text(), "unfinished");
    EXPECT_THAT(tags.input().items(), ::testing::ElementsAre(
        Dropdown::TextItem{"Alpha"}, Dropdown::TextItem{"Gamma"}));
    EXPECT_EQ(tags.flow().scroll_offset(), Vec2(0, retained_offset));
}

TEST(TagListTest, RemovePayloadSurvivesSynchronousControlledRefresh) {
    TagList tags;
    tags.set_tags(frame_tags({"Alpha", "Beta"}));
    std::string removed;
    tags.on_remove([&](size_t, const FrameTag& tag) {
        tags.set_tags(frame_tags({"Replacement"}));
        removed = tag.toStr();
    });

    tags.request_remove(1);
    EXPECT_EQ(removed, "Beta");
    EXPECT_THAT(serialized_tags(tags.tags()), ::testing::ElementsAre("Replacement"));
}

TEST(DynamicGUITagListTest, ResolvesSourcesStylesAndControlledActions) {
    constexpr std::string_view json = R"json(
{
  "type": "taglist",
  "var": "tags",
  "catalog": "catalog",
  "add_action": "add_tag",
  "remove_action": "remove_tag",
  "max_size": "{limit}",
  "allow_new": "{can_add}",
  "match_threshold": "{threshold}",
  "item": {
    "pad": [7, 3],
    "font": { "size": 0.7, "style": "bold" },
    "fill": "{chip_fill}",
    "line": [1, 2, 3, 255],
    "color": [250, 251, 252, 255],
    "corners": [6]
  },
  "input": {
    "size": "{input_size}",
    "list_size": [200, 140],
    "placeholder": "Search tags",
    "font": { "size": 0.55, "style": "italic" },
    "fill": [240, 241, 242, 255],
    "line": [20, 21, 22, 255],
    "color": [3, 4, 5, 255]
  }
}
)json";

    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    std::vector<std::string> selected{"beta", "alpha"};
    std::vector<std::string> catalog{"Alpha", "Beta", "Gamma"};
    Size2 limit{210, 80};
    bool can_add{false};
    float threshold{0.8f};
    Color chip_fill{12, 34, 56, 220};
    Size2 input_size{130, 26};
    std::vector<Action> add_actions;
    std::vector<Action> remove_actions;
    Context context{
        VarFunc("tags", [&selected](const VarProps&) -> std::vector<std::string> { return selected; }),
        VarFunc("catalog", [&catalog](const VarProps&) -> std::vector<std::string> { return catalog; }),
        VarFunc("limit", [&limit](const VarProps&) -> Size2 { return limit; }),
        VarFunc("can_add", [&can_add](const VarProps&) -> bool { return can_add; }),
        VarFunc("threshold", [&threshold](const VarProps&) -> float { return threshold; }),
        VarFunc("chip_fill", [&chip_fill](const VarProps&) -> Color { return chip_fill; }),
        VarFunc("input_size", [&input_size](const VarProps&) -> Size2 { return input_size; }),
        ActionFunc("add_tag", [&add_actions](Action action) { add_actions.push_back(std::move(action)); }),
        ActionFunc("remove_tag", [&remove_actions](Action action) { remove_actions.push_back(std::move(action)); })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<TagList>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    auto tag_list = root.to<TagList>();
    tag_list->update();

    EXPECT_THAT(serialized_tags(tag_list->tags()),
                ::testing::ElementsAre("beta", "alpha"));
    EXPECT_THAT(tag_list->catalog(), ::testing::ElementsAre("Alpha", "Beta", "Gamma"));
    EXPECT_EQ(tag_list->flow().max_size(), (attr::SizeLimit{210, 44}));
    EXPECT_FALSE(tag_list->allow_new());
    EXPECT_FLOAT_EQ(tag_list->match_threshold(), 0.8f);
    ASSERT_EQ(tag_list->flow().objects().size(), 2u);
    auto first_chip = tag_list->flow().objects()[0].to<HorizontalLayout>();
    EXPECT_EQ((Color)first_chip->bg_fill_color(), Color(12, 34, 56, 220));
    EXPECT_EQ((Color)first_chip->bg_line_color(), Color(1, 2, 3, 255));
    EXPECT_EQ(first_chip->corner_flags(), CornerFlags::Rounded(6));
    ASSERT_EQ(first_chip->objects().size(), 2u);
    auto first_label = first_chip->objects()[0].to<StaticText>();
    auto first_remove = first_chip->objects()[1].to<StaticText>();
    EXPECT_EQ((Bounds)first_label->margins(), Bounds(7, 3, 1, 3));
    EXPECT_EQ((Bounds)first_remove->margins(), Bounds(0, 3, 7, 3));
    EXPECT_FLOAT_EQ(first_label->font().size, 0.7f);
    EXPECT_EQ(first_label->font().style, Style::Bold);
    EXPECT_EQ(first_label->text_color(), Color(250, 251, 252, 255));
    EXPECT_EQ(tag_list->input().size(), Size2(130, 26));
    EXPECT_EQ(tag_list->input().list().size(), Size2(200, 140));
    EXPECT_FLOAT_EQ(tag_list->input().textfield()->font().size, 0.55f);
    EXPECT_EQ(tag_list->input().textfield()->font().style, Style::Italic);
    EXPECT_EQ(tag_list->input().textfield()->fill_color(), Color(240, 241, 242, 255));
    EXPECT_EQ(tag_list->input().textfield()->line_color(), Color(20, 21, 22, 255));
    EXPECT_EQ(tag_list->input().textfield()->text_color(), Color(3, 4, 5, 255));
    EXPECT_EQ((Color)tag_list->input().list().list_fill_clr(), Color(240, 241, 242, 255));
    EXPECT_EQ((Color)tag_list->input().list().list_line_clr(), Color(20, 21, 22, 255));
    EXPECT_EQ(tag_list->input().list().text_color(), Color(3, 4, 5, 255));
    ASSERT_THAT(tag_list->input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Gamma"}));

    tag_list->input().on_select()(Dropdown::RawIndex{0}, tag_list->input().items()[0]);
    ASSERT_EQ(add_actions.size(), 1u);
    EXPECT_THAT(add_actions[0].parameters, ::testing::ElementsAre("Gamma"));
    tag_list->request_remove(1);
    ASSERT_EQ(remove_actions.size(), 1u);
    EXPECT_THAT(remove_actions[0].parameters, ::testing::ElementsAre("alpha", "1"));

    selected = {"Gamma"};
    catalog = {"Gamma", "Delta"};
    limit = Size2(120, 40);
    can_add = true;
    threshold = 0.6f;
    chip_fill = Color(60, 70, 80, 230);
    input_size = Size2(170, 30);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    tag_list->update();
    EXPECT_THAT(serialized_tags(tag_list->tags()), ::testing::ElementsAre("Gamma"));
    EXPECT_THAT(tag_list->input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Delta"}));
    EXPECT_EQ(tag_list->flow().max_size(), (attr::SizeLimit{120, 0}));
    EXPECT_TRUE(tag_list->allow_new());
    EXPECT_FLOAT_EQ(tag_list->match_threshold(), 0.6f);
    EXPECT_EQ(tag_list->input().size(), input_size);
    EXPECT_EQ((Color)tag_list->flow().objects()[0].to<HorizontalLayout>()->bg_fill_color(), chip_fill);
}

TEST(DynamicGUITagListTest, SupportsLiteralAndGlobalCatalogsAndDisplayOnlyMode) {
    SETTING(dyngui_taglist_catalog) = std::vector<std::string>{"Global", "Other"};

    constexpr std::string_view literal_json = R"json(
{
  "type": "taglist",
  "var": "tags",
  "catalog": ["Literal", "Second"],
  "add_action": "add:{tag}:fixed",
  "remove_action": "remove:{index}:{tag}:fixed"
}
)json";
    constexpr std::string_view global_json = R"json(
{
  "type": "taglist",
  "var": "tags",
  "catalog": "{global.dyngui_taglist_catalog}",
  "add_action": "add"
}
)json";
    constexpr std::string_view display_json = R"json(
{
  "type": "taglist",
  "var": ["Current"]
}
)json";

    std::vector<std::string> selected{"Current"};
    std::vector<Action> actions;
    std::vector<Action> removals;
    Context context{
        VarFunc("tags", [&selected](const VarProps&) -> std::vector<std::string> { return selected; }),
        ActionFunc("add", [&actions](Action action) { actions.push_back(std::move(action)); }),
        ActionFunc("remove", [&removals](Action action) { removals.push_back(std::move(action)); })
    };
    DrawStructure graph(640, 480);

    auto parse_tag_list = [&](std::string_view source) {
        glz::json_t obj;
        const auto error = glz::read_json(obj, source);
        EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, source);
        State state;
        auto handler = std::make_shared<CurrentObjectHandler>();
        state._current_object_handler = handler;
        auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
        EXPECT_TRUE(root.is<TagList>());
        EXPECT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
        root.to<TagList>()->update();
        return std::pair{root, handler};
    };

    auto [literal, literal_handler] = parse_tag_list(literal_json);
    auto literal_tags = literal.to<TagList>();
    EXPECT_THAT(literal_tags->catalog(), ::testing::ElementsAre("Literal", "Second"));
    literal_tags->input().on_select()(Dropdown::RawIndex{0}, literal_tags->input().items()[0]);
    ASSERT_EQ(actions.size(), 1u);
    EXPECT_THAT(actions[0].parameters, ::testing::ElementsAre("Literal", "fixed"));
    literal_tags->request_remove(0);
    ASSERT_EQ(removals.size(), 1u);
    EXPECT_THAT(removals[0].parameters, ::testing::ElementsAre("0", "Current", "fixed"));

    auto [global, global_handler] = parse_tag_list(global_json);
    EXPECT_THAT(global.to<TagList>()->catalog(), ::testing::ElementsAre("Global", "Other"));

    auto [display, display_handler] = parse_tag_list(display_json);
    auto display_tags = display.to<TagList>();
    ASSERT_EQ(display_tags->flow().objects().size(), 1u);
    EXPECT_EQ(display_tags->flow().objects()[0].to<HorizontalLayout>()->objects().size(), 1u);
}

TEST(DynamicGUITagListTest, SupportsSetAndJsonArrayVariableSources) {
    constexpr std::string_view json = R"json(
{
  "type": "taglist",
  "var": "selected",
  "catalog": "catalog",
  "add_action": "add"
}
)json";
    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    std::set<std::string> selected{"Beta", "Alpha"};
    glz::json_t catalog;
    parse_error = glz::read_json(catalog, R"json(["Alpha", "Beta", "Gamma"])json");
    ASSERT_EQ(parse_error, glz::error_code::none);
    Context context{
        VarFunc("selected", [&selected](const VarProps&) -> std::set<std::string> { return selected; }),
        VarFunc("catalog", [&catalog](const VarProps&) -> glz::json_t { return catalog; }),
        ActionFunc("add", [](Action) {})
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<TagList>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    auto tag_list = root.to<TagList>();
    EXPECT_THAT(serialized_tags(tag_list->tags()),
                ::testing::ElementsAre("Alpha", "Beta"));
    EXPECT_THAT(tag_list->catalog(), ::testing::ElementsAre("Alpha", "Beta", "Gamma"));
    EXPECT_THAT(tag_list->input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Gamma"}));

    selected = {"Gamma"};
    parse_error = glz::read_json(catalog, R"json(["Gamma", "Delta"])json");
    ASSERT_EQ(parse_error, glz::error_code::none);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    EXPECT_THAT(serialized_tags(tag_list->tags()), ::testing::ElementsAre("Gamma"));
    EXPECT_THAT(tag_list->input().items(), ::testing::ElementsAre(Dropdown::TextItem{"Delta"}));
}

TEST(DynamicGUITagListTest, ReportsMissingControlledSourceAndMissingAddCatalog) {
    Context context;
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    auto parse = [&](std::string_view source) {
        glz::json_t obj;
        const auto error = glz::read_json(obj, source);
        EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, source);
        return parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    };

    EXPECT_TRUE(parse(R"json({"type":"taglist"})json").is<ErrorElement>());
    EXPECT_TRUE(parse(R"json({"type":"taglist","var":[],"add_action":"add"})json").is<ErrorElement>());
}

TEST(DynamicGUIFloatingLayoutTest, ParsesPolicyAndTracksDynamicMaxSize) {
    constexpr std::string_view json = R"json(
{
  "type": "floatinglayout",
  "policy": "vertical-first",
  "max_size": "{limit}",
  "pad": [1, 2, 3, 4],
  "outer_pad": [3, 4, 5, 6],
  "children": [
    { "type": "rect", "size": [10, 10] },
    { "type": "rect", "size": [10, 10] },
    { "type": "rect", "size": [10, 10] }
  ]
}
)json";
    glz::json_t obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    Size2 limit{25, 45};
    Context context{
        VarFunc("limit", [&limit](const VarProps&) -> Size2 { return limit; })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<FloatingLayout>());
    graph.wrap_object(*root);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    auto layout = root.to<FloatingLayout>();
    layout->update();
    EXPECT_EQ(layout->policy(), FloatingLayout::Policy::VerticalFirst);
    EXPECT_EQ(layout->max_size(), attr::SizeLimit{limit});
    ASSERT_EQ(layout->objects().size(), 3u);
    EXPECT_EQ(layout->objects()[0]->pos(), Vec2(4, 6));
    EXPECT_EQ(layout->objects()[1]->pos(), Vec2(4, 22));
    EXPECT_EQ(layout->objects()[2]->pos(), Vec2(18, 6));
    EXPECT_EQ(layout->scroll_axis(), ScrollAxis::Horizontal);

    limit = Size2(40, 25);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    layout->update();
    EXPECT_EQ(layout->max_size(), attr::SizeLimit{limit});
    EXPECT_EQ(layout->objects()[0]->pos(), Vec2(4, 6));
    EXPECT_EQ(layout->objects()[1]->pos(), Vec2(18, 6));
    EXPECT_EQ(layout->objects()[2]->pos(), Vec2(32, 6));
}

TEST(DynamicGUIFloatingLayoutTest, ReportsUnknownPolicyAsAnErrorElement) {
    constexpr std::string_view json = R"json(
{
  "type": "floatinglayout",
  "policy": "diagonal",
  "children": []
}
)json";
    glz::json_t obj;
    const auto error = glz::read_json(obj, json);
    ASSERT_EQ(error, glz::error_code::none) << glz::format_error(error, json);

    Context context;
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    auto result = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    EXPECT_TRUE(result.is<ErrorElement>());
}
