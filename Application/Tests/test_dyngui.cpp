#include <commons.pc.h>
#include "gtest/gtest.h"
#include <misc/parse_parameter_lists.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>
#include <gui/DrawBase.h>
#include <gui/DynamicGUI.h>
#include <gui/DynamicVariable.h>
#include <gui/LabeledField.h>
#include <gui/Passthrough.h>
#include <gui/dyn/ParseText.h>
#include <gui/dyn/ResolveVariable.h>
#include <gui/dyn/binders.h>
#include <gui/types/ListItemTypes.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/ErrorElement.h>
#include <gui/types/Layout.h>
#include <gui/types/SettingsTooltip.h>
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

/// Local settings are only readable as strings now, so parse the value back
/// into the declared type for the typed expectations below.
template<typename T>
static T local_value(const Context& context, std::string_view name) {
    return Meta::fromStr<T>(context.local_setting_value_string(name));
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
static_assert(cmn::gui::dyn::takes_attribute<TagList, ItemCornerFlags>);
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

    EXPECT_TRUE(local_value<bool>(context, "local.enabled"));
    EXPECT_EQ(local_value<int>(context, "local.count"), 4);
    EXPECT_DOUBLE_EQ(local_value<double>(context, "local.ratio"), 0.5);
    EXPECT_EQ(local_value<std::string>(context, "local.label"), "hello");
    EXPECT_EQ(local_value<file::Path>(context, "local.dataset"), file::Path("/tmp/data.yaml"));
    EXPECT_THAT(local_value<std::vector<int>>(context, "local.selected_ids"), ::testing::ElementsAre(1, 2, 3));
    EXPECT_EQ(local_value<Vec2>(context, "local.origin"), Vec2(2, 3));
    EXPECT_EQ(local_value<Size2>(context, "local.panel_size"), Size2(100, 40));
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
    EXPECT_EQ(local_value<std::string>(context, "local.local_string"), "abc");
    EXPECT_EQ(local_value<std::vector<int>>(context, "local.local_ids"), std::vector<int>({5, 6}));
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
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    EXPECT_EQ(parse_text("{local.title}", context, state), "before");
    EXPECT_EQ(parse_text("{if:{local.enabled}:yes:no}", context, state), "yes");

    context.update_local_setting(handler.get(), "local.title", [](sprite::Reference& ref) {
        ref.get().set_value_from_string("after");
    });
    context.update_local_setting(handler.get(), "local.enabled", [](sprite::Reference& ref) {
        ref.get() = false;
    });
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

    EXPECT_EQ(local_value<int>(gui.context, "local.tab"), 0);

    const auto button_center = center_of(*button);
    ASSERT_NO_THROW(graph.mouse_move(button_center.x, button_center.y));
    ASSERT_NO_THROW(graph.mouse_down(true));
    ASSERT_NO_THROW(graph.mouse_up(true));

    EXPECT_EQ(local_value<int>(gui.context, "local.tab"), 1);
}

TEST(DynamicGUILocalSettings, OptionalMapInsertionAcceptsFirstAndFalseLikeNames) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "new_name": { "type": "string", "value": "" }
  },
  "objects": [
    {
      "type": "button",
      "text": "Add entry",
      "action": "set:entries:{if:{nequal:{local.new_name}:''}:{insert:{if:{global.entries}:{+:1:{max_element:{keys:{global.entries}}}}:0}:{local.new_name}:{if:{global.entries}:{global.entries}:'{}'}}:{global.entries}}"
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    auto [defaults, objects] = std::move(loaded.value());
    const auto action = PreAction::fromStr(objects.get_array().front().get_object().at("action").get_string());

    using EntryMap = std::map<uint16_t, std::string>;
    using Entries = std::optional<EntryMap>;
    sprite::Map global;
    Context context{
        VarFunc("global", [&](const VarProps&) -> sprite::Map& { return global; })
    };
    context.apply_local_settings(defaults);
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    for(const auto& initial : std::vector<Entries>{std::nullopt, EntryMap{}, EntryMap{{2, "fish"}, {10, "ray"}}}) {
        SCOPED_TRACE(Meta::toStr(initial));
        for(const auto* name : {"", "fish", "0", "false", "null", R"(fish, juvenile: "spotted")"}) {
            SCOPED_TRACE(name);
            global["entries"] = initial;
            context.update_local_setting(handler.get(), "local.new_name", [&](sprite::Reference& ref) {
                ref.get() = std::string(name);
            });
            auto expected = initial;
            if(std::string_view(name) != "") {
                expected = initial.value_or(EntryMap{});
                const auto index = expected->empty() ? 0 : expected->rbegin()->first + 1;
                expected->emplace(index, name);
            }

            const auto parsed = action.parse(context, state);
            EXPECT_EQ(parsed.name, "set");
            ASSERT_EQ(parsed.parameters.size(), 2u);
            EXPECT_EQ(parsed.parameters.front(), "entries");
            EXPECT_EQ(Meta::fromStr<Entries>(parsed.parameters.back()), expected);
        }
    }
}

TEST(ListElementTest, GeneratedCommandPreservesMapParameter) {
    glz::generic obj = glz::generic::object_t{
        {"type", "list"},
        {"var", "[0]"},
        {"template", glz::generic::object_t{
            {"text", "Class"},
            {"action", "{if:true:{command}:noop}"}
        }}
    };

    std::string value = R"({0:"Fish-School"})";
    size_t evaluations = 0;
    std::vector<Action> received_actions;
    Context context{
        VarFunc("command", [&](const VarProps&) -> std::string {
            ++evaluations;
            return "set:detect_classes:" + value;
        }),
        ActionFunc("set", [&](Action action) {
            received_actions.push_back(std::move(action));
        })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<ScrollableList<DetailTooltipItem>>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    auto list = root.to<ScrollableList<DetailTooltipItem>>();
    ASSERT_EQ(list->items().size(), 1u);
    EXPECT_TRUE(received_actions.empty());

    value = R"({0:"Fish-School",3:"new_class"})";
    evaluations = 0;
    ASSERT_NO_THROW(list->select_item(0));
    ASSERT_EQ(received_actions.size(), 1u);
    EXPECT_EQ(received_actions[0].name, "set");
    EXPECT_THAT(received_actions[0].parameters, ::testing::ElementsAre("detect_classes", value));
    EXPECT_EQ(evaluations, 1u);
}

TEST(EventBindingTest, ResolvesCommandsOnceAcrossWidgetHandlers) {
    for(const auto* type : {"button", "checkbox", "rect", "textfield", "list"}) {
        SCOPED_TRACE(type);
        for(const auto* source : {"set:detect_classes:{value}", "{if:true:{command}:noop}"}) {
            SCOPED_TRACE(source);
            glz::generic obj = glz::generic::object_t{
                {"type", type},
                {"size", glz::generic::array_t{160, 40}},
                {"clickable", true}
            };
            if(std::string_view(type) == "list") {
                obj["items"] = glz::generic::array_t{
                    glz::generic::object_t{{"text", "Class"}, {"action", source}}
                };
            } else {
                obj[std::string_view(type) == "rect" ? "click" : "action"] = source;
            }

            std::string value = R"({0:"Fish-School"})";
            size_t evaluations = 0;
            std::vector<Action> received_actions;
            Context context{
                VarFunc("value", [&](const VarProps&) -> std::string {
                    ++evaluations;
                    return value;
                }),
                VarFunc("command", [&](const VarProps&) -> std::string {
                    ++evaluations;
                    return "set:detect_classes:" + value;
                }),
                ActionFunc("set", [&](Action action) {
                    received_actions.push_back(std::move(action));
                })
            };
            State state;
            auto handler = std::make_shared<CurrentObjectHandler>();
            state._current_object_handler = handler;
            DrawStructure graph(640, 480);
            auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
            ASSERT_TRUE(root);
            ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
            EXPECT_TRUE(received_actions.empty());

            value = R"({0:"Fish-School",3:"new_class"})";
            evaluations = 0;
            if(std::string_view(type) == "list") {
                ASSERT_NO_THROW(root.to<ScrollableList<DetailTooltipItem>>()->select_item(0));
            } else if(std::string_view(type) == "textfield") {
                ASSERT_NO_THROW(root.to<Textfield>()->enter());
            } else {
                graph.wrap_object(*root);
                const auto target_center = center_of(*root);
                ASSERT_NO_THROW(graph.mouse_move(target_center.x, target_center.y));
                ASSERT_NO_THROW(graph.mouse_down(true));
                ASSERT_NO_THROW(graph.mouse_up(true));
            }

            ASSERT_EQ(received_actions.size(), 1u);
            EXPECT_EQ(received_actions[0].name, "set");
            EXPECT_THAT(received_actions[0].parameters, ::testing::ElementsAre("detect_classes", value));
            EXPECT_EQ(evaluations, 1u);
        }
    }
}

TEST(DynamicGUIButtonTest, RoutesConditionalActionFromCurrentLocalValue) {
    constexpr std::string_view json = R"json(
{
  "locals": {
    "tab": { "type": "int", "value": 0 }
  },
  "objects": [
    {
      "type": "button",
      "name": "export-button",
      "text": "Export",
      "size": [160, 40],
      "action": "{if:{equal:{local.tab}:0}:export:export-behavior}"
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    size_t export_actions = 0;
    size_t behavior_actions = 0;
    Context context{
        ActionFunc("export", [&](const Action&) {
            ++export_actions;
        }),
        ActionFunc("export-behavior", [&](const Action&) {
            ++behavior_actions;
        })
    };
    context.apply_local_settings(defaults);
    context.defaults = std::move(defaults);

    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, objects.get_array().front().get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<Button>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    auto click_button = [&]() {
        graph.wrap_object(*root);
        const auto button_center = center_of(*root);
        EXPECT_NO_THROW(graph.mouse_move(button_center.x, button_center.y));
        EXPECT_NO_THROW(graph.mouse_down(true));
        EXPECT_NO_THROW(graph.mouse_up(true));
    };

    /// reset counters
    export_actions = behavior_actions = 0;
    click_button();
    EXPECT_EQ(export_actions, 1u);
    EXPECT_EQ(behavior_actions, 0u);

    context.update_local_setting(handler.get(), "local.tab", [](auto&ref) {
        ref.get() = 1;
    });

    /// reset counters, we have a local setting called "local.tab" now with value int(1).
    /// before, it was 0. so next we should select behavior:
    export_actions = behavior_actions = 0;

    click_button();
    EXPECT_EQ(export_actions, 0u);
    EXPECT_EQ(behavior_actions, 1u);
}

TEST(DynamicGUIButtonTest, KeepsInlineConditionTextAndActionInSyncAcrossMissingScopeChanges) {
    constexpr std::string_view json = R"json(
{
  "type": "button",
  "name": "conditional-button",
  "text": "{if:{not:{.missing}}:primary:fallback}",
  "size": [160, 40],
  "action": "{if:{not:{.missing}}:primary:fallback}"
}
)json";

    glz::generic object;
    const auto parse_error = glz::read_json(object, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    size_t primary_actions = 0;
    size_t fallback_actions = 0;
    Context context{
        ActionFunc("primary", [&](const Action&) { ++primary_actions; }),
        ActionFunc("fallback", [&](const Action&) { ++fallback_actions; })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<Button>());
    auto button = root.to<Button>();

    auto update_button = [&]() {
        EXPECT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
        graph.wrap_object(*root);
    };
    auto click_button = [&]() {
        const auto button_center = center_of(*root);
        EXPECT_NO_THROW(graph.mouse_move(button_center.x, button_center.y));
        EXPECT_NO_THROW(graph.mouse_down(true));
        EXPECT_NO_THROW(graph.mouse_up(true));
    };

    /// reset actions counter
    primary_actions = fallback_actions = 0;

    update_button();
    EXPECT_EQ(button->txt(), "primary");
    click_button();
    EXPECT_EQ(primary_actions, 1u);
    EXPECT_EQ(fallback_actions, 0u);

    {
        auto scope = handler->scope();
        scope.set("missing", VarFunc("missing", [](const VarProps&) -> bool {
            return true;
        }).second);

        /// reset actions counter. now we should get fallback because missing is true
        primary_actions = fallback_actions = 0;

        update_button();
        EXPECT_EQ(button->txt(), "fallback");
        click_button();
        EXPECT_EQ(primary_actions, 0u);
        EXPECT_EQ(fallback_actions, 1u);
    }

    /// reset actions counter. we should get primary again because
    /// the scope should have been destroyed
    primary_actions = fallback_actions = 0;

    update_button();
    EXPECT_EQ(button->txt(), "primary");
    click_button();
    EXPECT_EQ(primary_actions, 1u);
    EXPECT_EQ(fallback_actions, 0u);
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
    EXPECT_EQ(local_value<int>(gui.context, "local.tab"), 31);

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

    glz::generic object;
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

    glz::generic object;
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
    auto handler = std::make_shared<CurrentObjectHandler>();
    context.update_local_setting(handler.get(), "local.title", [](sprite::Reference& ref) {
        ref.get().set_value_from_string("edited");
    });

    auto second = load(std::string(json));
    ASSERT_TRUE(second.has_value()) << second.error();
    auto [reloaded_defaults, reloaded_objects] = std::move(second.value());
    (void)reloaded_objects;
    context.apply_local_settings(reloaded_defaults);

    EXPECT_EQ(local_value<std::string>(context, "local.title"), "edited");
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
    auto handler = std::make_shared<CurrentObjectHandler>();
    context.update_local_setting(handler.get(), "local.value", [](sprite::Reference& ref) {
        ref.get().set_value_from_string("edited");
    });

    auto second = load(std::string(int_json));
    ASSERT_TRUE(second.has_value()) << second.error();
    auto [reloaded_defaults, reloaded_objects] = std::move(second.value());
    (void)reloaded_objects;
    context.apply_local_settings(reloaded_defaults);

    EXPECT_EQ(local_value<int>(context, "local.value"), 7);
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

TEST(DynamicGUISettingsTest, PathFieldRefreshesAfterSettingChanges) {
    sprite::Map settings;
    settings["path"] = file::Path("before");
    GUITaskQueue_t queue;
    LabeledPath field(&queue, "path", "", file::Path("before"), &settings);
    ASSERT_EQ(field._dropdown->textfield()->text(), "before");

    settings["path"] = file::Path("after");
    field.update_ref_in_main_thread();
    EXPECT_EQ(field._dropdown->textfield()->text(), "after");
}

TEST(DynamicGUISettingsTest, OptionalCornersReachValueAndResetWithoutImplicitLabelOverride) {
    SETTING(dyngui_test_optional_corner_text) = std::optional<std::string>{"value"};
    SETTING(dyngui_test_optional_corner_path) = std::optional<file::Path>{file::Path("value")};

    constexpr std::string_view json = R"json(
{
  "objects": [
    { "type": "settings", "desc": "", "corners": ["top", 6] },
    { "type": "settings", "desc": "", "corners": ["none"], "label": { "color": [255,255,255] } },
    { "type": "settings", "desc": "", "corners": ["none"], "label": { "corners": ["right", 4] } }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    auto [defaults, objects] = std::move(loaded.value());
    const std::array<CornerFlags, 3> value_corners{
        CornerFlags(CornerFlags::TopLeft, 6), CornerFlags::Square(), CornerFlags(CornerFlags::None, 4)
    };
    const std::array<CornerFlags, 3> reset_corners{
        CornerFlags(CornerFlags::TopRight, 6), CornerFlags::Square(), CornerFlags::Right(4)
    };

    for(const auto* name : {"dyngui_test_optional_corner_text", "dyngui_test_optional_corner_path"}) {
        SCOPED_TRACE(name);
        for(size_t i = 0; i < objects.get_array().size(); ++i) {
            SCOPED_TRACE(i);
            auto object = objects.get_array().at(i);
            object["var"] = std::string(name);

            GUITaskQueue_t queue;
            Context context;
            context.defaults = defaults;
            State state;
            auto handler = std::make_shared<CurrentObjectHandler>();
            state._current_object_handler = handler;

            auto root = parse_object(&queue, object.get_object(), context, state, context.defaults);
            ASSERT_TRUE(root.is<HorizontalLayout>());
            const auto& children = root.to<HorizontalLayout>()->objects();
            ASSERT_EQ(children.size(), 2u);
            if(children.front().is<Dropdown>()) {
                EXPECT_EQ(children.front().to<Dropdown>()->textfield()->corner_flags(), value_corners.at(i));
            } else {
                ASSERT_TRUE(children.front().is<Textfield>());
                EXPECT_EQ(children.front().to<Textfield>()->corner_flags(), value_corners.at(i));
            }
            ASSERT_TRUE(children.back().is<Button>());
            EXPECT_EQ(children.back().to<Button>()->corner_flags(), reset_corners.at(i));
        }
    }
}

TEST(DynamicGUISettingsTest, UnsetOptionalPathInComboboxKeepsSharedEdgeSquare) {
    GlobalSettings::write([](Configuration& config) {
        config.values["dyngui_test_unset_optional_path"] = std::optional<file::Path>{};
        config.defaults["dyngui_test_unset_optional_path"] = std::optional<file::Path>{};
    });

    GUITaskQueue_t queue;
    DrawStructure graph(640, 480);
    Combobox combo(&queue, Box(0, 0, 500, 40));
    combo.set(ParmName{"dyngui_test_unset_optional_path"});
    graph.wrap_object(combo);
    graph.collect();

    Entangled* placeholder = nullptr;
    for(auto* child : combo.children()) {
        if(auto* layout = dynamic_cast<HorizontalLayout*>(child)) {
            const auto& objects = layout->objects();
            ASSERT_EQ(objects.size(), 2u);
            ASSERT_TRUE(objects.front().is<Dropdown>());
            ASSERT_TRUE(objects.back().is<Entangled>());
            ASSERT_FALSE(objects.back().is<Dropdown>());
            placeholder = objects.back().to<Entangled>();
        }
    }
    ASSERT_NE(placeholder, nullptr);
    EXPECT_EQ(combo.dropdown()->textfield()->corner_flags(), CornerFlags::Left());
    EXPECT_EQ(placeholder->corner_flags(), CornerFlags::Right());
}

TEST(DynamicGUISettingsTest, BooleanSelectionDoesNotLeaveTooltipHovered) {
    GlobalSettings::write([](Configuration& config) {
        config.values["dyngui_test_hover_bool"] = false;
        config.defaults["dyngui_test_hover_bool"] = false;
        config.docs["dyngui_test_hover_bool"] = "Boolean setting for the tooltip hover regression.";
    });

    constexpr std::string_view json = R"json(
{
  "objects": [
    {
      "type": "settings",
      "var": "",
      "desc": "",
      "pos": [10, 420],
      "size": [500, 40]
    }
  ]
}
)json";

    auto loaded = load(std::string(json));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();

    auto [defaults, objects] = std::move(loaded.value());
    GUITaskQueue_t queue;
    DrawStructure graph(640, 480);
    graph.set_dialog_window_size(Size2(640, 480));
    Rect background(Box(0, 0, 640, 480), Clickable{true});
    graph.wrap_object(background);

    Context context;
    context.defaults = std::move(defaults);
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    auto root = parse_object(&queue, objects.get_array().front().get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    auto combo = state._last_settings_box;
    ASSERT_TRUE(combo);
    combo->set(ParmName{"dyngui_test_hover_bool"});
    graph.wrap_object(*root);
    graph.collect();

    List* list = nullptr;
    for(auto* child : combo->children()) {
        if(auto* layout = dynamic_cast<Layout*>(child)) {
            for(const auto& object : layout->objects()) {
                if(object.is<List>())
                    list = object.to<List>();
            }
        }
    }
    ASSERT_NE(list, nullptr);
    ASSERT_TRUE(list->folded());
    ASSERT_EQ(list->items().size(), 2u);
    ASSERT_EQ(std::string(*list->items().at(1)), "true");

    list->set_folded(false);
    graph.collect();
    auto* true_row = list->child<Rect*>(1);
    ASSERT_NE(true_row, nullptr);
    const auto row_center = center_of(*true_row);
    ASSERT_EQ(graph.mouse_move(row_center.x, row_center.y), true_row);
    ASSERT_TRUE(list->hovered());
    ASSERT_TRUE(combo->hovered());
    handler->update_tooltips(graph);
    ASSERT_NE(handler->_tooltip_object, nullptr);
    ASSERT_EQ(handler->_tooltip_object->other().lock().get(), combo.get());

    ASSERT_EQ(graph.mouse_down(true), true_row);
    graph.mouse_up(true);
    EXPECT_TRUE(GlobalSettings::read_value<bool>("dyngui_test_hover_bool").value());
    EXPECT_TRUE(list->folded());

    // Mouse-up can hover an option again before the folded list removes its rows.
    // Process that update before any further mouse movement can send a leave event.
    queue.processTasks(nullptr, graph);
    graph.collect();
    EXPECT_FALSE(true_row->is_staged());
    EXPECT_FALSE(true_row->hovered());
    EXPECT_FALSE(list->hovered());
    EXPECT_FALSE(combo->hovered());

    ASSERT_EQ(graph.mouse_move(620, 20), &background);
    EXPECT_EQ(graph.hovered_object(), &background);
    handler->update_tooltips(graph);
    EXPECT_EQ(handler->_tooltip_object, nullptr);

    const auto label_center = center_of(*list);
    graph.mouse_move(label_center.x, label_center.y);
    ASSERT_TRUE(list->hovered());
    ASSERT_TRUE(combo->hovered());
    handler->update_tooltips(graph);
    ASSERT_NE(handler->_tooltip_object, nullptr);
    EXPECT_EQ(handler->_tooltip_object->other().lock().get(), combo.get());

    ASSERT_EQ(graph.mouse_move(620, 20), &background);
    handler->update_tooltips(graph);
    EXPECT_EQ(handler->_tooltip_object, nullptr);
}

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

TEST(ConditionElementTest, EvaluatesOnlySelectedBranch)
{
    constexpr std::string_view json = R"json(
{
  "type": "condition",
  "var": "{show_then}",
  "then": {
    "type": "stext",
    "text": "{then_value}"
  },
  "else": {
    "type": "stext",
    "text": "{else_value}"
  }
}
)json";

    glz::generic object;
    auto parse_error = glz::read_json(object, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(object.is_object());

    for(const bool show_then : {false, true}) {
        SCOPED_TRACE(show_then);

        size_t then_reads = 0;
        size_t else_reads = 0;
        Context context{
            VarFunc("show_then", [show_then](const VarProps&) -> bool {
                return show_then;
            }),
            VarFunc("then_value", [&](const VarProps&) -> std::string {
                ++then_reads;
                return "then";
            }),
            VarFunc("else_value", [&](const VarProps&) -> std::string {
                ++else_reads;
                return "else";
            })
        };

        State state;
        auto handler = std::make_shared<CurrentObjectHandler>();
        state._current_object_handler = handler;

        DrawStructure graph(640, 480);
        auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
        ASSERT_TRUE(root);
        ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

        if(show_then) {
            EXPECT_GT(then_reads, 0u);
            EXPECT_EQ(else_reads, 0u);
        } else {
            EXPECT_EQ(then_reads, 0u);
            EXPECT_GT(else_reads, 0u);
        }
    }
}

TEST(DynamicGUIErrorRecovery, RestoresCustomObjectsWithoutRecreatingThem)
{
    const std::array<std::string_view, 7> layouts{
        R"json({"type":"recoverable","name":"recoverable"})json",
        R"json({"type":"collection","children":[{"type":"collection","children":[{"type":"recoverable","name":"recoverable"}]}]})json",
        R"json({"type":"each","var":"items","do":{"type":"recoverable","name":"recoverable"}})json",
        R"json({"type":"condition","var":"{show_then}","then":{"type":"recoverable","name":"recoverable"},"else":{"type":"recoverable","name":"recoverable"}})json",
        R"json({"type":"condition","var":"{show_then}","then":{"type":"recoverable","name":"recoverable"},"else":{"type":"recoverable","name":"recoverable"}})json",
        R"json({"type":"recoverable","name":"recoverable"})json",
        R"json({"type":"each","var":"cached_items","do":{"type":"recoverable","name":"recoverable"}})json"
    };

    for(size_t i = 0; i < layouts.size(); ++i) {
        for(const bool initially_failing : {false, true}) {
            SCOPED_TRACE(i);
            SCOPED_TRACE(initially_failing);

            bool failing = initially_failing;
            size_t creations = 0, updates = 0;
            Layout::Ptr original;
            std::vector<std::shared_ptr<VarBase_t>> cached_items{
                VarFunc("item", [](const VarProps&) { return 1; }).second
            };
            Context context{
                VarFunc("items", [](const VarProps&) { return std::vector<int>{1}; }),
                VarFunc("cached_items", [&](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& { return cached_items; }),
                VarFunc("show_then", [i](const VarProps&) { return i == 3; })
            };
            context.custom_elements["recoverable"] = std::make_shared<CustomElement>(
                "recoverable",
                [&](LayoutContext&) -> Layout::Ptr {
                    ++creations;
                    original = Layout::Make<StaticText>{Str{"ready"}};
                    return original;
                },
                [&](Layout::Ptr& object, const Context&, State&, PatternMapType&) -> bool {
                    ++updates;
                    EXPECT_EQ(object.get(), original.get());
                    if(failing)
                        throw std::runtime_error("preview unavailable");
                    return false;
                });
            State state;
            auto handler = std::make_shared<CurrentObjectHandler>();
            state._current_object_handler = handler;
            DrawStructure graph(640, 480);

            glz::generic object;
            const auto parse_error = glz::read_json(object, layouts[i]);
            ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, layouts[i]);
            auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
            if(i == 5) {
                root = Layout::Make<PlaceinLayout>{std::vector<Layout::Ptr>{root}};
            }
            ASSERT_TRUE(root);
            ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
            ASSERT_TRUE(original);
            EXPECT_EQ(creations, 1u);
            EXPECT_EQ(handler->retrieve_named("recoverable").get(), original.get());

            auto displayed_child = [&]() {
                auto child = root;
                while(child) {
                    if(auto pass = dynamic_cast<Fallthrough*>(child.get())) {
                        child = pass->object();
                    } else if(auto layout = dynamic_cast<Layout*>(child.get());
                              layout && not layout->objects().empty())
                    {
                        child = layout->objects().front();
                        EXPECT_THAT(layout->current_children(), ::testing::Contains(child.get()));
                    } else
                        break;
                }
                return child;
            };

            if(not initially_failing)
                EXPECT_EQ(displayed_child().get(), original.get());
            failing = true;
            auto previous_updates = updates;
            ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
            EXPECT_EQ(updates, previous_updates + 1);
            auto error = displayed_child();
            ASSERT_TRUE(dynamic_cast<ErrorElement*>(error.get()));

            for(size_t attempt = 0; attempt < 2; ++attempt) {
                if(i == 0)
                    error.to<ErrorElement>()->set_content_changed(false);
                previous_updates = updates;
                ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
                EXPECT_EQ(updates, previous_updates + 1);
                EXPECT_EQ(displayed_child().get(), error.get());
                if(i == 0)
                    EXPECT_FALSE(error.to<ErrorElement>()->content_changed());
            }

            failing = false;
            previous_updates = updates;
            ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
            EXPECT_EQ(updates, previous_updates + 1);
            EXPECT_EQ(displayed_child().get(), original.get());
            EXPECT_EQ(handler->retrieve_named("recoverable").get(), original.get());
            EXPECT_EQ(creations, 1u);
        }
    }
}

TEST(DynamicGUIErrorRecovery, PointerReplacementDoesNotOverrideUpdateResult)
{
    bool replace = false;
    Context context;
    context.custom_elements["replaceable"] = std::make_shared<CustomElement>(
        "replaceable",
        [](LayoutContext&) -> Layout::Ptr {
            return Layout::Make<StaticText>{Str{"initial"}};
        },
        [&](Layout::Ptr& object, const Context&, State&, PatternMapType&) -> bool {
            if(replace)
                object = Layout::Make<StaticText>{Str{"replacement"}};
            return false;
        });
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);
    glz::generic object;
    ASSERT_EQ(glz::read_json(object, R"json({"type":"replaceable"})json"), glz::error_code::none);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    const auto original = root;

    replace = true;
    EXPECT_FALSE(DynamicGUI::update_objects(nullptr, graph, root, context, state));
    EXPECT_NE(root.get(), original.get());
    ASSERT_TRUE(root.is<StaticText>());
    EXPECT_EQ(root.to<StaticText>()->text(), "replacement");
}

TEST(ConditionElementTest, PropagatesChildUpdateWithoutPointerReplacement)
{
    constexpr std::string_view json = R"json({
        "type":"condition", "var":"{outer}",
        "then":{
            "type":"condition", "var":"{inner}",
            "then":{"type":"stext", "text":"first"},
            "else":{"type":"stext", "text":"second"}
        },
        "else":{
            "type":"condition", "var":"{inner}",
            "then":{"type":"stext", "text":"first"},
            "else":{"type":"stext", "text":"second"}
        }
    })json";
    for(const bool outer : {false, true}) {
        SCOPED_TRACE(outer);
        bool inner = true;
        Context context{
            VarFunc("outer", [outer](const VarProps&) { return outer; }),
            VarFunc("inner", [&](const VarProps&) { return inner; })
        };
        State state;
        auto handler = std::make_shared<CurrentObjectHandler>();
        state._current_object_handler = handler;
        DrawStructure graph(640, 480);
        glz::generic object;
        ASSERT_EQ(glz::read_json(object, json), glz::error_code::none);
        auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
        ASSERT_TRUE(DynamicGUI::update_objects(nullptr, graph, root, context, state));
        const auto original = root;
        const auto child = root.to<Fallthrough>()->object();
        ASSERT_TRUE(child.is<Fallthrough>());
        EXPECT_FALSE(DynamicGUI::update_objects(nullptr, graph, root, context, state));

        inner = false;
        EXPECT_TRUE(DynamicGUI::update_objects(nullptr, graph, root, context, state));
        EXPECT_EQ(root.get(), original.get());
        EXPECT_EQ(root.to<Fallthrough>()->object().get(), child.get());
        ASSERT_TRUE(child.to<Fallthrough>()->object().is<StaticText>());
        EXPECT_EQ(child.to<Fallthrough>()->object().to<StaticText>()->text(), "second");
    }
}

TEST(DynamicGUIErrorRecovery, RecoversInitialUpdateOfCachedCustomObject)
{
    bool failing = false;
    size_t creations = 0;
    Layout::Ptr original;
    Context context;
    context.custom_elements["recoverable"] = std::make_shared<CustomElement>(
        "recoverable",
        [&](LayoutContext&) -> Layout::Ptr {
            ++creations;
            original = Layout::Make<StaticText>{Str{"ready"}};
            return original;
        },
        [&](Layout::Ptr&, const Context&, State&, PatternMapType&) -> bool {
            if(failing)
                throw std::runtime_error("preview unavailable");
            return false;
        });
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);
    glz::generic object;
    ASSERT_EQ(glz::read_json(object, R"json({"type":"recoverable","name":"recoverable"})json"), glz::error_code::none);

    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults, 91);
    ASSERT_EQ(root.get(), original.get());
    failing = true;
    root = parse_object(nullptr, object.get_object(), context, state, context.defaults, 91);
    ASSERT_TRUE(dynamic_cast<ErrorElement*>(root.get()));
    EXPECT_EQ(handler->retrieve_named("recoverable").get(), original.get());

    failing = false;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    EXPECT_EQ(root.get(), original.get());
    EXPECT_EQ(creations, 1u);
}

TEST(DynamicGUIErrorRecovery, ConstructionFailureRemainsAnError)
{
    size_t creations = 0, updates = 0;
    Context context;
    context.custom_elements["broken"] = std::make_shared<CustomElement>(
        "broken",
        [&](LayoutContext&) -> Layout::Ptr {
            ++creations;
            throw std::runtime_error("invalid definition");
        },
        [&](Layout::Ptr&, const Context&, State&, PatternMapType&) -> bool {
            ++updates;
            return false;
        });
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);
    glz::generic object;
    ASSERT_EQ(glz::read_json(object, R"json({"type":"broken"})json"), glz::error_code::none);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    ASSERT_TRUE(dynamic_cast<ErrorElement*>(root.get()));
    const auto error = root;

    for(size_t attempt = 0; attempt < 2; ++attempt) {
        EXPECT_FALSE(DynamicGUI::update_objects(nullptr, graph, root, context, state));
        EXPECT_EQ(root.get(), error.get());
    }
    EXPECT_EQ(creations, 1u);
    EXPECT_EQ(updates, 0u);
}

TEST(ConditionElementTest, ResolutionFailureDisplaysErrorAndRecovers)
{
    constexpr std::string_view json = R"json(
{
  "type": "condition",
  "var": "{condition_value}",
  "then": {
    "type": "stext",
    "text": "then"
  },
  "else": {
    "type": "stext",
    "text": "else"
  }
}
)json";

    glz::generic object;
    const auto parse_error = glz::read_json(object, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    enum class ConditionMode {
        True,
        False,
        Fail
    };

    ConditionMode mode = ConditionMode::True;
    Context context{
        VarFunc("condition_value", [&](const VarProps&) -> bool {
            if(mode == ConditionMode::Fail)
                throw std::runtime_error("condition variable failed");
            return mode == ConditionMode::True;
        })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<Fallthrough>());
    const auto original = root;
    auto pass = root.to<Fallthrough>();

    mode = ConditionMode::True;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_TRUE(pass->object().is<StaticText>());
    EXPECT_EQ(pass->object().to<StaticText>()->text(), "then");

    mode = ConditionMode::False;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_TRUE(pass->object().is<StaticText>());
    EXPECT_EQ(pass->object().to<StaticText>()->text(), "else");

    mode = ConditionMode::Fail;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_TRUE(dynamic_cast<ErrorElement*>(root.get()));

    mode = ConditionMode::True;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    EXPECT_EQ(root.get(), original.get());
    ASSERT_TRUE(pass->object().is<StaticText>());
    EXPECT_EQ(pass->object().to<StaticText>()->text(), "then");
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

    glz::generic obj;
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

    glz::generic obj;
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
        VarFunc("annotations", [&source](const VarProps&) -> std::vector<glz::generic> {
            std::vector<glz::generic> result;
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

                result.push_back(glz::generic::object_t{
                    {"id", object.uid},
                    {"seed_frame", glz::generic{0}},
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

    glz::generic obj;
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

    glz::generic obj;
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

    glz::generic obj;
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

    glz::generic obj;
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

    glz::generic obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<glz::generic> items;
    {
        glz::generic first;
        first["x"] = 42;
        first["name"] = std::string("trex");
        items.push_back(first);
    }
    {
        glz::generic second;
        second["x"] = 7;
        second["name"] = std::string("raptor");
        items.push_back(second);
    }

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<glz::generic> { return items; })
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

TEST(EachElementTest, JsonCallbacksRetainItemsAcrossSourceUpdates) {
    for(int return_kind = 0; return_kind < 3; ++return_kind) {
        SCOPED_TRACE(return_kind);
        glz::generic source;
        ASSERT_EQ(glz::read_json(source, R"json({"rows":[{"name":"first"}]})json"), glz::error_code::none);

        Context context;
        if(return_kind == 0) {
            context.variables["items"] = VarFunc("items", [&](const VarProps& props) -> glz::generic {
                EXPECT_THAT(props.parameters, ::testing::ElementsAre("1"));
                return std::move(source);
            }).second;
        } else if(return_kind == 1) {
            context.variables["items"] = VarFunc("items", [&](const VarProps&) -> glz::generic& {
                return source;
            }).second;
        } else {
            context.variables["items"] = VarFunc("items", [&](const VarProps&) -> const glz::generic& {
                return source;
            }).second;
        }

        std::vector<std::function<std::string()>> callbacks;
        context.custom_elements["capture_item"] = std::make_shared<CustomElement>(
            "capture_item",
            [](LayoutContext&) -> Layout::Ptr {
                return Layout::Make<StaticText>{Str{"item"}};
            },
            [&](Layout::Ptr&, const Context& context, State& state, PatternMapType&) -> bool {
                callbacks.emplace_back(bind_with_state<>(state._current_object_handler, context,
                    [](const Context& context, State& state) {
                        return parse_text("{i.name}:{index}", context, state);
                    }));
                return false;
            });

        State state;
        auto handler = std::make_shared<CurrentObjectHandler>();
        state._current_object_handler = handler;
        DrawStructure graph(640, 480);
        glz::generic obj;
        ASSERT_EQ(glz::read_json(obj, R"json({
            "type":"each", "var":"{items.rows:1}", "do":{"type":"capture_item"}
        })json"), glz::error_code::none);
        auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
        ASSERT_TRUE(root);
        ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
        ASSERT_FALSE(callbacks.empty());
        auto first_callback = callbacks.back();
        EXPECT_TRUE(handler->capture_scoped_variable_values().empty());

        if(return_kind != 0) {
            ASSERT_TRUE(source["rows"].is_array());
            ASSERT_EQ(source["rows"].get_array().size(), 1u);
            EXPECT_EQ(source["rows"][0]["name"].get_string(), "first");
        }
        source = glz::generic{};
        ASSERT_EQ(glz::read_json(source, R"json({"rows":[{"name":"second"}]})json"), glz::error_code::none);
        EXPECT_EQ(first_callback(), "first:0");

        const auto previous_count = callbacks.size();
        ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
        ASSERT_GT(callbacks.size(), previous_count);
        source = glz::generic{};
        EXPECT_EQ(callbacks.back()(), "second:0");
        EXPECT_EQ(first_callback(), "first:0");
        EXPECT_TRUE(handler->capture_scoped_variable_values().empty());
    }
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

    glz::generic obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    std::vector<glz::generic> items;
    {
        glz::generic first;
        first["name"] = std::string("trex");
        items.push_back(first);
    }
    {
        glz::generic second;
        second["name"] = std::string("raptor");
        items.push_back(second);
    }

    Context context{
        VarFunc("items", [&items](const VarProps&) -> std::vector<glz::generic> { return items; })
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

    glz::generic obj;
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

TEST(EachElementTest, NestedStringArraysUseReferenceItems) {
    glz::generic obj;
    ASSERT_EQ(glz::read_json(obj, R"json({
        "type":"each", "var":"[[1,2]]",
        "do":{
            "type":"each", "var":"{i}",
            "do":{"type":"stext", "text":"{i}"}
        }
    })json"), glz::error_code::none);

    Context context;
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);
    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    std::vector<std::string> texts;
    collect_static_text_strings(root, texts);
    EXPECT_THAT(texts, ::testing::ElementsAre("1", "2"));
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

        glz::generic obj;
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

        glz::generic obj;
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

    glz::generic obj;
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

TEST(ListElementTest, DynamicListPreservesQuotedStringSourceItems) {
    constexpr std::string_view json = R"json(
{
  "type": "list",
  "var": "['a,b','c','{value}']",
  "template": { "text": "{i}" }
}
)json";

    glz::generic obj;
    const auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    Context context{
        VarFunc("value", [](const VarProps&) -> std::string { return "d,e"; })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root);
    ASSERT_TRUE(root.is<ScrollableList<DetailTooltipItem>>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    const auto list = root.to<ScrollableList<DetailTooltipItem>>();
    ASSERT_EQ(list->items().size(), 3u);
    EXPECT_EQ(list->items().at(0).value().name(), "a,b");
    EXPECT_EQ(list->items().at(1).value().name(), "c");
    EXPECT_EQ(list->items().at(2).value().name(),
              "{value}");
}

TEST(ListElementTest, DynamicListTemplateDispatchesActionsOnSelection) {
    glz::generic obj = glz::generic::object_t{
        {"type", "list"},
        {"var", "items"},
        {"template", glz::generic::object_t{
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

/*TEST(EventBindingTest, JsonLoopClickActionUsesUpdatedItem) {
    const auto object = Meta::fromStr<glz::generic>(R"json({
        "type":"each", "var":"items",
        "do":{
            "type":"rect", "size":"[{i.value},30]",
            "clickable":true, "click":"select:{i.value}"
        }
    })json");
    auto items = Meta::fromStr<glz::generic>(R"json([{"value":10}])json");
    std::string selected;
    Context context{
        VarFunc("items", [&](const VarProps&) -> glz::generic {
            return items;
        }),
        ActionFunc("select", [&](const Action& action) {
            selected = action.first();
        })
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;

    DrawStructure graph(100, 100);
    auto root = parse_object(nullptr, object.get_object(), context, state, context.defaults);
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_TRUE(root.is<Layout>());
    ASSERT_EQ(root.to<Layout>()->objects().size(), 1u);
    auto target = root.to<Layout>()->objects().front();
    EXPECT_EQ(target->width(), 10);
    target->mdown(1, 1, true);
    target->mup(1, 1, true, true);
    ASSERT_EQ(selected, "10");

    items.get_array().front()["value"] = 20;
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    ASSERT_EQ(root.to<Layout>()->objects().front().get(), target.get());
    EXPECT_EQ(target->width(), 20);
    ASSERT_TRUE(handler->capture_scoped_variable_values().empty());
    selected.clear();
    target->mdown(1, 1, true);
    target->mup(1, 1, true, true);
    EXPECT_EQ(selected, "20");
}*/

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

    glz::generic obj;
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
    ASSERT_EQ(received_actions.size(), 0u);

    ASSERT_NO_THROW(graph.mouse_up(true));
    ASSERT_EQ(received_actions.size(), 1u);
    EXPECT_EQ(received_actions.at(0).name, "select");
    ASSERT_EQ(received_actions.at(0).parameters.size(), 2u);
    EXPECT_EQ(received_actions.at(0).parameters.at(0), "beta");
    EXPECT_EQ(received_actions.at(0).parameters.at(1), "1");
}

TEST(EventBindingTest, NestedEachClickPreservesOuterAndInnerScopedContext) {
    constexpr std::string_view json = R"json(
{
  "type": "each",
  "var": "outer_items",
  "as": "outer",
  "do": {
    "type": "each",
    "var": "inner_items",
    "as": "inner",
    "do": {
      "type": "rect",
      "name": "cell-{outer.name}-{inner.name}",
      "pos": "[{*:40:{index}},{*:40:{pindex}}]",
      "size": [30, 30],
      "origin": [0, 0],
      "clickable": true,
      "click": "select:{outer.name}:{inner.name}:{index}:{pindex}:{runtime_value}"
    }
  }
}
)json";

    glz::generic obj;
    const auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    std::vector<sprite::Map> outer_data(2);
    outer_data[0]["name"] = std::string("outer-alpha");
    outer_data[1]["name"] = std::string("outer-beta");
    std::vector<std::shared_ptr<VarBase_t>> outer_items;
    for(size_t idx = 0; idx < outer_data.size(); ++idx) {
        outer_items.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &outer_data](const VarProps&) -> sprite::Map& {
            return outer_data[idx];
        })));
    }

    std::vector<sprite::Map> inner_data(2);
    inner_data[0]["name"] = std::string("inner-alpha");
    inner_data[1]["name"] = std::string("inner-beta");
    std::vector<std::shared_ptr<VarBase_t>> inner_items;
    for(size_t idx = 0; idx < inner_data.size(); ++idx) {
        inner_items.emplace_back(std::shared_ptr<VarBase_t>(new Variable([idx, &inner_data](const VarProps&) -> sprite::Map& {
            return inner_data[idx];
        })));
    }

    std::vector<Action> received_actions;
    Context context{
        VarFunc("outer_items", [&outer_items](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return outer_items;
        }),
        VarFunc("inner_items", [&inner_items](const VarProps&) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return inner_items;
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

    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));
    graph.wrap_object(*root);

    ASSERT_TRUE(handler->capture_scoped_variable_values().empty());

    auto* target = graph.find("cell-outer-beta-inner-alpha");
    ASSERT_NE(target, (Drawable*)NULL);

    const auto target_center = center_of(*target);
    ASSERT_NO_THROW(graph.mouse_move(target_center.x, target_center.y));

    {
        auto ambient_scope = handler->scope();
        ambient_scope.set("runtime_value", "ambient");
        ambient_scope.set("index", "99");
        ambient_scope.set("pindex", "98");
        const auto ambient_snapshot = handler->capture_scoped_variable_values();

        ASSERT_NO_THROW(graph.mouse_down(true));
        ASSERT_EQ(received_actions.size(), 0u);

        ASSERT_NO_THROW(graph.mouse_up(true));
        ASSERT_EQ(received_actions.size(), 1u);
        EXPECT_EQ(received_actions.front().name, "select");
        EXPECT_THAT(
            received_actions.front().parameters,
            ::testing::ElementsAre("outer-beta", "inner-alpha", "0", "1", "ambient"));

        EXPECT_EQ(handler->capture_scoped_variable_values().values, ambient_snapshot.values);
        EXPECT_EQ(parse_text("{runtime_value}:{index}:{pindex}", context, state), "ambient:99:98");
    }

    EXPECT_TRUE(handler->capture_scoped_variable_values().empty());
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

    glz::generic obj;
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

    glz::generic obj;
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

    glz::generic obj;
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
    void checkLineBreaks(const std::vector<cmn::gui::StaticText::RichString>& strings, float max_width, cmn::gui::Drawable* reference) {
        for (const auto& str : strings) {
            Bounds bounds = cmn::utils::calculate_bounds(str.str, reference, str.font);
            float width = cmn::utils::calculate_width(bounds);
            
            //Print("** ", utils::ShortenText(str.str, 15)," w=", width, " max=",max_width, " font=",str.font);
            
            EXPECT_LE(width, max_width) << "Line exceeds max width: " << str.str;
        }
    }

    // Utility function to check that all characters from input are found in the correct order
    void checkCharactersInOrder(const std::string& input, const std::vector<cmn::gui::StaticText::RichString>& strings) {
        Vec2 prev(FLT_MAX);
        size_t index = 0;
        for (const auto& str : strings) {
            bool new_line = str.pos.y != prev.y;
            prev = str.pos;
            
            for(size_t i = 0; i < str.str.size(); ++i, ++index) {
                ASSERT_LT(index, input.size());
                if(new_line && i == 0 && std::isspace(input[index]) && input[index] != str.str[i])
                    ++index;
                
                ASSERT_LT(index, input.size());
                EXPECT_EQ(str.str[i], input[index]);
            }
        }
        EXPECT_EQ(index, input.size());
    }

    void checkRenderedLayout(StaticText& label) {
        label.update();
        std::map<Float2_t, Font> line_fonts;
        for(const auto& span : label.rich_strings()) {
            auto [it, inserted] = line_fonts.try_emplace(span.pos.y, span.font);
            if(not inserted && span.font.size > it->second.size)
                it->second = span.font;
        }

        std::map<Float2_t, Float2_t> line_centers, line_widths;
        Float2_t top = label.margins().y;
        size_t row = 0;
        for(const auto& [line, font] : line_fonts) {
            EXPECT_EQ(line, row++);
            const auto height = Base::default_line_spacing(font);
            line_centers[line] = top + height * 0.5_F;
            top += height;
        }

        size_t child_index = 0;
        Vec2 extent;
        for(const auto& span : label.rich_strings()) {
            if(span.str.empty())
                continue;
            SCOPED_TRACE(std::string(span.str));
            ASSERT_LT(child_index, label.children().size());
            auto* text = dynamic_cast<Text*>(label.children()[child_index++]);
            ASSERT_NE(text, nullptr);
            const auto bounds = utils::calculate_bounds(span.str, text, span.font);
            EXPECT_EQ(text->txt(), span.str);
            EXPECT_EQ(text->font(), span.font);
            EXPECT_EQ(text->color(), span.clr.alpha(255 * label.alpha()));
            EXPECT_EQ(text->text_bounds(), bounds);
            EXPECT_EQ(text->size(), bounds.size());
            EXPECT_FLOAT_EQ(text->pos().x, label.margins().x + line_widths[span.pos.y]);
            EXPECT_FLOAT_EQ(text->pos().y, line_centers.at(span.pos.y));
            EXPECT_EQ(text->origin(), Vec2(0, 0.5));
            line_widths[span.pos.y] += utils::calculate_width(bounds);
            extent.x = max(extent.x, text->pos().x + bounds.width);
            extent.y = max(extent.y, text->pos().y + bounds.height * 0.5_F);
        }
        EXPECT_EQ(child_index, label.children().size());
        for(const auto& [line, width] : line_widths)
            EXPECT_LE(width, label.max_size().x - 2 * label.margins().x + 0.001_F) << "Line " << line;
        EXPECT_FLOAT_EQ(label.size().width, extent.x + label.margins().width);
        EXPECT_FLOAT_EQ(label.size().height, extent.y + label.margins().height);
    }
};

TEST_F(StaticTextTest, NoMaxWidth) {
    StaticText::Settings settings;
    settings.default_font = Font(0.5);
    settings.max_size.x = 0;  // No max width
    
    std::vector<StaticText::RichString> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    StaticText::RichString richString(input, cmn::gui::Font(), cmn::Vec2(), Red);
    
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

    std::vector<StaticText::RichString> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    StaticText::RichString richString(input, cmn::gui::Font(), cmn::Vec2(), Red);

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

    std::vector<StaticText::RichString> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    StaticText::RichString richString(input, cmn::gui::Font(), cmn::Vec2(), Red);

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

    std::vector<StaticText::RichString> strings;
    cmn::Vec2 offset(0, 0);

    std::string input = "This is a test string that should be split into multiple lines if it exceeds the max width";
    StaticText::RichString richString(input, cmn::gui::Font(), cmn::Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    // Verify that lines do not exceed the specified max width
    checkLineBreaks(strings, settings.max_size.x, nullptr);

    // Additional verification to ensure multiple lines are created
    EXPECT_GT(strings.size(), 1);

    // Verify that all characters are in the correct order
    checkCharactersInOrder(input, strings);
}

TEST_F(StaticTextTest, BoundedMeasurementPreservesExactFitsAndUtf8Boundaries) {
    const Font font(0.5, Style::Monospace);
    const std::string_view input = "∅ab";
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    const char* remaining = nullptr;
    for(const std::string_view prefix : {"", "∅", "∅a", "∅ab"}) {
        SCOPED_TRACE(prefix);
        const auto limit = width(prefix);
        const auto bounds = utils::calculate_bounds(input, nullptr, font, std::nextafter(limit, std::numeric_limits<Float2_t>::infinity()), &remaining);
        EXPECT_EQ(remaining, input.data() + prefix.size());
        EXPECT_FLOAT_EQ(utils::calculate_width(bounds), limit);
        if(not prefix.empty())
            EXPECT_EQ(utils::find_splitting_point(input, width(input), limit, nullptr, font), prefix.size());
    }
    EXPECT_EQ(utils::find_splitting_point(input, width(input), 0, nullptr, font), std::string_view("∅").size());

    const std::string_view empty;
    EXPECT_EQ(utils::calculate_width(utils::calculate_bounds(empty, nullptr, font, 1, &remaining)), 0);
    EXPECT_EQ(remaining, empty.data());
}

TEST_F(StaticTextTest, SplittingInsideWordPreservesRemainingWidth) {
    const Font font(0.5, Style::Monospace);
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    StaticText::Settings settings;
    settings.margins = Margins{0};
    settings.max_size.x = width("abc") + width(" ") * 0.5_F;
    Vec2 offset(0, 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {"abcdef", font, offset, White}, strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, "abc");
    EXPECT_EQ(strings[0].pos.y, 0);
    EXPECT_EQ(strings[1].str, "def");
    EXPECT_EQ(strings[1].pos.y, 1);
    EXPECT_FLOAT_EQ(offset.x, width("def"));
    EXPECT_EQ(offset.y, 1);
}

TEST_F(StaticTextTest, EmptyAnnotationLabelKeepsParenthesizedWordTogether) {
    const Font font(0.5), mono(0.5, Style::Monospace);
    const auto width = [](std::string_view str, const Font& font) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };

    // Resolved label from tracking_layout.json, with room only for " (emp" after the symbol.
    StaticText label(Str{"<c>Annotate this frame as </c>∅<c> (empty):</c>"}, font);
    label.set(SizeLimit{Size2{
        width("Annotate this frame as ", mono) + width("∅", font) + width(" (emp", mono)
            + 2 * label.margins().x, 0}});
    label.update();

    std::map<Float2_t, std::string> lines;
    /*for(auto* child : label.children()) {
        auto* text = dynamic_cast<Text*>(child);
        ASSERT_NE(text, nullptr);
        lines[text->pos().y] += text->txt();
    }*/
    for(auto& rstr : label.rich_strings()) {
        lines[rstr.pos.y] += std::string(rstr.str);
    }

    ASSERT_EQ(lines.size(), 2u);
    ASSERT_EQ(lines.begin()->first, 0);
    ASSERT_EQ(lines.rbegin()->first, 1);
    EXPECT_EQ(lines.begin()->second, "Annotate this frame as ∅");
    EXPECT_EQ(lines.rbegin()->second, "(empty):");
    for(const auto& span : label.rich_strings()) {
        if(span.str.empty())
            continue;
        EXPECT_EQ(span.font, span.str == "∅" ? font : mono);
        EXPECT_EQ(span.clr, label.text_color());
        if(span.pos.y == 1)
            EXPECT_FLOAT_EQ(width(span.str, span.font), width("(empty):", mono));
    }
    checkRenderedLayout(label);
}

TEST_F(StaticTextTest, TrimmedSpanMeasuresRemainingSpaceForFollowingText) {
    StaticText::Settings settings;
    const Font font(0.5, Style::Monospace);
    const Color color{45, 130, 210, 180};
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    const std::string_view suffix = " suffix";
    const auto content_width = width("(empty):") + width(suffix);
    settings.max_size.x = content_width + 2 * settings.margins.x;

    for(const std::string_view leading : {" ", "   ", "\xC2\xA0 "}) {
        const std::string input = std::string(leading) + "(empty):";
        SCOPED_TRACE(input);
        Vec2 offset(content_width - width(" (emp"), 0);
        std::vector<StaticText::RichString> strings;
        StaticText::add_string(nullptr, settings, {input, font, offset, color}, strings, offset);

        ASSERT_EQ(strings.size(), 1u);
        EXPECT_EQ(strings.front().str, "(empty):");
        EXPECT_EQ(strings.front().font, font);
        EXPECT_EQ(strings.front().clr, color);
        EXPECT_EQ(strings.front().pos.y, 1);
        EXPECT_EQ(offset.y, 1);
        EXPECT_FLOAT_EQ(offset.x, width("(empty):"));

        StaticText::add_string(nullptr, settings, {suffix, font, offset, color}, strings, offset);
        ASSERT_EQ(strings.size(), 2u);
        EXPECT_EQ(strings.back().str, suffix);
        EXPECT_EQ(strings.back().pos.y, 1);
        EXPECT_FLOAT_EQ(strings.back().pos.x, width("(empty):"));
        EXPECT_EQ(strings.back().font, font);
        EXPECT_EQ(strings.back().clr, color);
        EXPECT_EQ(offset.y, 1);
        EXPECT_FLOAT_EQ(offset.x, content_width);
    }
}

TEST_F(StaticTextTest, TwoWordsUseHalfEmptyLine) {
    const Font font(0.5, Style::Monospace);
    StaticText::Settings settings;
    settings.margins = Margins{0};
    settings.max_size.x = utils::calculate_width(utils::calculate_bounds("a b", nullptr, font));

    // A line holds "a b"; half is occupied, so "a" still fits on that line.
    Vec2 offset(settings.max_size.x * 0.5_F, 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {"a b", font, offset, White}, strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, "a");
    EXPECT_EQ(strings[0].pos.y, 0);
    EXPECT_EQ(strings[1].str, "b");
    EXPECT_EQ(strings[1].pos.y, 1);
}

TEST_F(StaticTextTest, MultiwordSpanUsesRemainingLineBeforeWrapping) {
    StaticText::Settings settings;
    const Font font(0.5, Style::Monospace);
    const std::string_view input = " alpha beta gamma";
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    const auto content_width = width(input) + width(" extra");
    settings.max_size.x = content_width + 2 * settings.margins.x;
    Vec2 offset(content_width - width(" alpha") - width(" ") * 0.5_F, 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {input, font, offset, Cyan}, strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, " alpha");
    EXPECT_EQ(strings[0].pos.y, 0);
    EXPECT_EQ(strings[1].str, "beta gamma");
    EXPECT_EQ(strings[1].pos.y, 1);
    EXPECT_FLOAT_EQ(offset.x, width("beta gamma"));
    EXPECT_EQ(offset.y, 1);
}

TEST_F(StaticTextTest, FirstWordMovesIntactToNextLine) {
    const Font font(0.5, Style::Monospace);
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    StaticText::Settings settings;
    settings.margins = Margins{0};
    settings.max_size.x = width("empty");

    // Only "emp" fits in the remaining space; "empty" fits on a fresh line.
    Vec2 offset(settings.max_size.x - width("emp"), 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {"empty a", font, offset, White}, strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, "empty");
    EXPECT_EQ(strings[0].pos.y, 1);
    EXPECT_EQ(strings[1].str, "a");
    EXPECT_EQ(strings[1].pos.y, 2);
}

TEST_F(StaticTextTest, TrimmedLeadingSpacesRebaseWordBreaks) {
    const Font font(0.5, Style::Monospace);
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    StaticText::Settings settings;
    settings.margins = Margins{0};
    settings.max_size.x = width("aa bb") + width(" ") * 0.5_F;
    Vec2 offset(width("aaaa"), 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {"  aa bb cc", font, offset, White}, strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, "aa bb");
    EXPECT_EQ(strings[0].pos.y, 1);
    EXPECT_EQ(strings[1].str, "cc");
    EXPECT_EQ(strings[1].pos.y, 2);
    EXPECT_FLOAT_EQ(offset.x, width("cc"));
}

TEST_F(StaticTextTest, LeadingSpaceDoesNotPreventWholeWordFromFittingNextLine) {
    const Font font(0.5, Style::Monospace);
    const auto width = [&](std::string_view str) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    StaticText::Settings settings;
    settings.margins = Margins{0};
    settings.max_size.x = width("empty") + width(" ") * 0.5_F;
    Vec2 offset(width("em"), 0);
    std::vector<StaticText::RichString> strings;
    StaticText::add_string(nullptr, settings, {" empty", font, offset, White}, strings, offset);

    ASSERT_EQ(strings.size(), 1u);
    EXPECT_EQ(strings[0].str, "empty");
    EXPECT_EQ(strings[0].pos.y, 1);
    EXPECT_EQ(offset.y, 1);
    EXPECT_FLOAT_EQ(offset.x, width("empty"));
}

/* These tests require keeping words and punctuation together across style boundaries.
TEST_F(StaticTextTest, TrailingPeriodStaysWithStyledWord) {
    const Font font(0.5), italic(0.5, Style::Italic);
    const auto width = [](std::string_view str, const Font& font) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    StaticText label(Str{"a <i>word</i>."}, font, Margins{0});
    // The word fits after "a "; keeping its period attached requires a new line.
    label.set(SizeLimit{Size2{width("a ", font) + width("word", italic) + width(".", font) * 0.5_F, 0}});

    const auto& strings = label.rich_strings();
    ASSERT_EQ(strings.size(), 3u);
    EXPECT_EQ(strings[0].pos.y, 0);
    EXPECT_EQ(strings[1].str, "word");
    EXPECT_EQ(strings[1].pos.y, 1);
    EXPECT_EQ(strings[2].str, ".");
    EXPECT_EQ(strings[2].pos.y, 1);
}

TEST_F(StaticTextTest, LongAnnotationLabelPreservesWordsStylesAndGeometryOnReflow) {
    const Font font(0.5), mono(0.5, Style::Monospace);
    const std::string input =
        "<c>Annotate this frame as </c>∅<c> (empty): review every visible object before confirming "
        "that this frame contains no detections.</c> "
        "<cyan><b>Keep uncertain objects</b> <i>for manual review</i></cyan>, "
        "and <orange><c>do not discard partially visible animals</c></orange>.\n"
        "<gray>Check <b>the image boundaries</b>, <i>overlapping animals</i>, and "
        "<cyan><b>small objects near the background</b></cyan>.</gray> "
        "<small>Use &lt;preview&gt; before saving.</small> "
        "<c>Only then add the </c>∅<c> (empty) annotation and continue to the next frame.</c>";
    const Color color{130, 170, 210, 180};
    StaticText reference(Str{input}, font, TextClr{color});
    StaticText label(Str{input}, font, TextClr{color}, Alpha{0.6_F}, Margins{7, 9, 7, 11});
    const auto width = [](std::string_view str, const Font& font) {
        return utils::calculate_width(utils::calculate_bounds(str, nullptr, font));
    };
    const auto content_width = width("Annotate this frame as ", mono) + width("∅", font) + width(" (emp", mono);

    // Compare words and styled characters with the unwrapped markup; whitespace at a wrap may be trimmed.
    const auto content = [](const StaticText& text) {
        std::string plain;
        std::vector<std::tuple<char, Font, Color>> characters;
        Float2_t line = 0;
        for(const auto& span : text.rich_strings()) {
            if(span.pos.y != line)
                plain += ' ';
            line = span.pos.y;
            plain += span.str;
            for(const unsigned char c : span.str) {
                if(not std::isspace(c))
                    characters.emplace_back(c, span.font, span.clr);
            }
        }
        std::vector<std::string> words;
        std::istringstream stream(plain);
        for(std::string word; stream >> word; )
            words.push_back(std::move(word));
        return std::pair{words, characters};
    };
    const auto expected = content(reference);
    for(const auto scale : {1_F, 1.5_F, 2_F, 1_F}) {
        SCOPED_TRACE(scale);
        label.set(SizeLimit{Size2{content_width * scale + 2 * label.margins().x, 0}});
        const auto actual = content(label);
        for(size_t i = 0; i < max(actual.first.size(), expected.first.size()); ++i) {
            const auto actual_word = i < actual.first.size() ? actual.first[i] : "<missing>";
            const auto expected_word = i < expected.first.size() ? expected.first[i] : "<missing>";
            EXPECT_EQ(actual_word, expected_word) << "Word " << i;
        }
        ASSERT_EQ(actual.second.size(), expected.second.size());
        for(size_t i = 0; i < expected.second.size(); ++i)
            EXPECT_EQ(actual.second[i], expected.second[i]) << "Character " << i;
        checkRenderedLayout(label);
        if(scale == 1_F) {
            std::map<Float2_t, std::string> lines;
            for(const auto& span : label.rich_strings())
                lines[span.pos.y] += span.str;
            EXPECT_EQ(lines.at(0), "Annotate this frame as ∅");
            EXPECT_TRUE(lines.at(1).starts_with("(empty):"));
        }
    }
}

*/

TEST_F(StaticTextTest, SkipsWhitespaceOnlyRemainder) {
    StaticText::Settings settings;
    const Font font;
    const auto width = utils::calculate_width(utils::calculate_bounds("word ", nullptr, font));
    settings.max_size.x = width + settings.margins.x + settings.margins.x;

    const std::string input = "word     ";
    std::vector<StaticText::RichString> strings;
    Vec2 offset(0, 0);
    StaticText::RichString richString(input, font, Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    ASSERT_EQ(strings.size(), 1u);
    EXPECT_EQ(strings.front().str, "word");
    EXPECT_EQ(strings.front().pos.y, 0);
    checkLineBreaks(strings, width, nullptr);
}

TEST_F(StaticTextTest, TrimsUnicodeWhitespaceAcrossCachedBreaks) {
    StaticText::Settings settings;
    const Font font;
    const auto remaining_width = utils::calculate_width(utils::calculate_bounds("b c", nullptr, font));
    const auto width = remaining_width * 1.1_F;
    settings.max_size.x = width + settings.margins.x + settings.margins.x;

    // The leading space, non-breaking space, and next space span two cache entries.
    const std::string input = "a \xC2\xA0 b c";
    std::vector<StaticText::RichString> strings;
    Vec2 offset(0, 0);
    StaticText::RichString richString(input, font, Vec2(), Red);

    StaticText::add_string(nullptr, settings, std::move(richString), strings, offset);

    ASSERT_EQ(strings.size(), 2u);
    EXPECT_EQ(strings[0].str, "a");
    EXPECT_EQ(strings[1].str, "b c");
    EXPECT_EQ(strings[0].pos.y, 0);
    EXPECT_EQ(strings[1].pos.y, 1);
    EXPECT_EQ(offset.y, 1);
    EXPECT_FLOAT_EQ(offset.x, remaining_width);
    checkLineBreaks(strings, width, nullptr);
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

    glz::generic obj;
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
    glz::generic obj;
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

TEST(ScrollableListTest, DefersRowGeometryUntilDrawing) {
    for(float padding_y : {0.f, 1.f, 3.f, 5.f}) {
        SCOPED_TRACE(padding_y);
        ScrollableList<DetailTooltipItem> list{ListDims_t{180, 110}};
        list.set_items({DetailTooltipItem{"One"}, DetailTooltipItem{"Two"}});
        list.before_draw();

        ASSERT_GE(list.children().size(), 2u);
        auto* row = dynamic_cast<Rect*>(list.children().front());
        ASSERT_NE(row, nullptr);
        const auto previous_height = row->height();

        list.set(ItemPadding_t{5, padding_y});
        list.set(ItemFont_t{Font{0.5f}});
        list.set(ItemCornerFlags{CornerFlags::Square()});
        EXPECT_FLOAT_EQ(row->height(), previous_height);
        EXPECT_FLOAT_EQ(list.line_height(), Base::default_line_spacing(Font{0.5f})
            + Base::default_line_spacing(Font{0.55f}) + padding_y * 2);

        list.before_draw();
        EXPECT_FLOAT_EQ(row->height(), list.line_height());
        EXPECT_EQ((CornerFlags)row->corners(), CornerFlags::Square());
        auto* text = dynamic_cast<StaticText*>(list.children()[1]);
        ASSERT_NE(text, nullptr);
        EXPECT_FLOAT_EQ(text->font().size, 0.5f);

        list.set(ItemFont_t{Font{0.6f}});
        EXPECT_TRUE(list.content_changed());
        list.before_draw();
        EXPECT_FLOAT_EQ(row->height(), list.line_height());
        EXPECT_FLOAT_EQ(text->font().size, 0.6f);
    }
}

TEST(ScrollableListTest, RefreshesPendingRowsBeforeKeyboardHighlight) {
    ScrollableList<std::string> list{ListDims_t{180, 40}};
    DrawStructure graph(320, 200);
    graph.wrap_object(list);
    list.before_draw();

    list.set_items({"One", "Two", "Three", "Four", "Five"});
    list.set(ItemPadding_t{5, 3});
    list.set(ItemFont_t{Font{0.5f}});
    list.set(ItemCornerFlags{CornerFlags::Square()});
    list.highlight_item(4);

    ASSERT_EQ(list.highlighted_item(), 4u);
    auto* row = dynamic_cast<Rect*>(graph.hovered_object());
    ASSERT_NE(row, nullptr);
    EXPECT_FLOAT_EQ(row->height(), list.line_height());
    EXPECT_EQ((CornerFlags)row->corners(), CornerFlags::Square());
    EXPECT_GT(list.scroll_offset().y, 0);
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
        ItemCornerFlags{CornerFlags::Rounded(6)},
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
        ItemCornerFlags{CornerFlags::Rounded(5)}, object));
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

    glz::generic obj;
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
        glz::generic obj;
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

TEST(DynamicGUITagListTest, PreservesQuotedNumericStringSources) {
    constexpr std::string_view json = R"json(
{
  "type": "taglist",
  "var": "['123','456']",
  "catalog": "['123','456','789']",
  "add_action": "add"
}
)json";

    glz::generic obj;
    const auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);
    ASSERT_TRUE(obj.is_object());

    Context context{
        ActionFunc("add", [](Action) {})
    };
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    DrawStructure graph(640, 480);

    auto root = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    ASSERT_TRUE(root.is<TagList>());
    ASSERT_NO_THROW((void)DynamicGUI::update_objects(nullptr, graph, root, context, state));

    const auto tag_list = root.to<TagList>();
    EXPECT_THAT(serialized_tags(tag_list->tags()), ::testing::ElementsAre("123", "456"));
    EXPECT_THAT(tag_list->catalog(), ::testing::ElementsAre("123", "456", "789"));
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
    glz::generic obj;
    auto parse_error = glz::read_json(obj, json);
    ASSERT_EQ(parse_error, glz::error_code::none) << glz::format_error(parse_error, json);

    std::set<std::string> selected{"Beta", "Alpha"};
    glz::generic catalog;
    parse_error = glz::read_json(catalog, R"json(["Alpha", "Beta", "Gamma"])json");
    ASSERT_EQ(parse_error, glz::error_code::none);
    Context context{
        VarFunc("selected", [&selected](const VarProps&) -> std::set<std::string> { return selected; }),
        VarFunc("catalog", [&catalog](const VarProps&) -> glz::generic { return catalog; }),
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
        glz::generic obj;
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
    glz::generic obj;
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
    glz::generic obj;
    const auto error = glz::read_json(obj, json);
    ASSERT_EQ(error, glz::error_code::none) << glz::format_error(error, json);

    Context context;
    State state;
    auto handler = std::make_shared<CurrentObjectHandler>();
    state._current_object_handler = handler;
    auto result = parse_object(nullptr, obj.get_object(), context, state, context.defaults);
    EXPECT_TRUE(result.is<ErrorElement>());
}
