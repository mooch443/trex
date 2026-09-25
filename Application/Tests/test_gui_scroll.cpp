#include "gtest/gtest.h"
#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/ListAttributes.h>
#include <gui/types/Button.h>
#include <gui/types/Entangled.h>
#include <gui/types/Layout.h>
#include <gui/types/ScrollableList.h>

using namespace cmn;
using namespace cmn::gui;

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

TEST(DrawStructureFindTest, CentralLookupIncludesRootSectionsAndChildren) {
    DrawStructure graph(200, 200);
    Rect child(Box(0, 0, 40, 40));
    child.set_name("named-child");
    child.set_clickable(true);

    Section* section = nullptr;
    {
        DrawStructure::SectionGuard root_guard(graph, "root");
        {
            DrawStructure::SectionGuard guard(graph, "named-section");
            section = guard._section;
            graph.wrap_object(child);
        }
    }

    ASSERT_NE(section, nullptr);
    auto* root = graph.find("root");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->name(), "root");
    EXPECT_EQ(graph.find("named-section"), section);
    EXPECT_EQ(graph.find("named-child"), &child);
    EXPECT_EQ(
        graph.find(20, 20, pointer::Events::Click),
        &child);

    // A section not visited during the next root update remains available for
    // structural lookup but is disabled for drawing and pointer hit testing.
    {
        DrawStructure::SectionGuard root_guard(graph, "root");
    }
    ASSERT_FALSE(section->enabled());
    EXPECT_EQ(
        graph.find(20, 20, pointer::Events::Click),
        nullptr);
    EXPECT_EQ(graph.find("named-section"), section);
    EXPECT_EQ(graph.find("named-child"), &child);
}

TEST(PointerEventsTest, MaskIsIndependentFromClickableAndDraggable) {
    using pointer::Events;

    Rect drawable(Box(0, 0, 40, 40));
    EXPECT_EQ(drawable.pointer_events(), Events::All);
    EXPECT_FALSE(drawable.does_receive(Events::Hover));

    const auto drag_hover = Events::Drag | Events::Hover;
    drawable.set(PointerEvents{drag_hover});
    drawable.set_clickable(true);

    EXPECT_TRUE(drawable.does_receive(Events::Drag));
    EXPECT_TRUE(drawable.does_receive(Events::Hover));
    EXPECT_FALSE(drawable.does_receive(Events::Click));
    EXPECT_TRUE(drawable.does_receive(Events::Drag | Events::Click));

    drawable.set_clickable(false);
    EXPECT_FALSE(drawable.does_receive(Events::Drag));
    EXPECT_TRUE(drawable.does_receive(Events::Hover));
    EXPECT_EQ(drawable.pointer_events(), drag_hover);

    drawable.set_draggable(true);
    EXPECT_TRUE(drawable.clickable());
    EXPECT_EQ(drawable.pointer_events(), drag_hover);
}

TEST(PointerEventsTest, EventSpecificHitTestingTraversesContainers) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set(PointerEvents{Events::None});

    auto click_target = Layout::Make<Rect>{Box(0, 0, 100, 100)}();
    click_target->set_clickable(true);
    click_target->set(PointerEvents{Events::Click});
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*click_target);
    });
    graph.wrap_object(parent);

    Rect drag_overlay(Box(0, 0, 100, 100));
    drag_overlay.set_clickable(true);
    drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});
    graph.wrap_object(drag_overlay);

    EXPECT_EQ(graph.find(20, 20, Events::Click), click_target.get());
    EXPECT_EQ(graph.find(20, 20, Events::Drag), &drag_overlay);
    EXPECT_EQ(graph.find(20, 20, Events::Hover), &drag_overlay);
    EXPECT_EQ(graph.find(20, 20, Events::Scroll), nullptr);
}

namespace {
struct CirclePointerScene {
    DrawStructure graph{200, 200};
    Rect click_target{Box(0, 0, 100, 100)};
    Entangled skeleton;
    Circle circle;
    Entangled parent;

    CirclePointerScene(pointer::Events events, bool clickable, int updates = 1) {
        click_target.set_clickable(true);
        click_target.set(PointerEvents{pointer::Events::Click});

        for(int i = 0; i < updates; ++i)
            update_circle(events, clickable);

        parent.update([&](Entangled& layout) {
            layout.advance_wrap(click_target);
            layout.advance_wrap(skeleton);
        });
        graph.wrap_object(parent);
    }

    void update_circle(pointer::Events events, bool clickable) {
        circle.create(
            Loc{20, 20},
            Radius{5},
            PointerEvents{events},
            Clickable{clickable});
        skeleton.update([&](Entangled& layout) {
            layout.advance_wrap(circle);
        });
    }
};
}

TEST(PointerEventsTest, NonClickableHoverCirclePassesClicksThroughAfterAdvanceWrap) {
    using pointer::Events;

    CirclePointerScene scene(Events::Hover, false, 3);
    int clicks = 0;
    scene.click_target.on_click([&](Event) {
        ++clicks;
    });

    EXPECT_FALSE(scene.circle.clickable());
    EXPECT_EQ(scene.circle.pointer_events(), Events::Hover);
    EXPECT_TRUE(scene.circle.does_receive(Events::Hover));
    EXPECT_FALSE(scene.circle.does_receive(Events::Click));
    EXPECT_EQ(scene.graph.find(20, 20, Events::Hover), &scene.circle);
    EXPECT_EQ(scene.graph.find(20, 20, Events::Click), &scene.click_target);

    EXPECT_EQ(scene.graph.mouse_move(20, 20), &scene.circle);
    EXPECT_TRUE(scene.circle.hovered());
    EXPECT_EQ(scene.graph.mouse_down(true), &scene.click_target);
    EXPECT_EQ(scene.graph.selected_object(), &scene.click_target);
    EXPECT_FALSE(scene.circle.selected());
    EXPECT_EQ(scene.graph.mouse_up(true), &scene.click_target);
    EXPECT_EQ(clicks, 1);
}

TEST(PointerEventsTest, RetainedCircleKeepsHoverButClearsSelection) {
    using pointer::Events;

    CirclePointerScene scene(Events::All, true);

    ASSERT_EQ(scene.graph.mouse_move(20, 20), &scene.circle);
    ASSERT_EQ(scene.graph.mouse_down(true), &scene.circle);
    ASSERT_EQ(scene.graph.mouse_up(true), &scene.circle);
    ASSERT_TRUE(scene.circle.hovered());
    ASSERT_TRUE(scene.circle.selected());

    scene.update_circle(Events::Hover, false);

    EXPECT_TRUE(scene.circle.hovered());
    EXPECT_FALSE(scene.circle.selected());
    EXPECT_EQ(scene.graph.mouse_position(), Vec2(20, 20));
    EXPECT_EQ(scene.graph.find(20, 20, Events::Hover), &scene.circle);
    EXPECT_EQ(scene.graph.find(20, 20, Events::Click), &scene.click_target);
}

TEST(PointerEventsTest, ShortSplitGestureCommitsUnderlyingClick) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect click_target(Box(0, 0, 100, 100));
    click_target.set_clickable(true);
    click_target.set(PointerEvents{Events::Click});

    Rect drag_overlay(Box(0, 0, 100, 100));
    drag_overlay.set_clickable(true);
    drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});

    int mouse_downs = 0;
    int clicks = 0;
    click_target.add_event_handler(MBUTTON, [&](Event event) {
        if(event.mbutton.pressed)
            ++mouse_downs;
    });
    click_target.on_click([&](Event) {
        ++clicks;
    });

    graph.wrap_object(click_target);
    graph.wrap_object(drag_overlay);

    ASSERT_EQ(graph.mouse_move(20, 20), &drag_overlay);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    EXPECT_TRUE(click_target.pressed());
    EXPECT_TRUE(drag_overlay.pressed());
    EXPECT_EQ(mouse_downs, 0);
    EXPECT_EQ(clicks, 0);

    ASSERT_NE(graph.mouse_move(23, 20), nullptr);
    EXPECT_TRUE(click_target.pressed());
    EXPECT_FALSE(drag_overlay.being_dragged());

    ASSERT_EQ(graph.mouse_up(true), &click_target);
    EXPECT_EQ(mouse_downs, 1);
    EXPECT_EQ(clicks, 1);
    EXPECT_EQ(graph.selected_object(), &click_target);
    EXPECT_FALSE(click_target.pressed());
    EXPECT_FALSE(drag_overlay.pressed());
}

TEST(PointerEventsTest, ClickReleaseUsesDragThreshold) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect background(Box(0, 0, 100, 100));
    background.set_clickable(true);
    //background.set(PointerEvents{Events::Click});

    int mouse_downs = 0;
    int mouse_ups = 0;
    int clicks = 0;
    bool release_started_here = true;
    background.add_event_handler(MBUTTON, [&](Event event) {
        if(event.mbutton.pressed) {
            ++mouse_downs;
            EXPECT_TRUE(event.mbutton.started_here);
        } else {
            ++mouse_ups;
            release_started_here = event.mbutton.started_here;
        }
    });
    background.on_click([&](Event) {
        ++clicks;
    });
    graph.wrap_object(background);

    ASSERT_EQ(graph.mouse_move(20, 20), &background);
    ASSERT_EQ(graph.mouse_down(true), &background);
    ASSERT_EQ(graph.mouse_move(30, 20), &background);
    ASSERT_EQ(graph.mouse_move(20, 20), &background);
    ASSERT_EQ(graph.mouse_up(true), &background);

    EXPECT_EQ(mouse_downs, 1);
    EXPECT_EQ(mouse_ups, 1);
    EXPECT_FALSE(release_started_here);
    EXPECT_EQ(clicks, 0);

    ASSERT_EQ(graph.mouse_move(20, 20), &background);
    ASSERT_EQ(graph.mouse_down(true), &background);
    ASSERT_EQ(graph.mouse_move(23, 20), &background);
    ASSERT_EQ(graph.mouse_up(true), &background);

    EXPECT_EQ(mouse_downs, 2);
    EXPECT_EQ(mouse_ups, 2);
    EXPECT_TRUE(release_started_here);
    EXPECT_EQ(clicks, 1);
}

TEST(PointerEventsTest, ShortSplitGestureDoesNotClickAfterLeavingTarget) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect click_target(Box(0, 0, 2, 20));
    click_target.set_clickable(true);
    click_target.set(PointerEvents{Events::Click});

    Rect drag_overlay(Box(0, 0, 100, 20));
    drag_overlay.set_clickable(true);
    drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});

    int clicks = 0;
    click_target.on_click([&](Event) {
        ++clicks;
    });

    graph.wrap_object(click_target);
    graph.wrap_object(drag_overlay);

    ASSERT_EQ(graph.mouse_move(1, 10), &drag_overlay);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    ASSERT_EQ(graph.mouse_move(4, 10), &drag_overlay);
    ASSERT_EQ(graph.mouse_up(true), &click_target);
    EXPECT_EQ(clicks, 0);
}

TEST(PointerEventsTest, BeingDraggedOnlyTracksMovableDrawables) {
    Rect movable(Box(0, 0, 100, 100));
    movable.set_draggable(true);

    DrawStructure graph(200, 200);
    graph.wrap_object(movable);

    ASSERT_EQ(graph.mouse_move(20, 20), &movable);
    ASSERT_EQ(graph.mouse_down(true), &movable);
    EXPECT_TRUE(movable.being_dragged());

    ASSERT_EQ(graph.mouse_move(30, 20), &movable);
    EXPECT_EQ(movable.pos(), Vec2(10, 0));

    ASSERT_EQ(graph.mouse_up(true), &movable);
    EXPECT_FALSE(movable.being_dragged());
}

TEST(PointerEventsTest, DragReleaseDoesNotBubbleMouseUpToParent) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_clickable(true);

    auto child = Layout::Make<Rect>{Box(0, 0, 60, 60)}();
    child->set_draggable(true);
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*child);
    });

    int parent_mouse_ups = 0;
    int parent_clicks = 0;
    parent.add_event_handler(MBUTTON, [&](Event event) -> bool {
        if(not event.mbutton.pressed)
            ++parent_mouse_ups;
        return false;
    });
    parent.on_click([&](Event) {
        ++parent_clicks;
    });

    graph.wrap_object(parent);

    ASSERT_EQ(graph.mouse_move(20, 20), child.get());
    ASSERT_EQ(graph.mouse_down(true), child.get());
    EXPECT_TRUE(child->pressed());
    EXPECT_FALSE(parent.pressed());
    ASSERT_EQ(graph.mouse_move(30, 20), child.get());
    EXPECT_EQ(child->pos(), Vec2(10, 0));

    ASSERT_EQ(graph.mouse_up(true), child.get());
    EXPECT_EQ(parent_mouse_ups, 0);
    EXPECT_EQ(parent_clicks, 0);
    EXPECT_FALSE(parent.pressed());
}

TEST(PointerEventsTest, ClickChildRetainsDraggableParentBehavior) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 100, 100));
    parent.set_draggable(true);

    auto child = Layout::Make<Rect>{Box(0, 0, 50, 50), Clickable{true}}();
    int child_mouse_ups = 0;
    child->add_event_handler(MBUTTON, [&](Event event) {
        if(not event.mbutton.pressed)
            ++child_mouse_ups;
    });
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*child);
    });
    graph.wrap_object(parent);

    ASSERT_EQ(graph.mouse_move(20, 20), child.get());
    ASSERT_EQ(graph.mouse_down(true), child.get());
    ASSERT_EQ(graph.mouse_move(30, 20), &parent);
    EXPECT_EQ(parent.pos(), Vec2(10, 0));
    EXPECT_TRUE(parent.being_dragged());

    ASSERT_EQ(graph.mouse_up(true), child.get());
    EXPECT_EQ(child_mouse_ups, 0);
    EXPECT_FALSE(parent.being_dragged());
}

TEST(PointerEventsTest, CapturedDragSuppressesCrossedTargetsAndClick) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect click_target(Box(0, 0, 120, 60));
    click_target.set_clickable(true);
    click_target.set(PointerEvents{Events::Click});

    Rect drag_overlay(Box(0, 0, 50, 60));
    drag_overlay.set_clickable(true);
    drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});

    Rect crossed_target(Box(50, 0, 70, 60));
    crossed_target.set_clickable(true);
    crossed_target.set(PointerEvents{Events::Drag | Events::Hover});

    int clicks = 0;
    int captured_updates = 0;
    int crossed_hovers = 0;
    int crossed_drags = 0;
    click_target.on_click([&](Event) {
        ++clicks;
    });
    drag_overlay.add_event_handler(DRAG, [&](Event) {
        ++captured_updates;
    });
    crossed_target.on_hover([&](Event event) {
        if(event.hover.hovered)
            ++crossed_hovers;
    });
    crossed_target.add_event_handler(DRAG, [&](Event) {
        ++crossed_drags;
    });

    graph.wrap_object(click_target);
    graph.wrap_object(drag_overlay);
    graph.wrap_object(crossed_target);

    ASSERT_EQ(graph.mouse_move(20, 20), &drag_overlay);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    EXPECT_TRUE(graph.has_active_pointer_gesture());
    ASSERT_EQ(graph.mouse_move(26, 20), &drag_overlay);
    EXPECT_TRUE(graph.has_active_pointer_gesture());

    EXPECT_FALSE(click_target.pressed());
    EXPECT_TRUE(drag_overlay.pressed());
    EXPECT_FALSE(drag_overlay.being_dragged());
    EXPECT_FALSE(drag_overlay.hovered());
    EXPECT_GT(captured_updates, 0);

    ASSERT_EQ(graph.mouse_move(80, 20), &drag_overlay);
    EXPECT_FALSE(crossed_target.hovered());
    EXPECT_EQ(crossed_hovers, 0);
    EXPECT_EQ(crossed_drags, 0);
    EXPECT_EQ(clicks, 0);

    ASSERT_NO_THROW(graph.mouse_up(true));
    EXPECT_FALSE(graph.has_active_pointer_gesture());
    EXPECT_FALSE(drag_overlay.pressed());
    EXPECT_FALSE(drag_overlay.being_dragged());
    EXPECT_TRUE(crossed_target.hovered());
    EXPECT_EQ(crossed_hovers, 1);
    EXPECT_EQ(crossed_drags, 0);
    EXPECT_EQ(clicks, 0);
}

TEST(PointerEventsTest, RemovingGestureTargetCancelsPendingAndCapturedState) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect click_target(Box(0, 0, 100, 100));
    click_target.set_clickable(true);
    click_target.set(PointerEvents{Events::Click});

    int clicks = 0;
    click_target.on_click([&](Event) {
        ++clicks;
    });
    graph.wrap_object(click_target);

    {
        Rect drag_overlay(Box(0, 0, 100, 100));
        drag_overlay.set_clickable(true);
        drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});
        graph.wrap_object(drag_overlay);

        ASSERT_EQ(graph.mouse_move(20, 20), &drag_overlay);
        ASSERT_NE(graph.mouse_down(true), nullptr);
        EXPECT_TRUE(click_target.pressed());
        EXPECT_TRUE(drag_overlay.pressed());
    }

    EXPECT_FALSE(click_target.pressed());
    EXPECT_EQ(clicks, 0);
    EXPECT_NO_THROW(graph.mouse_up(true));
    EXPECT_EQ(clicks, 0);

    {
        Rect drag_overlay(Box(0, 0, 100, 100));
        drag_overlay.set_clickable(true);
        drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});
        graph.wrap_object(drag_overlay);

        ASSERT_EQ(graph.mouse_move(20, 20), &drag_overlay);
        ASSERT_NE(graph.mouse_down(true), nullptr);
        ASSERT_EQ(graph.mouse_move(26, 20), &drag_overlay);
        EXPECT_FALSE(drag_overlay.being_dragged());
    }

    EXPECT_FALSE(click_target.pressed());
    EXPECT_NO_THROW(graph.mouse_up(true));
    EXPECT_EQ(clicks, 0);
}

TEST(PointerEventsTest, RightButtonUsesClickTargetWithoutDragArbitration) {
    using pointer::Events;

    DrawStructure graph(200, 200);
    Rect click_target(Box(0, 0, 100, 100));
    click_target.set_clickable(true);
    click_target.set(PointerEvents{Events::Click});

    Rect drag_overlay(Box(0, 0, 100, 100));
    drag_overlay.set_clickable(true);
    drag_overlay.set(PointerEvents{Events::Drag | Events::Hover});

    int right_events = 0;
    click_target.add_event_handler(MBUTTON, [&](Event event) {
        if(event.mbutton.button == 1)
            ++right_events;
    });

    graph.wrap_object(click_target);
    graph.wrap_object(drag_overlay);

    ASSERT_EQ(graph.mouse_move(20, 20), &drag_overlay);
    EXPECT_EQ(graph.mouse_down(false), &click_target);
    EXPECT_FALSE(drag_overlay.pressed());
    EXPECT_FALSE(drag_overlay.being_dragged());

    EXPECT_EQ(graph.mouse_up(false), &click_target);
    EXPECT_EQ(right_events, 2);
}

TEST(TestGuiScroll, ScrollEnabledButtonConsumesScroll) {
    DrawStructure graph(200, 200);
    Button button(Str("Owns scroll"), Box(10, 10, 90, 30));

    graph.wrap_object(button);
    ASSERT_TRUE(button.scroll_enabled());
    ASSERT_EQ(graph.mouse_move(20, 20), &button);
    ASSERT_EQ(graph.hovered_object(), &button);

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), &button);
    EXPECT_EQ(button.scroll_offset(), Vec2());
}

TEST(TestGuiScroll, ScrollEnabledButtonConsumesBeforeScrollableParent) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(), Rangef(0, 100));

    auto button = Layout::Make<Button>{
        Str("Child"), Box(10, 10, 90, 30)
    }();
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*button);
    });
    ASSERT_NE(button.get(), nullptr);

    graph.wrap_object(parent);
    ASSERT_TRUE(button->scroll_enabled());
    ASSERT_EQ(graph.mouse_move(20, 20), button.get());
    ASSERT_EQ(graph.hovered_object(), button.get());

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), button.get());
    EXPECT_EQ(button->scroll_offset(), Vec2());
    EXPECT_EQ(parent.scroll_offset(), Vec2());
}

TEST(TestGuiScroll, NonScrollableChildBubblesScrollToScrollableParent) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(), Rangef(0, 100));

    auto child = Layout::Make<Rect>{
        Box(10, 10, 90, 30), Clickable{true}
    }();
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*child);
    });

    graph.wrap_object(parent);
    ASSERT_EQ(graph.mouse_move(20, 20), child.get());

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), &parent);
    EXPECT_EQ(parent.scroll_offset(), Vec2(0, 10));
}

TEST(TestGuiScroll, InnerScrollableConsumesAtBoundary) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(), Rangef(0, 100));

    auto child = Layout::Make<Entangled>{Box(10, 10, 90, 90)}();
    child->set_clickable(true);
    child->set_scroll_enabled(true);
    child->set_scroll_limits(Rangef(), Rangef(0, 100));
    child->set_scroll_offset(Vec2(0, 100));
    parent.update([&](Entangled& layout) {
        layout.advance_wrap(*child);
    });

    graph.wrap_object(parent);
    ASSERT_TRUE(child->scroll_enabled());
    ASSERT_EQ(graph.mouse_move(20, 20), child.get());

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), child.get());
    EXPECT_EQ(child->scroll_offset(), Vec2(0, 100));
    EXPECT_EQ(parent.scroll_offset(), Vec2());
}

TEST(TestGuiScroll, HorizontalAxisMapsOrdinaryWheelMovementToX) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Horizontal);
    parent.set_scroll_enabled(true);
    parent.set_clickable(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(0, 20));

    graph.wrap_object(parent);
    ASSERT_EQ(graph.mouse_move(20, 20), &parent);

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), &parent);
    EXPECT_EQ(parent.scroll_offset(), Vec2(10, 20));
}

TEST(TestGuiScroll, HorizontalAxisRendersScrollbarAlongBottomEdge) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Horizontal);
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 0));
    parent.set_scroll_offset(Vec2(50, 0));
    graph.wrap_object(parent);

    parent.update([](Entangled&) {});
    ASSERT_EQ(parent.children().size(), 3u);
    ASSERT_TRUE(dynamic_cast<Rect*>(parent.children()[0]));
    ASSERT_TRUE(dynamic_cast<Rect*>(parent.children()[1]));
    EXPECT_EQ(parent.children()[0]->global_bounds(), Bounds(0, 112, 60, 8));
    EXPECT_EQ(parent.children()[1]->global_bounds(), Bounds(60, 112, 60, 8));
    EXPECT_EQ(parent.children()[2]->global_bounds(), Bounds(0, 112, 120, 8));
}

TEST(TestGuiScroll, DefaultAxisPreservesLegacyVerticalScrollbarBehavior) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(50, 50));
    graph.wrap_object(parent);

    parent.update([](Entangled&) {});
    ASSERT_EQ(parent.children().size(), 3u);
    EXPECT_EQ(parent.children()[0]->global_bounds(), Bounds(104, 0, 8, 60));
    EXPECT_EQ(parent.children()[1]->global_bounds(), Bounds(104, 60, 8, 60));
    EXPECT_EQ(parent.children()[2]->global_bounds(), Bounds(104, 0, 16, 120));
}

TEST(TestGuiScroll, HorizontalScrollbarDragPreservesVerticalOffset) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Horizontal);
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(50, 20));
    graph.wrap_object(parent);
    parent.update([](Entangled&) {});

    ASSERT_NE(graph.mouse_move(30, 116), nullptr);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    ASSERT_NE(graph.mouse_move(90, 116), nullptr);
    EXPECT_EQ(parent.scroll_offset().y, 20);
    EXPECT_NE(parent.scroll_offset().x, 50);
    ASSERT_NO_THROW(graph.mouse_up(true));
}

TEST(TestGuiScroll, VerticalScrollbarDragPreservesHorizontalOffset) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Vertical);
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(20, 50));
    graph.wrap_object(parent);
    parent.update([](Entangled&) {});

    ASSERT_NE(graph.mouse_move(108, 30), nullptr);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    ASSERT_NE(graph.mouse_move(108, 90), nullptr);
    EXPECT_EQ(parent.scroll_offset().x, 20);
    EXPECT_NE(parent.scroll_offset().y, 50);
    ASSERT_NO_THROW(graph.mouse_up(true));
}

TEST(TestGuiScroll, VerticalScrollbarClickJumpsImmediately) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Vertical);
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(20, 50));
    graph.wrap_object(parent);
    parent.update([](Entangled&) {});

    ASSERT_NE(graph.mouse_move(108, 90), nullptr);
    ASSERT_NE(graph.mouse_down(true), nullptr);
    EXPECT_EQ(parent.scroll_offset(), Vec2(20, 75));
    ASSERT_NO_THROW(graph.mouse_up(true));
}

TEST(TestGuiScroll, DropdownScrollbarDragSurvivesListRebuilds) {
    ScrollableList<std::string> list;
    list.set(Foldable_t{true});
    list.set(Folded_t{false});
    list.set(LabelDims_t{100, 30});
    list.set(ListDims_t{120, 80});
    list.set_items({
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9"
    });

    DrawStructure graph(300, 300);
    graph.set_dialog_window_size(Size2(300, 300));
    graph.wrap_object(list);
    graph.collect();
    // The first layout pass resolves the dropdown's content-dependent limits.
    list.set_content_changed(true);
    graph.collect();

    Entangled* dropdown = nullptr;
    for(auto* child : list.children()) {
        if(child->type() == Type::ENTANGLED) {
            auto* candidate = static_cast<Entangled*>(child);
            if(candidate->scroll_enabled()
               && candidate->size() == Size2(120, 80))
            {
                dropdown = candidate;
                break;
            }
        }
    }
    ASSERT_NE(dropdown, nullptr);

    Drawable* scrollbar = nullptr;
    Float2_t largest_scrollbar_area = 0;
    for(auto* child : dropdown->children()) {
        if(child->custom_data("scrollbar")) {
            const auto bounds = child->global_bounds();
            const auto area = bounds.width * bounds.height;
            if(area > largest_scrollbar_area) {
                scrollbar = child;
                largest_scrollbar_area = area;
            }
        }
    }
    ASSERT_NE(scrollbar, nullptr);

    const auto dropdown_bounds = dropdown->global_bounds();
    const auto scrollbar_bounds = scrollbar->global_bounds();
    const auto drag_x = scrollbar_bounds.x + min(4_F, scrollbar_bounds.width * 0.5_F);
    const auto drag_start_y = dropdown_bounds.y + dropdown_bounds.height - 10;
    const auto drag_middle_y = dropdown_bounds.y + dropdown_bounds.height * 0.625_F;
    const auto drag_end_y = dropdown_bounds.y + dropdown_bounds.height * 0.375_F;

    ASSERT_EQ(graph.mouse_move(drag_x, drag_start_y), scrollbar);
    ASSERT_EQ(graph.mouse_down(true), scrollbar);
    ASSERT_TRUE(graph.has_active_pointer_gesture());

    // Open dropdowns rebuild their visible rows after the initial scrollbar
    // update. The captured drag target must survive that rebuild.
    graph.collect();
    const auto after_click = dropdown->scroll_offset().y;

    ASSERT_EQ(graph.mouse_move(drag_x, drag_middle_y), scrollbar);
    const auto after_first_drag = dropdown->scroll_offset().y;
    EXPECT_LT(after_first_drag, after_click);
    graph.collect();
    ASSERT_EQ(graph.mouse_move(drag_x, drag_end_y), scrollbar);
    EXPECT_LT(dropdown->scroll_offset().y, after_first_drag);
    EXPECT_EQ(list.last_selected_item(), -1);
    EXPECT_TRUE(graph.has_active_pointer_gesture());
    ASSERT_NO_THROW(graph.mouse_up(true));
}

TEST(TestGuiScroll, DropdownScrollbarBlocksBorderSideItemClicks) {
    ScrollableList<std::string> list;
    list.set(Foldable_t{true});
    list.set(Folded_t{false});
    list.set(LabelDims_t{100, 30});
    list.set(ListDims_t{120, 80});
    list.set_items({"0", "1", "2", "3", "4"});

    DrawStructure graph(300, 300);
    graph.set_dialog_window_size(Size2(300, 300));
    graph.wrap_object(list);
    graph.collect();
    // The first layout pass resolves the dropdown's content-dependent limits.
    list.set_content_changed(true);
    graph.collect();

    Entangled* dropdown = nullptr;
    for(auto* child : list.children()) {
        if(child->type() == Type::ENTANGLED) {
            auto* candidate = static_cast<Entangled*>(child);
            if(candidate->scroll_enabled()
               && candidate->size() == Size2(120, 80))
            {
                dropdown = candidate;
                break;
            }
        }
    }
    ASSERT_NE(dropdown, nullptr);

    Drawable* visible_scrollbar = nullptr;
    for(auto* child : dropdown->children()) {
        if(child->custom_data("scrollbar")
           && child->global_bounds().height > 0
           && (!visible_scrollbar
               || child->global_bounds().width
                    < visible_scrollbar->global_bounds().width))
        {
            visible_scrollbar = child;
        }
    }
    ASSERT_NE(visible_scrollbar, nullptr);

    const auto dropdown_bounds = dropdown->global_bounds();
    const auto scrollbar_bounds = visible_scrollbar->global_bounds();
    const auto border_side_x =
        (scrollbar_bounds.x + scrollbar_bounds.width
         + dropdown_bounds.x + dropdown_bounds.width) * 0.5_F;
    const auto border_side_y =
        scrollbar_bounds.y + scrollbar_bounds.height * 0.5_F;

    auto* border_side = graph.mouse_move(border_side_x, border_side_y);
    ASSERT_NE(border_side, nullptr);
    ASSERT_NE(border_side->custom_data("scrollbar"), nullptr);
    ASSERT_EQ(graph.mouse_down(true), border_side);
    ASSERT_NO_THROW(graph.mouse_up(true));
    EXPECT_EQ(list.last_selected_item(), -1);
    EXPECT_FALSE(list.folded());
}

TEST(TestGuiScroll, HidingScrollbarDoesNotDisableScrolling) {
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Horizontal);
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 0));
    parent.set_scroll_show_bar(false);

    parent.update([](Entangled&) {});
    EXPECT_TRUE(parent.scroll_enabled());
    EXPECT_TRUE(parent.children().empty());
}

TEST(TestGuiScroll, TighteningLimitsClampsTheExistingOffsetImmediately) {
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(80, 70));

    parent.set_scroll_limits(Rangef(0, 30), Rangef(0, 40));
    EXPECT_EQ(parent.scroll_offset(), Vec2(30, 40));
}

TEST(FloatingLayoutTest, HorizontalFirstWrapsRowsAndCapsVerticalViewport) {
    auto first = Layout::Make<Rect>{Box{0, 0, 40, 10}}();
    auto second = Layout::Make<Rect>{Box{0, 0, 40, 10}}();
    auto third = Layout::Make<Rect>{Box{0, 0, 40, 10}}();
    FloatingLayout layout{
        std::vector<Layout::Ptr>{first, second, third},
        FloatingLayout::Policy::HorizontalFirst,
        attr::Margins{1, 2, 3, 4},
        OuterPadding{3, 4, 5, 6},
        attr::SizeLimit{Size2{100, 30}}
    };
    layout.update();

    EXPECT_EQ(first->pos(), Vec2(4, 6));
    EXPECT_EQ(second->pos(), Vec2(48, 6));
    EXPECT_EQ(third->pos(), Vec2(4, 22));
    EXPECT_EQ(layout.content_size(), Size2(96, 42));
    EXPECT_EQ(layout.size(), Size2(96, 30));
    EXPECT_TRUE(layout.scroll_enabled());
    EXPECT_EQ(layout.scroll_axis(), ScrollAxis::Vertical);
    EXPECT_EQ(layout.scroll_limit_y(), Rangef(0, 12));
}

TEST(FloatingLayoutTest, VerticalFirstWrapsColumnsAndCapsHorizontalViewport) {
    auto first = Layout::Make<Rect>{Box{0, 0, 10, 10}}();
    auto second = Layout::Make<Rect>{Box{0, 0, 10, 10}}();
    auto third = Layout::Make<Rect>{Box{0, 0, 10, 10}}();
    FloatingLayout layout{
        std::vector<Layout::Ptr>{first, second, third},
        FloatingLayout::Policy::VerticalFirst,
        attr::Margins{1, 2, 3, 4},
        OuterPadding{3, 4, 5, 6},
        attr::SizeLimit{Size2{25, 45}}
    };
    layout.update();

    EXPECT_EQ(first->pos(), Vec2(4, 6));
    EXPECT_EQ(second->pos(), Vec2(4, 22));
    EXPECT_EQ(third->pos(), Vec2(18, 6));
    EXPECT_EQ(layout.content_size(), Size2(36, 42));
    EXPECT_EQ(layout.size(), Size2(25, 42));
    EXPECT_TRUE(layout.scroll_enabled());
    EXPECT_EQ(layout.scroll_axis(), ScrollAxis::Horizontal);
    EXPECT_EQ(layout.scroll_limit_x(), Rangef(0, 11));
}

TEST(FloatingLayoutTest, IgnoresEmptyChildrenAndClampsAfterContentShrink) {
    auto empty = Layout::Make<Rect>{Box{0, 0, 0, 10}}();
    auto first = Layout::Make<Rect>{Box{0, 0, 80, 10}}();
    auto second = Layout::Make<Rect>{Box{0, 0, 80, 10}}();
    auto third = Layout::Make<Rect>{Box{0, 0, 80, 10}}();
    first->set_origin(Vec2(0.5f));
    FloatingLayout layout{
        std::vector<Layout::Ptr>{empty, first, second},
        FloatingLayout::Policy::HorizontalFirst,
        attr::SizeLimit{Size2{50, 15}}
    };
    DrawStructure graph(200, 200);
    graph.wrap_object(layout);
    layout.update();

    EXPECT_EQ(empty->pos(), Vec2());
    EXPECT_EQ(first->size(), Size2(80, 10));
    EXPECT_EQ(first->pos(), Vec2(40, 5));
    EXPECT_EQ(layout.content_size(), Size2(80, 20));
    // The primary-axis limit controls wrapping. An oversized item remains fully
    // visible; only the policy's scrolling axis is capped.
    EXPECT_EQ(layout.size(), Size2(80, 15));
    ASSERT_TRUE(layout.scroll_enabled());
    layout.set_scroll_offset(Vec2(0, 5));

    layout.set_children(std::vector<Layout::Ptr>{first, second, third});
    layout.update();
    EXPECT_EQ(layout.scroll_offset(), Vec2(0, 5));

    layout.set_children(std::vector<Layout::Ptr>{first});
    layout.update();
    EXPECT_EQ(layout.content_size(), Size2(80, 10));
    EXPECT_EQ(layout.size(), Size2(80, 10));
    EXPECT_FALSE(layout.scroll_enabled());
    EXPECT_EQ(layout.scroll_offset(), Vec2());
}

TEST(FloatingLayoutTest, VerticalFirstKeepsOversizedChildrenVisibleOnItsWrappingAxis) {
    auto first = Layout::Make<Rect>{Box{0, 0, 10, 80}}();
    auto second = Layout::Make<Rect>{Box{0, 0, 10, 80}}();
    FloatingLayout layout{
        std::vector<Layout::Ptr>{first, second},
        FloatingLayout::Policy::VerticalFirst,
        attr::SizeLimit{Size2{15, 50}}
    };
    layout.update();

    EXPECT_EQ(first->pos(), Vec2(0, 0));
    EXPECT_EQ(second->pos(), Vec2(10, 0));
    EXPECT_EQ(layout.content_size(), Size2(20, 80));
    EXPECT_EQ(layout.size(), Size2(15, 80));
    EXPECT_TRUE(layout.scroll_enabled());
    EXPECT_EQ(layout.scroll_axis(), ScrollAxis::Horizontal);
    EXPECT_EQ(layout.scroll_limit_x(), Rangef(0, 5));
}

TEST(FloatingLayoutTest, NonPositiveLimitsRemainUnboundedForEmptyAndNonEmptyContent) {
    auto empty = Layout::Make<Rect>{Box{0, 0, 0, 10}}();
    auto first = Layout::Make<Rect>{Box{0, 0, 40, 10}}();
    auto second = Layout::Make<Rect>{Box{0, 0, 40, 10}}();
    FloatingLayout layout{
        std::vector<Layout::Ptr>{empty, first, second},
        OuterPadding{3, 4, 5, 6},
        attr::SizeLimit{Size2{0, -1}}
    };
    layout.update();

    EXPECT_EQ(first->pos(), Vec2(3, 4));
    EXPECT_EQ(second->pos(), Vec2(43, 4));
    EXPECT_EQ(layout.content_size(), Size2(88, 20));
    EXPECT_EQ(layout.size(), Size2(88, 20));
    EXPECT_FALSE(layout.scroll_enabled());

    layout.set_children(std::vector<Layout::Ptr>{empty});
    layout.update();
    EXPECT_EQ(layout.content_size(), Size2(8, 10));
    EXPECT_EQ(layout.size(), Size2(8, 10));
}
