#include "gtest/gtest.h"
#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/ListAttributes.h>
#include <gui/types/Button.h>
#include <gui/types/Entangled.h>
#include <gui/types/Layout.h>

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

    Event scroll_event(Float2_t dy) {
        Event e(EventType::SCROLL);
        e.scroll.dx = 0;
        e.scroll.dy = dy;
        return e;
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

TEST(TestGuiScroll, InertClickableButtonDoesNotConsumeScroll) {
    DrawStructure graph(200, 200);
    Button button(Str("No scroll"), Box(10, 10, 90, 30));

    graph.wrap_object(button);
    ASSERT_EQ(graph.mouse_move(20, 20), &button);
    ASSERT_EQ(graph.hovered_object(), &button);

    EXPECT_FALSE(graph.event(scroll_event(-10)));
    EXPECT_EQ(button.scroll_offset(), Vec2());
}

TEST(TestGuiScroll, InertButtonBubblesScrollToScrollableParent) {
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
    ASSERT_EQ(graph.mouse_move(20, 20), button.get());
    ASSERT_EQ(graph.hovered_object(), button.get());

    EXPECT_EQ(graph.scroll(Vec2(0, -10)), &parent);
    EXPECT_EQ(button->scroll_offset(), Vec2());
    EXPECT_EQ(parent.scroll_offset(), Vec2(0, 10));
}

TEST(TestGuiScroll, HorizontalAxisMapsOrdinaryWheelMovementToX) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_axis(ScrollAxis::Horizontal);
    parent.set_scroll_enabled(true);
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
    ASSERT_EQ(parent.children().size(), 2u);
    ASSERT_TRUE(dynamic_cast<Rect*>(parent.children()[0]));
    ASSERT_TRUE(dynamic_cast<Rect*>(parent.children()[1]));
    EXPECT_EQ(parent.children()[0]->global_bounds(), Bounds(0, 112, 60, 8));
    EXPECT_EQ(parent.children()[1]->global_bounds(), Bounds(60, 112, 60, 8));
}

TEST(TestGuiScroll, DefaultAxisPreservesLegacyVerticalScrollbarBehavior) {
    DrawStructure graph(200, 200);
    Entangled parent(Box(0, 0, 120, 120));
    parent.set_scroll_enabled(true);
    parent.set_scroll_limits(Rangef(0, 100), Rangef(0, 100));
    parent.set_scroll_offset(Vec2(50, 50));
    graph.wrap_object(parent);

    parent.update([](Entangled&) {});
    ASSERT_EQ(parent.children().size(), 2u);
    EXPECT_EQ(parent.children()[0]->global_bounds(), Bounds(104, 0, 8, 60));
    EXPECT_EQ(parent.children()[1]->global_bounds(), Bounds(104, 60, 8, 60));
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
