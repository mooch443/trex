#include "gtest/gtest.h"
#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/ListAttributes.h>
#include <gui/types/Button.h>
#include <gui/types/Entangled.h>

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
