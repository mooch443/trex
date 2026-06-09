#include "gtest/gtest.h"
#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/types/Button.h>
#include <gui/types/Entangled.h>

using namespace cmn;
using namespace cmn::gui;

namespace {
    Event scroll_event(Float2_t dy) {
        Event e(EventType::SCROLL);
        e.scroll.dx = 0;
        e.scroll.dy = dy;
        return e;
    }
}

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
