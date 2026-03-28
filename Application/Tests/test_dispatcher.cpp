#include "gtest/gtest.h"

#include <core/TimingStatsCollector.h>
#include <gui/Dispatcher.h>
#include <gui/dyn/VarProps.h>
#include <gui/types/StaticText.h>
#include <misc/frame_t.h>
#include <ui/ImageGeneratorRegistry.h>

using namespace cmn;
using namespace cmn::gui::attr;

namespace {

using Dispatcher = cmn::gui::attr::Dispatcher;
using cmn::gui::StaticText;

const bool kStaticTextRegistration = Dispatcher::register_pending(+[](Dispatcher& dispatcher) {
    dispatcher.register_custom<StaticText, int>([](StaticText& text, const int& value) {
        text.set_txt(std::to_string(value));
        return true;
    });
});

} // namespace

TEST(TimingStatsCollectorTest, ExplicitCollectorRecordsEvents) {
    auto collector = std::make_shared<TimingStatsCollector>();
    auto handle = collector->startEvent(TimingMetric_t::FrameRender, 5_f);
    collector->endEvent(handle);

    auto events = collector->getEvents(std::chrono::seconds(5));
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.back().metric, TimingMetric_t::FrameRender);
    EXPECT_EQ(events.back().frameIndex, 5_f);
    EXPECT_TRUE(events.back().end.has_value());
}

TEST(ImageGeneratorRegistryTest, ExplicitRegistryReturnsRegisteredGenerator) {
    cmn::gui::ImageGeneratorRegistry registry;
    bool reset_called = false;
    registry.register_generator("demo", {
        .generate = [](const cmn::gui::dyn::VarProps&) {
            return cmn::Image::Make(1, 1, 4);
        },
        .reset = [&]() {
            reset_called = true;
        }
    });

    auto generator = registry.get_generator("demo");
    auto image = generator.generate(cmn::gui::dyn::VarProps{});
    ASSERT_TRUE(image != nullptr);

    registry.reset_generator("demo");
    EXPECT_TRUE(reset_called);
}

#if COMMONS_DISPATCHER_REQUIRE_EXPLICIT_INSTANCE
TEST(DispatcherTest, ExplicitModeRequiresSetupBeforeUse) {
    Dispatcher::set_instance(nullptr);
    EXPECT_EQ(Dispatcher::instance_if_set(), nullptr);
    EXPECT_ANY_THROW((void)Dispatcher::instance());

    Dispatcher canonical;
    Dispatcher::set_instance(&canonical);

    StaticText text(Str{"before"}, Box(cmn::Vec2(0, 0), cmn::Size2(50, 20)));
    EXPECT_TRUE(Dispatcher::instance().apply(text, 7));
    EXPECT_EQ(text.text(), "7");
}
#else
TEST(DispatcherTest, AutoModeReplaysPendingRegistrationsOnFirstUse) {
    Dispatcher::set_instance(nullptr);
    EXPECT_EQ(Dispatcher::instance_if_set(), nullptr);

    StaticText text(Str{"before"}, Box(cmn::Vec2(0, 0), cmn::Size2(50, 20)));
    EXPECT_TRUE(Dispatcher::instance().apply(text, 7));
    EXPECT_EQ(text.text(), "7");
    EXPECT_NE(Dispatcher::instance_if_set(), nullptr);
}
#endif
