#include <commons.pc.h>
#include <gtest/gtest.h>
#include <gui/DrawCVBase.h>
#include <ui/Scene.h>

using namespace cmn;
using namespace cmn::gui;

TEST(SceneManagerTest, PreparationIsCommittedOnlyAfterSuccessfulActivation) {
    struct TestScene : Scene {
        using Scene::Scene;
        std::function<void()> on_activate, on_deactivate;
        void activate() override { if(on_activate) on_activate(); }
        void deactivate() override { if(on_deactivate) on_deactivate(); }
    };
    cv::Mat pixels = cv::Mat::zeros(1, 1, CV_8UC4);
    CVBase window(pixels);
    TestScene settings(window, "rollback-settings", [](auto&, auto&) {});
    TestScene conversion(window, "rollback-conversion", [](auto&, auto&) {});
    auto& manager = SceneManager::getInstance();
    manager.register_scene(&settings);
    manager.register_scene(&conversion);
    manager.set_fallback(settings.name());

    for(const auto failure : {"none", "prepare", "activate", "discard"}) {
        SCOPED_TRACE(failure);
        manager.set_active(&settings);
        manager.update_queue();
        std::vector<std::string> events;
        settings.on_deactivate = [&] { events.emplace_back("deactivate"); };
        conversion.on_activate = [&] {
            events.emplace_back("activate");
            if(std::string_view(failure) == "activate")
                throw std::runtime_error("activation failed");
        };
        bool released = false;
        int fallback_activations = 0;
        settings.on_activate = [&] {
            EXPECT_TRUE(released);
            ++fallback_activations;
        };
        auto pending = std::shared_ptr<int>(new int(0), [&](int* value) {
            released = true;
            delete value;
        });
        if(std::string_view(failure) == "discard")
            manager.set_active(nullptr);
        manager.set_active(conversion.name(), [&] {
            events.emplace_back("prepare");
            if(std::string_view(failure) == "prepare")
                throw std::runtime_error("preparation failed");
        }, [&, pending = std::move(pending)] {
            (void)pending;
            events.emplace_back("commit");
        });
        EXPECT_TRUE(events.empty());
        EXPECT_FALSE(released);
        manager.update_queue();
        EXPECT_TRUE(released);
        EXPECT_EQ(fallback_activations,
                  std::string_view(failure) == "prepare" || std::string_view(failure) == "activate" ? 1 : 0);
        if(std::string_view(failure) == "none")
            EXPECT_EQ(events, (std::vector<std::string>{"deactivate", "prepare", "activate", "commit"}));
        else {
            EXPECT_EQ(std::count(events.begin(), events.end(), "commit"), 0);
            EXPECT_EQ(std::count(events.begin(), events.end(), "prepare"),
                      std::string_view(failure) == "discard" ? 0 : 1);
            EXPECT_EQ(std::count(events.begin(), events.end(), "activate"),
                      std::string_view(failure) == "activate" ? 1 : 0);
        }
        settings.on_activate = {};
        settings.on_deactivate = {};
        conversion.on_activate = {};
        manager.switching_error().read();
    }
    manager.set_fallback("");
    manager.set_active(nullptr);
    manager.update_queue();
    manager.unregister_scene(&conversion);
    manager.unregister_scene(&settings);
}
