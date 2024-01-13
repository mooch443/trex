#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/SettingsTooltip.h>
#include <misc/ThreadManager.h>
#include <misc/DetectionImageTypes.h>

class AbstractBaseVideoSource;

namespace gui {

class SettingsScene : public Scene {
    // showing preview video
    
    useMatPtr_t intermediate{nullptr};
    Timer _last_image_timer;
    std::shared_ptr<ExternalImage> _preview_image;
    
    SettingsTooltip _settings_tooltip;
    std::shared_ptr<VerticalLayout> _buttons_and_items = std::make_shared<VerticalLayout>();
    std::shared_ptr<Layout> _logo_title_layout = std::make_shared<Layout>();
    std::shared_ptr<HorizontalLayout> _button_layout;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    Size2 window_size;
    dyn::DynamicGUI dynGUI;
    CallbackCollection callback;
    ThreadGroupId group;
    
    Frame_t last_frame;
    std::unique_ptr<AbstractBaseVideoSource> _source;
    std::atomic<bool> video_changed{true};
    std::atomic<double> blur_percentage{0};
    std::atomic<Size2> max_resolution;
    double blur_target{1};
    std::mutex image_mutex;
    Image::Ptr transfer_image, local_image, return_image;
    Timer timer, animation_timer;
    std::atomic<size_t> allowances{0};
    
public:
    SettingsScene(Base& window);
    void activate() override;
    void deactivate() override;
    void _draw(DrawStructure& graph);
    bool on_global_event(Event) override;
};
}

