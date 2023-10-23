#include "TrackingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>

namespace gui {

TrackingScene::TrackingScene(Base& window)
: Scene(window, "tracking-scene", [this](auto&, DrawStructure& graph){ _draw(graph); })
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print("window dimensions", window.window_dimensions().mul(dpi));
}

void TrackingScene::activate() {
    using namespace dyn;
    for(size_t i=0; i<1000; ++i) {
        sprite::Map map;
        map["i"] = i;
        map["pos"] = Vec2(100, 100 + i * 10);
        map["name"] = std::string("Text");
        map["detail"] = std::string("detail");
        map["radius"] = float(rand()) / float(RAND_MAX) * 50 + 50;
        map["angle"] = float(rand()) / float(RAND_MAX) * RADIANS(180);
        map["acceleration"] = Vec2();
        map["velocity"] = Vec2();
        map["angular_velocity"] = float(0);
        map["mass"] = float(rand()) / float(RAND_MAX) * 1 + 0.1f;
        map["inertia"] = float(rand()) / float(RAND_MAX) * 1 + 0.1f;
        map.set_do_print(false);
        _data.emplace_back(std::move(map));
        
        _individuals.emplace_back(new Variable([i, this](VarProps) -> sprite::Map& {
            return _data.at(i);
        }));
    }
    
    SETTING(video_length) = uint64_t(100);
}

void TrackingScene::deactivate() {
    dynGUI.clear();
}

void TrackingScene::_draw(DrawStructure& graph) {
    static Timer timer;
    auto dt = saturate(timer.elapsed(), 0.001, 0.1);
    
    using namespace dyn;
    if(not dynGUI)
        dynGUI = {
            .path = "tracking_layout.json",
            .graph = &graph,
            .context = {
                ActionFunc("set", [](Action action) {
                    if(action.parameters.size() != 2)
                        throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                    
                    auto parm = Meta::fromStr<std::string>(action.parameters.front());
                    if(not GlobalSettings::has(parm))
                        throw InvalidArgumentException("No parameter ",parm," in global settings.");
                    
                    auto value = action.parameters.back();
                    GlobalSettings::get(parm).get().set_value_from_string(value);
                }),
                ActionFunc("change_scene", [](Action action) {
                    if(action.parameters.empty())
                        throw U_EXCEPTION("Invalid arguments for ", action, ".");

                    auto scene = Meta::fromStr<std::string>(action.parameters.front());
                    if(not SceneManager::getInstance().is_scene_registered(scene))
                        return false;
                    SceneManager::getInstance().set_active(scene);
                    return true;
                }),
                
                VarFunc("window_size", [this](VarProps) -> Vec2 {
                    return window_size;
                }),
                
                VarFunc("fishes", [this](VarProps)
                    -> std::vector<std::shared_ptr<VarBase_t>>&
                {
                    return _individuals;
                })
            }
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height);
    element_size = Size2((window()->window_dimensions().width - max_w - 50), window()->window_dimensions().height - 25 - 50);
    
    dynGUI.update(nullptr);
    
    const float target_angular_velocity = 0.01f * 2.0f * static_cast<float>(M_PI);
    auto mouse_position = graph.mouse_position();
    float time_factor = 0.5f; // Smaller values make the system more 'inertial', larger values make it more 'responsive'
    const float linear_damping = 0.95f;  // Close to 1: almost no damping, close to 0: strong damping
    const float angular_damping = 0.95f;
    
    for(auto &i : _data) {
        auto v = i["pos"].value<Vec2>();
        auto velocity = i["velocity"].value<Vec2>();
        auto radius = i["radius"].value<float>();
        auto angle = i["angle"].value<float>();
        auto angular_velocity = i["angular_velocity"].value<float>();
        auto mass = i["mass"].value<float>();
        auto inertia = i["inertia"].value<float>();
        
        // Apply damping
        velocity *= linear_damping;
        angular_velocity *= angular_damping;

        // Calculate new target angle based on mouse position
        float dx = mouse_position.x - v.x;
        float dy = mouse_position.y - v.y;
        float target_angle = atan2(dy, dx);
        
        // Calculate angular direction
        float angular_direction = target_angle - angle;

        // Update angular acceleration, angular_velocity and angle
        float angular_acceleration = angular_direction / inertia;
        angular_velocity += angular_acceleration * dt;
        angle += angular_velocity * dt;

        // Ensure angle is within bounds
        while (angle > 2.0f * static_cast<float>(M_PI)) {
            angle -= 2.0f * static_cast<float>(M_PI);
        }

        // Calculate new target position
        float target_x = mouse_position.x + cos(angle) * radius;
        float target_y = mouse_position.y + sin(angle) * radius;
        Vec2 target(target_x, target_y);

        // Calculate direction and update linear acceleration, velocity and position
        Vec2 direction = target - v;
        Vec2 acceleration = direction / mass;
        velocity += acceleration * dt;
        v += velocity * dt;

        // Update the attributes
        i["pos"] = v;
        i["velocity"] = velocity;
        i["angle"] = angle;
        i["angular_velocity"] = angular_velocity;
    }
    timer.reset();
    graph.root().set_dirty();
}

}
