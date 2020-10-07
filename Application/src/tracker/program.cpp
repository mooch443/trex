#include <gui/IMGUIBase.h>
#include <gui/DrawStructure.h>
#include <misc/GlobalSettings.h>
#include <imgui/imgui.h>
#include <gui/types/Button.h>
#include <gui/types/Checkbox.h>
#include <misc/Timer.h>
#include <gui/Graph.h>
#include <misc/PixelTree.h>
#include <processing/CPULabeling.h>

int main() {
    using namespace gui;
    
    Image image(300, 300, 1);
    cv::circle(image.get(), Vec2(50, 50), 10, White, -1);
    cv::imshow("raw", image.get());
    
    auto blobs = CPULabeling::run_fast(image.get());
    for (auto && [lines, pixels] : blobs) {
        auto blob = std::make_shared<pv::Blob>(lines, pixels);
        auto outlines = pixel::find_outer_points(blob, 0);
        
    }
    
    cv::rectangle(image.get(), Vec2(0,0), Vec2(300, 300), Blue, -1);
    
    Timer timer;
    
    Graph g(Bounds(250, 250, 500, 300), "", Rangef(0, M_PI*2), Rangef(-1, 1));
    g.add_function(Graph::Function("sin(x)", Graph::Type::CONTINUOUS, [](float x) {
        return sinf(x);
    }));
    g.add_function(Graph::Function("cos(x)", Graph::Type::CONTINUOUS, [](float x) {
        return cosf(x);
    }));
    g.set_draggable();
    
    DrawStructure graph(1024, 768);
    IMGUIBase base("Test", graph, [&](){
        std::lock_guard<std::recursive_mutex> lock(graph.lock());
        //graph.image(Vec2(10, 10), image);
        graph.circle(Vec2(100, 100), 50, Blue, Red);
        
        graph.section("tmp", [](DrawStructure&base, auto section) {
            static Button button("test", Bounds(300, 300, 100, 35));
            section->set_scale(Vec2(1));
            button.set_line_clr(White);
            //static Circle button(Vec2(300, 30), 50, Blue, Blue);
            base.wrap_object(button);
            
            static Rect rect(Bounds(), Transparent, White);
            base.wrap_object(rect);
            auto text = base.text("boundary_text", Vec2(50, 150));
            rect.set_bounds(text->bounds());
        });
        
        graph.wrap_object(g);
        
        static Checkbox checkbox(Vec2(50, 250), "Hi");
        graph.wrap_object(checkbox);
        
        auto str = DEBUG::format("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        str = Meta::toStr(DurationUS{uint64_t(timer.elapsed() * 1000 * 1000)});
        
        if(SETTING(terminate))
            return false;
        
        return true;
    }, [&](const gui::Event& e) {
        graph.event(e);
    });
    
    base.loop();
    
    Debug("Terminating");
    
    return 0;
}
