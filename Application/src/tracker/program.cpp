#include <commons.pc.h>
#include <gui/IMGUIBase.h>
#include <gui/DrawStructure.h>
#include <misc/GlobalSettings.h>
#include <gui/types/Button.h>
#include <gui/types/Checkbox.h>
#include <misc/Timer.h>
#include <gui/Graph.h>
#include <misc/PixelTree.h>
#include <processing/CPULabeling.h>
#include <misc/PVBlob.h>
#include <processing/DLList.h>

int main() {
    using namespace cmn;
    using namespace cmn::gui;
    
    SETTING(terminate) = false;
    
    Image image(300, 300, 1);
    cv::circle(image.get(), Vec2(50, 50), 10, White, -1);
    cv::imshow("raw", image.get());
    
    cmn::CPULabeling::DLList list;
    auto blobs = CPULabeling::run(list, image.get());
    for (auto && pair : blobs) {
        auto blob = std::make_shared<pv::Blob>(std::move(pair.lines), std::move(pair.pixels), pair.extra_flags, std::move(pair.pred));
        auto outlines = pixel::find_outer_points(blob.get(), 0);
        
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
    
    IMGUIBase base("Test", {1024,768}, [&](DrawStructure& graph){
        graph.circle(Loc(100, 100), Radius{50}, FillClr{Blue}, LineClr{Red});
        
        graph.section("tmp", [](DrawStructure&base, auto section) {
            static Button button(Str{"test"}, Box(300, 300, 100, 35));
            section->set_scale(Vec2(1));
            button.set_line_clr(White);
            //static Circle button(Vec2(300, 30), 50, Blue, Blue);
            base.wrap_object(button);
            
            static Rect rect(FillClr{Transparent}, LineClr{White});
            base.wrap_object(rect);
            auto text = base.text(Str("boundary_text"), Loc(50, 150));
            rect.set_bounds(text->bounds());
        });
        
        graph.wrap_object(g);
        
        static Checkbox checkbox(Loc(50, 250), Str("Hi"));
        graph.wrap_object(checkbox);
        
        auto str = format<FormatterType::NONE>("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        str = Meta::toStr(DurationUS{uint64_t(timer.elapsed() * 1000 * 1000)});
        
        if(SETTING(terminate))
            return false;
        
        return true;
    }, [&](DrawStructure& graph, const gui::Event& e) {
        graph.event(e);
    });
    
    base.loop();
    
    Print("Terminating");
    
    return 0;
}
