#include "DrawGraph.h"
#include <tracking/Tracker.h>
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <tracking/OutputLibrary.h>

using namespace track;

namespace cmn::gui {
    void PropertiesGraph::Graph::before_draw() {
        /*if(content_changed()) {
            LockGuard guard(ro_t{}, "PropertiesGraph::Graph::before_draw()",100);
            if(guard.locked()) {
                gui::Graph::update();
                _content_changed = false;
            }
        }
        //gui::Graph::update();*/
        gui::Graph::before_draw();
    }
    
    PropertiesGraph::PropertiesGraph()
    : _graph(Size2(800, 500), "Individual")
    {
        _graph.set_background(Transparent, Transparent);
        set_background(Black.alpha(150), White.alpha(200));
        set_draggable();
        set_clickable(true);
        set_bounds(Bounds(Vec2(100), _graph.size()));
        _graph.set_pos(Vec2());
        //_graph.set_scale(0.8);
        add_event_handler(EventType::DRAG, [this](Event){
            _graph.set_content_changed(true);
        });
        on_hover([this](Event e){
            if(e.hover.hovered)
                set_background(Black.alpha(180), White.alpha(200));
            else
                set_background(Black.alpha(150), White.alpha(200));
        });
    }
    
    void PropertiesGraph::draw(gui::DrawStructure &base) {
        //base.wrap_object(_graph);
        base.wrap_object(*this);
    }

void PropertiesGraph::update() {
    OpenContext([this](){
        advance_wrap(_graph);
    });
}
    
    void PropertiesGraph::setup_graph(Frame_t frameNr, const Range<Frame_t>& range, const Individual* fish, Output::LibraryCache::Ptr cache) {
        if(_fdx == fish->identity().ID()
           && frameNr == _frameNr)
        {
            return;
        }
        
        Print("Updating ", fish->identity().ID());
        
        if(not fish->empty()) {
            if(!range.empty()) {
                _graph.set_ranges(Rangef{
                    (float)max(range.start, fish->start_frame()).get(),
                    (float)min(range.end, fish->end_frame()).get()
                },
                Rangef{
                    RADIANS(-180),
                    RADIANS(180)
                });
            } else {
                _graph.set_ranges(Rangef{
                    (float)fish->start_frame().get(),
                    (float)fish->end_frame().get()
                },
                Rangef{
                    RADIANS(-180),
                    RADIANS(180)
                });
            }
        }
        
        _graph.set_title("Individual "+fish->identity().ID().toStr());
        _graph.set_zero(frameNr.valid() ? frameNr.get() : 0);
        _graph.clear();
        
        _fdx = fish->identity().ID();
        _frameNr = frameNr;
        
        using namespace Output;
        Library::init_graph(_graph, fish, cache);
        //_graph.gui::Graph::update();
    }

    void PropertiesGraph::reset() {
        _graph.set_dirty();
        _fdx = {};
    }
}
