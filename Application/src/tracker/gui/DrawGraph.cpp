#include "DrawGraph.h"
#include <tracking/Tracker.h>
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <tracking/OutputLibrary.h>
#include <gui/types/Button.h>

using namespace track;

namespace cmn::gui {
    void PropertiesGraph::Graph::before_draw() {
        /*if(content_changed()) {
            LockGuard guard(ro_t{}, "PropertiesGraph::Graph::before_draw()",100);
            if(guard.locked()) {
                gui::Graph::before_draw();
                _content_changed = false;
            }
        }*/
        //gui::Graph::update();
        Entangled::before_draw();
    }

void PropertiesGraph::Graph::set_bounds_changed() {
    set_content_changed(true);
    gui::Graph::set_bounds_changed();
}

void PropertiesGraph::Graph::set_parent(SectionInterface*parent)
{
    if(parent != this->parent()) {
        set_content_changed(true);
        gui::Graph::set_parent(parent);
        
        LockGuard guard(ro_t{}, "PropertiesGraph::Graph::before_draw()");
        if(guard.locked()) {
            set_content_changed(true);
            gui::Graph::update();
            _content_changed = false;
        }
    }
}

    PropertiesGraph::PropertiesGraph()
        : _graph(Size2(880, 550), ""),
        _close(new Button{
            attr::Size{30,30},
            Str{"<sym>✕</sym>"},
            FillClr{100,50,50,255},
            TextClr{White}, Font{0.55}, Margins{-5,0,0,0}, Origin{1,0}
        })
    {
        _graph.reset_bg();
        _graph.set(Loc{10,10});
        _graph.set(Origin{0, 0});
        _graph.set(Graph::DisplayLabels::Outside);
        set(FillClr{Black.alpha(150)});
        set(LineClr{White.alpha(200)});
        set_draggable();
        set_clickable(true);
        set(Origin{0.5, 0.5});
        set_bounds(Bounds(Vec2(200 + _graph.size().width * 0.5,150 + _graph.size().height * 0.5), _graph.size() + Size2(20, 20)));
        
        //_graph.set_scale(0.8);
        add_event_handler(EventType::DRAG, [this](Event){
            _graph.set_content_changed(true);
        });
        on_hover([this](Event e){
            if(e.hover.hovered) {
                set(FillClr{Black.alpha(180)});
            } else {
                set(FillClr{Black.alpha(150)});
            }
        });
        _close->on_click([](Event){
            SETTING(gui_show_graph) = false;
        });
    }
    
    void PropertiesGraph::draw(gui::DrawStructure &base) {
        //base.wrap_object(_graph);
        base.wrap_object(*this);
    }

void PropertiesGraph::update() {
    OpenContext([this](){
        advance_wrap(_graph);
        advance_wrap(*_close);
    });
    
    _close->set(Loc{width() - 10, 10});
}
    
    void PropertiesGraph::setup_graph(const Output::cached_output_fields_t& output_fields, Frame_t frameNr, const Range<Frame_t>& range, const Individual* fish, Output::LibraryCache::Ptr cache) {
        if(_fdx == fish->identity().ID()
           && frameNr == _frameNr)
        {
            return;
        }
        
        //Print("Updating ", fish->identity().ID());
        
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
        
        _graph.set_title(fish->identity().name());
        _graph.set_zero(frameNr.valid() ? frameNr.get() : 0);
        _graph.clear();
        
        _fdx = fish->identity().ID();
        _frameNr = frameNr;
        
        using namespace Output;
        Library::init_graph(output_fields, _graph, fish, cache);
        //_graph.gui::Graph::update();
        
        _graph.gui::Graph::update();
        _graph.set_content_changed(false);
        set_content_changed(false);
    }

    void PropertiesGraph::reset() {
        _graph.set_dirty();
        _fdx = {};
    }
}
