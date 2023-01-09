#include "DrawGraph.h"

using namespace track;

namespace gui {
    void PropertiesGraph::Graph::before_draw() {
        if(content_changed()) {
            LockGuard guard(ro_t{}, "PropertiesGraph::Graph::before_draw()",100);
            if(guard.locked()) {
                gui::Graph::update();
                _content_changed = false;
            }
        }
    }
    
    PropertiesGraph::PropertiesGraph(const Tracker& tracker, const Vec2& mouse_position)
    : _tracker(tracker), _mouse_position(mouse_position), _graph(Size2(1024, 900), "Individual")
    {
        _graph.set_background(Black.alpha(150), Transparent);
        _graph.set_draggable();
        _graph.set_pos(SETTING(video_size).value<Size2>() - _graph.global_bounds().size());
        //_graph.set_scale(0.8);
    }
    
    void PropertiesGraph::draw(gui::DrawStructure &base) {
        Vec2 mpos(_mouse_position.x / float(_tracker.average().cols),
                  _mouse_position.y / float(_tracker.average().rows));
        Vec2 gpos(_graph.pos().x / float(_tracker.average().cols),
                  _graph.pos().y / float(_tracker.average().rows));
        
        //float d = length(mpos);
        //_graph.set_scale(max(0.2f, min(1.f, d*d)));
        //_pgraph.set_scale(1);
        //_graph.set_pos(Vec2(_tracker.average().cols - _graph.width() * _graph.scale(),
        //                    _tracker.average().rows - _graph.height() * _graph.scale()));
        
        
        /*if(!_graph.) {
            Section::reuse_objects();
            return;
        }*/
        
        base.wrap_object(_graph);
        //_graph.display(base, _pgraph.pos, _pgraph.scale, _pgraph.scale);
        
        //_pgraph.changed = false;
    }
    
    void PropertiesGraph::setup_graph(long_t frameNr, const Rangel& range, const Individual* fish, Output::LibraryCache::Ptr cache) {
        if(_fish != fish || frameNr != _frameNr) {
            if(!range.empty()) {
                _graph.set_ranges(Rangef(max(range.start, fish->start_frame().get()),
                                         min(range.end, fish->end_frame().get())),
                                  Rangef(RADIANS(-180), RADIANS(180)));
            } else {
                _graph.set_ranges(Rangef(fish->start_frame().get(), fish->end_frame().get()),
                                  Rangef(RADIANS(-180), RADIANS(180)));
            }
            
            _graph.set_title("Individual "+std::to_string(fish->identity().ID()));
            _graph.set_zero(frameNr);
            _graph.clear();
            _graph.set_scroll_enabled(true);
            _graph.set_scroll_limits(Rangef(0, 0), Rangef(0, 0));
            
            _fish = fish;
            _frameNr = frameNr;
            
            using namespace Output;
            Library::init_graph(_graph, fish, cache);
        }
    }
}
