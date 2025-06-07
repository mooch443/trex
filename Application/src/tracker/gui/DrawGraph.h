#ifndef _DRAWGRAPH_H
#define _DRAWGRAPH_H

#include <commons.pc.h>
#include <gui/Graph.h>
#include <misc/idx_t.h>
#include <tracking/OutputLibraryTypes.h>

namespace track {
class Tracker;
class Individual;
}

namespace cmn::gui {
    class DrawStructure;
    class Button;

    class PropertiesGraph : public Entangled {
        class Graph : public gui::Graph {
        public:
            Graph(const Size2& dim, const std::string& name) : gui::Graph(Bounds(dim), name, Rangef(), Rangef())
            { }
            
            void before_draw() override;
            void update() override {}
            void set_parent(SectionInterface*) override;
            void set_bounds_changed() override;
            void update_sample_cache_automatically() override;
        };
        
    protected:
        track::Idx_t _fdx;
        Frame_t _frameNr;
        
        //! The graph that was displayed last
        GETTER_NCONST(Graph, graph);
        derived_ptr<Button> _close;
        
    public:
        PropertiesGraph();
        void update() override;
        void draw(DrawStructure& d);
        void setup_graph(const Output::cached_output_fields_t&, Frame_t frameNr, const Range<Frame_t>& range, const track::Individual* fish, std::shared_ptr<Output::LibraryCache> cache);
        void reset();
    };
}

#endif
