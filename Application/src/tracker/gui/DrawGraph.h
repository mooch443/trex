#ifndef _DRAWGRAPH_H
#define _DRAWGRAPH_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <gui/Graph.h>
#include <tracking/OutputLibrary.h>

namespace gui {
    class PropertiesGraph {
        class Graph : public gui::Graph {
        public:
            Graph(const Size2& dim, const std::string& name) : gui::Graph(Bounds(dim), name, Rangef(), Rangef())
            {
                
            }
            
            void before_draw() override;
        };
        
    protected:
        const track::Individual* _fish;
        long_t _frameNr;
        
        const track::Tracker& _tracker;
        const Vec2& _mouse_position;
        
        //! The graph that was displayed last
        GETTER_NCONST(Graph, graph);
        
    public:
        PropertiesGraph(const track::Tracker& tracker, const Vec2& mouse_position);
        void draw(DrawStructure& d);
        void setup_graph(long_t frameNr, const Rangel& range, const track::Individual* fish, Output::LibraryCache::Ptr cache);
        void reset() { _graph.set_dirty(); _fish = NULL; }
    };
}

#endif
