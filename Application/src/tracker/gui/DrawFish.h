#ifndef _DRAW_FISH_H
#define _DRAW_FISH_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <gui/Timeline.h>
#include <misc/EventAnalysis.h>
#include <gui/Graph.h>
#include <misc/OutputLibrary.h>

namespace pv {
struct CompressedBlob;
}

namespace gui {
    class Fish : public DrawableCollection {
        track::Individual& _obj;
        const track::PPFrame* _frame;
        GETTER(long_t, idx)
        long_t _safe_idx;
        double _time;
        std::unique_ptr<ExternalImage> _image;
        Midline::Ptr _cached_midline;
        MinimalOutline::Ptr _cached_outline;
        GETTER(Vec2, fish_pos)
        Circle _circle;
        //Image _image;
        //Image *_probabilities;
        const EventAnalysis::EventMap* _events;
        
        Vec2 _position;
        float _plus_angle;
        ColorWheel _wheel;
        Color _color;
        Vec2 _v;
        std::shared_ptr<std::vector<Vec2>> _polygon_points;
        std::shared_ptr<Polygon> _polygon;
        
        Rangel _prev_frame_range;
        
        struct FrameVertex {
            long_t frame;
            Vertex vertex;
            float speed_percentage;
        };
        
        std::deque<FrameVertex> frame_vertices;
        std::vector<Vertex> vertices;
        std::shared_ptr<Circle> _recognition_circle;
        std::vector<Vec2> points;
        
        pv::CompressedBlob *_blob;
        Bounds _blob_bounds;
        int _match_mode;
        IndividualCache _next_frame_cache;
        std::shared_ptr<Individual::BasicStuff> _basic_stuff;
        std::shared_ptr<Individual::PostureStuff> _posture_stuff;
        int _avg_cat = -1;
        Output::Library::LibInfo _info;
        double _library_y = Graph::invalid();
        //ExternalImage _colored;
        
        Graph _graph;
        Entangled _posture;
        
    public:
        Fish(track::Individual& obj);
        void update(DrawStructure& d) override;
        //void draw_occlusion(DrawStructure& window);
        void set_data(long_t frameIndex, double time, const track::PPFrame& frame, const EventAnalysis::EventMap* events);
        
    private:
        //void paint(cv::Mat &target, int max_frames = 1000) const;
        void paintPath(DrawStructure& window, const Vec2& offset, long_t to = -1, long_t from = -1, const Color& = Transparent);
        //void paintPixels() const;
        void update_recognition_circle(DrawStructure&);
    public:
        void label(DrawStructure&);
        void shadow(DrawStructure&);
    };
}

#endif
