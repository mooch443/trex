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
#include <gui/Coordinates.h>

namespace pv {
struct CompressedBlob;
}

namespace gui {
    class Label;
    class Skelett;

    class Fish {
        Entangled _view;
        Label* _label { nullptr };

        GETTER(Frame_t, frame)
        Frame_t _safe_frame;
        double _time;
        ExternalImage _image;
        int32_t _probability_radius = 0;
        Vec2 _probability_center;
        Midline::Ptr _cached_midline;
        MinimalOutline::Ptr _cached_outline;
        GETTER(Vec2, fish_pos)
        Circle _circle;

        std::vector<Vertex> _vertices;
        std::vector<std::unique_ptr<Vertices>> _paths;
        const EventAnalysis::EventMap* _events;
        
        Vec2 _position;
        float _plus_angle;
        ColorWheel _wheel;
        Color _color;
        Vec2 _v;
        std::shared_ptr<std::vector<Vec2>> _polygon_points;
        std::shared_ptr<Polygon> _polygon;
        
        Range<Frame_t> _prev_frame_range;
        
        struct FrameVertex {
            Frame_t frame;
            Vertex vertex;
            float speed_percentage;
        };
        
        float _color_start, _color_end;
        std::deque<FrameVertex> frame_vertices;
        std::shared_ptr<Circle> _recognition_circle;
        std::vector<Vec2> points;
        
        blob::Pose _average_pose;
        Bounds _blob_bounds;
        int _match_mode;
        IndividualCache _next_frame_cache;
        
        Identity _id;
        std::optional<BasicStuff> _basic_stuff;
        std::optional<PostureStuff> _posture_stuff;
        double _ML{0}; // midline length
        Midline::Ptr _pp_midline;
        bool _empty{true};
        Range<Frame_t> _range;
        
        std::tuple<bool, FrameRange> _has_processed_segment;
        decltype(Individual::average_recognition_segment)::mapped_type processed_segment;
        std::shared_ptr<SegmentInformation> _segment;
        Individual::IDaverage _qr_code;
        std::vector<float> _pred;
        
        Color _previous_color;
        Output::Library::LibInfo _info;
        double _library_y = Graph::invalid();
        std::string circle_animator{ "recognition-circle-"+Meta::toStr((uint64_t)this) };
        bool _path_dirty{false};
        //ExternalImage _colored;
        
        /// Categorization information
        int _avg_cat = -1;
        int _cat = -1;
        std::string _cat_name;
        std::string _avg_cat_name;
        
        Graph _graph;
        Entangled _posture, _label_parent;
        std::unique_ptr<Skelett> _skelett;
        
    public:
        Fish(track::Individual& obj);
        ~Fish();
        void update(const FindCoord&, Entangled& p, DrawStructure& d);
        //void draw_occlusion(DrawStructure& window);
        void set_data(Individual& obj, Frame_t frameIndex, double time, const EventAnalysis::EventMap* events);
        
    private:
        //void paint(cv::Mat &target, int max_frames = 1000) const;
        void paintPath(const Vec2& offset);
        void updatePath(Individual&, Frame_t to = {}, Frame_t from = {});
        //void paintPixels() const;
        void update_recognition_circle();
        Color get_color(const BasicStuff*) const;
    public:
        void label(const FindCoord&, Entangled&);
        Drawable* shadow();
        void check_tags();
    };
}

#endif
