#ifndef _DRAW_FISH_H
#define _DRAW_FISH_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <tracking/EventAnalysis.h>
#include <tracking/OutputLibrary.h>
#include <misc/Coordinates.h>
#include <tracking/Outline.h>
#include <gui/Graph.h>
#include <tracking/IndividualCache.h>
#include <tracking/TrackletInformation.h>
#include <misc/Identity.h>
#include <misc/idx_t.h>
#include <misc/Timer.h>
#include <misc/TrackingSettings.h>

namespace pv {
struct CompressedBlob;
}

namespace track {
class Individual;
}

namespace cmn::gui {
    class Label;
    class Skelett;

struct UpdateSettings {
    bool gui_show_outline;
    bool gui_show_midline;
    bool gui_happy_mode;
    bool gui_show_probabilities;
    bool gui_show_shadows;
    bool gui_show_texts;
    bool gui_show_selections;
    bool gui_show_boundary_crossings;
    bool gui_show_paths;
    bool gui_highlight_categories;
    bool gui_show_match_modes;
    bool gui_show_cliques;
    std::string gui_fish_label;
    int panic_button;
    uint8_t gui_outline_thickness;
    std::string gui_fish_color;
    Color gui_single_identity_color;
    Frame_t gui_pose_smoothing;
    float gui_max_path_time;
};

    class Fish {
        Entangled _view;
        Label* _label { nullptr };

        GETTER(Frame_t, frame);
        Frame_t _safe_frame;
        double _time;
        ExternalImage _image;
        int32_t _probability_radius = 0;
        Vec2 _probability_center;
        track::Midline::Ptr _cached_midline;
        const track::Midline* _pp_midline{nullptr};
        const track::MinimalOutline* _cached_outline;
        GETTER(Vec2, fish_pos);
        
        std::variant<std::monostate, Rect, Polygon, Circle> _selection;
        Polygon _tight_selection;
        double _radius{0};
        Timer _frame_change;
        std::vector<Vec2> _current_points, _current_corners, _cached_points, _cached_circle;
        //std::unique_ptr<Line> _lines;

        std::vector<Vertex> _vertices;
        std::vector<std::unique_ptr<Vertices>> _paths;
        const track::EventAnalysis::EventMap* _events;
        
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
            Float2_t speed_percentage;
        };
        
        float _color_start, _color_end;
        std::deque<FrameVertex> frame_vertices;
        std::shared_ptr<Circle> _recognition_circle;
        std::vector<Vec2> points;
        
        UpdateSettings _options;
        
        //blob::Pose _average_pose;
        Bounds _blob_bounds;
        std::optional<default_config::matching_mode_t::Class> _match_mode;
        std::optional<track::IndividualCache> _next_frame_cache;
        
        track::Identity _id;
        std::optional<track::BasicStuff> _basic_stuff;
        std::optional<track::PostureStuff> _posture_stuff;
        Float2_t _ML{0}; // midline length
        std::map<Frame_t, Float2_t> _previous_midline_angles, _previous_midline_angles_d, _previous_midline_angles_dd;
        bool _empty{true};
        Vec2 posture_direction_, midline_direction;
        std::vector<Vec2> _posture_directions;
        Range<Frame_t> _range;
        
        std::tuple<bool, FrameRange> _has_processed_tracklet;
        std::tuple<size_t, std::map<track::Idx_t, float>> processed_tracklet;
        std::shared_ptr<track::TrackletInformation> _tracklet;
        track::IDaverage _qr_code;
        std::vector<float> _pred;
        
        Color _previous_color;
        Output::Library::LibInfo _info;
        double _library_y = GlobalSettings::invalid();
        std::string circle_animator{ "recognition-circle-"+Meta::toStr((uint64_t)this) };
        bool _path_dirty{false};
        //ExternalImage _colored;
        Float2_t last_scale{0};
        
        /// Categorization information
        track::MaybeLabel _avg_cat{};
        track::MaybeLabel _cat{};
        
        std::string _cat_name;
        std::string _avg_cat_name;
        
        std::string _recognition_str;
        FrameRange _recognition_tracklet;
        
        struct Data;
        std::unique_ptr<Data> _data;
        std::optional<std::map<track::Idx_t, float>> _raw_preds;
        
        Entangled _posture, _label_parent;
        std::unique_ptr<Skelett> _skelett;
        
    public:
        Fish(track::Individual& obj);
        ~Fish();
        void update(const FindCoord&, Entangled& p, DrawStructure& d);
        //void draw_occlusion(DrawStructure& window);
        void set_data(const UpdateSettings& settings, track::Individual& obj, Frame_t frameIndex, double time, const track::EventAnalysis::EventMap* events);
        
    private:
        //void paint(cv::Mat &target, int max_frames = 1000) const;
        void paintPath(const Vec2& offset);
        void updatePath(track::Individual&, Frame_t to = {}, Frame_t from = {});
        //void paintPixels() const;
        void update_recognition_circle();
        Color get_color(const track::BasicStuff*) const;
        bool setup_rotated_bbx(const FindCoord&, const Vec2& offset, const Vec2& c_pos, Float2_t angle);
        void selection_hovered(Event);
        void selection_clicked(Event);
    public:
        void label(const FindCoord&, Entangled&);
        Drawable* shadow();
        void check_tags();
    };
}

#endif
