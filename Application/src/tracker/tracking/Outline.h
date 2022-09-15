#ifndef _OUTLINE_H
#define _OUTLINE_H

#include <types.h>
#include <misc/GlobalSettings.h>
#include <file/DataFormat.h>
#include <gui/Transform.h>
#include <tracker/misc/default_config.h>
#include <misc/ranges.h>

namespace Output {
    class ResultsFormat;
}

namespace track {
    class Posture;
    class MinimalOutline;
    struct MovementInformation;
    
    struct DebugInfo {
        Frame_t frameIndex;
        uint32_t fdx;
        bool debug;
        //Vec2 previous_position;
    };

    struct MidlineSegment {
        float height;
        float l_length;
        Vec2 pos;
        //Vec2 pt_l;
        
        bool operator==(const MidlineSegment& other) const {
            return height == other.height && l_length == other.l_length && pos == other.pos;
        }
    };

    class Midline {
    public:
        typedef std::shared_ptr<Midline> Ptr;
        
    private:
        GETTER_NCONST(float, len)
        GETTER_NCONST(float, angle)
        GETTER_NCONST(Vec2, offset)
        GETTER_NCONST(Vec2, front)
        GETTER_NCONST(std::vector<MidlineSegment>, segments)
        GETTER_NCONST(long_t, head_index)
        GETTER_NCONST(long_t, tail_index)
        GETTER_NCONST(bool, inverted_because_previous)
        
        GETTER_NCONST(bool, is_normalized)
        
    public:
        bool empty() const { return _segments.empty(); }
        size_t size() const { return _segments.size(); }
        
        Midline();
        Midline(const Midline& other);
        ~Midline();
//#ifdef _DEBUG_MEMORY
        static size_t saved_midlines();
//#endif
        
        void post_process(const MovementInformation& movement, DebugInfo info);
        Ptr normalize(float fix_length = -1, bool debug = false) const;
        static void fix_length(float len, std::vector<MidlineSegment>& segments, bool debug = false);
        size_t memory_size() const;
        
        std::array<Vec2, 2> both_directions() const;
        float original_angle() const;
        
        /**
         * if to_real_world is true, the returned transform will transform points
         * to the same coordinate system as outline points / the video coordinate system.
         * if its set to false, the returned transform is meant to transform coordinates
         * from the global system to the midline coordinate system (in order to e.g.
         * normalize an image)
         **/
        gui::Transform transform(const default_config::individual_image_normalization_t::Class &type, bool to_real_world = false) const;
        
    private:
        friend class Outline;
        static float calculate_angle(const std::vector<MidlineSegment>& segments);
    };
    
    struct MovementInformation {
        Vec2 position, direction, velocity;
        std::vector<Vec2> directions;
    };

    class Outline : public Minimizable {
        friend class Individual;
        friend class DebugDrawing;
        
        //! Structure used to save the area under the curvature curve
        //  (as used in offset_to_middle)
        struct Area {
            long_t start, end;
            long_t extremum;
            float extremum_height;
            float area;
            
            Area() : start(-1), end(-1), extremum(-1), extremum_height(0), area(0) {}
            
            void clear() {
                start = end = -1;
                extremum = -1;
                area = extremum_height = 0;
            }
        };
        
    public:
        Frame_t frameIndex;
        static float average_curvature();
        static float max_curvature();
        static uint8_t get_outline_approximate();
        
    protected:
        /**
         * Persistent memory
         * (cannot be reduced without losing information)
         */
        std::shared_ptr<std::vector<Vec2>> _points;
        
        //! confidence in the results
        GETTER_NCONST(float, confidence)
        
        //! the uncorrected angle of the posture detection
        GETTER(float, original_angle)
        GETTER(bool, inverted_because_previous)
        
        //GETTER(long_t, tail_index)
        //GETTER(long_t, head_index)
        
        //! When set to true, this Outline cannot be changed anymore.
        GETTER(bool, concluded)
        
        int curvature_range;
        
        /**
         * Temporary memory
         */
        std::vector<float> _curvature;
        //GETTER(bool, needs_invert)
        
    public:
        Outline(std::shared_ptr<std::vector<Vec2>> points, Frame_t f = {});
        ~Outline();
        
        void clear();
        
        inline const Vec2& at(size_t index) const { return operator[](index); }
        Vec2& operator[](size_t index);
        const Vec2& operator[](size_t index) const;
        
        void push_back(const Vec2& pt); // inserts at the back
        void push_front(const Vec2& pt); // inserts at the front
        
        template<typename Iterator>
        void insert(size_t index, const Iterator& begin, const Iterator& end)
        {
            _points->insert(_points->begin() + index, begin, end);
            if(!_curvature.empty())
                throw U_EXCEPTION("Cannot insert points after calculating curvature.");
        }
        
        void insert(size_t index, const Vec2& pt);
        void remove(size_t index);
        
        void finish();
        
        const Vec2& back() const { return _points->back(); }
        const Vec2& front() const { return _points->front(); }
        
        //float slope(size_t index) const;// { assert(index < _slope.size()); return _slope[index]; }
        //float curvature(size_t index) const { assert(index < _curvature.size()); return _curvature[index]; }
        
        void resample(const float distance = 1.0f);
        
        size_t size() const { return _points->size(); }
        bool empty() const { return _points->empty(); }
        
        std::vector<Vec2>& points() { return *_points; }
        const std::vector<Vec2>& points() const { return *_points; }
        
        float angle() const;
        
        //! Rotates the midline so that the angle between the first and 0.2*size
        //  point is zero (horizontal). That's the rigid part (hopefully).
        //static const Midline* normalized_midline();
        void calculate_midline(Midline &midline, const DebugInfo& info);
        
        virtual void minimize_memory() override;
        
        //static float calculate_slope(const std::vector<Vec2>&, size_t index);
        static float calculate_curvature(const int curvature_range, const std::vector<Vec2>&, size_t index, float scale = 1);
        static void smooth_array(const std::vector<float>& input, std::vector<float>& output, float * max_curvature = NULL);
        
        size_t memory_size() const;
        
        static int calculate_curvature_range(size_t number_points);
        void replace_points(decltype(_points) ptr) { _points = ptr; }
        static float get_curvature_range_ratio();
        
    protected:
        void smooth();
        
        //void calculate_slope(size_t index);
        void calculate_curvature(size_t index);
        std::tuple<long_t, long_t> offset_to_middle(const DebugInfo& info);
        
        //! Smooth the curvature array.
        std::vector<float> smoothed_curvature_array(float& max_curvature) const;
        
        //! Tries to find the tail by looking at the outline/curvature.
        long_t find_tail(const DebugInfo& info);
        
        //! Ensures the globals are loaded
        static void check_constants();
        friend Midline::Midline();
    };
    
    class MinimalOutline {
    private:
        static constexpr int factor = 10;
        
    protected:
        GETTER(Vec2, first)
        std::vector<uint16_t> _points;
        //GETTER_NCONST(long_t, tail_index)
        //GETTER_NCONST(long_t, head_index)
        
        friend class Output::ResultsFormat;
        friend class cmn::Data;
        
    public:
        typedef std::shared_ptr<MinimalOutline> Ptr;
        
        MinimalOutline();
        MinimalOutline(const Outline& outline);
        ~MinimalOutline();
        inline size_t memory_size() const { return sizeof(MinimalOutline) + sizeof(decltype(_points)::value_type) * _points.size() + sizeof(decltype(_first)); }
        std::vector<Vec2> uncompress() const;
        size_t size() const { return _points.size(); }
        void convert_from(const std::vector<Vec2>& array);
    };
}

#endif
