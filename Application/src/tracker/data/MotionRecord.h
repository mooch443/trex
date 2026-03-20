#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track {

using namespace cmn;
class Individual;

class PairDistance {
    GETTER_SETTER_PTR(const Individual*, fish0)
    GETTER_SETTER_PTR(const Individual*, fish1)
    GETTER_SETTER(float, d);
        
public:
    PairDistance(const Individual* fish0, const Individual* fish1, float d)
        : _fish0(fish0), _fish1(fish1), _d(d)
    {}
        
    bool operator==(const PairDistance& other) const {
        return ((other.fish0() == _fish0 || other.fish1() == _fish0)
            && other.fish1() != other.fish0()
            && (other.fish1() == _fish1 || other.fish0() == _fish1));
    }
    bool operator<(const PairDistance& other) const {
        return _d < other.d();
    }
    bool operator>(const PairDistance& other) const {
        return _d > other.d();
    }
    bool operator<=(const PairDistance& other) const {
        return _d <= other.d();
    }
    bool operator>=(const PairDistance& other) const {
        return _d >= other.d();
    }
};

struct FrameProperties {
    using Ptr = std::unique_ptr<FrameProperties>;
    
    template<typename... Args>
    static auto Make(Args... args) {
        return std::make_unique<FrameProperties>(std::forward<Args>(args)...);
    }
    
    double _time{-1};
    timestamp_t _org_timestamp{0};
    GETTER(Frame_t, frame);
    GETTER(long_t, active_individuals){-1};
    
public:
    FrameProperties(Frame_t frame, double t, timestamp_t ot)
        : _time(t), _org_timestamp(ot), _frame(frame)
    {}
    FrameProperties() noexcept = default;
    
    bool operator<(Frame_t frame) const {
        return this->_frame < frame;
    }
    
    void set_timestamp(uint64_t ts);
    void set_active_individuals(long_t);
    
    timestamp_t timestamp() const;
    double time() const;
    std::string toStr() const;
    static std::string class_name() { return "FrameProperties"; }
};

struct CacheHints {
    Frame_t current;
    std::vector<const FrameProperties *> _last_second;
        
    CacheHints(size_t size = 0);
    void push(Frame_t index, const FrameProperties* ptr);
    //void push_front(Frame_t index, const FrameProperties* ptr);
    void clear(size_t size = 0);
    size_t size() const;
    bool full() const;
    void remove_after(Frame_t);
    const FrameProperties* properties(Frame_t) const;
};

enum class Units {
    //PX_AND_FRAMES,
    PX_AND_SECONDS,
    CM_AND_SECONDS,
        
    DEFAULT = PX_AND_SECONDS
};
   
/**
 * This class describes the motion-related properties with respect
 * to a reference point (such as the head, centroid, or similar).
 * Each record is therefore part of a trajectory (or chain) of records
 * that describe how an individuals motion evolves over time.
 */
class MotionRecord {
public:
    static constexpr const size_t max_derivatives = 3;

protected:
    friend class DataFormat;
    GETTER(double, time);

    std::array<Vec2, MotionRecord::max_derivatives> _pos;
    std::array<Float2_t, MotionRecord::max_derivatives> _angle;
        
public:
    void init(const MotionRecord* previous, double time, const Vec2& pos, Float2_t angle);
        
    template<Units to> Float2_t speed(bool smooth) const { return v<to>(smooth).length(); }
    template<Units to> Float2_t speed() const { return v<to>().length(); }
        
    template<Units to> Float2_t acceleration(bool smooth) const { return length(a<to>(smooth)); }
    template<Units to> Float2_t acceleration() const { return length(a<to>()); }
        
    Float2_t angle(bool smooth) const { return value<Float2_t>(0, smooth); }
    Float2_t angle() const { return get<Float2_t>(0); }
        
    template<Units to> Float2_t angular_velocity(bool smooth) const { return value<to, Float2_t>(1, smooth); }
    template<Units to> Float2_t angular_velocity() const { return value<to, Float2_t>(1); }
        
    template<Units to> Float2_t angular_acceleration(bool smooth) const { return value<to, Float2_t>(2, smooth); }
    template<Units to> Float2_t angular_acceleration() const { return value<to, Float2_t>(2); }
        
    template<Units to> Vec2 pos(bool smooth) const { return value<to, Vec2>(0, smooth); }
    template<Units to> Vec2 pos() const { return value<to, Vec2>(0); }
        
    template<Units to> Vec2 v(bool smooth) const { return value<to, Vec2>(1, smooth); }
    template<Units to> Vec2 v() const { return value<to, Vec2>(1); }
        
    template<Units to> Vec2 a(bool smooth) const { return value<to, Vec2>(2, smooth); }
    template<Units to> Vec2 a() const { return value<to, Vec2>(2); }
        
    void flip(const MotionRecord* previous);
    static Float2_t cm_per_pixel();
        
private:
    template<typename T>
    const T& get(size_t derivative = 0) const {
        if constexpr (std::is_same_v<T, Vec2>)
            return _pos[derivative];
        else if constexpr (std::is_same_v<T, Float2_t>)
            return _angle[derivative];
        else
            static_assert(std::same_as<T, Float2_t>);
    }

    template<typename T>
    constexpr T& get(size_t derivative = 0) {
        if constexpr (std::same_as<T, Vec2>)
            return _pos[derivative];
        else if constexpr (std::is_same_v<T, Float2_t>)
            return _angle[derivative];
        else
            static_assert(std::same_as<T, Float2_t>);
    }

    template<typename T>
    T value(size_t derivative, bool smooth) const
    {
        if (smooth)
            return value<Units::DEFAULT, Units::DEFAULT, true, T>(derivative);
        else
            return value<Units::DEFAULT, Units::DEFAULT, false, T>(derivative);
    }

    template<Units from, Units to, typename T>
    T value(size_t derivative, bool smooth) const
    {
        if (smooth)
            return value<from, to, true, T>(derivative);
        else
            return value<from, to, false, T>(derivative);
    }

    template<Units to, typename T>
    T value(size_t derivative, bool smooth) const
    {
        return value<Units::DEFAULT, to, T>(derivative, smooth);
    }

    template<Units from, Units to, bool smooth, typename T>
    T value(size_t derivative = 0) const
    {
        return convert<from, to>(get<T>(derivative));
    }

    template<Units from, Units to, typename T>
    T value(size_t derivative = 0) const
    {
        return convert<from, to, false>(get<T>(derivative));
    }

    template<Units to, typename T>
    T value(size_t derivative = 0) const
    {
        return value<Units::DEFAULT, to, false, T>(derivative);
    }

    template<Units from, typename T>
    void value(const MotionRecord* previous, const T& val, size_t derivative = 0)
    {
        // save
        set<T>(derivative, convert<from, Units::DEFAULT>(val));

        // calculate the next higher derivative
        for (size_t i = derivative + 1; i < MotionRecord::max_derivatives; i++) {
            calculate_derivative<T>(previous, i);
        }
    }

    template<track::Units from, track::Units to, typename T>
    static T convert(const T& val)
    {
        if constexpr (from == Units::PX_AND_SECONDS) {
            if constexpr (to == Units::CM_AND_SECONDS) {
                return val * MotionRecord::cm_per_pixel();
            }

        }
        else {
            if constexpr (to == Units::PX_AND_SECONDS) {
                return val / MotionRecord::cm_per_pixel();
            }
        }

        return val;
    }

    size_t memory_size() const {
        return (sizeof(Vec2) + sizeof(float)) * max_derivatives;
    }

private:
    template<typename T>
    void calculate_derivative(const MotionRecord* prev, size_t index) {
        if (index >= MotionRecord::max_derivatives)
            return;

        assert(index > 0);

        if (!prev) {
            set<T>(index, T(0));
            return;
        }

        Float2_t tdelta = /*abs*/(time() - prev->time());
        const T& current_value = get<T>(index - 1);
        const T& prev_value = prev->get<T>(index - 1);

        assert(tdelta > 0);
        set<T>(index, (current_value - prev_value) / tdelta);
    }

protected:
    template<typename T>
    void set(size_t derivative, const T& value) {
        assert(derivative < max_derivatives);
        assert(!cmn::isnan(value));
        get<T>(derivative) = value;
    }
    
};

}

