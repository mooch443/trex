#ifndef PHYSICAL_PROPERTIES_H
#define PHYSICAL_PROPERTIES_H

#include <types.h>
#include <misc/Blob.h>
#include <misc/GlobalSettings.h>

#define SMOOTH_RECURSIVELY false

namespace track {
    class Individual;

    class PairDistance {
        GETTER_SETTER_PTR(const Individual*, fish0)
        GETTER_SETTER_PTR(const Individual*, fish1)
        GETTER_SETTER(float, d)
        
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
        double time;
        uint64_t org_timestamp;
        long_t frame;
        long_t active_individuals;
        std::vector<PairDistance> _pair_distances;
        
        FrameProperties(long_t frame, double t, uint64_t ot, std::vector<PairDistance> pair_distances = {})
            : time(t), org_timestamp(ot), frame(frame), active_individuals(-1), _pair_distances(pair_distances)
        {}
        
        FrameProperties()
            : time(-1), org_timestamp(0), frame(-1), active_individuals(-1)
        {}
        
        bool operator<(long_t frame) const {
            return this->frame < frame;
        }
    };

    struct CacheHints {
        long_t current;
        std::vector<const FrameProperties *> _last_second;
        
        CacheHints(size_t size = 0);
        void push(long_t index, const FrameProperties* ptr);
        void push_front(long_t index, const FrameProperties* ptr);
        void clear(size_t size = 0);
        size_t size() const;
        bool full() const;
        const FrameProperties* properties(long_t) const;
    };

    enum class Units {
        //PX_AND_FRAMES,
        PX_AND_SECONDS,
        CM_AND_SECONDS,
        
        DEFAULT = CM_AND_SECONDS
    };
    
    enum class PropertyType { POSITION, ANGLE };
    
    /**
     * This class describes the physical properties of a moving entity
     * for a specific point in time.
     * TODO: Make linked list where it knows its predecessor so that
     * more complicated properties can be calculated and saved
     */
    class PhysicalProperties {
    public:
        static constexpr const size_t max_derivatives = 3;
        typedef PropertyType Type;
        
        //! Basic properties
        template<typename T>
        class Property;
        
        class PropertyBase {
        protected:
            PhysicalProperties* _mother;
            GETTER(Type, type)
            
        public:
            PropertyBase(PhysicalProperties* mother, const Type& type) : _mother(mother), _type(type) {}
            virtual ~PropertyBase() {}
            
            template<typename T>
            T value(const Units& units = Units::DEFAULT, size_t derivative = 0, bool smooth = false) const;
            
            template<typename T>
            void value(const T& val, const Units& input_units = Units::DEFAULT, size_t derivative = 0, const track::CacheHints* hints = nullptr);
            
            template<typename T>
            Property<T>* is_type() {
                return dynamic_cast<Property<T>*>(this);
            }
            
            template<typename T>
            const Property<T>* is_type() const {
                return dynamic_cast<const Property<T>*>(this);
            }
            
            template<typename T>
            static T convert(const T& val, const Units& from, const Units& to);
            
            static float cm_per_pixel();
            virtual size_t memory_size() const {
                return sizeof(PropertyBase);
            }
            
        private:
            template<typename T>
            void calculate_derivative(Property<T> &property, size_t index, const CacheHints* hints);
        };
        
        template<typename T>
        class Property : public PropertyBase {
        private:
            std::array<T, PhysicalProperties::max_derivatives> _values;
            //std::array<T, PhysicalProperties::max_derivatives> _smooth_values;
            
        public:
            constexpr Property(PhysicalProperties* mother, const Type& type)
                : PropertyBase(mother, type)
            { }
            
            virtual ~Property() { }
            
            constexpr const T& value(size_t derivative) const {
                return _values[derivative];
            }
            
            T smooth_value(size_t derivative) const {
                return update_smooth(derivative);
            }
            
            size_t memory_size() const override {
                return sizeof(Property<T>);
            }
            
        protected:
            friend class PropertyBase;
            friend class PhysicalProperties;
            
            void set_value(size_t derivative, const T& value);
            T update_smooth(size_t derivative) const;
            
            static T _update_smooth(const PhysicalProperties::Property<T> *prop, size_t derivative);
        };
        
    private:
        friend class DataFormat;
        
        //GETTER_PTR(PhysicalProperties*, prev)
        //GETTER_PTR(PhysicalProperties*, next)
        GETTER_PTR(Individual*, fish)
        GETTER(long_t, frame)
        //GETTER(double, time)
        
        // contains either float or Point2f pointers
        std::array<PropertyBase*, 2> _derivatives;
        //std::map<Type, PropertyBase*> _derivatives;
        
    public:
        PhysicalProperties(Individual* fish, PhysicalProperties* prev, long_t frame, const Vec2& pos, float angle, const CacheHints* hints = nullptr);
        ~PhysicalProperties();
        
        const decltype(_derivatives)& derivatives() const { return _derivatives; }
        static size_t saved_midlines();
        
        //! Gets a properties value with given derivative depth,
        //  type is either float or Vec2
        inline const PropertyBase& get(const Type& name) const {
            return *_derivatives[(size_t)name];
        }
        inline PropertyBase& get(const Type& name) {
            return *_derivatives[(size_t)name];
        }
        
        //void set_next(PhysicalProperties* ptr);
        
        float speed(const Units& units = Units::DEFAULT, bool smooth = false) const { return length(v(units, smooth)); }
        float acceleration(const Units& units = Units::DEFAULT, bool smooth = false) const { return length(a(units, smooth)); }
        float angle(bool smooth = false) const { return get(PropertyType::ANGLE).value<float>(Units::DEFAULT, 0, smooth); };
        float angular_velocity(const Units& units = Units::DEFAULT, bool smooth = false) const { return get(PropertyType::ANGLE).value<float>(units, 1, smooth); };
        float angular_acceleration(const Units& units = Units::DEFAULT, bool smooth = false) const { return get(PropertyType::ANGLE).value<float>(units, 2, smooth); };
        
        Vec2 pos(const Units& units = Units::DEFAULT, bool smooth = false) const { return get(PropertyType::POSITION).value<Vec2>(units, 0, smooth); }
        Vec2 v(const Units& units = Units::DEFAULT, bool smooth = false) const { return get(PropertyType::POSITION).value<Vec2>(units, 1, smooth); }
        Vec2 a(const Units& units = Units::DEFAULT, bool smooth = false) const { return get(PropertyType::POSITION).value<Vec2>(units, 2, smooth); }
        
        void flip();
        size_t memory_size() const;
        static uint32_t smooth_window();
        
    private:
        void update_derivatives();
    };
    
    template<typename T>
    T PhysicalProperties::PropertyBase::value(const Units& units, size_t derivative, bool smooth) const
    {
        assert(dynamic_cast<const Property<T>*>(this));
        auto ptr = static_cast<const Property<T>*>(this);
        //if(!ptr)
        //    U_EXCEPTION("Wrong data type for property.");
        return convert(smooth ? ptr->smooth_value(derivative) : ptr->value(derivative), Units::DEFAULT, units);
    }
    
    template<typename T>
    void PhysicalProperties::PropertyBase::value(const T& val, const Units& input_units, size_t derivative, const CacheHints* hints)
    {
        assert(dynamic_cast<Property<T>*>(this));
        auto ptr = static_cast<Property<T>*>(this);
        //if(!ptr)
        //    U_EXCEPTION("Wrong data type for property.");
        
        // save
        ptr->set_value(derivative, convert(val, input_units, Units::DEFAULT));
        
        // calculate the next higher derivative
        for(size_t i=derivative+1; i<PhysicalProperties::max_derivatives; i++) {
            calculate_derivative(*ptr, i, hints);
        }
    }
    
    template<typename T>
    T PhysicalProperties::PropertyBase::convert(const T &val, const track::Units &from, const track::Units &to)
    {
        switch (from) {
            case Units::PX_AND_SECONDS: {
                if (to == Units::CM_AND_SECONDS) {
                    return val * cm_per_pixel();
                }
                break;
            }
                
            case Units::CM_AND_SECONDS: {
                if (to == Units::PX_AND_SECONDS) {
                    return val / cm_per_pixel();
                }
                break;
            }
                
            default:
                U_EXCEPTION("unknown conversion.");
                break;
        }
        
        return val;
    }


template<> Vec2 PhysicalProperties::Property<Vec2>::update_smooth(size_t derivative) const;
template<> float PhysicalProperties::Property<float>::update_smooth(size_t derivative) const;
}

#endif
