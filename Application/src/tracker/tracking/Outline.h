#ifndef _OUTLINE_H
#define _OUTLINE_H

#include <commons.pc.h>
//#include <file/DataFormat.h>
#include <gui/Transform.h>
#include <tracker/misc/default_config.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>

namespace Output {
    class ResultsFormat;
}

namespace cmn {
class Data;

/**
 * @brief A lightweight, simplified vector-like container for small dynamic arrays of simple types.
 *
 * The SmallVector class is designed for efficient storage and management of small arrays, with a
 * focus on minimizing memory overhead. It uses inline storage for small arrays and dynamically
 * allocates memory only when necessary, which provides both performance benefits and reduced
 * memory usage.
 *
 * This particular implementation is optimized for use with simple types, such as `uint16_t`,
 * and assumes that the elements do not require construction or destruction, making it suitable
 * for integral and floating-point types.
 *
 * Key Features:
 * - **Inline Storage**: Stores a small number of elements directly within the object itself
 *   (in this case, up to `InlineCapacity` elements), avoiding dynamic memory allocation for
 *   small vectors.
 * - **Dynamic Storage**: Automatically transitions to dynamic memory allocation when the
 *   vector grows beyond its inline storage capacity.
 * - **Memory Efficient**: The total size of the SmallVector object is kept small, making it
 *   ideal for use cases where memory footprint is a critical concern.
 *
 * Memory Layout (with `T = uint16_t` and `InlineCapacity = 4`):
 * - **size_**: 4 bytes (std::uint32_t) to track the number of elements currently stored.
 * - **inline_storage**: 8 bytes (array of 4 `uint16_t` elements).
 * - **dynamic_storage**: 8 bytes (pointer to dynamically allocated storage when needed).
 * - The union ensures that either `inline_storage` or `dynamic_storage` is used, occupying
 *   the larger of the two sizes.
 * - **Padding**: The structure is padded to align to an 8-byte boundary, resulting in a total
 *   size of 16 bytes.
 *
 * Limitations:
 * - This class is specifically designed for simple types and does not support complex types
 *   that require custom constructors or destructors.
 * - The implementation is optimized for small vectors and may not be as efficient for larger
 *   dynamic arrays where frequent resizing is required.
 *
 * Example Usage:
 * ```
 * SmallVector<uint16_t, 4> vec;
 * vec.push_back(1);
 * vec.push_back(2);
 * vec.push_back(3);
 * vec.push_back(4);
 * vec.push_back(5); // This will trigger dynamic storage allocation.
 * ```
 *
 * @tparam T Type of elements stored in the vector (e.g., `uint16_t`).
 * @tparam InlineCapacity The number of elements that can be stored in the inline storage
 *         before transitioning to dynamic storage.
 */
template<typename T, std::size_t InlineCapacity = 4>
struct SmallVector {
    using value_type = T;
    
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "SmallVector only supports simple types (integral or floating-point types).");

    std::uint32_t size_{0};  ///< Number of elements currently stored.
    union {
        T inline_storage[InlineCapacity];  ///< Inline storage for small sizes.
        T* dynamic_storage;                ///< Pointer to dynamically allocated storage.
    };
    
    constexpr SmallVector() noexcept = default;
    
    // Copy constructor
    SmallVector(const SmallVector& other)
        : size_(other.size_) {
        if (size_ > InlineCapacity) {
            dynamic_storage = static_cast<T*>(malloc(sizeof(T) * size_));
            std::memcpy(dynamic_storage, other.dynamic_storage, sizeof(T) * size_);
        } else {
            std::memcpy(inline_storage, other.inline_storage, sizeof(T) * size_);
        }
    }

    // Copy assignment operator
    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            // Clean up current storage
            if (size_ > InlineCapacity) {
                free(dynamic_storage);
            }

            size_ = other.size_;
            if (size_ > InlineCapacity) {
                dynamic_storage = static_cast<T*>(malloc(sizeof(T) * size_));
                std::memcpy(dynamic_storage, other.dynamic_storage, sizeof(T) * size_);
            } else {
                std::memcpy(inline_storage, other.inline_storage, sizeof(T) * size_);
            }
        }
        return *this;
    }

    // Move constructor
    constexpr SmallVector(SmallVector&& other) noexcept
        : size_(other.size_) {
        if (size_ > InlineCapacity) {
            dynamic_storage = other.dynamic_storage;
            other.dynamic_storage = nullptr;
        } else {
            std::memcpy(inline_storage, other.inline_storage, sizeof(T) * size_);
        }
        other.size_ = 0;
    }

    // Move assignment operator
    constexpr SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            // Clean up current storage
            if (size_ > InlineCapacity) {
                free(dynamic_storage);
            }

            size_ = other.size_;
            if (size_ > InlineCapacity) {
                dynamic_storage = other.dynamic_storage;
                other.dynamic_storage = nullptr;
            } else {
                std::memcpy(inline_storage, other.inline_storage, sizeof(T) * size_);
            }
            other.size_ = 0;
        }
        return *this;
    }

    // Destructor
    ~SmallVector() {
        if (size_ > InlineCapacity) {
            free(dynamic_storage);
        }
    }

    // Access operator
    constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size_);
        return (size_ <= InlineCapacity) ? inline_storage[index] : dynamic_storage[index];
    }

    constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size_);
        return (size_ <= InlineCapacity) ? inline_storage[index] : dynamic_storage[index];
    }

    constexpr std::size_t size() const noexcept {
        return size_;
    }

    constexpr std::size_t capacity() const noexcept {
        return (size_ <= InlineCapacity) ? InlineCapacity : size_;
    }

    void push_back(const T& value) {
        resize(size_ + 1);
        operator[](size_ - 1) = value;
    }

    void resize(std::size_t new_size) {
        if (new_size == size_) return;

        if (new_size < size_) {
            // Shrinking, no need to destroy elements for simple types
            if (new_size <= InlineCapacity && size_ > InlineCapacity) {
                // Transition from dynamic storage back to inline storage
                std::memmove(inline_storage, dynamic_storage, sizeof(T) * new_size);
                free(dynamic_storage);
            }
        } else {
            // Expanding
            if (new_size > InlineCapacity && size_ <= InlineCapacity) {
                // Transition from inline storage to dynamic storage
                T* new_storage = static_cast<T*>(malloc(sizeof(T) * new_size));
                std::memcpy(new_storage, inline_storage, sizeof(T) * size_);
                dynamic_storage = new_storage;
            } else if (new_size > InlineCapacity) {
                // Expand dynamic storage
                dynamic_storage = static_cast<T*>(realloc(dynamic_storage, sizeof(T) * new_size));
            }
            // No need to construct new elements for simple types, just increase size_
        }

        size_ = static_cast<std::uint32_t>(new_size);
    }

    void clear() noexcept {
        if (size_ > InlineCapacity) {
            free(dynamic_storage);
            dynamic_storage = nullptr;
        }
        size_ = 0;
    }

    constexpr bool empty() const noexcept {
        return size_ == 0;
    }
    
    // Data method: provides access to the underlying array
    T* data() noexcept {
        return (size_ <= InlineCapacity) ? inline_storage : dynamic_storage;
    }

    const T* data() const noexcept {
        return (size_ <= InlineCapacity) ? inline_storage : dynamic_storage;
    }
};

}

namespace track {
    class MinimalOutline;
    struct MovementInformation;
    using namespace cmn;
    
    struct DebugInfo {
        Frame_t frameIndex;
        Idx_t fdx;
        bool debug;
        //Vec2 previous_position;
    };

    struct MidlineSegment {
        Float2_t height;
        Float2_t l_length;
        Vec2 pos;
        //Vec2 pt_l;
        
        bool operator==(const MidlineSegment& other) const {
            return height == other.height && l_length == other.l_length && pos == other.pos;
        }
    };

    class Midline {
    public:
        typedef std::unique_ptr<Midline> Ptr;
        
    private:
        GETTER_NCONST(Float2_t, len){0};
        GETTER_NCONST(Float2_t, angle){0};
        GETTER_NCONST(Vec2, offset);
        GETTER_NCONST(Vec2, front);
        GETTER_NCONST(std::vector<MidlineSegment>, segments);
        GETTER_NCONST(long_t, head_index){-1};
        GETTER_NCONST(long_t, tail_index){-1};
        GETTER_NCONST(bool, inverted_because_previous){false};
        
        GETTER_NCONST(bool, is_normalized){false};
        
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
        Ptr normalize(Float2_t fix_length = -1, bool debug = false) const;
        static void fix_length(Float2_t len, std::vector<MidlineSegment>& segments, bool debug = false);
        size_t memory_size() const;
        
        Vec2 midline_direction() const;
        Float2_t original_angle() const;
        
        /**
         * if to_real_world is true, the returned transform will transform points
         * to the same coordinate system as outline points / the video coordinate system.
         * if its set to false, the returned transform is meant to transform coordinates
         * from the global system to the midline coordinate system (in order to e.g.
         * normalize an image)
         **/
        gui::Transform transform(const default_config::individual_image_normalization_t::Class &type, bool to_real_world = false) const;
        
        Vec2 real_point(const Bounds& bounds, size_t index) const;
        Vec2 real_point(const Bounds& bounds, const Vec2& pt) const;
        
    private:
        friend class Outline;
        static Float2_t calculate_angle(const std::vector<MidlineSegment>& segments);
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
            Float2_t extremum_height;
            Float2_t area;
            
            Area() : start(-1), end(-1), extremum(-1), extremum_height(0), area(0) {}
            
            void clear() {
                start = end = -1;
                extremum = -1;
                area = extremum_height = 0;
            }
        };
        
    public:
        Frame_t frameIndex;
        static Float2_t average_curvature();
        static Float2_t max_curvature();
        static uint8_t get_outline_approximate();
        
    protected:
        /**
         * Persistent memory
         * (cannot be reduced without losing information)
         */
        std::unique_ptr<std::vector<Vec2>> _points;
        
        //! confidence in the results
        //GETTER_NCONST(float, confidence);
        
        //! the uncorrected angle of the posture detection
        GETTER(Float2_t, original_angle);
        GETTER(bool, inverted_because_previous);
        
        //GETTER(long_t, tail_index);
        //GETTER(long_t, head_index);
        
        //! When set to true, this Outline cannot be changed anymore.
        GETTER(bool, concluded){false};
        
        int curvature_range;
        
        /**
         * Temporary memory
         */
        std::vector<Float2_t> _curvature;
        //GETTER(bool, needs_invert);
        
    public:
        Outline() = default;
        Outline(std::unique_ptr<std::vector<Vec2>>&& points, Frame_t f = {});
        Outline(Outline&&) = default;
        ~Outline();
        
        Outline& operator=(Outline&&) = default;
        
        void clear();
        
        inline const Vec2& at(size_t index) const { return operator[](index); }
        Vec2& operator[](size_t index);
        const Vec2& operator[](size_t index) const;
        
        void push_back(const Vec2& pt); // inserts at the back
        void push_front(const Vec2& pt); // inserts at the front
        
        template<typename Iterator>
        void insert(size_t index, const Iterator& begin, const Iterator& end)
        {
            if(not _points)
                _points = std::make_unique<std::vector<Vec2>>();
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
        
        void resample(const Float2_t distance = 1.0_F);
        
        size_t size() const { return _points ? _points->size() : 0; }
        bool empty() const { return size() == 0; }
        
        std::vector<Vec2>& points() { return *_points; }
        const std::vector<Vec2>& points() const { return *_points; }
        
        Float2_t angle() const;
        
        //! Rotates the midline so that the angle between the first and 0.2*size
        //  point is zero (horizontal). That's the rigid part (hopefully).
        //static const Midline* normalized_midline();
        tl::expected<Midline::Ptr, const char*> calculate_midline(const DebugInfo& info);
        
        virtual void minimize_memory() override;
        
        //static float calculate_slope(const std::vector<Vec2>&, size_t index);
        static Float2_t calculate_curvature(const int curvature_range, const std::vector<Vec2>&, size_t index, Float2_t scale = 1);
        static void smooth_array(const std::vector<Float2_t>& input, std::vector<Float2_t>& output, Float2_t * max_curvature = NULL);
        
        size_t memory_size() const;
        
        static int calculate_curvature_range(size_t number_points);
        void replace_points(decltype(_points)&& ptr) { _points = std::move(ptr); }
        void replace_points(decltype(_points)::element_type&& vec) {
            if(not _points)
                _points = std::make_unique<decltype(_points)::element_type>(std::move(vec));
            else
                *_points = std::move(vec);
        }
        static Float2_t get_curvature_range_ratio();
        
    protected:
        void smooth();
        
        //void calculate_slope(size_t index);
        void calculate_curvature(size_t index);
        tl::expected<std::tuple<long_t, long_t>, const char*> offset_to_middle(const DebugInfo& info);
        
        //! Smooth the curvature array.
        std::vector<Float2_t> smoothed_curvature_array(Float2_t& max_curvature) const;
        
        //! Tries to find the tail by looking at the outline/curvature.
        tl::expected<long_t, const char*> find_tail(const DebugInfo& info);
        
        friend Midline::Midline();
        
    public:
        //! Ensures the globals are loaded
        static void check_constants();
    };
    
//#pragma pack(push, 1)
    class MinimalOutline {
    protected:
        SmallVector<uint16_t> _points;
        GETTER(Vec2, first);
        Float2_t scale;
        //GETTER_NCONST(long_t, tail_index);
        //GETTER_NCONST(long_t, head_index);
        
        friend class Output::ResultsFormat;
        friend class cmn::Data;
        
    public:
        //typedef std::unique_ptr<MinimalOutline> Ptr;
        
        MinimalOutline();
        MinimalOutline(const Outline& outline);
        ~MinimalOutline();
        inline size_t memory_size() const { return sizeof(MinimalOutline) + sizeof(decltype(_points)::value_type) * _points.size() + sizeof(decltype(_first)); }
        std::vector<Vec2> uncompress() const;
        std::vector<Vec2> uncompress(Float2_t factor) const;
        size_t size() const { return _points.size(); }
        void convert_from(const std::vector<Vec2>& array);
        
        constexpr operator bool() const noexcept {
            return not _points.empty();
        }
    };
//#pragma pack(pop)

}

#endif
