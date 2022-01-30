#pragma once
#include <misc/defines.h>
#include <misc/metastring.h>
#include <misc/frame_t.h>

namespace cmn {

template<typename T>
concept range_type_needs_get_method = requires (T t) {
    (!std::convertible_to<T, size_t>);
    { t.get() } -> std::convertible_to<size_t>;
};

template<typename T>
class arange {
public:
    T first; T last; T step;

    constexpr arange(T first = T(0), T last = T(0), T step = T(1)) : first(first), last(last), step(step)
    { }

private:
    template<typename K = T>
        requires (!range_type_needs_get_method<K>)
    constexpr size_t num_steps() const {
        return size_t((last - first) / step + K(1));
    }
    
    template<typename K = T>
        requires range_type_needs_get_method<K>
    constexpr size_t num_steps() const {
        return (size_t)((last - first) / step + K(1)).get();
    }

public:
    template<typename ValueType, typename NodeType>
    struct _iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = ValueType;
        using pointer = value_type*;
        using reference = value_type;
        using iterator_category = std::bidirectional_iterator_tag;

        _iterator(size_t value = 0, NodeType* ptr = nullptr) : ptr_(ptr), value(value)
        {

        }
        _iterator(const _iterator& rhs) {
            operator=(rhs);
        }
        _iterator(_iterator&& rhs) {
            operator=(rhs);
        }
        _iterator& operator=(const _iterator& rhs) {
            ptr_ = rhs.ptr_;
            current_ = rhs.current_;
            end_ = rhs.end_;
            value = rhs.value;
            return *this;
        }
        _iterator& operator=(_iterator&& rhs) {
            ptr_ = rhs.ptr_;
            current_ = rhs.current_;
            end_ = rhs.end_;
            value = rhs.value;
            return *this;
        }
        void swap(_iterator& iter) {
            std::swap(iter.value, value);
            std::swap(iter.ptr_, ptr_);
            std::swap(iter.current_, current_);
            std::swap(iter.end_, end_);
        }

        bool operator==(const _iterator& rhs) const { return value == rhs.value; }
        bool operator!=(const _iterator& rhs) const { return value != rhs.value; }

        reference operator*() {
            return value_type(ptr_->first + value_type(value) * value_type(ptr_->step));
        }
        pointer operator->() {
            return *ptr_;
        }

        _iterator& operator++() {
            ++value;
            return *this;
        }
        _iterator operator++(int) {
            _iterator i(*this);
            ++i;
            return i;
        }
        _iterator& operator--() {
            --value;
            return *this;
        }
        _iterator operator--(int) {
            _iterator i(*this);
            --i;
            return i;
        }

    private:
        NodeType* ptr_;
        size_t value;
        value_type current_;
        value_type end_;
    };

    typedef _iterator<T, arange<T>> iterator;
    typedef _iterator<T, const arange<T>> const_iterator;

    constexpr iterator begin() { return iterator(0, this); }
    constexpr iterator end() { return iterator(num_steps(), this); }

#if !defined(_MSC_VER) || _MSC_VER >= 1920
    template<typename K, typename V = typename std::enable_if< is_container<K>::value, void >::type>
    operator K() { return K(begin(), end()); }

    template<typename K, typename V = typename std::enable_if< is_container<K>::value, void >::type>
    operator K() const { return K(begin(), end()); }
#else
    template<typename K, typename V = typename std::enable_if< is_container<K>::value, void >::type>
    operator K() {
        std::vector<T> v;
        v.reserve(size());
        for (T i : *this)
            v.push_back(i);
        return v;
    }

    template<typename K, typename V = typename std::enable_if< is_container<K>::value, void >::type>
    operator K() const {
        std::vector<T> v;
        v.reserve(size());
        for (T i : *this)
            v.push_back(i);
        return v;
    }
#endif

    constexpr const_iterator begin() const { return const_iterator(0, this); }
    constexpr const_iterator end() const { return const_iterator(num_steps(), this); }

    constexpr bool contains(T value) const { return value >= first && value <= last; }
    constexpr bool empty() const { return first == last; }
    constexpr size_t size() const { return num_steps(); }
};

typedef arange<int> irange;
typedef arange<float> frange;
typedef arange<long_t> lrange;

template<typename T>
struct Range {
    T start, end;

    constexpr Range() noexcept : Range(T(), T()) {}
    explicit constexpr Range(T s, T e = T()) noexcept : start(s), end(e) { assert(s <= e); }
    constexpr bool empty() const { return start == end; }
    constexpr bool contains(T v) const {
        return v >= start && v < end;
    }
    constexpr bool overlaps(const Range<T>& v) const {
        return contains(v.start) || contains(v.end)
            || v.contains(start) || v.contains(end)
            || v.start == end || start == v.end;
    }

    constexpr bool operator<(const Range<T>& other) const {
        return start < other.start || (start == other.start && end < other.end);
    }

    constexpr T length() const { return T(1) + end - start; }
    constexpr arange<T> iterable() const { return arange<T>(start, end); }
    constexpr bool operator==(const Range<T>& other) const {
        return other.start == start && other.end == end;
    }
    constexpr bool operator!=(const Range<T>& other) const {
        return other.start != start || other.end != end;
    }

    constexpr Range<T> operator*(T number) const {
        return Range<T>(start * number, end * number);
    }

    constexpr Range<T> operator/(T number) const {
        return Range<T>(start / number, end / number);
    }

    std::string toStr() const {
        return "[" + Meta::toStr(start) + "," + Meta::toStr(end) + "]";
    }
    static std::string class_name() { return "range<" + Meta::template name<T>() + ">"; }

    static Range<T> fromStr(const std::string& str)
    {
        auto parts = util::parse_array_parts(util::truncate(str));
        if (parts.size() != 2) {
            throw CustomException<std::invalid_argument>("Illegal Rangel format.");
        }

        auto x = Meta::template fromStr<T>(parts[0]);
        auto y = Meta::template fromStr<T>(parts[1]);

        return Range(x, y);
    }
};

using Rangef = Range<float>;
using Rangei = Range<int>;
using Rangel = Range<long_t>;

struct FrameRange {
    Range<Frame_t> range;
    Frame_t first_usable;

    constexpr explicit FrameRange(Range<Frame_t> range = Range<Frame_t>(Frame_t(), Frame_t()), Frame_t first_usable = Frame_t())
        : range(range), first_usable(first_usable)
    {}

    static constexpr FrameRange merge(const FrameRange& A, const FrameRange& B) {
        return FrameRange { Range<Frame_t>{
            min(A.start(), B.start()),
            max(A.end(), B.end())
        }, A.first_usable.valid() ? (B.first_usable.valid() ? min(A.first_usable, B.first_usable) : A.first_usable) : B.first_usable };
    }
    
    constexpr Frame_t::number_t length(bool usable_only = false) const {
        if (!range.start.valid())
            return 0;

        if (usable_only)
            return (first_usable >= range.start ? (range.end - first_usable) : range.length()).get();
        return range.length().get();
    }

    constexpr Frame_t start() const {
        return range.start;
    }

    constexpr Frame_t end() const {
        return range.end;
    }

    constexpr bool contains(Frame_t frame) const {
        if (empty())
            return false;
        return frame >= start() && frame <= end();
    }

    constexpr bool overlaps(const FrameRange& v) const {
        return contains(v.start()) || contains(v.end())
            || v.contains(start()) || v.contains(end())
            || v.start() == end() || start() == v.end();
    }

    template<typename T>
    constexpr bool overlaps(const Range<T>& v) const {
        return contains(v.start) || contains(v.end)
            || (v.contains(start()) || v.end == start()) || (v.contains(end()) || v.end == end())
            || v.start == end() || start() == v.end;
    }

    constexpr bool empty() const {
        return !range.start.valid();
    }

    constexpr bool operator<(const FrameRange& other) const {
        return range < other.range;
    }
    constexpr bool operator==(const FrameRange& other) const {
        return range == other.range;
    }

    arange<Frame_t> iterable() const {
        return arange<Frame_t>(start(), end());
    }

    std::string toStr() const {
        return "[" + Meta::toStr(start()) + "," + Meta::toStr(end()) + "]";
    }
    static std::string class_name() {
        return "FrameRange";
    }

};

}

