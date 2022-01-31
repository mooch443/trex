#pragma once

#include <misc/defines.h>
#include <misc/metastring.h>

namespace cmn {

struct Frame_t {
    using number_t = long_t;
    static constexpr number_t invalid = -1;

    long_t _frame = invalid;
    
    constexpr Frame_t() = default;
    explicit constexpr Frame_t(number_t frame)
        : _frame(frame)
    {
        //Debug("Initializing with %d", frame);
    }
    
    constexpr void invalidate() { _frame = invalid; }
    
    constexpr number_t get() const { return (long_t)_frame; }
    //constexpr operator long_t() const { return _frame; }
    constexpr bool valid() const { return _frame >= 0; }
    constexpr Frame_t& operator+=(const Frame_t& other) {
        _frame += other._frame;
        return *this;
    }
    /*constexpr Frame_t& operator+=(long_t&& other) {
        assert(valid() && other != invalid);
        if(!valid())
            _frame = other;
        else
            _frame += other;
        return *this;
    }*/
    /*constexpr Frame_t& operator=(Frame value) {
        _frame = value;
        return *this;
    }*/
    
    constexpr bool operator<(const Frame_t& other) const {
        //return (valid() ^ other.valid() && valid()) || (valid() && get() < other.get());
        return get() < other.get(); // invalid is fine - can be <<<< 0, but still sorted
    }
    constexpr bool operator>(const Frame_t& other) const {
        return get() > other.get();
        //return (other.valid() ^ valid() && other.valid()) || (valid() && get() > other.get());
    }
    constexpr bool operator<=(const Frame_t& other) const {
        return *this < other || *this == other;
    }
    constexpr bool operator>=(const Frame_t& other) const {
        return *this > other || *this == other;
    }
    
    constexpr Frame_t operator-(const Frame_t& other) const {
        return Frame_t(get() - other.get());
    }
    constexpr Frame_t operator-() const {
        return Frame_t(-get());
    }
    constexpr Frame_t& operator--() {
        --_frame;
        return *this;
    }
    
    constexpr Frame_t& operator++() {
        ++_frame;
        return *this;
    }
    
    constexpr bool operator==(const Frame_t& other) const {
        return other.get() == get();
    }
    constexpr Frame_t operator+(const Frame_t& other) const {
        return Frame_t(get() + other.get());
    }
    constexpr Frame_t operator/(const Frame_t& other) const {
        return Frame_t(get() / other.get());
    }
    constexpr Frame_t operator*(const Frame_t& other) const {
        return Frame_t(get() * other.get());
    }

    std::string toStr() const {
        return Meta::toStr<number_t>(_frame);
    }
    static std::string class_name() {
        return "frame";
    }
    static Frame_t fromStr(const std::string& str) {
        return Frame_t(cmn::Meta::fromStr<number_t>(str));
    }
};

constexpr Frame_t operator""_f(const unsigned long long int value) {
    // intentional static cast, but probably unsafe.
    return Frame_t(static_cast<Frame_t::number_t>(value));
}

constexpr inline Frame_t min(Frame_t A, Frame_t B) {
    return std::min(A, B);
}

constexpr inline Frame_t max(Frame_t A, Frame_t B) {
    return std::max(A, B);
}

}

namespace std {

template<>
struct hash<cmn::Frame_t>
{
    size_t operator()(const cmn::Frame_t& k) const
    {
        return std::hash<cmn::Frame_t::number_t>{}(k.get());
    }
};

}
