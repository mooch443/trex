#pragma once
#include <misc/defines.h>
#include <misc/checked_casts.h>
#include <misc/metastring.h>

namespace track {
struct Idx_t {
    uint32_t _identity = cmn::infinity<uint32_t>();
    constexpr Idx_t() = default;
    template<typename T>
    explicit constexpr Idx_t(T ID) : _identity(cmn::narrow_cast<uint32_t>(ID)) {}
    constexpr operator uint32_t() const { return _identity; }
    constexpr bool valid() const { return _identity != cmn::infinity<uint32_t>(); }
    
    static std::string class_name() { return "Idx_t"; }
    static Idx_t fromStr(const std::string&);
};

struct Frame_t {
    static constexpr long_t invalid = -1;

    long_t _frame = invalid;
    constexpr Frame_t() = default;
    explicit constexpr Frame_t(long_t frame) : _frame(frame) {}
    constexpr operator long_t() const { return _frame; }
    constexpr bool valid() const { return _frame != invalid; }
    constexpr Frame_t& operator+=(Frame_t&& other) {
        assert(valid() && other.valid());
        _frame += other._frame;
        return *this;
    }
    constexpr Frame_t& operator+=(long_t&& other) {
        assert(valid() && other != invalid);
        _frame += other;
        return *this;
    }
    constexpr Frame_t& operator=(long_t value) {
        _frame = value;
        return *this;
    }

    //std::string toStr() const {
    //    return Meta::toStr<long_t>(_frame);
    //}
    static std::string class_name() {
        return "frame";
    }
    static Frame_t fromStr(const std::string& str) {
        return Frame_t(cmn::Meta::fromStr<long_t>(str));
    }
};

}

namespace std
{
    template<>
    struct hash<track::Idx_t>
    {
        size_t operator()(const track::Idx_t& k) const
        {
            return std::hash<uint32_t>{}((uint32_t)k);
        }
    };
    template<>
    struct hash<track::Frame_t>
    {
        size_t operator()(const track::Frame_t& k) const
        {
            return std::hash<long_t>{}((long_t)k);
        }
    };
}
