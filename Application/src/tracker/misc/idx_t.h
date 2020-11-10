#pragma once
#include <commons.pc.h>
#include <misc/checked_casts.h>

namespace track {

struct Idx_t {
    uint32_t _identity = infinity<uint32_t>();
    constexpr Idx_t() = default;
    template<typename T>
    explicit constexpr Idx_t(T ID) : _identity(cmn::narrow_cast<uint32_t>(ID)) {}
    constexpr operator uint32_t() const { return _identity; }
    constexpr bool valid() const { return _identity != infinity<uint32_t>(); }
    
    static std::string class_name() { return "Idx_t"; }
    //operator std::string() const;
    static Idx_t fromStr(const std::string&);
};

struct Frame_t {
    long_t _frame = infinity<long_t>();
    constexpr Frame_t() = default;
    explicit constexpr Frame_t(long_t frame) : _frame(frame) {}
    constexpr operator long_t() const { return _frame; }
    constexpr bool valid() const { return _frame != infinity<long_t>(); }
    constexpr Frame_t& operator+=(Frame_t&& other) {
        assert(valid() && other.valid());
        _frame += other._frame;
        return *this;
    }
    constexpr Frame_t& operator+=(long_t&& other) {
        assert(valid() && other != infinity<long_t>());
        _frame += other;
        return *this;
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
}
