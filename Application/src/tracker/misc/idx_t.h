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
