#pragma once
#include <misc/defines.h>
#include <misc/checked_casts.h>
#include <misc/metastring.h>

namespace track {
struct Idx_t {
    uint32_t _identity = cmn::infinity<uint32_t>();
    constexpr Idx_t() = default;
    Idx_t(Idx_t const &ID) = default;
    Idx_t& operator=(const Idx_t&) = default;
    Idx_t& operator=(Idx_t&&) = default;
    
    template<typename T>
        requires std::convertible_to<T, uint32_t>
    explicit constexpr Idx_t(T ID) : _identity((uint32_t)ID) {}
    
    explicit constexpr Idx_t(uint32_t ID) : _identity(ID) {}
    constexpr operator uint32_t() const { return _identity; }
    constexpr bool valid() const { return _identity != cmn::infinity<uint32_t>(); }
    
    constexpr auto operator<=>(const Idx_t& other) const {
        return _identity <=> other._identity;
    }
    
    static std::string class_name() { return "Idx_t"; }
    static Idx_t fromStr(const std::string&);
    std::string toStr() const { return !valid() ? "-1" : std::to_string((uint32_t)_identity); }
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
