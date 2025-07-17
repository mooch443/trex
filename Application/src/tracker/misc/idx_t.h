#pragma once
#include <commons.pc.h>

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
    constexpr uint32_t get() const { return _identity; }
    explicit constexpr operator bool() const noexcept { return valid(); }
    constexpr bool valid() const { return _identity != cmn::infinity<uint32_t>(); }
    
    constexpr auto operator<=>(const Idx_t& other) const {
#ifndef NDEBUG
        if(!valid() || !other.valid())
            throw std::invalid_argument("Comparing to an invalid Idx_t does not produce the desired outcome.");
#endif
        return _identity <=> other._identity;
    }
    constexpr bool operator==(const Idx_t& other) const {
        if(not other.valid() && not valid())
            return true;
        else if(other.valid() != valid())
            return false;
        
        return other.get() == get();
    }
    
    constexpr bool operator!=(const Idx_t& other) const {
        return not operator==(other);
    }
    
    constexpr Idx_t operator-(const Idx_t& other) const {
        return Idx_t(get() - other.get());
    }
    constexpr Idx_t operator+(const Idx_t& other) const {
        return Idx_t(get() + other.get());
    }
    constexpr Idx_t operator/(const Idx_t& other) const {
        return Idx_t{ get() / other.get() };
    }
    constexpr Idx_t operator*(const Idx_t& other) const {
        return Idx_t(get() * other.get());
    }
    
    static std::string class_name() { return "Idx_t"; }
    static Idx_t fromStr(cmn::StringLike auto&& str) {
        if(std::string_view(str) == "-1")
            return Idx_t();
        return Idx_t(cmn::Meta::fromStr<uint32_t>(str));
    }
    glz::json_t to_json() const;
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
            return std::hash<uint32_t>{}(k.get());
        }
    };
}
