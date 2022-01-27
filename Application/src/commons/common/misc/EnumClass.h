#pragma once

#define CHOOSE_MAP_START(count) MAP ## count

#define APPLY(macro, ...) IDENTITY(macro(__VA_ARGS__))

#define IDENTITY(x) x

#define MAP1(m, x)      m(x)
#define MAP2(m, x, ...) m(x) IDENTITY(MAP1(m, __VA_ARGS__))
#define MAP3(m, x, ...) m(x) IDENTITY(MAP2(m, __VA_ARGS__))
#define MAP4(m, x, ...) m(x) IDENTITY(MAP3(m, __VA_ARGS__))
#define MAP5(m, x, ...) m(x) IDENTITY(MAP4(m, __VA_ARGS__))
#define MAP6(m, x, ...) m(x) IDENTITY(MAP5(m, __VA_ARGS__))
#define MAP7(m, x, ...) m(x) IDENTITY(MAP6(m, __VA_ARGS__))
#define MAP8(m, x, ...) m(x) IDENTITY(MAP7(m, __VA_ARGS__))
#define MAP9(m, x, ...) m(x) IDENTITY(MAP8(m, __VA_ARGS__))
#define MAP10(m, x, ...) m(x) IDENTITY(MAP9(m, __VA_ARGS__))
#define MAP11(m, x, ...) m(x) IDENTITY(MAP10(m, __VA_ARGS__))
#define MAP12(m, x, ...) m(x) IDENTITY(MAP11(m, __VA_ARGS__))
#define MAP13(m, x, ...) m(x) IDENTITY(MAP12(m, __VA_ARGS__))
#define MAP14(m, x, ...) m(x) IDENTITY(MAP13(m, __VA_ARGS__))
#define MAP15(m, x, ...) m(x) IDENTITY(MAP14(m, __VA_ARGS__))
#define MAP16(m, x, ...) m(x) IDENTITY(MAP15(m, __VA_ARGS__))
#define MAP17(m, x, ...) m(x) IDENTITY(MAP16(m, __VA_ARGS__))
#define MAP18(m, x, ...) m(x) IDENTITY(MAP17(m, __VA_ARGS__))
#define MAP19(m, x, ...) m(x) IDENTITY(MAP18(m, __VA_ARGS__))
#define MAP20(m, x, ...) m(x) IDENTITY(MAP19(m, __VA_ARGS__))

#define EVALUATE_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, count, ...) \
count

#define COUNT(...) \
IDENTITY(EVALUATE_COUNT(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))

#define MAP(macro, ...) \
IDENTITY( \
APPLY(CHOOSE_MAP_START, COUNT(__VA_ARGS__)) \
(macro, __VA_ARGS__))

#define STRINGIZE(...) IDENTITY(MAP(STRINGIZE_SINGLE, __VA_ARGS__))
#define STRINGIZE_SINGLE(e) #e,

#define PP_NARG(...)    PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...)  IDENTITY( PP_ARG_N(__VA_ARGS__) )

#define PP_ARG_N( \
_1, _2, _3, _4, _5, _6, _7, _8, _9,_10,  \
_11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
_21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
_31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
_41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
_51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
_61,_62,_63,N,...) N

#define PP_RSEQ_N() \
63,62,61,60,                   \
59,58,57,56,55,54,53,52,51,50, \
49,48,47,46,45,44,43,42,41,40, \
39,38,37,36,35,34,33,32,31,30, \
29,28,27,26,25,24,23,22,21,20, \
19,18,17,16,15,14,13,12,11,10, \
9,8,7,6,5,4,3,2,1,0

/* need extra level to force extra eval */
#define Paste(a,b) a ## b
#define XPASTE(a,b) Paste(a,b)

#define X_PASTE_FUNC_(x, y) y(data :: values :: y)
#define X_MOD_NAME(y) IDENTITY( _ ## y ),

/* APPLYXn variadic X-Macro by M Joshua Ryan      */
/* Free for all uses. Don't be a jerk.            */
/* I got bored after typing 15 of these.          */
/* You could keep going upto 64 (PPNARG's limit). */
#define XSEP ,

#define _APPLYX1(a)         X_PASTE_FUNC_(a)
#define _APPLYX2(a,b)         X_PASTE_FUNC_(a, b)
#define _APPLYX3(a,b,c)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c)
#define _APPLYX4(a,b,c,d)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d)
#define _APPLYX5(a,b,c,d,e)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e)
#define _APPLYX6(a,b,c,d,e,f)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f)
#define _APPLYX7(a,b,c,d,e,f,g)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g)
#define _APPLYX8(a,b,c,d,e,f,g,h)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h)
#define _APPLYX9(a,b,c,d,e,f,g,h,i)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i)
#define _APPLYX10(a,b,c,d,e,f,g,h,i,j)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j)
#define _APPLYX11(a,b,c,d,e,f,g,h,i,j,k)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k)
#define _APPLYX12(a,b,c,d,e,f,g,h,i,j,k,l)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l)
#define _APPLYX13(a,b,c,d,e,f,g,h,i,j,k,l,m)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m)
#define _APPLYX14(a,b,c,d,e,f,g,h,i,j,k,l,m,n)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n)
#define _APPLYX15(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o)
#define _APPLYX16(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o) XSEP X_PASTE_FUNC_(a, p)
#define _APPLYX17(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o) XSEP X_PASTE_FUNC_(a, p) XSEP X_PASTE_FUNC_(a, q)
#define _APPLYX18(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o) XSEP X_PASTE_FUNC_(a, p) XSEP X_PASTE_FUNC_(a, q) XSEP X_PASTE_FUNC_(a, r)
#define _APPLYX19(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o) XSEP X_PASTE_FUNC_(a, p) XSEP X_PASTE_FUNC_(a, q) XSEP X_PASTE_FUNC_(a, r) XSEP X_PASTE_FUNC_(a, s)
#define _APPLYX20(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t)       X_PASTE_FUNC_(a, b) XSEP X_PASTE_FUNC_(a, c) XSEP X_PASTE_FUNC_(a, d) XSEP X_PASTE_FUNC_(a, e) XSEP X_PASTE_FUNC_(a, f) XSEP X_PASTE_FUNC_(a, g) XSEP X_PASTE_FUNC_(a, h) XSEP X_PASTE_FUNC_(a, i) XSEP X_PASTE_FUNC_(a, j) XSEP X_PASTE_FUNC_(a, k) XSEP X_PASTE_FUNC_(a, l) XSEP X_PASTE_FUNC_(a, m) XSEP X_PASTE_FUNC_(a, n) XSEP X_PASTE_FUNC_(a, o) XSEP X_PASTE_FUNC_(a, p) XSEP X_PASTE_FUNC_(a, q) XSEP X_PASTE_FUNC_(a, r) XSEP X_PASTE_FUNC_(a, s) XSEP X_PASTE_FUNC_(a, t)

#define APPLYX_(M, ...) IDENTITY( M(__VA_ARGS__) )
#define _APPLYXn(...) APPLYX_(XPASTE(_APPLYX, PP_NARG(__VA_ARGS__)), __VA_ARGS__)

constexpr inline bool strings_equal(char const * a, char const * b) {
    return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}


template <typename ValueType, size_t N, typename _names>
class Enum {
public:
    constexpr static const size_t num_values = N;
    typedef Enum<ValueType, N, _names> self_type;
    typedef _names Data;
    
protected:
    ValueType _value;
    
public:
    constexpr Enum(const ValueType& value = (ValueType)0) noexcept
        : _value(value)
    {}
    
    constexpr const char * name() const { return _names::str()[(size_t)_value]; }
    explicit constexpr operator const char*() const { return name(); }
    //operator std::string() const { return std::string(name()); }
    //constexpr uint32_t toInt() const { return (uint32_t)_value; }
    constexpr explicit operator uint32_t() const { return (uint32_t)_value; }
    constexpr operator ValueType() const { return _value; }
    constexpr const ValueType& value() const { return _value; }
    
    // comparison operators
    constexpr bool operator==(const Enum& other) const { return other._value == _value; }
    constexpr bool operator==(const ValueType& other) const { return other == _value; }
    constexpr bool operator!=(const Enum& other) const { return other._value != _value; }
    constexpr bool operator!=(const ValueType& other) const { return other != _value; }
    std::string toStr() const { return name(); }
    static std::string class_name() { return std::string(_names::class_name());}
    
    template<typename T = std::string>
    static const self_type& get(T name) {
        return _names::get(name);
    }
    static auto fields() { return _names::str(); }
    
    constexpr bool operator==(const std::string& other) const { return other == _names::str()[(size_t)_value]; }
};

/*namespace cmn {
    template<typename T>
    constexpr bool known_enum() {
        return false;
    }
}*/

/*struct enum_has_docs \
{ enum { value = sizeof docs_exist(  ) == sizeof( std::array<const char*, num_elements> ) }; }; \
 */
#undef ENUM_CLASS
#define ENUM_CLASS(NAME, ...) \
namespace NAME { \
    namespace data { \
        enum class values { \
            __VA_ARGS__ \
        }; \
        \
        constexpr const char *name = #NAME; \
        constexpr static const size_t num_elements = PP_NARG( __VA_ARGS__ ) ; \
        std::array<const char*, num_elements> docs_exist(); \
         \
        struct names {						       \
            template<class T> struct enum_has_docs : public std::false_type {}; \
            template<typename T> \
            static std::array<const char*, num_elements> docs( \
                typename std::enable_if_t<names::enum_has_docs<T>::value, char> * = nullptr) \
            { \
                return docs_exist(); \
            } \
            constexpr static const std::array<const char *, num_elements> str() {	\
                const std::array<const char*, num_elements> _names = {	\
                  {IDENTITY( STRINGIZE(__VA_ARGS__) )}			\
                };								\
                return _names;						\
            }								\
            constexpr static std::string_view class_name() { return std::string_view( #NAME ); } \
            template<typename T = std::string> static inline const Enum<NAME :: data::values, NAME :: data::num_elements, NAME :: data::names>& get(T name); \
        }; \
    } \
    \
    typedef Enum<data::values, data::num_elements, data::names> Class; \
    \
    constexpr const Class _APPLYXn(NAME, __VA_ARGS__); \
    constexpr std::array<Class, data::num_elements> values = {{ __VA_ARGS__ }}; \
    constexpr std::array<const char*, data::num_elements> names = data::names::str(); \
    template<typename T = std::string > \
    static inline bool has(T name) { \
        for(size_t i=0; i<data::num_elements; i++) \
            if(utils::lowercase(name) == utils::lowercase(data::names::str()[i])) return true; \
        return false; \
    } \
    template<typename T = std::string> \
    static inline const Class& get(T name) { \
        UNUSED(names) \
        for(auto &v : values) \
            if(utils::lowercase(v.name()) == utils::lowercase(name)) return v; \
        U_EXCEPTION("Cannot find value '%S' in enum '%s'", &name, NAME :: data :: name ); \
    } \
    \
template<typename T> const Class& data::names::get(T name) { return NAME :: get(name); }\
}

#define ENUM_CLASS_HAS_DOCS(NAME) \
namespace NAME { \
    namespace data { \
        template<> struct names::enum_has_docs<Enum<data::values, data::num_elements, data::names>> : public std::true_type {}; \
    } \
}

#define ENUM_CLASS_DOCS(NAME, ...) \
namespace NAME { \
    namespace data { \
        std::array<const char*, num_elements> docs_exist( ) { \
            return std::array<const char*, num_elements> { \
                __VA_ARGS__ \
            }; \
        } \
    } \
}

namespace cmn {
    template<class T> struct is_enum : public std::false_type {};
    template<class Alloc, size_t N, typename B>
    struct is_enum< Enum<Alloc, N, B> > : public std::true_type {};
}

namespace std
{
    template <typename ValueType, size_t N, typename names>
    struct hash<Enum<ValueType, N, names>>
    {
        size_t operator()(const Enum<ValueType, N, names>& k) const
        {
            return std::hash<uint32_t>{}((uint32_t)k);
        }
    };
}

//template<> constexpr bool cmn::known_enum<NAME :: Class>() { return true; } 
