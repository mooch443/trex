#pragma once

#include <misc/defines.h>

namespace cmn {
    
class illegal_syntax : public std::logic_error {
public:
    illegal_syntax(const std::string& str) : std::logic_error(str) { }
    ~illegal_syntax() throw() { }
};
    
struct DurationUS {
    //! A duration in microseconds.
    uint64_t timestamp;
        
    std::string to_string() const {
        static constexpr std::array<std::string_view, 5> names{{"us", "ms", "s", "min", "h"}};
        static constexpr std::array<double, 5> ratios{{1000, 1000, 60, 60, 24}};
            
        double scaled = static_cast<double>(timestamp), previous_scaled = 0;
        size_t i = 0;
        while(i < ratios.size()-1 && scaled >= ratios[i]) {
            scaled /= ratios[i];
                
            previous_scaled = scaled - size_t(scaled);
            previous_scaled *= ratios[i];
                
            i++;
        }
            
        size_t sub_part = (size_t)previous_scaled;
            
        std::stringstream ss;
        ss << std::fixed << std::setprecision(0) << scaled;
        if(i>0 && i > 2)
            ss << ":" << std::setfill('0') << std::setw(2) << sub_part;
        ss << std::string(names[i].begin(), names[i].end());
        return ss.str();
    }
        
    std::string to_html() const {
        static constexpr std::array<std::string_view, 5> names{{"us", "ms", "s", "min", "h"}};
        static constexpr std::array<double, 5> ratios{{1000, 1000, 60, 60, 24}};
            
        double scaled = static_cast<double>(timestamp), previous_scaled = 0;
        size_t i = 0;
        while(i < ratios.size()-1 && scaled >= ratios[i]) {
            scaled /= ratios[i];
                
            previous_scaled = scaled - size_t(scaled);
            previous_scaled *= ratios[i];
                
            i++;
        }
            
        size_t sub_part = (size_t)previous_scaled;
            
        std::stringstream ss;
        ss << "<nr>" << std::fixed << std::setprecision(0) << scaled << "</nr>";
        if(i>0 && i > 2)
            ss << ":<nr>" << std::setfill('0') << std::setw(2) << sub_part << "</nr>";
        ss << std::string(names[i].begin(), names[i].end());
        return ss.str();
    }
        
    static DurationUS fromStr(const std::string&) {
        U_EXCEPTION("Not implemented.");
        return DurationUS{0};
    }
    static std::string class_name() { return "duration"; }
    std::string toStr() const {
        return to_string();
    }
};
    
struct FileSize {
    size_t bytes;
        
    FileSize(size_t b = 0) : bytes(b) {}
        
    std::string to_string() const {
        std::vector<std::string> descriptions = {
            "bytes", "KB", "MB", "GB", "TB"
        };
            
        size_t i=0;
        auto scaled = static_cast<double>(bytes);
        while (scaled >= 1000 && i < descriptions.size()-1) {
            scaled /= 1000.0;
            i++;
        }
            
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << scaled << descriptions[i];
            
        return ss.str();
    }
        
    static FileSize fromStr(const std::string&) {
        U_EXCEPTION("Not implemented.");
    }

    static std::string class_name() {
        return "filesize";
    }
    std::string toStr() const {
        return to_string();
    }
};

// <concepts>
#pragma region concepts
template<template<class...>class Template, class T>
struct is_instantiation : std::false_type {};
template<template<class...>class Template, class... Ts>
struct is_instantiation<Template, Template<Ts...>> : std::true_type {};

template<typename T, typename U>
concept _clean_same =
    std::same_as<T, typename std::remove_cv<U>::type>;

template <template<class...>class T, class U>
concept _is_instance = (is_instantiation<T, U>::value);

template <typename T>
concept is_shared_ptr
    = _is_instance<std::shared_ptr, T>;
template <typename T>
concept is_unique_ptr 
    = _is_instance<std::unique_ptr, T>;

template<typename T>
concept _has_tostr_method = requires(T t) {
    { t.toStr() } -> std::convertible_to<std::string>;
};
template<typename T>
concept _has_fromstr_method = requires() {
    { T::fromStr(std::string()) }
        -> _clean_same<T>;
};

template<typename T, typename K = typename std::remove_cv<T>::type>
concept _has_class_name = requires() {
    { K::class_name() } -> std::convertible_to<std::string>;
};

template<typename T>
concept _is_smart_pointer = 
    (is_shared_ptr<T> || is_unique_ptr<T>);

template<typename T>
concept _is_dumb_pointer =
    (std::is_pointer<T>::value) && (!_is_smart_pointer<T>);

template<typename T>
concept _is_number = 
    (!_clean_same<bool, T>) && (std::floating_point<T> || std::integral<T> || std::convertible_to<T, int>);

template<typename T>
concept is_numeric = (!_clean_same<bool, T>) && (std::floating_point<T> || std::integral<T>);

#pragma region concepts
// </concepts>

// <util>
#pragma region util
#pragma mark
namespace util {

template <typename T>
    requires std::floating_point<T>
std::string to_string(const T& t) {
    std::string str{std::to_string (t)};
    size_t offset = min(str.size()-1, size_t(1));
    if (str.find_last_not_of('0') == str.find('.')) {
        offset = 0;
    }
            
    str.erase(str.find_last_not_of('0') + offset, std::string::npos);
    return str;
}

template <typename T>
    requires std::convertible_to<T, std::string>
std::string to_string(const T& t) {
    return "\""+(std::string)t+"\"";
}
template <typename T>
    requires (!std::convertible_to<T, std::string>)
             && (!std::floating_point<T>)
std::string to_string(const T& t) {
    return std::to_string (t);
}

inline std::string truncate(const std::string& str) {
    if ((utils::beginsWith(str, '{') && utils::endsWith(str, '}'))
        || (utils::beginsWith(str, '[') && utils::endsWith(str, ']')))
        return utils::trim(str.substr(1, str.length() - 2));

    throw CustomException<illegal_syntax>("Cannot parse array '%S'", &str);
}

inline std::string escape(std::string str) {
    str = utils::find_replace(str, "\\", "\\\\");
    return utils::find_replace(str, "\"", "\\\"");
}

inline std::string unescape(std::string str) {
    str = utils::find_replace(str, "\\\"", "\"");
    return utils::find_replace(str, "\\\\", "\\");
}

inline std::vector<std::string> parse_array_parts(const std::string& str, const char delimiter = ',') {
    std::deque<char> brackets;
    std::stringstream value;
    std::vector<std::string> ret;

    char prev = 0;
    for (size_t i = 0; i < str.length(); i++) {
        char c = str.at(i);
        bool in_string = !brackets.empty() && (brackets.front() == '\'' || brackets.front() == '"');

        if (in_string) {
            auto s = value.str();
            if (prev != '\\' && c == brackets.front()) {
                brackets.pop_front();
            }

            value << c;

        }
        else {
            switch (c) {
            case '[':
            case '{':
                brackets.push_front(c);
                break;

            case '"':
            case '\'':
                brackets.push_front(c);
                break;
            case ']':
                if (!brackets.empty() && brackets.front() == '[')
                    brackets.pop_front();
                break;
            case '}':
                if (!brackets.empty() && brackets.front() == '{')
                    brackets.pop_front();
                break;
            default:
                break;
            }

            if (brackets.empty()) {
                if (c == delimiter) {
                    ret.push_back(utils::trim(value.str()));
                    value.str("");
                }
                else
                    value << c;

            }
            else {
                value << c;
            }
        }

        if (prev == '\\' && c == '\\')
            prev = 0;
        else
            prev = c;
    }

    std::string s = utils::trim(value.str());
    if (!s.empty()) {
        ret.push_back(s);
    }

    return ret;
}

namespace tuple_tools {
    template <class F, size_t... Is>
    constexpr auto index_apply_impl(F f,
        std::index_sequence<Is...>) {
        return f(std::integral_constant<size_t, Is> {}...);
    }

    template <size_t N, class F>
    constexpr auto index_apply(F f) {
        return index_apply_impl(f, std::make_index_sequence<N>{});
    }

    template <class Tuple, class F>
    constexpr auto apply(Tuple t, F f) {
        return index_apply < std::tuple_size<Tuple>{} > (
            [&](auto... Is) { return f(Is..., std::get<Is>(t)...); });
    }
}

}
#pragma endregion util
// </util>

// <Meta prototypes>
#pragma region Meta prototypes
namespace Meta {
template<typename Q>
inline std::string name(const typename std::enable_if< std::is_pointer<Q>::value && !std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q>
inline std::string toStr(Q value, const typename std::enable_if< std::is_pointer<Q>::value && !std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q>
inline std::string name(const typename std::enable_if< std::is_pointer<Q>::value && std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q>
inline std::string toStr(Q value, const typename std::enable_if< std::is_pointer<Q>::value && std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q, typename T = typename remove_cvref<Q>::type>
inline T fromStr(const std::string& str, const typename std::enable_if< std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q>
inline std::string name(const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q>
inline std::string toStr(const Q& value, const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* =nullptr);
        
template<typename Q, typename T = typename remove_cvref<Q>::type>
inline T fromStr(const std::string& str, const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* =nullptr);
}
#pragma endregion Meta prototypes
// </Meta prototypes>

// <_Meta>
#pragma region _Meta
namespace _Meta {
        
/**
    * These methods return the appropriate type name as a string.
    * Such as "vector<pair<int,float>>".
    */
//template<class Q> std::string name(const typename std::enable_if< std::is_integral<typename std::remove_cv<Q>::type>::value && !std::is_same<bool, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return sizeof(Q) == sizeof(long) ? "long" : "int"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<int, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "int"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<short, typename std::remove_cv<Q>::type>::value && !std::is_same<int16_t, short>::value, Q >::type* =nullptr) { return "short"; }
template<class Q> std::string name(const typename std::enable_if< !std::is_same<int32_t, int>::value && std::is_same<int32_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "int32"; }
template<class Q> std::string name(const typename std::enable_if< !std::is_same<uint32_t, unsigned int>::value && std::is_same<uint32_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "uint32"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<int16_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "int16"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<uint16_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "uint16"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<unsigned int, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "uint"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<unsigned short, typename std::remove_cv<Q>::type>::value && !std::is_same<uint16_t, unsigned short>::value, Q >::type* =nullptr) { return "ushort"; }
template<class Q> std::string name(const typename std::enable_if< !std::is_same<uint64_t, unsigned long>::value && std::is_same<uint64_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "uint64"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<unsigned long, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "ulong"; }
template<class Q> std::string name(const typename std::enable_if< !std::is_same<int64_t, long>::value && std::is_same<int64_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "int64"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<long, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "long"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<uint8_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "uchar"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<int8_t, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "char"; }
    
template<class Q> std::string name(const typename std::enable_if< std::is_floating_point<typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return sizeof(double) == sizeof(Q) ? "double" : "float"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<std::string, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "string"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<std::wstring, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "wstring"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<bool, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "bool"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<cv::Mat, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "mat"; }
template<class Q> std::string name(const typename std::enable_if< std::is_same<cv::Range, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "range"; }
        
/**
    * chrono:: time objects
    */
template<class Q> std::string name(const typename std::enable_if< std::is_same<DurationUS, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "time"; }
        
template<class Q>
    requires _has_class_name<Q>
std::string name() {
    return Q::class_name();
}
    
template< template < typename...> class Tuple, typename ...Ts >
std::string tuple_name (Tuple< Ts... >&& tuple)
{        
    std::stringstream ss;
    ss << "tuple<";
    std::apply([&](auto&&... args) {
        ((ss << Meta::name<decltype(args)>() << ","), ...);
    }, std::forward<Tuple<Ts...>>(tuple));
        
    ss << ">";
        
    return ss.str();
}
    
template<class Q>
    requires (is_instantiation<std::tuple, Q>::value)
std::string name() { 
    return tuple_name(Q{});
}
    
template<class Q>
std::string name(const typename std::enable_if< is_container<Q>::value, Q >::type* =nullptr) {
    return "array<"+Meta::name<typename Q::value_type>()+">";
}
template<class Q>
std::string name(const typename std::enable_if< is_queue<Q>::value, Q >::type* =nullptr) {
    return "queue<"+Meta::name<typename Q::value_type>()+">";
}
template<class Q>
std::string name(const typename std::enable_if< is_set<Q>::value, Q >::type* =nullptr) {
    return "set<"+Meta::name<typename Q::value_type>()+">";
}
template<class Q>
std::string name(const typename std::enable_if< is_map<Q>::value, Q >::type* =nullptr) {
    return "map<"+Meta::name<typename Q::key_type>()+","+name<typename Q::mapped_type>()+">";
}
template<class Q>
std::string name(const typename std::enable_if< is_pair<Q>::value, Q >::type* =nullptr) {
    return "pair<"+Meta::name<typename Q::first_type>()
            +","+Meta::name<typename Q::second_type>()+">";
}
template<class Q>
    requires (is_instantiation<cv::Size_, Q>::value)
std::string name() {
    return "size<"+Meta::name<typename Q::value_type>()+">";
}
template<class Q>
    requires (is_instantiation<cv::Rect_, Q>::value)
std::string name() {
    return "rect<"+Meta::name<typename Q::value_type>()+">";
}
        
/**
    * The following methods convert any given value to a string.
    * All objects should be parseable by JavaScript as JSON.
    */
        
template<class Q>
    requires (is_container<Q>::value || is_set<Q>::value || is_deque<Q>::value)
std::string toStr(const Q& obj) {
    std::stringstream ss;
    auto start = obj.begin(), end = obj.end();
    for(auto it=start; it != end; ++it) {
        if(it != start)
            ss << ",";
        ss << Meta::toStr(*it);
    }
    return "[" + ss.str() + "]";
}
    
template<class Q>
    requires (is_queue<Q>::value) && (!is_deque<Q>::value)
std::string toStr(const Q& obj) {
    return "queue<size:"+Meta::toStr(obj.size())+">";
}
        
template<class Q>
    requires (std::convertible_to<Q, std::string> || (std::is_constructible_v<std::string, Q>))
        && (!(is_instantiation<std::tuple, Q>::value))
        && (!_has_tostr_method<Q>)
std::string toStr(const Q& obj) {
    return "\"" + util::escape(std::string(obj)) + "\"";
}
        
template<class Q>
    requires _clean_same<cv::Range, Q>
std::string toStr(const Q& obj) {
    return "[" + Meta::toStr(obj.start) + "," + Meta::toStr(obj.end) + "]";
}
        
/*template<class Q, class type>
std::string toStr(const cmn::Range<type>& obj) {
    return "[" + Meta::toStr(obj.start) + "," + Meta::toStr(obj.end) + "]";
}*/

template<class Q>
    requires _clean_same<bool, Q>
std::string toStr(Q obj) {
    return obj == true ? "true" : "false";
}
        
template<class TupType, size_t... I>
std::string tuple_str(const TupType& _tup, std::index_sequence<I...>)
{
            
    std::stringstream str;
    str << "[";
    (..., (str << (I == 0? "" : ",") << Meta::toStr(std::get<I>(_tup))));
    str << "]";
    return str.str();
}
        
template<class... Q>
std::string tuple_str (const std::tuple<Q...>& _tup)
{
    return tuple_str(_tup, std::make_index_sequence<sizeof...(Q)>());
}
        
template<class Q>
    requires (is_instantiation<std::tuple, Q>::value)
std::string toStr(const Q& obj) {
    return tuple_str(obj);
}
        
template<class Q>
    requires _is_number<Q>
std::string toStr(const Q& obj) {
    return util::to_string(obj);
}
        
template<class Q>
    requires _clean_same<std::wstring, Q>
std::string toStr(const Q& obj) {
    return ws2s(obj);
}
    
template<class Q>
    requires (is_pair<Q>::value)
std::string toStr(const Q& obj) {
    return "[" + Meta::toStr(obj.first) + "," + Meta::toStr(obj.second) + "]";
}
        
template<class Q>
    requires (is_map<Q>::value)
std::string toStr(const Q& obj) {
    std::stringstream ss;
    ss << "{";
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if(it != obj.begin())
            ss << ',';
        ss << Meta::toStr(it->first);
        ss << ':';
        ss << Meta::toStr(it->second);
                
    }
    ss << "}";
            
    return ss.str();
}
        
template<class Q>
    requires (is_instantiation<cv::Rect_, Q>::value)
std::string toStr(const Q& obj) {
    return "[" + Meta::toStr(obj.x) + "," + Meta::toStr(obj.y) + "," + Meta::toStr(obj.width) + "," + Meta::toStr(obj.height) + "]";
}
        
template<class Q>
    requires _is_smart_pointer<Q> && _has_class_name<typename Q::element_type>
std::string toStr(const Q& obj) {
    using K = typename Q::element_type;
    return "ptr<"+K::class_name() + ">" + (obj == nullptr ? "null" : Meta::toStr<K>(*obj));
}

template<class Q, 
    class C = typename std::remove_reference<Q>::type, 
    class K = typename std::remove_pointer<C>::type>
  requires _is_dumb_pointer<C> && _has_class_name<K> && _has_tostr_method<K>
std::string toStr(C obj) {
    return "ptr<"+K::class_name()+">" + obj->toStr();
}
        
template<class Q>
    requires _has_tostr_method<Q>
std::string toStr(const Q& obj) {
    return obj.toStr();
}
    
template<class Q>
    requires _is_smart_pointer<Q> && (!_has_tostr_method<typename Q::element_type>)
std::string toStr(const Q& obj) {
    return "ptr<?>0x" + Meta::toStr(uint64_t(obj.get()));//MetaType<typename std::remove_pointer<typename Q::element_type>::type>::toStr(*obj);
}
        
template<class Q>
    requires (is_instantiation<cv::Size_, Q>::value)
std::string toStr(const Q& obj) {
    return "[" + Meta::toStr(obj.width) + "," + Meta::toStr(obj.height) + "]";
}
        
template<class Q>
    requires _clean_same<cv::Mat, Q>
std::string toStr(const Q& obj)
{
    auto mat = *((const cv::Mat*)&obj);
    std::stringstream ss;
            
    ss << "[";
    for (int i=0; i<mat.rows; i++) {
        if(mat.rows > 15 && (i > 5 && i < mat.rows-6)) {
            //if(i < 7 || i > mat.rows-8)
            //    ss << ",";
            //continue;
        }
                
        ss << "[";
        for (int j=0; j<mat.cols; j++) {
            if(mat.cols > 15 && (j > 5 && j < mat.cols-6)) {
                //if(j < 8 || j > mat.cols-9)
                //    ss << ",";
                //continue;
            }
                    
            switch(mat.type()) {
                case CV_8UC1:
                    ss << mat.at<uchar>(i, j);
                    break;
                case CV_32FC1:
                    ss << mat.at<float>(i, j);
                    break;
                case CV_64FC1:
                    ss << mat.at<double>(i, j);
                    break;
                default:
                    U_EXCEPTION("unknown matrix type");
            }
            if(j < mat.cols-1)
                ss << ",";
        }
        ss << "]";
        if (i < mat.rows-1) {
            ss << ",";
        }
    }
            
    ss << "]";
    return ss.str();
}
        
/**
    * The following methods convert a string like "5.0" to the native
    * datatype (in this case float or double).
    * Values in the string are expected to represent datatype T.
    * Invalid values will throw exceptions.
    */
template<class Q>
    requires std::signed_integral<typename std::remove_cv<Q>::type> 
             && (!_clean_same<bool, Q>)
Q fromStr(const std::string& str)
{
    if(!str.empty() && str[0] == '\'' && str.back() == '\'')
        return Q(std::stoll(str.substr(1,str.length()-2)));
    return Q(std::stoll(str));
}
template<class Q>
    requires std::unsigned_integral<typename std::remove_cv<Q>::type> 
             && (!_clean_same<bool, Q>)
Q fromStr(const std::string& str)
{
    if (!str.empty() && str[0] == '\'' && str.back() == '\'')
        return Q(std::stoull(str.substr(1, str.length() - 2)));
    return Q(std::stoull(str));
}
    
template<class Q>
    requires _has_fromstr_method<Q>
Q fromStr(const std::string& str) {
    return Q::fromStr(str);
}
        
template<class Q>
    requires std::floating_point<typename std::remove_cv<Q>::type>
Q fromStr(const std::string& str)
{
    if(!str.empty() && str[0] == '\'' && str.back() == '\'')
        return Q(std::stod(str.substr(1,str.length()-2)));
    return (Q)std::stod(str);
}
        
template<class Q>
    requires (is_pair<typename std::remove_cv<Q>::type>::value)
Q fromStr(const std::string& str)
{
    using namespace util;
    auto parts = parse_array_parts(truncate(str));
    if(parts.size() != 2) {
        std::string x = name<typename Q::first_type>();
        std::string y = name<typename Q::second_type>();
        throw CustomException<std::invalid_argument>("Illegal pair<%S, %S> format ('%S').", &x, &y, &str);
    }
            
    auto x = Meta::fromStr<typename Q::first_type>(parts[0]);
    auto y = Meta::fromStr<typename Q::second_type>(parts[1]);
            
    return Q(x, y);
}
        
template<class Q>
    requires _clean_same<std::string, Q>
Q fromStr(const std::string& str)
{
    if((utils::beginsWith(str, '"') && utils::endsWith(str, '"'))
        || (utils::beginsWith(str, '\'') && utils::endsWith(str, '\'')))
        return util::unescape(utils::trim(str.substr(1, str.length()-2)));
    return util::unescape(utils::trim(str));
}
    
template<class Q>
    requires _clean_same<std::wstring, Q>
Q fromStr(const std::string& str)
{
    return s2ws(str);
}
        
template<class Q>
    requires _clean_same<DurationUS, Q>
Q fromStr(const std::string& str)
{
    return DurationUS::fromStr(str);
}
        
template<class Q>
    requires (is_container<Q>::value || is_deque<Q>::value)
Q fromStr(const std::string& str)
{
    std::vector<typename Q::value_type> ret;
    std::queue<char> brackets;
    std::stringstream value;
            
    auto parts = util::parse_array_parts(util::truncate(str));
    for(auto &s : parts) {
        if(s.empty()) {
            ret.push_back(typename Q::value_type());
            Warning("Empty value in '%S'.", &str);
        }
        else {
            auto v = Meta::fromStr<typename Q::value_type>(s);
            ret.push_back(v);
        }
    }
            
    return ret;
}
    
template<class Q>
    requires (!is_deque<Q>::value && is_queue<Q>::value)
Q fromStr(const std::string& str)
{
    std::vector<typename Q::value_type> ret;
    std::queue<char> brackets;
    std::stringstream value;
            
    auto parts = util::parse_array_parts(util::truncate(str));
    for(auto &s : parts) {
        if(s.empty()) {
            ret.push(typename Q::value_type());
            Warning("Empty value in '%S'.", &str);
        }
        else {
            auto v = Meta::fromStr<typename Q::value_type>(s);
            ret.push(v);
        }
    }
            
    return ret;
}
        
template<class Q>
    requires (is_set<Q>::value)
Q fromStr(const std::string& str)
{
    std::set<typename Q::value_type> ret;
    std::queue<char> brackets;
    std::stringstream value;
            
    auto parts = util::parse_array_parts(util::truncate(str));
    for(auto &s : parts) {
        if(s.empty()) {
            ret.insert(typename Q::value_type());
            Warning("Empty value in '%S'.", &str);
        }
        else {
            auto v = Meta::fromStr<typename Q::value_type>(s);
            ret.insert(v);
        }
    }
            
    return ret;
}
        
template<class Q>
    requires (is_map<Q>::value)
Q fromStr(const std::string& str)
{
    Q r;
            
    auto parts = util::parse_array_parts(util::truncate(str));
            
    for(auto &p: parts) {
        auto value_key = util::parse_array_parts(p, ':');
        if(value_key.size() != 2)
            throw CustomException<std::invalid_argument>("Illegal value/key pair: '%S'", &p);
                
        auto x = Meta::fromStr<typename Q::key_type>(value_key[0]);
        try {
            auto y = Meta::fromStr<typename Q::mapped_type>(value_key[1]);
            r[x] = y;
        } catch(const std::logic_error&) {
            auto name = Meta::name<Q>();
            Warning("Empty/illegal value in %S['%S'] = '%S'", &name, &value_key[0], &value_key[1]);
        }
    }
            
    return r;
}

// <_Meta::detail>
#pragma region _Meta::detail
namespace detail {

template <class F, typename... Args, size_t... Is>
auto transform_each_impl(const std::tuple<Args...>& t, F&& f, std::index_sequence<Is...>) {
    return std::make_tuple(
        f(Is, std::get<Is>(t) )...
    );
}

}
#pragma endregion _Meta::detail
// </_Meta::detail>
    
template <class F, typename... Args>
auto transform_each(const std::tuple<Args...>& t, F&& f) {
    return detail::transform_each_impl(
        t, std::forward<F>(f), std::make_index_sequence<sizeof...(Args)>{});
}
    
template<class Q>
    requires (is_instantiation<std::tuple, Q>::value)
Q fromStr(const std::string& str) {
    auto parts = util::parse_array_parts(util::truncate(str));
    //size_t i=0;
    if(parts.size() != std::tuple_size<Q>::value)
        throw CustomException<illegal_syntax>("tuple has %d parts instead of %d.",parts.size(), std::tuple_size<Q>::value);
            
    Q tup;
    return transform_each(tup, [&](size_t i, auto&& obj) {
        return Meta::fromStr<decltype(obj)>(parts.at(i));
    });
}
        
template<class Q>
    requires _clean_same<cv::Mat, Q>
Q fromStr(const std::string&)
{
    U_EXCEPTION("Not supported.");
}
        
template<class Q>
    requires _clean_same<bool, Q>
Q fromStr(const std::string& str)
{
    return str == "true" ? true : false;
}
        
template<class Q>
    requires _clean_same<cv::Range, Q>
Q fromStr(const std::string& str)
{
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 2) {
        throw CustomException<std::invalid_argument>("Illegal cv::Range format.");
    }
            
    int x = Meta::fromStr<int>(parts[0]);
    int y = Meta::fromStr<int>(parts[1]);
            
    return cv::Range(x, y);
}
        
template<typename ValueType, size_t N, typename _names>
Enum<ValueType, N, _names> fromStr(const std::string& str, const Enum<ValueType, N, _names>* =nullptr)
{
    return Enum<ValueType, N, _names>::get(Meta::fromStr<std::string>(str));
}
        
template<class Q>
    requires _clean_same<const Q&, decltype(Q::get(std::string_view()))>
const Q& fromStr(const std::string& str) {
    return Q::get(Meta::fromStr<std::string>(str));
}
        
/*template<class Q>
    requires _clean_same<Range<double>, Q>
Q fromStr(const std::string& str)
{
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 2) {
        throw CustomException<std::invalid_argument>("Illegal Rangef format.");
    }
            
    auto x = Meta::fromStr<double>(parts[0]);
    auto y = Meta::fromStr<double>(parts[1]);
            
    return Range<double>(x, y);
}
    
template<class Q>
    requires _clean_same<Range<float>, Q>
Q fromStr(const std::string& str)
{
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 2) {
        throw CustomException<std::invalid_argument>("Illegal Rangef format.");
    }
            
    auto x = Meta::fromStr<float>(parts[0]);
    auto y = Meta::fromStr<float>(parts[1]);
            
    return Rangef(x, y);
}
    
template<class Q>
    requires _clean_same<Range<long_t>, Q>
Q fromStr(const std::string& str)
{
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 2) {
        throw CustomException<std::invalid_argument>("Illegal Rangel format.");
    }
            
    auto x = Meta::fromStr<long_t>(parts[0]);
    auto y = Meta::fromStr<long_t>(parts[1]);
            
    return Range(x, y);
}*/
        
template<class Q>
    requires (is_instantiation<cv::Size_, Q>::value)
Q fromStr(const std::string& str)
{
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 2) {
        std::string x = name<typename Q::value_type>();
        throw CustomException<std::invalid_argument>("Illegal cv::Size_<%S> format.", &x);
    }
            
    auto x = Meta::fromStr<typename Q::value_type>(parts[0]);
    auto y = Meta::fromStr<typename Q::value_type>(parts[1]);
            
    return cv::Size_<typename Q::value_type>(x, y);
}
        
template<class Q>
    requires (is_instantiation<cv::Rect_, Q>::value)
Q fromStr(const std::string& str)
{
    using C = typename Q::value_type;
            
    auto parts = util::parse_array_parts(util::truncate(str));
    if(parts.size() != 4) {
        std::string x = name<C>();
        throw CustomException<std::invalid_argument>("Illegal cv::Rect_<%S> format.", &x);
    }
            
    auto x = Meta::fromStr<C>(parts[0]);
    auto y = Meta::fromStr<C>(parts[1]);
    auto w = Meta::fromStr<C>(parts[2]);
    auto h = Meta::fromStr<C>(parts[3]);
            
    return Q(x, y, w, h);
}

}
#pragma endregion _Meta
// </_Meta>

// <Meta implementation>
#pragma region Meta implementation
namespace Meta {

template<typename Q>
inline std::string name(const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* ) {
    return _Meta::name<typename remove_cvref<Q>::type>();
}
        
template<typename Q>
inline std::string toStr(const Q& value, const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* ) {
    return _Meta::toStr<typename remove_cvref<Q>::type>(value);
}
        
template<typename Q, typename T>
inline T fromStr(const std::string& str, const typename std::enable_if< !std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* ) {
    return _Meta::fromStr<T>(str);
}
        
template<typename Q, typename T>
inline T fromStr(const std::string& str, const typename std::enable_if< std::is_pointer<Q>::value, typename remove_cvref<Q>::type >::type* ) {
    return new typename std::remove_pointer<typename remove_cvref<Q>::type>(str);
}
        
template<typename Q>
inline std::string name(const typename std::enable_if< std::is_pointer<Q>::value && std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* ) {
    return "c_str";
}
        
template<typename Q>
inline std::string toStr(Q value, const typename std::enable_if< std::is_pointer<Q>::value && std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* ) {
    return Meta::toStr(std::string(value));
}
        
template<typename Q>
inline std::string name(const typename std::enable_if< std::is_pointer<Q>::value && !std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* ) {
    return Meta::name<typename std::remove_pointer<typename remove_cvref<Q>::type>::type>();
}
        
template<typename Q>
inline std::string toStr(Q value, const typename std::enable_if< std::is_pointer<Q>::value && !std::is_same<Q, const char*>::value, typename remove_cvref<Q>::type >::type* ) {
    return "("+Meta::name<Q>()+"*)"+Meta::toStr<const typename std::remove_pointer<typename remove_cvref<Q>::type>::type&>(*value);
}
}
#pragma endregion Meta implementation
// </Meta implementation>

}

