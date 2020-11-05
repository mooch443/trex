#pragma once

#include <types.h>
#include <file/Path.h>
#include <gui/types/Basic.h>
#include <misc/MetaObject.h>
#include <gui/types/Basic.h>

namespace cmn {
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
            return index_apply<std::tuple_size<Tuple>{}>(
                [&](auto... Is) { return f(Is..., std::get<Is>(t)...); });
        }
    }
    
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
            
            double scaled = timestamp, previous_scaled = 0;
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
            
            double scaled = timestamp, previous_scaled = 0;
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
    };
    
    struct FileSize {
        size_t bytes;
        
        FileSize(size_t b = 0) : bytes(b) {}
        
        std::string to_string() const {
            std::vector<std::string> descriptions = {
                "bytes", "KB", "MB", "GB", "TB"
            };
            
            size_t i=0;
            double scaled = bytes;
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
    };
    
    inline std::string truncate(const std::string& str) {
        if((utils::beginsWith(str, '{') && utils::endsWith(str, '}'))
           || (utils::beginsWith(str, '[') && utils::endsWith(str, ']')))
            return utils::trim(str.substr(1, str.length()-2));
        
        throw CustomException<illegal_syntax> ("Cannot parse array '%S'", &str);
    }

    inline std::string escape(std::string str) {
        str = utils::find_replace(str, "\\", "\\\\");
        return utils::find_replace(str, "\"", "\\\"");
    }

    inline std::string unescape(std::string str) {
        str = utils::find_replace(str, "\\\"", "\"");
        return utils::find_replace(str, "\\\\", "\\");
    }

    inline std::vector<std::string> parse_array_parts(const std::string& str, const char delimiter=',') {
        std::deque<char> brackets;
        std::stringstream value;
        std::vector<std::string> ret;
        
        char prev = 0;
        for(size_t i=0; i<str.length(); i++) {
            char c = str.at(i);
            bool in_string = !brackets.empty() && (brackets.front() == '\'' || brackets.front() == '"');
            
            if(in_string) {
                auto s = value.str();
                if(prev != '\\' && c == brackets.front()) {
                    brackets.pop_front();
                }
                
                value << c;
                
            } else {
                switch(c) {
                    case '[':
                    case '{':
                        brackets.push_front(c);
                        break;
                        
                    case '"':
                    case '\'':
                        brackets.push_front(c);
                        break;
                    case ']':
                        if(!brackets.empty() && brackets.front() == '[')
                            brackets.pop_front();
                        break;
                    case '}':
                        if(!brackets.empty() && brackets.front() == '{')
                            brackets.pop_front();
                        break;
                    default:
                        break;
                }
                
                if(brackets.empty()) {
                    if(c == delimiter) {
                        ret.push_back(utils::trim(value.str()));
                        value.str("");
                    } else
                        value << c;
                    
                } else {
                    value << c;
                }
            }
            
            if(prev == '\\' && c == '\\')
                prev = 0;
            else
                prev = c;
        }
        
        std::string s = utils::trim(value.str());
        if(!s.empty()) {
            ret.push_back(s);
        }
        
        return ret;
    }

    template<template<class...>class Template, class T>
    struct is_instantiation : std::false_type {};
    template<template<class...>class Template, class... Ts>
    struct is_instantiation<Template, Template<Ts...>> : std::true_type {};

    namespace util {
        template <typename T>
        std::string to_string(const T& t, const typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true) {
            std::string str{std::to_string (t)};
            size_t offset = min(str.size()-1, size_t(1));
            if (str.find_last_not_of('0') == str.find('.')) {
                offset = 0;
            }
            
            str.erase(str.find_last_not_of('0') + offset, std::string::npos);
            return str;
        }
        
        template <typename T>
        std::string to_string(const T& t, const typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = true) {
            return std::to_string (t);
        }
    }
    
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
        
        template <typename T>
        struct has_tostr_method;
    
        template <typename T>
        struct has_internal_tostr_method;
    
        template <typename T>
        struct has_fromstr_method;
    }

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
        template<class Q> std::string name(const typename std::enable_if< std::is_same<bool, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "bool"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<cv::Mat, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "mat"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<cv::Range, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "range"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<Rangef, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "rangef"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<Range<double>, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "range<double>"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<Rangel, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "rangel"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<file::Path, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "path"; }
        template<class Q> std::string name(const typename std::enable_if< std::is_same<FileSize, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "filesize"; }
        
        /**
         * chrono:: time objects
         */
        template<class Q> std::string name(const typename std::enable_if< std::is_same<DurationUS, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr) { return "time"; }
        
        template<class Q>
        std::string name(const typename std::enable_if< std::is_convertible<Q, MetaObject>::value, Q >::type* =nullptr) {
            //MetaObject obj = Q();
            //return obj.class_name();
            return Q::class_name();
        }
    
    /*template<class TupType, size_t... I>
    std::string tuple_name(const TupType&&, std::index_sequence<I...>)
    {
        
        std::stringstream str;
        str << "tuple<";
        (..., (str << (I == 0? "" : ",") << Meta::name<typename std::tuple_element<I, TupType>::type>()));
        str << ">";
        return str.str();
    }*/
    
    //template<class... Q>
    template< template < typename...> class Tuple, typename ...Ts >
    std::string tuple_name (Tuple< Ts... >&& tuple)
    {
        /*std::initializer_list<std::string> strs{
            (Meta::name<Ts>(), 0)...
        };*/
        
        std::stringstream ss;
        ss << "tuple<";
        std::apply([&](auto&&... args) {
            ((ss << Meta::name<decltype(args)>() << ","), ...);
        }, std::forward<Tuple<Ts...>>(tuple));
        
        ss << ">";
        
        return ss.str();
        //return tuple_name<Q...> (std::make_index_sequence<sizeof...(Q)>());
    }
    
    template<class Q>
    std::string name(const typename std::enable_if< is_instantiation<std::tuple, Q>::value, Q >::type* =nullptr) {
    //std::string name(const typename std::enable_if< is_instantiation<std::tuple, Q>::value, Q >::type* =nullptr) {
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
        std::string name(const typename std::enable_if< is_instantiation<cv::Size_, Q>::value, Q >::type* =nullptr) {
            return "size<"+Meta::name<typename Q::value_type>()+">";
        }
        template<class Q>
        std::string name(const typename std::enable_if< is_instantiation<cv::Rect_, Q>::value, Q >::type* =nullptr) {
            return "rect<"+Meta::name<typename Q::value_type>()+">";
        }
        template<class Q>
        std::string name(const typename std::enable_if< std::is_same<Vec2, Q>::value, Q >::type* =nullptr) {
            return "vec";
        }
        template<class Q>
        std::string name(const typename std::enable_if< std::is_same<gui::Color, Q>::value, Q >::type* =nullptr) {
            return "color";
        }
        template<class Q>
        std::string name(const typename std::enable_if< std::is_same<Bounds, Q>::value, Q >::type* =nullptr) {
            return "bounds";
        }
        template<class Q>
        std::string name(const typename std::enable_if< std::is_same<Size2, Q>::value, Q >::type* =nullptr) {
            return "size";
        }
        
        /**
         * The following methods convert any given value to a string.
         * All objects should be parseable by JavaScript as JSON.
         */
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_container<Q>::value || is_set<Q>::value || is_deque<Q>::value, Q >::type& obj) {
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
        std::string toStr(const typename std::enable_if< is_queue<Q>::value && !is_deque<Q>::value, Q >::type& obj) {
            return "queue<size:"+Meta::toStr(obj.size())+">";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_convertible<typename std::remove_cv<Q>::type, std::string>::value, Q >::type& obj) {
            return "\"" + escape(obj) + "\"";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_convertible<typename std::remove_cv<Q>::type, MetaObject>::value, Q >::type& obj) {
            MetaObject m(obj);
            return m.value();
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<typename std::remove_cv<Q>::type, file::Path>::value, Q >::type& obj) {
            return Meta::toStr(obj.str());
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<typename std::remove_cv<Q>::type, DurationUS>::value || std::is_same<Q, FileSize>::value, Q >::type& obj) {
            return obj.to_string();
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<typename std::remove_cv<Q>::type, cv::Range>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.start) + "," + Meta::toStr(obj.end) + "]";
        }
        
        template<class Q, class type>
        std::string toStr(const cmn::Range<type>& obj) {
            return "[" + Meta::toStr(obj.start) + "," + Meta::toStr(obj.end) + "]";
        }
        
        template<class Q>
        std::string toStr(const cmn::FrameRange& obj) {
            return "[" + Meta::toStr(obj.start()) + "," + Meta::toStr(obj.end()) + "]";
        }
        
        template<class Q>
        std::string toStr(typename std::enable_if< std::is_same<typename std::remove_cv<Q>::type, bool>::value, Q >::type obj) {
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
        std::string toStr(const typename std::enable_if< is_instantiation<std::tuple, Q>::value, Q >::type& obj) {
            return tuple_str(obj);
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< !std::is_same<typename std::remove_cv<Q>::type, bool>::value && std::is_same<std::string, decltype(std::to_string(typename std::remove_cv<Q>::type())) >::value, Q >::type& obj) {
            return util::to_string(obj);
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_pair<Q>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.first) + "," + Meta::toStr(obj.second) + "]";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_map<Q>::value, Q >::type& obj) {
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
        
        template<class T, class Q = typename std::remove_cv<T>::type>
        std::string toStr(const typename std::enable_if< std::is_same<Float2<true>, Q>::value || std::is_same<Float2<false>, Q>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.A()) + "," + Meta::toStr(obj.B()) + "]";
        }
    
        template<class T, class Q = typename std::remove_cv<T>::type>
        std::string toStr(const typename std::enable_if< std::is_same<gui::Color, Q>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.r) + "," + Meta::toStr(obj.g) + "," + Meta::toStr(obj.b) + "," + Meta::toStr(obj.a) + "]";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_instantiation<cv::Rect_, Q>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.x) + "," + Meta::toStr(obj.y) + "," + Meta::toStr(obj.width) + "," + Meta::toStr(obj.height) + "]";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_instantiation<std::shared_ptr, Q>::value && (Meta::has_tostr_method<typename Q::element_type>::value || std::is_convertible<typename Q::element_type, MetaObject>::value), Q >::type& obj) {
            return "ptr<"+Meta::name<typename Q::element_type>()+">" + (obj == nullptr ? "null" : Meta::toStr<typename Q::element_type>(*obj));//MetaType<typename std::remove_pointer<typename Q::element_type>::type>::toStr(*obj);
        }
    
        template<class Q>
        std::string toStr(const typename std::enable_if< is_instantiation<std::shared_ptr, Q>::value && (Meta::has_internal_tostr_method<typename Q::element_type>::value), Q >::type& obj) {
            return "ptr<"+Q::class_name()+">" + (obj == nullptr ? "null" : obj->toStr());
        }
    
        template<class Q>
        std::string toStr(const typename std::enable_if< !is_instantiation<std::shared_ptr, Q>::value && (Meta::has_internal_tostr_method<Q>::value), Q >::type& obj) {
            return "ptr<"+Q::class_name()+">" + obj.toStr();
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_instantiation<std::shared_ptr, Q>::value && !Meta::has_tostr_method<typename Q::element_type>::value && !std::is_convertible<typename Q::element_type, MetaObject>::value, Q >::type& obj) {
            return "ptr<?>0x" + Meta::toStr(uint64_t(obj.get()));//MetaType<typename std::remove_pointer<typename Q::element_type>::type>::toStr(*obj);
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<Bounds, typename std::remove_cv<Q>::type>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.x) + "," + Meta::toStr(obj.y) + "," + Meta::toStr(obj.width) + "," + Meta::toStr(obj.height) + "]";
        }
        
        /*template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<pv::Blob, typename std::remove_cv<Q>::type>::value, Q >::type& obj) {
            typedef MetaType<long> M;
            return "blob{" + M::toStr(obj.blob_id()) + "}";
        }*/
        
        template<class Q>
        std::string toStr(const typename std::enable_if< is_instantiation<cv::Size_, Q>::value, Q >::type& obj) {
            return "[" + Meta::toStr(obj.width) + "," + Meta::toStr(obj.height) + "]";
        }
        
        template<class Q>
        std::string toStr(const typename std::enable_if< std::is_same<cv::Mat, typename std::remove_cv<Q>::type>::value, Q >::type& obj)
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
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_integral<typename std::remove_cv<Q>::type>::value && !std::is_same<bool, Q>::value, Q >::type* =nullptr)
        {
            if(!str.empty() && str[0] == '\'' && str.back() == '\'')
                return Q(std::stol(str.substr(1,str.length()-2)));
            return Q(std::stol(str));
        }
    
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if<Meta::has_fromstr_method<Q>::value, Q>::type * = nullptr) {
            return Q::fromStr(str);
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_floating_point<typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            if(!str.empty() && str[0] == '\'' && str.back() == '\'')
                return Q(std::stod(str.substr(1,str.length()-2)));
            return (Q)std::stod(str);
        }
        
        template<class T, class Q = typename std::remove_cv<T>::type>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<Float2<true>, Q>::value || std::is_same<Float2<false>, Q>::value, Q >::type* =nullptr)
        {
            auto vec = Meta::fromStr<std::vector<float>>(str);
            if(vec.empty())
                return Q();
            if(vec.size() != 2)
                throw CustomException<std::invalid_argument>("Can only initialize Vec2 with two or no elements. ('%S')", &str);
            return Q(vec[0], vec[1]);
        }
        
        template<class T, class Q = typename std::remove_cv<T>::type>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<gui::Color, Q>::value, Q >::type* =nullptr)
        {
            auto s = utils::lowercase(str);
            if(s == "red") return gui::Red;
            if(s == "blue") return gui::Blue;
            if(s == "green") return gui::Green;
            if(s == "yellow") return gui::Yellow;
            if(s == "cyan") return gui::Cyan;
            if(s == "white") return gui::White;
            if(s == "black") return gui::Black;
            
            auto vec = Meta::fromStr<std::vector<uchar>>(str);
            if(vec.empty())
                return Q();
            if(vec.size() != 4 && vec.size() != 3)
                throw CustomException<std::invalid_argument>("Can only initialize Color with three or four elements. ('%S')", &str);
            return Q(vec[0], vec[1], vec[2], vec.size() == 4 ? vec[3] : 255);
        }
    
        template<class T, class Q = typename std::remove_cv<T>::type>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<Bounds, Q>::value, Q >::type* =nullptr)
        {
            auto vec = Meta::fromStr<std::vector<float>>(str);
            if(vec.empty())
                return Q();
            if(vec.size() != 4)
                throw CustomException<std::invalid_argument>("Can only initialize Bounds with exactly four or no elements. ('%S')", &str);
            return Q(vec[0], vec[1], vec[2], vec[3]);
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< is_pair<Q>::value, Q >::type* =nullptr)
        {
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
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<std::string, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            if((utils::beginsWith(str, '"') && utils::endsWith(str, '"'))
               || (utils::beginsWith(str, '\'') && utils::endsWith(str, '\'')))
                return unescape(utils::trim(str.substr(1, str.length()-2)));
            return unescape(utils::trim(str));
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<file::Path, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            return file::Path(Meta::fromStr<std::string>(str));
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<DurationUS, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            return DurationUS::fromStr(str);
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< is_container<Q>::value || is_deque<Q>::value, Q >::type* =nullptr)
        {
            std::vector<typename Q::value_type> ret;
            std::queue<char> brackets;
            std::stringstream value;
            
            auto parts = parse_array_parts(truncate(str));
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
        Q fromStr(const std::string& str, const typename std::enable_if< !is_deque<Q>::value && is_queue<Q>::value, Q >::type* =nullptr)
        {
            std::vector<typename Q::value_type> ret;
            std::queue<char> brackets;
            std::stringstream value;
            
            auto parts = parse_array_parts(truncate(str));
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
        Q fromStr(const std::string& str, const typename std::enable_if< is_set<Q>::value, Q >::type* =nullptr)
        {
            std::set<typename Q::value_type> ret;
            std::queue<char> brackets;
            std::stringstream value;
            
            auto parts = parse_array_parts(truncate(str));
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
        Q fromStr(const std::string& str, const typename std::enable_if< is_map<Q>::value, Q >::type* =nullptr)
        {
            Q r;
            
            auto parts = parse_array_parts(truncate(str));
            
            for(auto &p: parts) {
                auto value_key = parse_array_parts(p, ':');
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
    
    namespace detail {

    template <class F, typename... Args, size_t... Is>
    auto transform_each_impl(const std::tuple<Args...>& t, F&& f, std::index_sequence<Is...>) {
        return std::make_tuple(
            f(Is, std::get<Is>(t) )...
        );
    }

    } // namespace detail
    
    template <class F, typename... Args>
    auto transform_each(const std::tuple<Args...>& t, F&& f) {
        return detail::transform_each_impl(
            t, std::forward<F>(f), std::make_index_sequence<sizeof...(Args)>{});
    }
    
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< is_instantiation<std::tuple, Q>::value, Q >::type* =nullptr) {
            auto parts = parse_array_parts(truncate(str));
            //size_t i=0;
            if(parts.size() != std::tuple_size<Q>::value)
                throw CustomException<illegal_syntax>("tuple has %d parts instead of %d.",parts.size(), std::tuple_size<Q>::value);
            
            Q tup;
            return transform_each(tup, [&](size_t i, auto&& obj) {
                return Meta::fromStr<decltype(obj)>(parts.at(i));
            });
        }
        
        template<class Q>
        Q fromStr(const std::string&, const typename std::enable_if< std::is_same<cv::Mat, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            U_EXCEPTION("Not supported.");
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<bool, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            return str == "true" ? true : false;
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<cv::Range, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            auto parts = parse_array_parts(truncate(str));
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
        const Q& fromStr(const std::string& str, const typename std::enable_if< std::is_same<const Q&, decltype(Q::get(std::string_view()))>::value, Q >::type** =nullptr)
        {
            return Q::get(Meta::fromStr<std::string>(str));
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<Range<double>, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            auto parts = parse_array_parts(truncate(str));
            if(parts.size() != 2) {
                throw CustomException<std::invalid_argument>("Illegal Rangef format.");
            }
            
            auto x = Meta::fromStr<double>(parts[0]);
            auto y = Meta::fromStr<double>(parts[1]);
            
            return Range<double>(x, y);
        }
    
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<Rangef, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            auto parts = parse_array_parts(truncate(str));
            if(parts.size() != 2) {
                throw CustomException<std::invalid_argument>("Illegal Rangef format.");
            }
            
            auto x = Meta::fromStr<float>(parts[0]);
            auto y = Meta::fromStr<float>(parts[1]);
            
            return Rangef(x, y);
        }
    
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< std::is_same<Rangel, typename std::remove_cv<Q>::type>::value, Q >::type* =nullptr)
        {
            auto parts = parse_array_parts(truncate(str));
            if(parts.size() != 2) {
                throw CustomException<std::invalid_argument>("Illegal Rangel format.");
            }
            
            auto x = Meta::fromStr<long_t>(parts[0]);
            auto y = Meta::fromStr<long_t>(parts[1]);
            
            return Range(x, y);
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< is_instantiation<cv::Size_, Q>::value, Q >::type* =nullptr)
        {
            auto parts = parse_array_parts(truncate(str));
            if(parts.size() != 2) {
                std::string x = name<typename Q::value_type>();
                throw CustomException<std::invalid_argument>("Illegal cv::Size_<%S> format.", &x);
            }
            
            auto x = Meta::fromStr<typename Q::value_type>(parts[0]);
            auto y = Meta::fromStr<typename Q::value_type>(parts[1]);
            
            return cv::Size_<typename Q::value_type>(x, y);
        }
        
        template<class Q>
        Q fromStr(const std::string& str, const typename std::enable_if< is_instantiation<cv::Rect_, Q>::value, Q >::type* =nullptr)
        {
            using C = typename Q::value_type;
            
            auto parts = parse_array_parts(truncate(str));
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
        
        template <typename T>
        struct has_tostr_method
        {
            struct dummy {  };
            
            template <typename C, typename P>
            static auto test(P * p) -> decltype(static_cast<void>(sizeof(decltype(_Meta::name<C>()))), std::true_type());
            
            template <typename, typename>
            static std::false_type test(...);
            
            typedef decltype(test<T, dummy>(nullptr)) type;
            static const bool value = std::is_same<std::true_type, decltype(test<T, dummy>(nullptr))>::value;
        };
    
        template <typename T>
        struct has_internal_tostr_method
        {
            struct dummy {  };
            
            template <typename C, typename P>
            static auto test(C * p) -> decltype(static_cast<void>(sizeof(decltype(p->toStr()))), std::true_type());
            
            template <typename, typename>
            static std::false_type test(...);
            
            typedef decltype(test<T, dummy>(nullptr)) type;
            static const bool value = std::is_same<std::true_type, decltype(test<T, dummy>(nullptr))>::value;
        };
    
        template <typename T>
        struct has_fromstr_method
        {
            struct dummy {  };
            
            template <typename C, typename P>
            static auto test(P * p) -> decltype(static_cast<void>(sizeof(decltype(&C::fromStr))), std::true_type());
            
            template <typename, typename>
            static std::false_type test(...);
            
            typedef decltype(test<T, dummy>(nullptr)) type;
            static const bool value = std::is_same<std::true_type, decltype(test<T, dummy>(nullptr))>::value;
        };
    }
}

namespace cmn {
namespace tag {
struct warn_on_error {};
struct fail_on_error {};
}

template<typename T>
using try_make_unsigned =
typename std::conditional<
	std::is_integral<T>::value,
	std::make_unsigned<T>,
	double
>::type;

template<typename T>
using try_make_signed =
typename std::conditional<
	std::is_integral<T>::value,
	std::make_signed<T>,
	double
>::type;

template<typename To, typename From>
void fail_type(From&& value) {
    using FromType = typename remove_cvref<From>::type;
    using ToType = typename remove_cvref<To>::type;
    
    auto type1 = Meta::name<FromType>();
    auto type2 = Meta::name<ToType>();
    
    auto value1 = Meta::toStr(value);
    
    auto start1 = Meta::toStr(std::numeric_limits<FromType>::min());
    auto end1 = Meta::toStr(std::numeric_limits<FromType>::max());
    
    auto start2 = Meta::toStr(std::numeric_limits<ToType>::min());
    auto end2 = Meta::toStr(std::numeric_limits<ToType>::max());
    
    Warning("Failed converting %S(%S) [%S,%S] -> type %S [%S,%S]", &type1, &value1, &start1, &end1, &type2, &start2, &end2);
}

template<typename To, typename From>
constexpr To sign_cast(From&& value) {
    using FromType = typename remove_cvref<From>::type;
    using ToType = typename remove_cvref<To>::type;
    
    if constexpr(!std::is_floating_point<ToType>::value
                 && std::is_integral<ToType>::value)
    {
        if constexpr(std::is_signed<ToType>::value) {
            if constexpr(value > std::numeric_limits<ToType>::max())
                fail_type<To, From>(std::forward<FromType>(value));
            
        } else if constexpr(std::is_signed<FromType>::value) {
            if (value < 0)
                fail_type<To, From>(std::forward<From>(value));
            
            using bigger_type = typename std::conditional<(sizeof(FromType) > sizeof(ToType)), FromType, ToType>::type;
            if (bigger_type(value) > bigger_type(std::numeric_limits<ToType>::max()))
                fail_type<To, From>(std::forward<From>(value));
        }
    }
    
    return static_cast<To>(std::forward<From>(value));
}

template<typename To, typename From>
constexpr bool check_narrow_cast(const From& value) {
    using FromType = typename remove_cvref<From>::type;
    using ToType = typename remove_cvref<To>::type;

    auto str = Meta::toStr(value);
    if constexpr (
        std::is_floating_point<ToType>::value
        || (std::is_signed<FromType>::value == std::is_signed<ToType>::value && !std::is_floating_point<FromType>::value)
        )
    {
        // unsigned to unsigned
#ifdef _NARROW_PRINT_VERBOSE
        auto tstr0 = Meta::name<FromType>();
        auto tstr1 = Meta::name<ToType>();
        Debug("Narrowing %S -> %S (same) = %S.", &tstr0, &tstr1, &str);
#endif
        return true;
    }
    else if constexpr (std::is_floating_point<FromType>::value && std::is_signed<ToType>::value) {
        using signed_t = int64_t;
#ifdef _NARROW_PRINT_VERBOSE
        auto tstr0 = Meta::name<FromType>();
        auto tstr1 = Meta::name<ToType>();
        auto tstr2 = Meta::name<signed_t>();
        Debug("Narrowing %S -> %S | converting to %S and comparing (fs) = %S.", &tstr0, &tstr1, &tstr2, &str);
#endif
        return static_cast<signed_t>(value) >= static_cast<signed_t>(std::numeric_limits<To>::min())
            && static_cast<signed_t>(value) <= static_cast<signed_t>(std::numeric_limits<To>::max());
    }
    else if constexpr (std::is_floating_point<FromType>::value && std::is_unsigned<ToType>::value) {
        using unsigned_t = uint64_t;
#ifdef _NARROW_PRINT_VERBOSE
        auto tstr0 = Meta::name<FromType>();
        auto tstr1 = Meta::name<ToType>();
        auto tstr2 = Meta::name<unsigned_t>();
        Debug("Narrowing %S -> %S | converting to %S and comparing (fs) = %S.", &tstr0, &tstr1, &tstr2, &str);
#endif
        return value >= FromType(0)
            && static_cast<unsigned_t>(value) <= static_cast<unsigned_t>(std::numeric_limits<To>::max());
    }
    else if constexpr (std::is_unsigned<FromType>::value && std::is_signed<ToType>::value) {
        // unsigned to signed
        using signed_t = int64_t;
#ifdef _NARROW_PRINT_VERBOSE
        auto tstr0 = Meta::name<FromType>();
        auto tstr1 = Meta::name<ToType>();
        auto tstr2 = Meta::name<signed_t>();
        Debug("Narrowing %S -> %S | converting to %S and comparing (us) = %S.", &tstr0, &tstr1, &tstr2, &str);
#endif
        return static_cast<signed_t>(value) < static_cast<signed_t>(std::numeric_limits<To>::max());

    }
    else {
        static_assert(std::is_signed<FromType>::value && std::is_unsigned<ToType>::value, "Expecting signed to unsigned conversion");
        // signed to unsigned
        using unsigned_t = typename try_make_unsigned<FromType>::type;
#ifdef _NARROW_PRINT_VERBOSE
        auto tstr0 = Meta::name<FromType>();
        auto tstr1 = Meta::name<ToType>();
        auto tstr2 = Meta::name<unsigned_t>();
        Debug("Narrowing %S -> %S | converting to %S and comparing (su) = %S.", &tstr0, &tstr1, &tstr2, &str);
#endif
        return value >= 0 && static_cast<unsigned_t>(value) <= static_cast<unsigned_t>(std::numeric_limits<To>::max());
    }
}

template<typename To, typename From>
constexpr To narrow_cast(From&& value, struct tag::warn_on_error) {
    if (!check_narrow_cast<To, From>(value)) {
        auto vstr = Meta::toStr(value);
        auto lstr = Meta::toStr(std::numeric_limits<To>::min());
        auto rstr = Meta::toStr(std::numeric_limits<To>::max());

        auto tstr = Meta::name<To>();
        auto fstr = Meta::name<From>();
        Warning("Value '%S' in narrowing conversion of %S -> %S is not within limits [%S,%S].", &vstr, &fstr, &tstr, &lstr, &rstr);
    }

    return static_cast<To>(std::forward<From>(value));
}

template<typename To, typename From>
constexpr To narrow_cast(From&& value, struct tag::fail_on_error) {
    if (!check_narrow_cast<To, From>(value)) {
        auto vstr = Meta::toStr(value);
        auto lstr = Meta::toStr(std::numeric_limits<To>::min());
        auto rstr = Meta::toStr(std::numeric_limits<To>::max());

        auto tstr = Meta::name<To>();
        auto fstr = Meta::name<From>();
        U_EXCEPTION("Value '%S' in narrowing conversion of %S -> %S is not within limits [%S,%S].", &vstr, &fstr, &tstr, &lstr, &rstr);
    }

    return static_cast<To>(std::forward<From>(value));
}

template<typename To, typename From>
constexpr To narrow_cast(From&& value) {
    return narrow_cast<To, From>(std::forward<From>(value), tag::warn_on_error{});
}
}
