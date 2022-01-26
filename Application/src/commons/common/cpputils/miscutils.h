#ifndef _MISC_UTILS
#define _MISC_UTILS

#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <set>
#include "cpputils.h"

#define UNUSED(VAR) { (void)VAR; }

#include "utilsexception.h"

/**
 * Provides a wrapper for a vector of enum values.
 * Items can be pushed (and are saved only once) and then the list can
 * be asked whether it contains a specific value.
 * Values can also be removed.
 */
template<typename T>
class OptionsList {
    std::vector<T> list;
    
public:
    OptionsList() {}
    OptionsList(const std::vector<T> &values) : list(values) { }
	const std::vector<T>& values() const { return list;  }
    
    void push(const T& v) { if(is(v)) return; list.push_back(v); }
    void remove(const T& v) {
        for(auto it = list.begin(); it != list.end(); ++it) {
            if(*it == v) {
                list.erase(it);
                return;
            }
        }
        
        //U_EXCEPTION("Cannot find value '%d' of enum in this OptionsList.", v);
        //U_EXCEPTION("Cannot find value '%S' of enumeration '%s' in this OptionsList.", &v.toString(), T::name());
    }
    void clear() {
        list.clear();
    }
    
    size_t size() const { return list.size(); }
    
    bool is(const T& v) const {
        for(auto &l : list)
            if(l == v)
                return true;
        return false;
    }
    
    bool operator==(const OptionsList<T> &other) const {
		return list == other.list;
    }
};

template<typename T1, typename T2, typename T3>
class triplet {
public:
    triplet(const T1& t1, const T2& t2, const T3& t3)
        : first(t1), second(t2), third(t3)
    { }
    
public:
    T1 first;
    T2 second;
    T3 third;
};

template<typename T1, typename T2, typename T3>
bool operator==(const triplet<T1, T2, T3>& first, const triplet<T1, T2, T3>& second) {
    return first.first == second.first && first.second == second.second && first.third == second.third;
}

/*template<typename T>
inline bool contains(const std::vector<T>& v, T obj) {
    return std::find(v.begin(), v.end(), obj) != v.end();
}*/

template<typename T, typename Q>
inline bool contains(const Q& v, const T& obj) {
    return std::find(v.begin(), v.end(), obj) != v.end();
}

template<typename T>
inline bool contains(const std::set<T>& v, T obj) {
    //static_assert(!std::is_same<T, typename decltype(v)::value_type>::value, "We should not use this for sets.");
#if __cplusplus >= 202002L
    return v.contains(obj);
#else
    return v.find(obj) != v.end();
#endif
}

template<typename T, typename V>
inline bool contains(const std::map<T, V>& v, T key) {
    //static_assert(!std::is_same<T, typename decltype(v)::value_type>::value, "We should not use this for sets.");
#if __cplusplus >= 202002L
    return v.contains(key);
#else
    return v.find(key) != v.end();
#endif
}

// -------------------------------
//          FILE UTILS
// -------------------------------

#include <fstream>

namespace utils {

inline std::string read_file(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if(!input.is_open())
        U_EXCEPTION("Cannot read file '%S'.", &filename);
    
    std::stringstream ss;
    ss << input.rdbuf();
    
    return ss.str();
}
    
}

#endif
