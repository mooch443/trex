#include "stringutils.h"
#include <iomanip>
#include <algorithm>
#include <locale>
#include <functional>
#include <cstring>
#include <unordered_set>
#include <map>

namespace utils {
    /*
     * Begins with.
     */
    bool beginsWith(const std::string &str, const char needle) {
        return str.empty() ? false : (str.at(0) == needle);
    }
    bool beginsWith(const std::wstring &str, const wchar_t needle) {
        return str.empty() ? false : (str.at(0) == needle);
    }
    
    bool beginsWith(const std::string &str, const std::string &needle) {
        return str.compare(0, needle.length(), needle) == 0;
    }
    bool beginsWith(const std::wstring &str, const std::wstring &needle) {
        return str.compare(0, needle.length(), needle) == 0;
    }
    
    bool beginsWith(const std::string &str, const char *needle) {
        return str.compare(0, strlen(needle), needle) == 0;
    }
    
    /*
     * Ends with.
     */
    bool endsWith(const std::string &str, const char needle) {
        return str.empty() ? false : (str.back() == needle);
    }
    bool endsWith(const std::wstring &str, const wchar_t needle) {
        return str.empty() ? false : (str.back() == needle);
    }
    
    bool endsWith(const std::string &str, const std::string &needle) {
        if (needle.length() > str.length()) {
            return false;
        }
        return str.compare(str.length()-needle.length(), needle.length(), needle) == 0;
    }
    bool endsWith(const std::wstring &str, const std::wstring &needle) {
        if (needle.length() > str.length()) {
            return false;
        }
        return str.compare(str.length()-needle.length(), needle.length(), needle) == 0;
    }
    
    bool endsWith(const std::string &str, const char *needle) {
        const size_t len = strlen(needle);
        if (len > str.length()) {
            return false;
        }
        return str.compare(str.length()-len, len, needle) == 0;
    }

	bool contains(const std::string &str, const std::string &needle) {
		return str.find(needle) != std::string::npos;
	}
    
    bool contains(const std::string &str, const char &needle) {
        return str.find(needle) != std::string::npos;
    }

    bool contains(const std::wstring &str, const std::wstring &needle) {
        return str.find(needle) != std::wstring::npos;
    }

    bool contains(const std::wstring &str, const wchar_t &needle) {
        return str.find(needle) != std::wstring::npos;
    }

    // find/replace
    template<typename T>
    T _find_replace(const T& str, const T& oldStr, const T& newStr)
    {
        T result = str;
        
        size_t pos = 0;
        while((pos = result.find(oldStr, pos)) != T::npos)
        {
            result.replace(pos, oldStr.length(), newStr);
            pos += newStr.length();
        }
        
        return result;
    }
    std::string find_replace(const std::string& str, const std::string& oldStr, const std::string& newStr)
    {
        return _find_replace<std::string>(str, oldStr, newStr);
    }
    std::wstring find_replace(const std::wstring& str, const std::wstring& oldStr, const std::wstring& newStr)
    {
        return _find_replace<std::wstring>(str, oldStr, newStr);
    }

    std::string find_replace(const std::string& str, std::vector<std::tuple<std::string, std::string>> search_strings)
    {
        if(str.empty())
            return "";
        
        std::string result = str;
        
        // sort array so that the longest sequences come first, and we can abort early in the following loop
        std::sort(search_strings.begin(), search_strings.end(), [](auto& A, auto &B){
            return std::get<0>(A).length() > std::get<0>(B).length();
        });
        
        using map_t = std::unordered_map<int, std::vector<size_t>, std::hash<int>, std::equal_to<const int>, std::allocator<std::pair<const int, std::vector<size_t>>>>;
        std::vector<map_t> viable;
        std::unordered_set<size_t> _still;
        for (size_t j=0; j<search_strings.size(); ++j) {
            _still.insert(j);
        }
        
        size_t max_L = std::get<0>(search_strings.front()).length();
        viable.resize(max_L);
        
        // search for possible combinations of chars from multiple search strings at the same distance from 0
        // we will use this to determine the following things:
        //      - are we encountering a char that does not exist in search patterns at the current offset
        //      - can we thus break out of the search loop
        //      - did we reach the end of a certain search string (and thus found a match)
        for(size_t i=0; i<max_L; ++i) {
            for(size_t j=0; j<search_strings.size(); ++j) {
                auto & [from, to] = search_strings[j];
                if(from.length() > i) {
                    // still long enough for given offset
                    viable[i][from.at(i)].push_back(j);
                    
                } else
                    break;
            }
        }
        
        for (size_t i=0; i <result.size(); ) {
            int64_t match = -1;
            
            for (size_t offset=0; offset<max_L && offset+i < result.size(); ++offset)
            {
                auto c = result.at(i+offset);
                auto &map = viable.at(offset);
                auto it = map.find(c);
                
                if(it != map.end()) {
                    for(auto idx : it->second) {
                        auto &mot = std::get<0>(search_strings.at(idx));
                        if(mot.at(offset) == c) {
                            if(offset+1 == mot.length() && result.substr(i, offset + 1) == mot)
                            {
                                match = idx;
                                break;
                            }
                        }
                    }
                    
                } else
                    break;
            }
            
            if(match == -1) {
                // if there was no match, just skip one
                ++i;
                
            } else {
                // otherwise, do the replacement and skip the replaced letters (possibly also just next)
                auto &[from, to] = search_strings.at(match);
                result.replace(i, from.length(), to);
                i+=to.length();
            }
        }
        
        return result;
    }
    
    std::string repeat(const std::string& s, size_t N) {
        std::string output;
        for(size_t i=0; i<N; i++)
            output += s;
        return output;
    }

    // split string using delimiter
    template<typename Str>
    std::vector<Str> _split(Str const& s, char c) {
        const size_t len = s.length();
        using Char = typename Str::value_type;
        
        std::vector<Str> ret;
        Char *tmp = new Char[len+1];
        
        for (size_t i = 0, j = 0; i <= len; i++) {
            if(i == len || s[i] == c) {
                tmp[j] = 0;
                ret.push_back(tmp);
                j = 0;
                
            } else {
                tmp[j++] = s[i];
            }
        }
        
		delete[] tmp;
        return ret;
    }
    std::vector<std::string> split(std::string const& s, char c) {
        return _split(s, c);
    }
    std::vector<std::wstring> split(std::wstring const& s, char c) {
        return _split(s, c);
    }
}
