#include "stringutils.h"
#include <iomanip>
#include <algorithm>
#include <locale>
#include <functional>
#include <cstring>

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
