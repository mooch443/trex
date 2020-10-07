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
    
    bool beginsWith(const std::string &str, const std::string &needle) {
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
    
    bool endsWith(const std::string &str, const std::string &needle) {
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

    // find/replace
    std::string find_replace(const std::string& str, const std::string& oldStr, const std::string& newStr)
    {
        std::string result = str;
        
        size_t pos = 0;
        while((pos = result.find(oldStr, pos)) != std::string::npos)
        {
            result.replace(pos, oldStr.length(), newStr);
            pos += newStr.length();
        }
        
        return result;
    }
    
    // to lower case
    std::string lowercase(std::string const& original) {
        std::string s = original;
#ifndef WIN32
        std::transform(s.begin(), s.end(), s.begin(), (int(*)(int))std::tolower);
#else
		std::transform(s.begin(), s.end(), s.begin(), ::tolower);
#endif
        return s;
    }

    // to upper case
    std::string uppercase(std::string const& original) {
        std::string s = original;
#ifndef WIN32
        std::transform(s.begin(), s.end(), s.begin(), (int(*)(int))std::toupper);
#else
		std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#endif
        return s;
    }

    // trim from start
    std::string &ltrim(std::string &s) {
        const static std::function<bool(int)> pred = [](int c) -> bool {return !std::isspace(c);};
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), pred));
        return s;
    }

    // trim from end
    std::string &rtrim(std::string &s) {
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
        return s;
    }
    
    // trim from start
    std::string ltrim(const std::string &str) {
        std::string s(str);
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), (pred)));
        return s;
    }
    
    // trim from end
    std::string rtrim(const std::string &str) {
        std::string s(str);
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
        return s;
    }
    
    // trim from both ends
    std::string trim(std::string const& s) {
        std::string tmp = s;
        ltrim(rtrim(tmp));
        
        return tmp;
    }
    
    std::string repeat(const std::string& s, size_t N) {
        std::string output;
        for(size_t i=0; i<N; i++)
            output += s;
        return output;
    }

    // split string using delimiter
    std::vector<std::string> split(std::string const& s, char c) {
        const size_t len = s.length();
        
        std::vector<std::string> ret;
        char *tmp = new char[len+1];
        
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
}
