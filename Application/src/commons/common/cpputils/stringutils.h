#ifndef _STRINGUTILS_H
#define _STRINGUTILS_H

#include <string>
#include <vector>
#include <cctype>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <functional>
#include <unordered_map>

namespace utils {
    /**
     * Detects whether the given \p str begins with a given needle character.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string starts with exactly the given needle
     */
    bool beginsWith(const std::string &str, const char needle);
    bool beginsWith(const std::wstring &str, const wchar_t needle);
    
    /**
     * Detects whether the given \p str begins with a given needle string.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string starts with exactly the given needle
     */
    bool beginsWith(const std::string &str, const std::string &needle);
    bool beginsWith(const std::wstring &str, const std::wstring &needle);
    
    /**
     * Detects whether the given \p str begins with a given needle string.
     * The string \p needle has to be NULL terminated in order to work (this
     * method will use strlen() to detect the length).
     * @param str haystack
     * @param needle the needle
     * @return true if the given string starts with exactly the given needle
     */
    bool beginsWith(const std::string &str, const char *needle);
    
    /**
     * Detects whether the given \p str ends with a given needle character.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string ends with exactly the given needle
     */
    bool endsWith(const std::string &str, const char needle);
    bool endsWith(const std::wstring &str, const wchar_t needle);
    
    /**
     * Detects whether the given \p str ends with a given needle string.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string ends with exactly the given needle
     */
    bool endsWith(const std::string &str, const std::string &needle);
    bool endsWith(const std::wstring &str, const std::wstring &needle);
    
    /**
     * Detects whether the given \p str ends with a given needle string.
     * The string \p needle has to be NULL terminated in order to work (this
     * method will use strlen() to detect the length).
     * @param str haystack
     * @param needle the needle
     * @return true if the given string ends with exactly the given needle
     */
    bool endsWith(const std::string &str, const char *needle);

	/**
	 * Finds a given needle inside the \p str given as first parameter.
	 * Case-sensitive.
	 * @param str haystack
	 * @param needle the needle
	 * @return true if it is found
	 */
	bool contains(const std::string &str, const std::string &needle);
    bool contains(const std::string &str, const char &needle);
    bool contains(const std::wstring &str, const std::wstring &needle);
    bool contains(const std::wstring &str, const wchar_t &needle);
    
    //! find and replace string in another string
    /**
     * @param str the haystack
     * @param oldStr the needle
     * @param newStr replacement of needle
     * @return str with all occurences of needle replaced by newStr
     */
    std::string find_replace(const std::string& str, const std::string& oldStr, const std::string& newStr);
    std::wstring find_replace(const std::wstring& str, const std::wstring& oldStr, const std::wstring& newStr);

    std::string find_replace(const std::string& str, std::vector<std::tuple<std::string, std::string>>);
    
    // trim from start
    template<typename Str>
    Str &ltrim(Str &s) {
        const static std::function<bool(int)> pred = [](int c) -> bool {return !std::isspace(c);};
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), pred));
        return s;
    }

    // trim from end
    template<typename Str>
    Str &rtrim(Str &s) {
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
        return s;
    }

    // trim from start
    template<typename Str>
    Str ltrim(const Str &str) {
        Str s(str);
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), (pred)));
        return s;
    }

    // trim from end
    template<typename Str>
    Str rtrim(const Str &str) {
        Str s(str);
        const static std::function<bool(int)> pred = [](int c) -> bool  {return !std::isspace(c);};
        s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
        return s;
    }

    // trim from both ends
    template<typename Str>
    Str trim(Str const& s) {
        Str tmp = s;
        ltrim(rtrim(tmp));
        
        return tmp;
    }
    

// to lower case
template<typename Str>
Str lowercase(Str const& original) {
    Str s = original;
#ifndef WIN32
    std::transform(s.begin(), s.end(), s.begin(), (int(*)(int))std::tolower);
#else
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
#endif
    return s;
}

inline std::string lowercase(const char* original) {
    return lowercase(std::string(original));
}

// to upper case
template<typename Str>
Str uppercase(Str const& original) {
    Str s = original;
#ifndef WIN32
    std::transform(s.begin(), s.end(), s.begin(), (int(*)(int))std::toupper);
#else
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#endif
    return s;
}

inline std::string uppercase(const char* original) {
    return uppercase(std::string(original));
}

    // repeats a string N times
    std::string repeat(const std::string& s, size_t N);

    // split string using delimiter
    std::vector<std::string> split(std::string const& s, char c);
    std::vector<std::wstring> split(std::wstring const& s, char c);
}

inline std::wstring s2ws(const std::string& str)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.from_bytes(str);
}

inline std::string ws2s(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}

// convert UTF-8 string to wstring
inline std::wstring utf82ws (const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

// convert wstring to UTF-8 string
inline std::string ws2utf8 (const std::wstring& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
}

#endif
