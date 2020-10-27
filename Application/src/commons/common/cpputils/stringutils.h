#ifndef _STRINGUTILS_H
#define _STRINGUTILS_H

#include <string>
#include <vector>
#include <cctype>

namespace utils {
    /**
     * Detects whether the given \p str begins with a given needle character.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string starts with exactly the given needle
     */
    bool beginsWith(const std::string &str, const char needle);
    
    /**
     * Detects whether the given \p str begins with a given needle string.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string starts with exactly the given needle
     */
    bool beginsWith(const std::string &str, const std::string &needle);
    
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
    
    /**
     * Detects whether the given \p str ends with a given needle string.
     * @param str haystack
     * @param needle the needle
     * @return true if the given string ends with exactly the given needle
     */
    bool endsWith(const std::string &str, const std::string &needle);
    
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
    
    //! find and replace string in another string
    /**
     * @param str the haystack
     * @param oldStr the needle
     * @param newStr replacement of needle
     * @return str with all occurences of needle replaced by newStr
     */
    std::string find_replace(const std::string& str, const std::string& oldStr, const std::string& newStr);
    
    // to lower case
    std::string lowercase(std::string const& s);

    // to upper case
    std::string uppercase(std::string const& s);

    // trim from start
    std::string &ltrim(std::string &s);
    std::string ltrim(const std::string &s);

    // trim from end
    std::string &rtrim(std::string &s);
    std::string rtrim(const std::string &s);

    // trim from both ends
    std::string trim(std::string const& s);
    
    // repeats a string N times
    std::string repeat(const std::string& s, size_t N);

    // split string using delimiter
    std::vector<std::string> split(std::string const& s, char c);
}

#endif
