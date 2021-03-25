#ifndef DEBUG_H_
#define DEBUG_H_

#ifdef _WIN32
#include <Windows.h>
#endif

#include <ctime>
#include <cstdio>
#include <sstream>
#include <cstdarg>
#include <cstring>
#include <string>
#include <cstdlib>
#include <vector>
#include <functional>
#include <mutex>

#if __cpp_lib_string_view
#include <string_view>
#elif __cpp_lib_experimental_string_view
#include <experimental/string_view>
#endif

class DebugException : virtual std::exception {
public:
    
	DebugException(std::string str) : std::exception() {
        msg = str;
	}
    
	DebugException(const char*fmt, ...) : std::exception() {
		char * BUFFER;
		
		va_list args;
		va_start(args, fmt);
        
#ifdef _WIN32
		size_t n = (size_t)vsnprintf_s(NULL, 0, 0, fmt, args) + 1;
#else
		size_t n = (size_t)vsnprintf(NULL, 0, fmt, args) + 1;
#endif
		va_end(args);
		
		va_start(args, fmt);
		BUFFER = (char*)calloc(n, sizeof(char));
        if (!BUFFER) {
            msg = "Cannot allocate memory for message '" + std::string(fmt) + "'.";
            return;
        }

#ifdef _WIN32
		vsnprintf_s(BUFFER, n, n, fmt, args);
#else
		vsnprintf(BUFFER, n, fmt, args);
#endif
		va_end(args);
		
		msg = BUFFER;
		
		free(BUFFER);
	}
    
    DebugException() throw() {
    }
    
    ~DebugException() throw() {
    }
    
    virtual const char * what() const throw() {
        return msg.c_str();
    }
    
private:
    std::string msg;
};

namespace DEBUG {

	enum CONSOLE_COLORS {
		BLACK = 0, DARK_BLUE = 1, DARK_GREEN = 2, DARK_CYAN = 3, DARK_RED = 4, PURPLE = 5, DARK_YELLOW = 6, 
		GRAY = 7, DARK_GRAY = 8, BLUE = 9, GREEN = 10, CYAN = 11, RED = 12, PINK = 13, YELLOW = 14, 
		WHITE = 15, LIGHT_BLUE
	};

	enum PARSE_OBJECTS {
		NORMAL = 0,
		BRACKETS = '(', END = 'E',
		CLASSSEPERATOR = ':',
		CLASSNAME = ' ',
		NUMBER = '1', HEXDECIMAL = 'x',
		STRING = '\'',
		DESCRIPTION = '<',
		KEY = 'K',
        KEYWORD = 'Y'
		
	};

	enum DEBUG_TYPE
	{
		TYPE_INFO,
		TYPE_ERROR,
		TYPE_WARNING,
        TYPE_EXCEPTION,
        TYPE_CUSTOM
	};
    
    std::mutex& debug_mutex();
    void set_runtime_quiet();

	struct StatusMsg {
#if __cpp_lib_string_view
        std::string_view buf;
#elif __cpp_lib_experimental_string_view
		std::experimental::string_view buf;
#else
		std::string buf;
#endif
		clock_t c;
		DEBUG_TYPE type;
        int line;
        const char *file;
        bool force_callback;
        CONSOLE_COLORS color = BLACK;
        std::string prefix;

#if __cpp_lib_string_view
        StatusMsg(DEBUG_TYPE t, const std::string_view& b, int line_, const char *file_)
            : buf(b), c(clock()), type(t), line(line_), file(file_), force_callback(false)
        { }
#elif __cpp_lib_experimental_string_view
        StatusMsg(DEBUG_TYPE t, const std::experimental::string_view& b, int line_, const char *file_)
            : buf(b), c(clock()), type(t), line(line_), file(file_), force_callback(false)
        { }
#else
        StatusMsg(DEBUG_TYPE t, const std::string& b, int line_, const char *file_)
            : buf(b), c(clock()), type(t), line(line_), file(file_), force_callback(false)
        { }
#endif

		~StatusMsg() { }
	};
    
    
    void* SetDebugCallback(const std::vector<DEBUG_TYPE>& types, const std::function<void(const StatusMsg*, const std::string&)>& callback);
    void UnsetDebugCallback(void*);

	void ParseStatusMessage(StatusMsg*msg);
    void Init();

    void ParseFormatString(std::string& output, const char *cmd, va_list args);
    std::string format(const char *cmd, ...);
}

#ifndef _WIN32
#define __FILE_NO_PATH__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#else
#define __FILE_NO_PATH__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#endif

#define MESSAGE_TYPE(NAME) void NAME(const char*cmd, ...);
#define MESSAGE_TYPE_DBG(NAME) void NAME(const char *file, int line, const char*cmd, ...);

MESSAGE_TYPE_DBG(ERROR_)
#define Error(...) ERROR_(__FILE_NO_PATH__, __LINE__, __VA_ARGS__)

MESSAGE_TYPE_DBG(EXCEPTION_)
#define Except(...) EXCEPTION_(__FILE_NO_PATH__, __LINE__, __VA_ARGS__)

MESSAGE_TYPE_DBG(WARNING_)
#define Warning(...) WARNING_(__FILE_NO_PATH__, __LINE__, __VA_ARGS__)

MESSAGE_TYPE(Debug)
MESSAGE_TYPE(DebugCallback)

MESSAGE_TYPE(DebugHeader)

template<typename T>
inline T CustomException(const char *cmd, ...) {
    va_list args;
    va_start(args, cmd);
    
    std::string str;
    DEBUG::ParseFormatString(str, cmd, args);
    
    va_end(args);
    
    return T(str);
}

void StatusMessage(DEBUG::DEBUG_TYPE type,
#if __cpp_lib_string_view
   const std::string_view& buf,
#elif __cpp_lib_experimental_string_view
   const std::experimental::string_view& buf,
#else
   const std::string& buf,
#endif
   int line = -1, const char *file = NULL, bool force_callback = false
);

#endif
