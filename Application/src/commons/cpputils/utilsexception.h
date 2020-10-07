#ifndef _EXCEPTION_UTILS_H
#define _EXCEPTION_UTILS_H

#include <stdarg.h>
#include <exception>
#include <string>

#include "debug/Debug.h"

class UtilsException : public std::exception {
public:

	UtilsException(const std::string& str);
	UtilsException(const char*fmt, ...);

	~UtilsException() throw();

	virtual const char * what() const throw();

private:
	std::string msg;
};

#define U_EXCEPTION(...) { EXCEPTION_(__FILE_NO_PATH__, __LINE__, __VA_ARGS__); throw UtilsException(__VA_ARGS__); }

#endif
