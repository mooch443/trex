#include "utilsexception.h"
#include <stdlib.h>

UtilsException::UtilsException(const std::string& str) : std::exception() {
	msg = str;
}

UtilsException::UtilsException(const char*fmt, ...) : std::exception() {
    va_list args;
    va_start(args, fmt);
    DEBUG::ParseFormatString(msg, fmt, args);
    va_end(args);
}

UtilsException::~UtilsException() throw() {
}

const char * UtilsException::what() const throw() {
	return msg.c_str();
}
