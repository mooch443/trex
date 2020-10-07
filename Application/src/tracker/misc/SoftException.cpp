#include "SoftException.h"
#include <stdlib.h>

SoftException::SoftException(const std::string& str) : std::exception() {
    msg = str;
}

SoftException::SoftException(const char*fmt, ...) : std::exception() {
    va_list args;
    va_start(args, fmt);
    DEBUG::ParseFormatString(msg, fmt, args);
    va_end(args);
}

SoftException::~SoftException() throw() {
}

const char * SoftException::what() const throw() {
    return msg.c_str();
}
