#pragma once

#include <commons/common/commons.pc.h>

class SoftException : public std::exception {
public:

    SoftException(const std::string& str);
    SoftException(const char*fmt, ...);

    ~SoftException() throw();

    virtual const char * what() const throw();

private:
    std::string msg;
};

#define SOFT_EXCEPTION(...) { EXCEPTION_(__FILE_NO_PATH__, __LINE__, __VA_ARGS__); throw SoftException(__VA_ARGS__); }
