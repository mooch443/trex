#ifndef _PRINTABLE_H
#define _PRINTABLE_H

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

enum WarningMessages {
	NO_TEXTURE_BUT_SHADER_REQUIRES,
	NOT_ENOUGH_TEXTURES_FOR_SHADER
};


#define _TOSTRING_RETURNTYPE const std::string
#define _TOSTRING_HEAD toStdString() const

#ifdef _TOSTRING
#undef _TOSTRING
#endif
#define _TOSTRING \
    _TOSTRING_RETURNTYPE _TOSTRING_HEAD

#define TOSTRING_RAW virtual _TOSTRING override

#define UTILS_TOSTRING(ARGS) TOSTRING_RAW { \
    std::stringstream ss; \
    ss << ARGS; \
    return ss.str(); \
}

#define _PRINT_NAME_RETURN_TYPE const std::string
#define _PRINT_NAME_HEAD print_name() const
#define PRINT_NAME_HEADER \
    virtual _PRINT_NAME_RETURN_TYPE _PRINT_NAME_HEAD

#define PRINT_NAME(ARGS) PRINT_NAME_HEADER override { return ARGS; }

class Printable {
public:
    Printable() { }
    virtual ~Printable() { }

    //! a const char version of toStdString
    const char* toString();

    /**
    * Returns true, if the warning has already been displayed.
    */
    bool hasWarnedFor(WarningMessages ident);

    //! Will be called if a certain warning has been given for this object
    void warnedFor(WarningMessages ident);

    //! returns a printable info string about the object
    virtual _TOSTRING;

    //! returns a printable name of the class
    PRINT_NAME_HEADER;
    
    //! prints out the TOSTRING methods output
    void print_object() const;
        
protected:
	std::string _tmpstr;

private:
	std::vector<WarningMessages> displayedWarnings;

};

//#undef _TOSTRING

#endif
