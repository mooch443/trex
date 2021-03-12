#include "Printable.h"
#include "Debug.h"

const char* Printable::toString() {
    _tmpstr = toStdString();
    return _tmpstr.c_str();
}

/**
 * Returns true, if the warning has already been displayed.
 */
bool Printable::hasWarnedFor(WarningMessages ident) {
    std::stringstream ss;
    _tmpstr = ss.str();
    return std::find(displayedWarnings.begin(), displayedWarnings.end(), ident) != displayedWarnings.end();
}

//! Will be called if a certain warning has been given for this object
void Printable::warnedFor(WarningMessages ident) {
    if (!hasWarnedFor(ident)) { displayedWarnings.insert(displayedWarnings.end(), ident); }
}

//! returns a printable info string about the object
_TOSTRING_RETURNTYPE Printable:: _TOSTRING_HEAD {
    std::stringstream ss;
    ss << "Printable<" << std::hex << (uint64_t)this << ">";
    return ss.str();
}
    
//! returns a printable name of the class
_PRINT_NAME_RETURN_TYPE Printable:: _PRINT_NAME_HEAD {
    return "Printable";
}

void Printable::print_object() const {
    Debug("%@", this);
}
