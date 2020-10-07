#pragma once

#include <commons/common/commons.pc.h>

namespace cmn {
    class MetaObject {
        std::string _value;
        std::string _class_name;
        
    public:
        MetaObject(const std::string_view& value, const std::string_view& class_name)
            : _value(value), _class_name(class_name)
        { }
        
        std::string value() const { return (std::string)_value; }
        std::string class_name() const { return (std::string)_class_name; }
    };
}
