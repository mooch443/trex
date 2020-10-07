#ifndef _SERIALIZABLE_H
#define _SERIALIZABLE_H

#include <types.h>
#include <sstream>
#include <assert.h>

namespace gui {
    class Serializable {
    public:
        constexpr Serializable() {}
        virtual std::ostream &operator <<(std::ostream &os) = 0;
        virtual ~Serializable() {}
    };
    
    inline std::ostream &operator<<(std::ostream &os, Serializable& obj)
    {
        return obj.operator<<(os);
    }
}

#endif
