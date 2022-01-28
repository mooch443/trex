#include "SpriteMap.h"

namespace std {
size_t hash<cmn::sprite::PNameRef>::operator()(const cmn::sprite::PNameRef& k) const
{
    return std::hash<std::string>{}(k.name());
}
}

namespace cmn {
namespace sprite {

Map::Map() : _do_print(true) {
}

Map::Map(const Map& other) {
    _props = other._props;
}

void Map::register_callback(const char *obj, const callback_func &func) {
    if(!obj) {
        Except("nullptr in register_callback");
        return;
    }
    
    LockGuard guard(this);
    if(_callbacks.count(obj) > 0)
        U_EXCEPTION("Object %s (%x) already in map callbacks.", obj, obj);
    
    _callbacks[obj] = func;
#ifndef NDEBUG
    Debug("Registered map callback %s (%x)", obj, obj);
#endif
}

void Map::unregister_callback(const char *obj) {
    if(!obj) {
        Except("nullptr in unregister_callback");
        return;
    }
    
    LockGuard guard(this);
    if(_callbacks.count(obj) == 0) {
        printf("[EXCEPTION] Cannot find obj %s (%x) in map callbacks.\n", obj, (uint32_t)(uint64_t)obj);
        return;
    }
    
#ifndef NDEBUG
    printf("Unregistering obj %s (%x) from map callbacks.\n", obj, (uint32_t)(uint64_t)obj);
#endif
    _callbacks.erase(obj);
}

Map::~Map() {
    {
        std::unique_lock guard(_mutex);
        auto c = _callbacks;
        
        guard.unlock();
        
        for(auto && [ptr, cb] : c) {
            //printf("Calling '%s'\n", ptr);
            cb(sprite::Map::Signal::EXIT, *this, "", Property<bool>::InvalidProp);
        }
    }
    
    {
        std::lock_guard guard(_mutex);
        _callbacks.clear();
        _props.clear();
        _print_key.clear();
    }
}
    
    // --------- PNameRef
    
    LockGuard::LockGuard(Map*map) {
        obj = std::make_shared<decltype(obj)::element_type>(map->mutex());
    }
    LockGuard::~LockGuard() {
    }
    
    PNameRef::PNameRef(const std::string& name)
    : _nameOnly(name), _ref(NULL)
    {
    }
    
    PNameRef::PNameRef(const std::shared_ptr<PropertyType>& ref)
    : _nameOnly(std::string()), _ref(ref)
    {
    }
    
    //PNameRef::PNameRef(const PropertyType& ref) : PNameRef(&ref) { }
    
    const std::string& PNameRef::name() const {
        return _ref ? _ref->name() : _nameOnly;
    }
    
    bool PNameRef::operator==(const PNameRef& other) const {
        return (_ref && _ref == other._ref)
        || (*this == other.name());
    }
    
    bool PNameRef::operator==(const std::string& other) const {
        return name() == other;
    }
    
    bool PNameRef::operator<(const PNameRef& other) const {
        return name() < other.name();
    }
    
    
    // -------- REFERENCE
    
    Reference::Reference(Map& container, PropertyType& type)
    : Reference(container, type, type.name()) {}

    _TOSTRING_RETURNTYPE Reference:: _TOSTRING_HEAD {
        return _type.toStdString();
    }
    
    _PRINT_NAME_RETURN_TYPE Reference:: _PRINT_NAME_HEAD {
        return _type.print_name();
    }
    
    bool Reference::operator==(const PropertyType& other) const {
        return get().operator==(other);
    }
    
    bool Reference::operator!=(const PropertyType& other) const {
        return get().operator!=(other);
    }
    
    _TOSTRING_RETURNTYPE ConstReference:: _TOSTRING_HEAD {
        return _type.toStdString();
    }
    
    _PRINT_NAME_RETURN_TYPE ConstReference:: _PRINT_NAME_HEAD {
        return _type.print_name();
    }
    
    bool ConstReference::operator==(const PropertyType& other) const {
        return get().operator==(other);
    }
    
    bool ConstReference::operator!=(const PropertyType& other) const {
        return get().operator!=(other);
    }

}
}
