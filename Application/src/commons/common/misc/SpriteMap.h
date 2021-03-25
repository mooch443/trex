#ifndef SPRITE_MAP_H
#define SPRITE_MAP_H

#include <commons/common/commons.pc.h>
#include <misc/metastring.h>

namespace cmn {
namespace sprite {
    
    class Map;
    class PropertyType;
    
    struct LockGuard {
        std::shared_ptr<std::lock_guard<std::mutex>> obj;
        LockGuard(Map*);
        ~LockGuard();
    };
    
    template<typename T>
    class Property;
    
    class ConstReference : public Printable {
    private:
        const PropertyType& _type;
        const Map& _container;
        
    public:
        ConstReference(const Map& container, const PropertyType& type)
        : _type(type), _container(container)//, _name(name)
        { }
        
        bool operator==(const PropertyType& other) const;
        bool operator==(const ConstReference& other) const { return operator==(other.get()); }
        
        bool operator!=(const PropertyType& other) const;
        bool operator!=(const ConstReference& other) const { return operator!=(other.get()); }
        
        template<typename T>
        const Property<T>& toProperty() const;
        
        template<typename T>
        operator Property<T>&() const {
            return toProperty<T>();
        }
        
        template<typename T>
        operator const T() const;
        
        template<typename T>
        const T& value() const {
            return toProperty<T>().value();
        }
        
        template<typename T>
        bool is_type() const;
        
        const PropertyType& get() const { return _type; }
        const Map& container() const { return _container; }
        
        TOSTRING_RAW;
        PRINT_NAME_HEADER override;
    };
    
    class Reference : public Printable {
    private:
        PropertyType& _type;
        Map& _container;
        const std::string& _name;
        
    public:
        Reference(Map& container, PropertyType& type);
        Reference(Map& container, PropertyType& type, const std::string& name)
        : _type(type), _container(container), _name(name)
        { }
        
        bool operator==(const PropertyType& other) const;
        bool operator==(const Reference& other) const { return operator==(other.get()); }
        
        bool operator!=(const PropertyType& other) const;
        bool operator!=(const Reference& other) const { return operator!=(other.get()); }
        
        template<typename T>
        Property<T>& toProperty() const;
        
        template<typename T>
        operator Property<T>&() const {
            return toProperty<T>();
        }
        
        template<typename T>
        operator const T();
        
        template<typename T>
        T& value() const {
            return toProperty<T>().value();
        }

		template<typename T>
		void value(const T& v) {
			operator=(v);
		}
        
        template<typename T>
        bool is_type() const;
        
        PropertyType& get() const { return _type; }
        Map& container() const { return _container; }

		double speed() const;
		Reference& speed(double s);
        
        template<typename T>
        void operator=(const T& value);
        
        TOSTRING_RAW;
        PRINT_NAME_HEADER override;
    };
}
}

#include "SpriteProperty.h"

namespace cmn {
namespace sprite {
    struct PNameRef {
        PNameRef(const std::string& name);
        PNameRef(const std::shared_ptr<PropertyType>& ref);
        
        std::string _nameOnly;
        std::shared_ptr<PropertyType> _ref;
        
        const std::string& name() const;
        
        bool operator==(const PNameRef& other) const;
        bool operator==(const std::string& other) const;
        bool operator<(const PNameRef& other) const;
    };
}
}

namespace std
{
    template <>
    struct hash<cmn::sprite::PNameRef>
    {
        size_t operator()(const cmn::sprite::PNameRef& k) const;
    };
}

namespace cmn {
namespace sprite {

    class Map : public Printable {
    public:
        enum class Signal {
            NONE,
            EXIT
        };
        
        typedef std::shared_ptr<PropertyType> Store;
        typedef std::function<void(Signal, Map&, const std::string&, const PropertyType&)> callback_func;
        
    private:
        
    protected:
        std::unordered_map<PNameRef, Store> _props;
        std::unordered_map<const char*, callback_func> _callbacks;
        std::unordered_map<std::string, bool> _print_key;
        
        GETTER_NCONST(std::mutex, mutex)
        GETTER_SETTER(bool, do_print)
        
    public:
        //! value of the property name has changed
        void changed(const PropertyType& var) {
            decltype(_callbacks) copy;
            {
                LockGuard guard(this);
                copy = _callbacks;
            }
            
            for(auto c: copy)
                c.second(Signal::NONE, *this, var.name(), var);
            
            if(_do_print) {
                LockGuard guard(this);
                auto it = _print_key.find(var.name());
                if(it == _print_key.end() || it->second)
                    var.print_object();
            }
        }
        
        void dont_print(const std::string& name) {
            _print_key[name] = false;
        }
        
        void do_print(const std::string& name) {
            auto it = _print_key.find(name);
            if(it != _print_key.end())
                _print_key.erase(it);
        }
        
        friend PropertyType;
        
    public:
        Map();
        Map(const Map& other);
        
        ~Map();
        
        bool empty() const { return _props.empty(); }
        size_t size() const { return _props.size(); }
        
        bool operator==(const Map& other) const {
            if(other.size() != size())
                return false;
            
            auto it0 = other._props.begin();
            auto it1 = _props.begin();
            
            for(; it0 != other._props.end(); ++it0, ++it1) {
                if(!(it0->first == it1->first) || !(*it0->second == *it1->second)) {
                    Debug("'%S' != '%S' || ", &it0->first.name(), &it1->first.name());
                    return false;
                }
            }
            
            return true;
        }
        
        void register_callback(const char *obj, const callback_func &func) {
            LockGuard guard(this);
            if(_callbacks.count(obj) > 0)
                U_EXCEPTION("Object %x already in map callbacks.", obj);
            _callbacks[obj] = func;
        }
        
        void unregister_callback(const char *obj) {
            LockGuard guard(this);
            if(_callbacks.count(obj) == 0) {
                Except("Cannot find obj %x in map callbacks.", obj);
                return;
            }
            _callbacks.erase(obj);
        }
        
        Reference operator[](const std::string& name) {
            LockGuard guard(this);
            auto it = _props.find(name);
            if(it != _props.end()) {
                return Reference(*this, *it->second);
            }
            
            return Reference(*this, Property<bool>::InvalidProp, name);
        }
        
        const ConstReference operator[](const std::string& name) const {
            auto it = _props.find(name);
            if(it != _props.end()) {
                return ConstReference(*this, *it->second);
            }
            
            return ConstReference(*this, Property<bool>::InvalidProp);
        }
        
        template<typename T>
        Property<T>& get(const std::string& name) {
            LockGuard guard(this);
            auto it = _props.find(name);
            if(it != _props.end()) {
				return (Property<T>&)it->second->toProperty<T>();
            }
            
            return Property<T>::InvalidProp;
        }
        
        template<typename T>
        Property<T>& get(const std::string& name) const {
            auto it = _props.find(name);
            if(it != _props.end()) {
                return (Property<T>&)it->second->toProperty<T>();
            }
            
            return Property<T>::InvalidProp;
        }
        
        bool has(const std::string& name) const {
            auto count = _props.count(name);
            return count > 0;
        }
        
        template<typename T>
        bool is_type(const std::string& name, const T&) const {
            return get<T>(name).valid();
        }
        
        template<typename T>
        bool is_type(const std::string& name) const {
            return get<T>(name).valid();
        }
        
        bool has(const PropertyType& prop) const {
            return has(prop.name());
        }
        
        template<typename T>
        Property<T>& set(const std::string& name, const T& value) {
            operator[](name) = value;
        }
        
        template<typename T>
        void operator<<(const std::pair<std::string, const T> pair) {
            Property<T>& type = get(pair.first);
            
            if(type) {
                type = pair.second;
                
            } else {
                insert(Property<T>(this, pair.first, pair.second));
            }
        }
        
        template<typename T>
        Property<T>& insert(const std::string& name, const T& value) {
            return insert(Property<T>(this, name, value));
        }
        
        template<typename T>
        Property<T>& insert(const Property<T>& property) {
            Property<T> *property_;
            
            {
                LockGuard guard(this);
                if(has(property)) {
                    std::string e = "Property already "+((const PropertyType&)property).toStdString()+" already exists.";
                    Error(e.c_str());
                    throw new PropertyException(e);
                }
                
                property_ = new Property<T>(this, property.name(), property.value());
                {
                    auto ptr = Store(property_);
                    _props[ptr] = ptr;
                }
            }
            
            changed(*property_);
            return *property_;
        }
        
        std::vector<std::string> keys() const {
            std::vector<std::string> result;
            result.reserve(_props.size());

            for (auto &p: _props)
                result.push_back(p.first.name());

            std::sort(result.begin(), result.end());
            return result;
        }
        
        
        
        UTILS_TOSTRING("Map<size: " << _props.size() << ">");
    };
    
    template<typename T>
    Property<T>& Reference::toProperty() const {
        return _type.toProperty<T>();
    }
    
    template<typename T>
    Reference::operator const T() {
        LockGuard guard(&_container);
        Property<T> *tmp = dynamic_cast<Property<T>*>(&_type);
        if (tmp) {
            return tmp->value();
        }
        
        std::string e = "Cannot cast " + _type.toStdString() + " to value type "+ Meta::name<T>() +".";
        Error(e.c_str());
        throw new PropertyException(e);
    }
    
    template<typename T>
    bool Reference::is_type() const {
        return _type.toProperty<T>().valid();
    }
    
    template<typename T>
    void Reference::operator=(const T& value) {
        if (_type.valid()) {
            _type.operator=(value);
            
        }
        else {
            PropertyType& tmp = _container.insert(_name, value);
            if (_type.speed() != PROPERTY_INVALID_SPEED)
                tmp.speed(_type.speed());
            
            //Debug("Inserting into map %@: %@", &_container, (PropertyType*)&tmp);
        }
    }
    
    template<typename T>
    const Property<T>& ConstReference::toProperty() const {
        return _type.toProperty<T>();
    }
    
    template<typename T>
    ConstReference::operator const T() const {
        Property<T> *tmp = dynamic_cast<Property<T>*>(&_type);
        if (tmp) {
            return tmp->value();
        }
        
        std::string e = "Cannot cast " + _type.toStdString() + " to value type "+ Meta::name<T>() +" .";
        Error(e.c_str());
        throw new PropertyException(e);
    }
    
    template<typename T>
    bool ConstReference::is_type() const {
        return _type.toProperty<T>().valid();
    }

    template<typename T>
    void PropertyType::operator=(const T& value) {
        Property<T>& ref = *this;
        if(ref.valid()) {
            ref = value;
            
        } else {
            std::stringstream ss;
            ss << "Reference to "+toStdString()+" cannot be cast to type of ";
            ss << Meta::name<T>();
            Error(ss.str().c_str());
            throw new PropertyException(ss.str());
        }
    }

    template<typename T>
    void Property<T>::value(const T& v) {
        {
            std::lock_guard<std::mutex> guard(_property_mutex);
            _value = v;
        }
        if(_map)
            _map->changed(*this);
    }
}
}

#endif
