#ifndef _SPRITE_PROPERTY_H
#define _SPRITE_PROPERTY_H

#include <misc/defines.h>
#include <misc/metastring.h>
#include <misc/vec2.h>

namespace cmn {
    namespace sprite {
        
        class PropertyException : public virtual UtilsException {
        public:
            PropertyException(const std::string& str) : UtilsException(str) { }
            ~PropertyException() throw() {}
        };
        
        template<typename ValueType = double>
        class Property;
        class Map;
        
        class PropertyType : public virtual Printable {
        protected:
            std::string _name;
            bool _valid;
            std::mutex _property_mutex;
            
            GETTER(bool, is_array)
            GETTER(bool, is_enum)
            
            std::function<void(const std::string&)> _set_value_from_string;
            std::function<std::string()> _type_name;
            GETTER(std::function<std::vector<std::string>()>, enum_values)
            GETTER(std::function<size_t()>, enum_index)
            
            GETTER_PTR(Map*, map)
            
        public:
            PropertyType(Map *map)
                : _name("<invalid>"), _valid(false), _is_array(false), _is_enum(false),
                _set_value_from_string([this](const std::string&){U_EXCEPTION("Uninitialized function set_value_from_string (%S).", &_name);}),
                _type_name([](){return "unknown";}),
                _enum_values([]() -> std::vector<std::string> {     U_EXCEPTION("PropertyType::enum_values() not initialized."); }),
                _enum_index([]() -> size_t{ U_EXCEPTION("PropertyType::enum_index() not initialized"); }),
                _map(map)
            { }
            
            PropertyType(Map *map, const std::string& name)
                : _name(name), _valid(true), _is_array(false), _is_enum(false),
                _set_value_from_string([this](const std::string&){U_EXCEPTION("Uninitialized function set_value_from_string (%S).", &_name);}),
                _type_name([](){return "unknown";}),
                _enum_values([]() -> std::vector<std::string> {     U_EXCEPTION("PropertyType::enum_values() not initialized."); }),
                _enum_index([]() -> size_t{ U_EXCEPTION("PropertyType::enum_index() not initialized"); }),
                _map(map)
            { }
            
            void set_value_from_string(const std::string& str) {
                try {
                    _set_value_from_string(str);
                } catch(const std::invalid_argument& e) {
                    U_EXCEPTION("Cannot set %@ to '%S' ('%s')", this, &str, e.what());
                }
            }
            
            virtual bool operator==(const PropertyType& other) const = 0;
            bool operator!=(const PropertyType& other) const {
                return !operator==(other);
            }
            
            template<typename T>
            void operator=(const T& value);
            
            const std::string& name() const { return _name; }
            
            void valid(bool valid) { _valid = valid; }
            bool valid() const { return _valid; }

            template<typename T>
            Property<T>& toProperty() {
                Property<T> *tmp = dynamic_cast<Property<T>*>(this);
                return tmp ? *tmp : Property<T>::InvalidProp;
            }

            template<typename T>
            const Property<T>& toProperty() const {
                const Property<T> *tmp = dynamic_cast<const Property<T>*>(this);
                return tmp ? *tmp : Property<T>::InvalidProp;
            }
            
            template<typename T>
            operator Property<T>&() {
                return toProperty<T>();
            }
            
            template<typename T>
            operator const Property<T>&() const {
                return toProperty<T>();
            }

            template<typename T>
            operator const T&() const {
                return value<T>();
            }
            
            template<typename T>
            const T& value() const {
                const Property<T>& p = toProperty<T>();
                if (p.valid())
                    return p.value();
                
                throw new PropertyException("Cannot cast " + toStdString() + " to const reference type.");
            }

            template<typename T>
            operator typename std::remove_const<T>::type &() {
                Property<T>& p = toProperty<T>();
                if (p.valid())
                    return p.value();

                throw new PropertyException("Cannot cast " + toStdString() + " to reference type.");
            }
            
            virtual const std::string type_name() const {
                return _type_name();
            }
            
            virtual std::string valueString() const {
                if(!valid())
                    throw new PropertyException("ValueString of invalid PropertyType.");
                throw new PropertyException("Cannot use valueString of PropertyType directly.");
            }
            
            UTILS_TOSTRING((
                      "Property<"+type_name()+">")
                     << "('" << _name << "'"
                     << ")");
        };
        
        class Reference;
        
        namespace detail {
            struct No {};
            template<typename T, typename Arg> No operator== (const T&, const Arg&);
        }
        
        template<typename T, typename Arg = T>
        struct has_equals
        {
            constexpr static const bool value = !std::is_same<decltype(*(T*)(0) == *(Arg*)(0)), detail::No>::value;
        };

        template<typename ValueType>
        class Property : public PropertyType {
        public:
            static Property<ValueType> InvalidProp;
            
        protected:
            ValueType _value;
            
        public:
            Property(Map *map)
            : PropertyType(map), _value(ValueType())
            {
                init();
            }
            
            Property(Map *map, const std::string& name, const ValueType& v = ValueType())
            : PropertyType(map, name), _value(v)
            {
                init();
            }
            
            void init() {
                _set_value_from_string = [this](const std::string& str) {
                    auto v = Meta::fromStr<ValueType>(str);
                    *this = v;
                };
                
                _type_name = [](){ return Meta::name<ValueType>(); };
                
                _is_enum = cmn::is_enum<ValueType>::value;
                _is_array = is_container<ValueType>::value;
                
                init_enum();
            }
            
            template<typename T = ValueType>
            void init_enum(typename std::enable_if_t<cmn::is_enum<T>::value, T> * = nullptr) {
                auto options = ValueType::Data::str();
                auto values = std::vector<std::string>(options.begin(), options.end());
                _enum_values = [values]() { return values; };
                _enum_index = [this]() -> size_t { return (size_t)this->value().value(); };
            }
            
            template<typename T = ValueType>
            void init_enum(typename std::enable_if_t<!cmn::is_enum<T>::value, T> * = nullptr) {
                _enum_values = []() -> std::vector<std::string> { U_EXCEPTION("This type is not an Enum, so enum_values() cannot be called."); };
            }
            
            ~Property() { };
            
            //using PropertyType::operator==;
            
            template<typename K>
            bool equals(const K& other, const typename std::enable_if< !std::is_same<cv::Mat, K>::value && has_equals<K>::value, K >::type* = NULL) const {
                return other == _value;
            }
            
            template<typename K>
            bool equals(const K& other, const typename std::enable_if< std::is_same<cv::Mat, K>::value, K >::type* = NULL) const {
                return cv::countNonZero(_value != other) == 0;
            }
            
            /*template<typename K>
            bool equals(const K& other, const typename std::enable_if< ! has_equals<K>::value, K >::type* = NULL) const {
                return &other == &_value;
            }*/
            
            bool operator==(const PropertyType& other) const override {
				const Property& other_ = (const Property<ValueType>&) other;
                return other_.valid() && equals(other_._value);
            }
            
            void operator=(const Reference& other);

            void operator=(const PropertyType& other);
            
        public:
            void operator=(const ValueType& v) { value(v); }
            void value(const ValueType& v);
            
            operator const ValueType() const { return value(); }
            const ValueType& value() const { return _value; }
            ValueType& value() { return _value; }
            std::string valueString() const override { return Meta::toStr<ValueType>(value()); }
            
            _TOSTRING override {
                if(!valid())
                    return PropertyType::toStdString();
                    
                std::stringstream ss;
                ss << PropertyType::toStdString();
                auto str = Meta::toStr<ValueType>(value());
                if(str.length() > 1000)
                    str = str.substr(0, 1000) + " (shortened)...";
                ss << " = " << str;
                return ss.str();
            }
        };
                
        template<typename T>
        Property<T> Property<T>::InvalidProp(NULL);
    }
}
            
#include "SpriteMap.h"

namespace cmn {
    namespace sprite {
        template<typename T>
        void Property<T>::operator=(const Reference& other) {
            LockGuard guard(map());
            const Property& _other = other.toProperty<T>();
            if (_other)
                *this = _other;
            else
                throw new PropertyException("Cannot assign " + other.toStdString() + " to " + this->toStdString());
        }
        
        template<typename T>
        void Property<T>::operator=(const PropertyType& other) {
            LockGuard guard(map());
            const Property& _other = other.operator const cmn::sprite::Property<T>&();
            if(_other.valid()) {
                *this = _other.value();
                
            } else {
                throw new PropertyException("Cannot assign "+other.toStdString()+" to "+this->toStdString());
            }
        }
    }
}
            
#endif
