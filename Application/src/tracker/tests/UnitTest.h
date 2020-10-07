#ifdef ASSERT
#undef ASSERT
#endif

#define ASSERT(condition) { condition_result({ std::string(__FILE__), __LINE__, std::string( #condition ) }, condition); } if(!(condition)) abort()

#pragma once

#include <types.h>
#include <misc/pretty.h>
#include <misc/metastring.h>

enum PrintType {
    info,
    error
};

void print(PrintType type, const char* cmd, ...) {
    std::string output;
    
    va_list args;
    va_start(args, cmd);
    
    DEBUG::ParseFormatString(output, cmd, args);
    va_end(args);
    
    printf("[%s] %s\n", type == info ? "." : "!", output.c_str());
}

using namespace cmn;
using namespace gui;

static_assert(std::is_same<decltype(HorizontalLine::x0), ushort>::value, "Expecting coordinates in HorizontalLine to be unsigned short.");

class UnitTest {
    GETTER(std::string, title)
    
    struct Condition {
        std::string path;
        int line;
        std::string name;
        
        inline bool operator <(const Condition& other) const {
            return path < other.path || line < other.line || name < other.name;
        }
        
        inline operator MetaObject() const {
            std::string fname = file::Path(path).filename().to_string();
            return MetaObject(fname + ":" + Meta::toStr(line)+ " '" + name + "'", "Condition");
        }
        static std::string class_name() {
            return "Condition";
        }
    };
    using cond_t = std::map<Condition, bool>;
    GETTER(cond_t, conditions)
    
public:
    UnitTest(const std::string& title) : _title(title) {}
    virtual ~UnitTest() {}
    virtual void evaluate() = 0;
    
protected:
    void condition_result(const Condition& condition, bool result) {
        if(_conditions.find(condition) != _conditions.end()) {
            if(!result) {
                _conditions[condition] = false;
            }
        } else
            _conditions[condition] = result;
    }
};

template<typename T>
static bool run() {
    static_assert(std::is_convertible<T*, UnitTest*>::value, "");
    
    T obj;
    print(info, "Running '%S' ...", &obj.title());
    
    obj.evaluate();
    
    size_t failed = 0;
    for (auto && [condition, result] : obj.conditions()) {
        if(!result) {
            print(error, "\tFailed test '%S'", &condition.name);
            ++failed;
        }
    }
    
    auto str = prettify_array(Meta::toStr(obj.conditions()));
    print(info, "'%S' results:\n%S", &obj.title(), &str);
    
    return failed == 0;
}
