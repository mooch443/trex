#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <cpputils/cpputils.h>
#include <misc/GlobalSettings.h>
#include <misc/create_struct.h>

#if WIN32
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif

std::string conda_environment_path(const char* argv) {
#ifdef TREX_CONDA_PACKAGE_INSTALL
    auto conda_prefix = getenv("CONDA_PREFIX");
    std::string envs = "envs";
    envs += OS_SEP;
    
    std::string home;
    if(conda_prefix) {
        // we are inside a conda environment
        home = conda_prefix;
    } else if(utils::contains(argv, envs)) {
        auto folders = utils::split(argv, OS_SEP);
        std::string previous = "";
        home = "";
        
        for(auto &folder : folders) {
            home += folder;
            
            if(previous == "envs") {
                break;
            }
            
            home += OS_SEP;
            previous = folder;
        }
    }
    return home;
#else
    return "";
#endif
}

const char * FOR_EACH_1 = "#define FOR_EACH_%d(NAM, what, x, ...)\\\n" \
                                "\twhat(NAM, x)\\\n" \
                                "\tEXPAND(FOR_EACH_%d(NAM, what,  __VA_ARGS__))\n\n";

struct Temp {
    enum Variables {
        Variable1,
        Variable2
    };
    
    static constexpr const char* VariableNames[] {
        "Variable1",
        "Variable2"
    };
    
    static inline struct Members {
        uint32_t Variable1;
        std::string Variable2;
    } _detail;
    
    using Variable1_t = uint32_t;
    using Variable2_t = std::string;
    
    static inline const std::array<std::function<const cmn::sprite::PropertyType&()>, sizeof(VariableNames)> _getters {
        []() -> const cmn::sprite::PropertyType& { return cmn::GlobalSettings::get("Variable1").get(); },
        []() -> const cmn::sprite::PropertyType& { return cmn::GlobalSettings::get("Variable2").get(); }
    };
    
    static Members& impl() {
        return _detail;
    }
    
    template<Variables M, typename T>
    static void set(T v, typename std::enable_if_t<M == Variables::Variable1 && std::is_convertible<T, decltype(Members::Variable1)>::value, void>* = nullptr) {
        impl().Variable1 = v;
        update<M>();
    }
    template<Variables M>
    static const uint32_t& get(typename std::enable_if_t<M == Variables::Variable1, void>* = nullptr) {
        return impl().Variable1;
    }
    
    template<Variables M>
    static const char* name() {
        return VariableNames[M];
    }
    
    template<Variables M, typename T>
    static void set(T v, typename std::enable_if_t<M == Variables::Variable2 && std::is_convertible<T, decltype(Members::Variable2)>::value, void>* = nullptr) {
        impl().Variable2 = v;
        update<M>();
    }
    
    template<Variables M>
    static void update() {
        // do nothing
    }
};

#include <misc/BlobSizeRange.h>

//#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...) N
//#define FOR_EACH_RSEQ_N() 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0"
using namespace cmn;
CREATE_STRUCT(Test,
  (int, Variable1),
  (int, Variable2),
  (int, Variable3),
  (int, frame_rate)
)

//template<> void Test::update<Test::Variable1>() {
//    Debug("Custom update<Test::Variable1>()");
//}

//STRUCT_META_EXTENSIONS(Test)

/*template<>
void Temp::update<Temp::Variable1>() {
    printf("Updating variable1!\n");
}


uint32_t __attribute__ ((noinline)) getDouble() {
    Test::set<Test::Variable1>(1337.0);
    return Test::get<Test::Variable1>();
}*/


int main(int argc, char** argv) {
    Debug("%d", Test::get(Test::frame_rate).value<Test::frame_rate_t>());
    std::stringstream ss;
    GlobalSettings::get("").get().type_name();
    Test::set_callback(Test::Variable1, [](auto&, auto&){
        Debug("Custom callback Variable1");
    });
    /*typename std::conditional< std::is_same<int, int>::value, int, float>::type v;
    Temp::set<Temp::Variable1>(1337);
    Temp::set<Temp::Variable2>("abc");
    Temp::get<Temp::Variable1>();
    getDouble();
    printf("Name = %s, Value = %d\n", Temp::name<Temp::Variable1>(), Temp::get<Temp::Variable1>());
    Test::set<Test::Variable1>(0.0);
    Test::name<Test::Variable1>();
    auto d = Test::get<Test::Variable1>();*/
    
    /*{
        auto str = Meta::toStr(Test::Variable2);
        Debug("%S", &str);
        Meta::fromStr<Test::Variables>("Variable1");
    }*/
    
    printf("#define FOR_EACH_1(NAM, what, x) what(NAM, x)\n");
    for (size_t i=2; i<128; ++i) {
        printf(FOR_EACH_1, i, i-1);
    }
    
    
    ss << "#define FOR_EACH_ARG_N( ";
    for(size_t i=1; i<128; ++i) {
        ss << "_" << std::to_string(i) << ", ";
    }
    ss << " N, ...) N";
    
    printf("%s\n", ss.str().c_str());
    ss.str("");
    
    printf("#define FOR_EACH_RSEQ_N() ");
    for(int i=128; i>0; --i) {
        printf("%d, ", i);
    }
    printf("0\n");
    exit(0);
    std::string target_path = "";
    auto conda_prefix = conda_environment_path(argv[0]);
    if(!conda_prefix.empty()) {
        target_path = conda_prefix;
        target_path += "/bin/";
        
        printf("Using conda prefix '%s'.\n", target_path.c_str());
    } else {
        target_path = argv[0];
        for(long i=(long)target_path.size(); i>0; --i) {
            if (target_path[i] == '/' || target_path[i] == '\\') {
                target_path = target_path.substr(0, i);
                break;
            }
        }
        target_path += "/";
        printf("Using argv[0] = '%s'...\n", target_path.c_str());
    }
    
#if __APPLE__
    ss << "open '";
#endif
    ss << target_path;
#if __APPLE__
    ss << "TRex.app' --args";
#else
    U_EXCEPTION("Only apple supported.");
#endif

    for(auto i=1; i<argc; ++i)
        ss << " " << argv[i];
    
    auto str = ss.str();
    printf("Calling '%s'...\n", str.c_str());
    fflush(stdout);
    system(str.c_str());
    
    return 0;
}
