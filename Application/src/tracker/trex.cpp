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
#ifdef TREX_PYTHON_PATH
    auto compiled_path = file::Path(TREX_PYTHON_PATH).is_regular() ? file::Path(TREX_PYTHON_PATH).remove_filename().str() : file::Path(TREX_PYTHON_PATH).str();
#else
    std::string compiled_path = "";
#endif
    
#ifdef TREX_CONDA_PACKAGE_INSTALL
    return compiled_path;
#else
    auto conda_prefix = getenv("CONDA_PREFIX");
    std::string envs = "envs";
    envs += OS_SEP;
    
    std::string home = compiled_path;
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
#endif
}

int main(int argc, char** argv) {
    std::stringstream ss;
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
    ss << "";
#endif
    ss << target_path;
#if __APPLE__
    ss << "TRex.app/Contents/MacOS/TRex";
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
