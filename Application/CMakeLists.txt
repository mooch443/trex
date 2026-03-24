cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
message("-- USER environment variable is set to: " $ENV{USER})

project(TRex LANGUAGES CXX C)
if(POLICY CMP0114)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0114 NEW)
endif()
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# only for linux
if(LINUX AND NOT TREX_CONDA_PACKAGE_INSTALL)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
include(ExternalProject)
include(FetchContent)

enable_testing()

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

set(MACOSX_BUNDLE_GUI_IDENTIFIER "com.trexing.trex")
set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "tristan")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
set(CMAKE_XCODE_ATTRIBUTE_GCC_INCREASE_PRECOMPILED_HEADER_SHARING "YES")
set(CMAKE_XCODE_ATTRIBUTE_GCC_PRECOMPILE_PREFIX_HEADER "YES")
set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
#set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym")
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

function(trex_log_option option_name)
    if(DEFINED ${option_name})
        message(STATUS "[TREX CONFIG]   ${option_name}=${${option_name}}")
    else()
        message(STATUS "[TREX CONFIG]   ${option_name}=<UNDEFINED>")
    endif()
endfunction()

function(trex_log_selected_options)
    message(STATUS "[TREX CONFIG] Selected options:")
    foreach(option_name IN ITEMS
        TREX_NO_PYTHON
        TREX_BUILD_DOCS
        TREX_WITH_TESTS
        TREX_BUILD_TOOLS
        TREX_CONDA_PACKAGE_INSTALL
        TREX_ENABLE_SHARED_INTERNAL_LIBS
        TREX_ENABLE_MODULES
        WITH_GITSHA1
        BUILD_LEGACY_TREX
    )
        trex_log_option(${option_name})
    endforeach()

    if(APPLE)
        trex_log_option(BUILD_FOR_ARM64)
    endif()

    if(DEFINED COMMONS_NO_PYTHON)
        trex_log_option(COMMONS_NO_PYTHON)
    endif()
endfunction()

#set(CMAKE_CXX_VISIBILITY_PRESET hidden)
option(TREX_NO_PYTHON OFF)
option(TREX_BUILD_DOCS OFF)
option(TREX_WITH_TESTS OFF)
option(TREX_BUILD_TOOLS OFF)
option(TREX_CONDA_PACKAGE_INSTALL OFF)
option(TREX_ENABLE_SHARED_INTERNAL_LIBS "Build TRex internal libraries as shared libraries on macOS for dependency hardening." OFF)
option(TREX_ENABLE_MODULES "Build TRex semantic libraries with C++ named module interfaces." OFF)
option(WITH_GITSHA1 "include git hash" ON)
option(BUILD_LEGACY_TREX "Build the TRex (legacy) target" OFF)

if(TREX_ENABLE_SHARED_INTERNAL_LIBS AND NOT APPLE)
    message(FATAL_ERROR
        "TREX_ENABLE_SHARED_INTERNAL_LIBS is currently supported only on macOS. "
        "Disable it or configure on Apple platforms."
    )
endif()

if(TREX_ENABLE_MODULES)
    if(NOT COMMONS_ENABLE_MODULES)
        message(FATAL_ERROR
            "TREX_ENABLE_MODULES requires COMMONS_ENABLE_MODULES=ON because the TRex semantic modules build on top of the commons module graph."
        )
    endif()

    if(${CMAKE_VERSION} VERSION_LESS "3.28.0")
        message(FATAL_ERROR "TREX_ENABLE_MODULES requires CMake 3.28 or newer.")
    endif()

    if(NOT CMAKE_GENERATOR MATCHES "^(Ninja|Ninja Multi-Config|Visual Studio 17 2022)$")
        message(FATAL_ERROR
            "TREX_ENABLE_MODULES is not supported with the '${CMAKE_GENERATOR}' generator. "
            "Current CMake C++ module support requires Ninja, Ninja Multi-Config, or Visual Studio 17 2022. "
            "Use -G Ninja (or disable TREX_ENABLE_MODULES)."
        )
    endif()
endif()

if(APPLE)
    option(BUILD_FOR_ARM64 "Build for arm64 architecture" ON)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(BUILD_FOR_ARM64 ON)
    else()
        set(BUILD_FOR_ARM64 OFF)
    endif()

    if(BUILD_FOR_ARM64)
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture for macOS" FORCE)
    else()
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architecture for macOS" FORCE)
    endif()

    message(STATUS "Architectures: ${CMAKE_OSX_ARCHITECTURES}")
endif()

if(COMMONS_NO_PYTHON)
    set(TREX_NO_PYTHON ON)
endif()

trex_log_selected_options()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG")
endif()

message(STATUS "[TREX ENV] Generator='${CMAKE_GENERATOR}' WIN32='${WIN32}'")
message(STATUS "[TREX ENV] Incoming ENV{CONDA_PREFIX}='$ENV{CONDA_PREFIX}'")

set(_trex_vs_debugger_env "")

if(WIN32 AND DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(_trex_conda_prefix "$ENV{CONDA_PREFIX}")
    get_filename_component(_trex_conda_root "${_trex_conda_prefix}" DIRECTORY)

    # If CMake is launched from an activated conda env, ensure configure-time
    # lookups (find_package/find_program/custom steps) resolve against that env.
    set(_trex_conda_path_prefix
        "${_trex_conda_prefix};${_trex_conda_prefix}/Library/mingw64/bin;${_trex_conda_prefix}/Library/mingw-w64/bin;${_trex_conda_prefix}/Library/usr/bin;${_trex_conda_prefix}/Library/bin;${_trex_conda_prefix}/Scripts;${_trex_conda_prefix}/bin;${_trex_conda_root}/condabin")
    set(ENV{PATH} "${_trex_conda_path_prefix};$ENV{PATH}")
    set(ENV{CONDA_PREFIX} "${_trex_conda_prefix}")

    # This is used for generated Visual Studio target debugging properties.
    set(_trex_vs_debugger_env
        "CONDA_PREFIX=${_trex_conda_prefix}\nPATH=${_trex_conda_path_prefix};%PATH%")
    set(CMAKE_VS_DEBUGGER_ENVIRONMENT "${_trex_vs_debugger_env}")

    string(FIND "$ENV{PATH}" "${_trex_conda_prefix}" _trex_conda_in_path)
    message(STATUS "[TREX ENV] Applied CONDA_PREFIX='$ENV{CONDA_PREFIX}'")
    message(STATUS "[TREX ENV] PATH now contains CONDA_PREFIX at index ${_trex_conda_in_path}")
    message(STATUS "[TREX ENV] CMAKE_VS_DEBUGGER_ENVIRONMENT has been set for new targets")
else()
    if(WIN32)
        message(WARNING "[TREX ENV] ENV{CONDA_PREFIX} is empty during configure. Conda PATH injection was skipped.")
    endif()
endif()

set(CONDA_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "CONDA_PREFIX if set during cmake")
if(CONDA_PREFIX)
    message(STATUS "CONDA_PREFIX=${CONDA_PREFIX}")
    if(NOT TREX_CONDA_PACKAGE_INSTALL AND LINUX)
        list(APPEND CMAKE_BUILD_RPATH "${CONDA_PREFIX}/lib")
        list(APPEND CMAKE_INSTALL_RPATH "${CONDA_PREFIX}/lib")
    endif()
endif()

if(TREX_CONDA_PACKAGE_INSTALL)
    add_definitions(-DTREX_CONDA_PACKAGE_INSTALL=1)
    set(CMAKE_BUILD_RPATH "${CONDA_PREFIX}/lib")
    
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif("${isSystemDir}" STREQUAL "-1")
endif()

add_subdirectory(src/commons)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

include(CheckCXXCompilerFlag)

if(MSVC)
    check_cxx_compiler_flag("/std:c++23" COMPILER_SUPPORTS_CXX23)
    if(NOT COMPILER_SUPPORTS_CXX23)
        set(COMPILER_SUPPORTS_CXX23 ON) # we dont have a way to check for C++23 in MSVC?
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /MP")
else()
    check_cxx_compiler_flag("-std=c++2b" COMPILER_SUPPORTS_CXX23)  # c++2b is often used for C++23 in GCC and Clang
endif()

if(COMPILER_SUPPORTS_CXX23)
    set(CMAKE_CXX_STANDARD 23)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()

if(MSVC)
    message(STATUS "Using latest C++ standard.")
else()
    message(STATUS "Using c++${CMAKE_CXX_STANDARD} standard.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug;DEBUG;")

set(CMAKE_DEBUG_POSTFIX -d)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    if(NOT ${TREX_ENABLE_SHARED_INTERNAL_LIBS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden -fstrict-aliasing")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden -fstrict-aliasing")
        message(STATUS "Using hidden visibility for non-shared TRex build on Apple Clang.")
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-undefined,error")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-undefined,error")
endif()

set(ORIGINAL_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS "ORIGINAL_CXX_FLAGS=${ORIGINAL_CXX_FLAGS}")

if(MSVC)
    # Enable incremental linking for Debug builds
    #set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL")
    
    # Enable incremental linking for Release builds (optional)
    #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL")
    #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /PROFILE")
endif()

if(WIN32)
    set(BUILD_SHARED_LIBS OFF)
    string(REPLACE " " ";" VERSION_LIST ${CMAKE_GENERATOR})
    list(GET VERSION_LIST 2 VSTD_VERSION_MAJOR)
    
    math (EXPR MSVC_TOOLSET_MAJOR "${MSVC_VERSION}/10")
    message(STATUS "CMAKE_GENERATOR_TOOLSET ${MSVC_TOOLSET_MAJOR} ${CMAKE_GENERATOR} ${VSTD_VERSION_MAJOR}")
    
    #if(${MSVC_TOOLSET_MAJOR} LESS 190)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_ATOMIC_ALIGNMENT_FIX")
    #endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -DCURL_STATICLIB")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
    #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG")
    
    #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF /LTCG")
    #set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
    #set(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
    #set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
    #set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
    #set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /O2 /Oi /Oy /Ot /Ob2 /Gy /GF")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd /bigobj")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2 /Zi")
    
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /O2 /Oi /Oy /Ot /Ob2 /Gy /GF")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd /bigobj")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2 /Zi")
    
    set(GLEW_STATIC ON)
else()
    set(BUILD_SHARED_LIBS OFF)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Emscripten") #-pthread -s PTHREAD_POOL_SIZE=8  -s TOTAL_MEMORY=250MB
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -O3 -s FETCH=1 -s ALLOW_MEMORY_GROWTH=0 -s WASM=1 -s FULL_ES3=1 -s USE_GLFW=3 -s LEGACY_GL_EMULATION=0 -s \"BINARYEN_METHOD='native-wasm'\" -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -s PTHREAD_POOL_SIZE=4 -s GL_PREINITIALIZED_CONTEXT=1") # 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -pthread")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --preload-file fonts --preload-file group_1.pv -s TOTAL_MEMORY=1000MB")
    #set(CMAKE_CONFIGURATION_TYPES Debug;Release FORCE)
    set(CMAKE_CONFIGURATION_TYPES "Release" FORCE)
endif()

set(TREX_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(TREX_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Set output paths
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake" )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DTT_DEBUG_ENABLED")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(TREX_CONDA_PACKAGE_INSTALL)
    set(TREX_CMAKE_BUILD_TYPE Release)
else()
    if(APPLE)
        set(TREX_CMAKE_BUILD_TYPE $<$<CONFIG:Debug>:Release>$<$<CONFIG:Release>:Debug>)
    else()
        set(TREX_CMAKE_BUILD_TYPE $<IF:$<CONFIG:Debug>,Debug,Release>)
    endif()
endif()

#if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
if (${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
    message(STATUS "Not adding any options.")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")

    #if(TREX_CONDA_PACKAGE_INSTALL)
        set(M1OPTIONS "")
    #else
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(M1OPTIONS "-mcpu=apple-m1")
    elseif(NOT TREX_CONDA_PACKAGE_INSTALL)
        set(M1OPTIONS "-march=native")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(M1OPTIONS "")
    endif()

    set(TREX_WARNING_FLAGS "-Wmove -Werror=return-stack-address") #-Wthread-safety")

    set(TREX_RELEASE_LTO_FLAG "-flto")
    if(APPLE AND TREX_ENABLE_MODULES)
        set(TREX_RELEASE_LTO_FLAG "-fno-lto")
    endif()

    set(C_SANITIZE_OPTIONS "")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g ${C_SANITIZE_OPTIONS} -fno-lto")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${M1OPTIONS} -O3 ${TREX_RELEASE_LTO_FLAG} -ffunction-sections -fdata-sections ${TREX_WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${M1OPTIONS} -O3 -g3 -gdwarf-5 -fno-omit-frame-pointer -fno-inline -fno-inline-functions -fno-lto ${TREX_WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic -Wall -Wextra -pedantic ${TREX_WARNING_FLAGS}")
    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=return-stack-address")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${M1OPTIONS} -O3 ${TREX_RELEASE_LTO_FLAG} -ffunction-sections -fdata-sections ${TREX_WARNING_FLAGS}")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${M1OPTIONS} -O3 -g ${C_SANITIZE_OPTIONS} -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat-pedantic")
    message(STATUS "Apple Silicon: ${CMAKE_SYSTEM_PROCESSOR} -> ${M1OPTIONS}")
    
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-nullability-extension")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-nullability-extension")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,-dead_strip_dylibs -Wl,-dead_strip")
elseif(MSVC)
else()
    message(STATUS "***** compiler ID ${CMAKE_CXX_COMPILER_ID}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    # Check if the compiler is GCC, and then verify its version.
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "GCC version 11 or newer is required.")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-local-addr -Wno-stringop-overflow -static-libstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic -Wall -Wextra -Wno-long-long -ggdb -DOM_DEBUG -O0 -no-pie")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -no-pie -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -ggdb -no-pie -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -no-pie -fPIC -Werror=return-local-addr")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -g0 -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -ggdb")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic -Wall -Wextra -Wno-long-long -ggdb -DOM_DEBUG -O0")

    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections")

    if(NOT TREX_CONDA_PACKAGE_INSTALL AND NOT LINUX)
        set(CMAKE_SKIP_RPATH TRUE)
    endif()
endif()

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DRELWITHDEBINFO")

if(XCODE)
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++23")
    #set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
    
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_RETURN_TYPE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_MISSING_NEWLINE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VALUE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VARIABLE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_SIGN_COMPARE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_FUNCTION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_INITIALIZER_NOT_FULLY_BRACKETED "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_HIDDEN_VIRTUAL_FUNCTIONS "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNINITIALIZED_AUTOS "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_BOOL_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_EMPTY_BODY "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_SUSPICIOUS_MOVE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_INFINITE_RECURSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_ASSIGN_ENUM "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_TIDY_BUGPRONE_REDUNDANT_BRANCH_CONDITION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_TIDY_MISC_REDUNDANT_EXPRESSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_MISSING_FIELD_INITIALIZERS "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_SUSPICIOUS_IMPLICIT_CONVERSION "NO")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_UNREACHABLE_CODE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN__DUPLICATE_METHOD_MATCH "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN__DUPLICATE_METHOD "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_ENUM_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_INT_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_CONSTANT_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_NON_VIRTUAL_DESTRUCTOR "YES")
endif()

set(trex_subproject_CMAKE_ARGS "")
message(STATUS "TREX PYTHON VARS: lib:${PYTHON_LIBRARY} inc:${PYTHON_INCLUDE_DIR} exec:${PYTHON_EXECUTABLE}")

SET(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
    -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
    -DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
    -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
    -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
    #-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_CXX_FLAGS=${ORIGINAL_CXX_FLAGS}
    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
    -DBUILD_SHARED_LIBS=OFF 
    -DSKIP_INSTALL_FILES=1
    -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS_DEBUG=${CMAKE_EXE_LINKER_FLAGS_DEBUG}
    -DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}
    -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY:PATH=${PYTHON_LIBRARY}
    -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
)

if(APPLE)
    set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
        -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
        -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
else()
    set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

#if(APPLE AND CMAKE_COMPILER_IS_GNUCXX)
#  set(trex_subproject_CMAKE_ARGS "-DWITH_AVFOUNDATION=OFF" "-DCMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH}")
#endif()

message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET = ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "MACOSX_DEPLOYMENT_TARGET = $ENV{MACOSX_DEPLOYMENT_TARGET}")


if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set_property (TARGET imgui APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc -D'NS_FORMAT_ARGUMENT(A)=' -D_Nullable_result=_Nullable")
endif()


#find_package(commons)
include_directories(src/commons)
include_directories(src/commons/common)


add_subdirectory(src)

include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/cpputils/src/)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


if(NOT WIN32 AND ${TREX_BUILD_DOCS})
    message(STATUS "Searching Sphinx at ${CONDA_PREFIX} ${CONDA_PREFIX}/bin")
    find_package(Sphinx REQUIRED) #PATHS ${CONDA_PREFIX} ${CONDA_PREFIX}/bin )
    if(NOT Sphinx_FOUND)
        message(ERROR "Cannot find Sphinx executable")
    else()
        get_filename_component(SPHINX_DIR ${SPHINX_EXECUTABLE} DIRECTORY)
        message(STATUS "Found sphinx at ${SPHINX_EXECUTABLE} (dir: ${SPHINX_DIR}).")

        add_custom_target(
          doc ALL
          COMMAND $<TARGET_FILE:TRex> -d "${CMAKE_SOURCE_DIR}/../docs" -h rst
          COMMAND export PATH="$ENV{PATH}:${SPHINX_DIR}"
          COMMAND echo "CONDA_PREFIX=${CONDA_PREFIX} PATH=${PATH}"
          COMMAND make html
          DEPENDS
            TRex
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../docs
          COMMENT "Generating API documentation with Doxygen"
          VERBATIM
        )
    endif()
elseif(${TREX_BUILD_DOCS})
    message(STATUS "Searching Sphinx at ${CONDA_PREFIX} ${CONDA_PREFIX}/bin")
    find_package(Sphinx REQUIRED) #PATHS ${CONDA_PREFIX} ${CONDA_PREFIX}/bin )
    if(NOT Sphinx_FOUND)
        message(ERROR "Cannot find Sphinx executable")
    else()
        get_filename_component(SPHINX_DIR ${SPHINX_EXECUTABLE} DIRECTORY)
        message(STATUS "Found sphinx at ${SPHINX_EXECUTABLE} (dir: ${SPHINX_DIR}).")
        add_custom_target(
            #C:\Users\tristan\Videos\f9_f10_f11_emptytank_20180220_120156.pv
          doc ALL #-i Z:\work\f9_f10_f11_emptytank_20180220_120156\%256d.mp4 -averaging_method mode -threshold 30 -use_closing -closing_size 5 -correct_luminance
          COMMAND ${CMAKE_SOURCE_DIR}/../docs/make.bat html "${SPHINX_DIR}" "${CMAKE_SOURCE_DIR}/../docs" "$<TARGET_FILE:trex>"
          DEPENDS
            trex
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../docs
          COMMENT "Generating API documentation with Doxygen PATH=$ENV{PATH}:${SPHINX_DIR} ${CMAKE_SOURCE_DIR}/../docs/make.bat"
          VERBATIM
        )
    endif()
endif()

add_subdirectory(Tests)

if(WIN32 AND NOT "${_trex_vs_debugger_env}" STREQUAL "")
    # Ensure existing executable targets also get debugger env entries in the
    # generated .vcxproj Debugging properties.
    get_property(_trex_all_targets GLOBAL PROPERTY TARGETS)
    foreach(_trex_tgt ${_trex_all_targets})
        get_target_property(_trex_is_imported ${_trex_tgt} IMPORTED)
        if(_trex_is_imported)
            continue()
        endif()

        get_target_property(_trex_type ${_trex_tgt} TYPE)
        if(_trex_type STREQUAL "EXECUTABLE")
            set_property(TARGET ${_trex_tgt} PROPERTY VS_DEBUGGER_ENVIRONMENT "${_trex_vs_debugger_env}")
        endif()
    endforeach()
    message(STATUS "[TREX ENV] Applied VS debugger environment to executable targets")
endif()

if(POLICY CMP0114)
  cmake_policy(POP)
endif()

message(STATUS "LINK_OPTIONS = ${CMAKE_CXX_LINK_FLAGS} ${CMAKE_CXX_LINK_FLAGS_RELEASE} ")
message(STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_STANDARD_LIBRARIES = ${CMAKE_CXX_STANDARD_LIBRARIES}")
message(STATUS "CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES = ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
