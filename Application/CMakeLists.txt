cmake_minimum_required(VERSION 3.0)
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    message(STATUS "* Please consider to switch to CMake 3.11.0")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug;DEBUG;")

message("-- USER environment variable is set to: " $ENV{USER})

project(TRex)
if(POLICY CMP0114)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0114 NEW)
endif()
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
include(ExternalProject)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0")
    include(FetchContent)
endif()

enable_testing ()

set(MACOSX_BUNDLE_GUI_IDENTIFIER "com.tristan.tracker")
set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "tristan")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")

#set(CMAKE_CXX_VISIBILITY_PRESET hidden)

option(TREX_BUILD_DOCS OFF)
option(TREX_WITH_TESTS OFF)
if(APPLE)
    option(TREX_WITH_OPENGL OFF)
else()
    option(TREX_WITH_OPENGL ON)
endif()
option(TREX_BUILD_GLFW OFF)
option(TREX_BUILD_PNG OFF)
option(TREX_BUILD_ZIP OFF)
option(TREX_BUILD_ZLIB OFF)
option(TREX_BUILD_OPENCV ON)
option(TREX_CONDA_PACKAGE_INSTALL OFF)
option(WITH_PYLON "use pylon support" OFF)
option(WITH_GITSHA1 "include git hash" ON)
option(CMN_USE_OPENGL2 ON)
option(WITH_FFMPEG "Compile with ffmpeg streaming support" ON)
option(WITH_HTTPD "Compile with libmicrohttpd (enable web features)" OFF)
option(NO_PUT_TIME "Disables std::put_time usage" OFF)
option(OPENCV_EXTRA_FLAGS "OpenCV extra CMake flags" "")
option(TREX_ENABLE_CPP20 "Enables CPP20 language standard (if available). Only for GCC and Clang at the moment. Experimental feature." OFF)

if(APPLE)
    if(CMAKE_OSX_DEPLOYMENT_TARGET AND NOT ${CMAKE_OSX_DEPLOYMENT_TARGET} MATCHES "")
        message(STATUS "Found OSX deployment target: '${CMAKE_OSX_DEPLOYMENT_TARGET}'")
        set(TREX_MACOS_VERSION ${CMAKE_OSX_DEPLOYMENT_TARGET})
    else()
        message(STATUS "Need to detect OSX version.")
        if(NOT CMAKE_OSX_SYSROOT)
            message(FATAL_ERROR "No OSX sysroot was set.")
        else()
            get_filename_component(osx_dir ${CMAKE_OSX_SYSROOT} NAME)
            string(REPLACE ".sdk" "" osx_dir ${osx_dir})
            string(REPLACE "MacOSX" "" osx_dir ${osx_dir})
            message(STATUS "OSX Target: ${osx_dir}")
            set(TREX_MACOS_VERSION ${osx_dir})
        endif()
    endif()
endif()

if(APPLE AND "${TREX_MACOS_VERSION}" VERSION_LESS "10.12")
    message(STATUS "This system does not support MetalKit (detected version: ${TREX_MACOS_VERSION}), enabling OpenGL.")
    SET(TREX_WITH_OPENGL ON CACHE BOOL "" FORCE)
elseif(APPLE)
    message(STATUS "MetalKit support detected (version: ${TREX_MACOS_VERSION}).")
else()
    message(STATUS "Non-Apple systems use OpenGL.")
    SET(TREX_WITH_OPENGL ON CACHE BOOL "" FORCE)
endif()

if(NO_PUT_TIME)
    add_definitions(-DNO_PUT_TIME=true)
endif()

set(CONDA_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "CONDA_PREFIX if set during cmake")
if(CONDA_PREFIX)
    message(STATUS "CONDA_PREFIX=${CONDA_PREFIX}")
endif()

if(TREX_CONDA_PACKAGE_INSTALL)
    add_definitions(-DTREX_CONDA_PACKAGE_INSTALL=1)
    set(CMAKE_BUILD_RPATH "${CONDA_PREFIX}/lib")
    
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif("${isSystemDir}" STREQUAL "-1")
endif()

set(CMAKE_DEBUG_POSTFIX -d)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden -D'NS_FORMAT_ARGUMENT(A)=' -D_Nullable_result=_Nullable")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden -D'NS_FORMAT_ARGUMENT(A)=' -D_Nullable_result=_Nullable")
endif()
set(ORIGINAL_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS "ORIGINAL_CXX_FLAGS=${ORIGINAL_CXX_FLAGS}")

if(WIN32)
    set(BUILD_SHARED_LIBS OFF)
    string(REPLACE " " ";" VERSION_LIST ${CMAKE_GENERATOR})
    list(GET VERSION_LIST 2 VSTD_VERSION_MAJOR)
    
    math (EXPR MSVC_TOOLSET_MAJOR "${MSVC_VERSION}/10")
    message(STATUS "CMAKE_GENERATOR_TOOLSET ${MSVC_TOOLSET_MAJOR} ${CMAKE_GENERATOR} ${VSTD_VERSION_MAJOR}")
    
    #if(${MSVC_TOOLSET_MAJOR} LESS 190)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_ATOMIC_ALIGNMENT_FIX")
    #endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DCURL_STATICLIB")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd /bigobj")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2")
    
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd /bigobj")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /MT /bigobj /O2 /Oi /Oy /Ot /Ob2")
    
    set(GLEW_STATIC ON)
else()
    set(BUILD_SHARED_LIBS OFF)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/cnpy/)

add_library(TRex::X11 INTERFACE IMPORTED)
if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(TRex::X11 INTERFACE X11::X11 ${X11_LIBRARIES} dl)
    
    message(STATUS "X11_LIBRARIES: ${X11_LIBRARIES} ${X11_xcb_FOUND} DIRS:${X11_LIBRARY_DIRS} CFLAGS:${X11_CFLAGS}")
    link_directories(${X11_LIBRARY_DIRS})
    include_directories(${X11_INCLUDE_DIRS})

    #FIND_PATH(UUID_INCLUDE_DIR uuid/uuid.h
    #     /usr/local/include
    #/usr/include
    #)

    #FIND_LIBRARY(UUID_LIBRARY
    #  NAMES uuid
    #  PATHS /lib /usr/lib /usr/local/lib
    #  )

    #message(STATUS "UUID: ${UUID_LIBRARY} ${UUID_INCLUDE_DIR} ${X11_xcb_LIB}")
endif()

add_library(TRex::OpenGL INTERFACE IMPORTED)
if(TREX_BUILD_GLFW OR UNIX)
    if(WIN32)
        find_package(OpenGL REQUIRED)
        target_link_libraries(TRex::OpenGL INTERFACE OpenGL::GL)
        message(STATUS "OpenGL library ${OPENGL_LIBRARY}")
        include_directories(${OPENGL_INCLUDE_DIR})
        add_definitions(-DTREX_HAS_OPENGL=true)
    elseif(UNIX AND NOT APPLE)
        find_package(OpenGL REQUIRED)
        target_link_libraries(TRex::OpenGL INTERFACE OpenGL::GL)
        SET(OPENGL_LIBRARY OpenGL::GL)
        message(STATUS "OpenGL library ${OPENGL_LIBRARY}")
        include_directories(${OPENGL_INCLUDE_DIR})
        add_definitions(-DTREX_HAS_OPENGL=true)
    else()
        if(TREX_WITH_OPENGL)
            find_library(OPENGL_LIBRARY OpenGL)
            target_link_libraries(TRex::OpenGL INTERFACE ${OPENGL_LIBRARY})
            add_definitions(-DTREX_HAS_OPENGL=true)
        else()
            add_definitions(-DTREX_HAS_OPENGL=false)
        endif()
    endif()
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(TRex::OpenGL INTERFACE OpenGL::GL)
    message(STATUS "OpenGL library ${OPENGL_LIBRARY}")
    include_directories(${OPENGL_INCLUDE_DIR})
    add_definitions(-DTREX_HAS_OPENGL=true)
endif()

if(APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-undefined,error -framework IOKit -framework CoreFoundation -framework CoreGraphics -framework CoreVideo -framework OpenCL -framework Cocoa -framework AppKit -framework AVKit -framework AVFoundation -framework CoreMedia -framework QuartzCore -framework Accelerate")
    
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    
    if(CMAKE_OSX_SYSROOT AND NOT ${CMAKE_OSX_SYSROOT} STREQUAL "")
        find_library(METALKIT_LIBRARY MetalKit ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH)
        find_library(METAL_LIBRARY Metal ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH)
        message(STATUS "CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks NO_DEFAULT_PATH")
    else()
        find_library(METALKIT_LIBRARY MetalKit)
        find_library(METAL_LIBRARY Metal)
        message(STATUS "Trying to find Metal in default paths.")
    endif()
    
    find_library(QUARTZ_LIBRARY QuartzCore)
    if(TREX_WITH_OPENGL)
        set(EXTRA_LIBS glad imgui TRex::OpenGL)
    else()
        set(EXTRA_LIBS imgui)
    endif()
    set(EXTRA_LIBS
        ${EXTRA_LIBS}
        ${COCOA_LIBRARY}
        ${IOKIT_LIBRARY}
        ${COREVIDEO_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
    
    if(NOT METAL_LIBRARY)
        message(STATUS "Did not find metal")
        
    else()
        message(STATUS "Did find metal")
        set(EXTRA_LIBS ${EXTRA_LIBS}
            ${METALKIT_LIBRARY}
            ${METAL_LIBRARY}
        )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Metal")
    endif()
    
    message(STATUS "Mac extra libs: ${EXTRA_LIBS}")
else()
    if(TREX_WITH_OPENGL)
        set(EXTRA_LIBS glad imgui TRex::OpenGL)
    else()
        set(EXTRA_LIBS imgui)
    endif()
endif(APPLE)

set(TREX_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(TREX_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Set output paths
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake" )

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wall)

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
if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
    
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat-pedantic")
    message(STATUS "hidden visibility in main project ${CMAKE_CXX_FLAGS}")
    
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-nullability-extension")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-nullability-extension")
elseif(MSVC)
else()
    message(STATUS "***** compiler ID ${CMAKE_CXX_COMPILER_ID}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if(TREX_ENABLE_CPP20)
    set(cxx_flag_std_number "")
    if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-std=c++2a HAVE_FLAG_STD_CXX20)
        if(HAVE_FLAG_STD_CXX20)
            message(STATUS "Using support for C++20")
            set(CMAKE_CXX_STANDARD 20)
            set(cxx_flag_std_number "-std=c++2a")
        endif()

        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${cxx_flag_std_number}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${cxx_flag_std_number}")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${cxx_flag_std_number}")
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic -Wall -Wextra -Wno-long-long -ggdb -DOM_DEBUG -O0 -no-pie")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -no-pie")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -ggdb -no-pie")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -no-pie -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -ggdb")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic -Wall -Wextra -Wno-long-long -ggdb -DOM_DEBUG -O0")
endif()

if(XCODE)
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++17")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
    
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_RETURN_TYPE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_MISSING_NEWLINE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VALUE "YES")
    #set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VARIABLE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_SIGN_COMPARE "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_FUNCTION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_INITIALIZER_NOT_FULLY_BRACKETED "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_HIDDEN_VIRTUAL_FUNCTIONS "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNINITIALIZED_AUTOS "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_BOOL_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_EMPTY_BODY "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_ENUM_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_INT_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_CONSTANT_CONVERSION "YES")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_NON_VIRTUAL_DESTRUCTOR "YES")
endif()

set(trex_subproject_CMAKE_ARGS "")

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
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF 
    -DSKIP_INSTALL_FILES=1
    -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}
    -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY:PATH=${PYTHON_LIBRARY}
    -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
)

if(APPLE)
    set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
        -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
        -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

#if(APPLE AND CMAKE_COMPILER_IS_GNUCXX)
#  set(trex_subproject_CMAKE_ARGS "-DWITH_AVFOUNDATION=OFF" "-DCMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH}")
#endif()

message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET = ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "MACOSX_DEPLOYMENT_TARGET = $ENV{MACOSX_DEPLOYMENT_TARGET}")

if(NOT APPLE)
    set(PY_VERSION "3.6")
    find_package(PythonLibs ${PY_VERSION} REQUIRED)
    find_package(PythonInterp ${PY_VERSION} REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})

    message(STATUS "${PYTHON_INCLUDE_DIRS} ${PYTHON_LIBRARIES} ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    message(STATUS "Using python version ${PYTHON_EXECUTABLE}")
endif()

if (DEFINED TRACKER_GLOBAL_THREADS)
    ADD_DEFINITIONS(-DTRACKER_GLOBAL_THREADS=${TRACKER_GLOBAL_THREADS})
endif()
ADD_DEFINITIONS(-DCV_STATIC_ANALYSIS=0)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/src/commons/CMakeModules" )

if(APPLE AND WITH_PYLON)
    FIND_LIBRARY(PYLON_LIBRARIES Pylon)
    if(PYLON_LIBRARIES STREQUAL "PYLON_LIBRARIES-NOTFOUND" OR NOT ${WITH_PYLON})
        set(PYLON_FOUND false)
        add_definitions(-DWITH_PYLON=false)
    else()
        set(PYLON_FOUND true)
        add_definitions(-DWITH_PYLON=true)
        message(STATUS "Pylon found. ${PYLON_LIBRARIES} lol")
        include_directories(/Library/Frameworks/pylon.framework/Headers/GenICam)
    endif()
    
elseif(WITH_PYLON)
    find_package(BaslerPylon)
    if(${PYLON_FOUND})
        include_directories(${PYLON_INCLUDE_DIRS})
        add_definitions(-DWITH_PYLON=true)
        message(STATUS "Pylon found.")
    else()
        add_definitions(-DWITH_PYLON=false)
    endif()
endif()

set(GLFW_add_flags "")
set(GLFW_CXX_FLAGS "${ORIGINAL_CXX_FLAGS}")
set(GLFW_C_FLAGS "${CMAKE_C_FLAGS}")

if(WIN32)
    set(GLFW_add_flags ${GLFW_add_flags} -DCMAKE_CXX_FLAGS_RELEASE="/MT" -DCMAKE_CXX_FLAGS_DEBUG="/MTd" -DUSE_MSVC_RUNTIME_LIBRARY_DLL=OFF)
endif()

add_library (TRex::GLFW INTERFACE IMPORTED)
if(NOT TREX_BUILD_GLFW)
    find_package(GLFW)
    if(NOT GLFW_FOUND)
        message(STATUS "Cannot find GLFW. Trying to build it instead.")
        set(TREX_BUILD_GLFW ON CACHE BOOL "" FORCE)
    else()
        target_link_libraries (TRex::GLFW INTERFACE ${GLFW_LIBRARY})
        include_directories(${GLFW_INCLUDE_DIR})
        
        get_filename_component(GLFW_DIR ${GLFW_LIBRARY} DIRECTORY)
        get_filename_component(GLFW_LIBRARY ${GLFW_LIBRARY} NAME)
        message(STATUS "Linking GLFW: ${GLFW_LIBRARY} includes: ${GLFW_INCLUDE_DIR} dir: ${GLFW_DIR}")
        
        target_link_directories(TRex::GLFW INTERFACE ${GLFW_DIR})
    endif()
endif()

if(TREX_BUILD_GLFW)
    set(GLFW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/glfw)
    ExternalProject_Add(glfw PREFIX glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.4
        CMAKE_ARGS
            ${trex_subproject_CMAKE_ARGS}
            "-DCMAKE_INSTALL_PREFIX=${GLFW_INSTALL_DIR}"
            "-DGLFW_BUILD_EXAMPLES=OFF"
            "-DGLFW_BUILD_TESTS=OFF"
            "-DGLFW_BUILD_DOCS=OFF"
            ${GLFW_add_flags}
        UPDATE_COMMAND ""
    )
    
    set(GLFW_INCLUDE_DIR ${GLFW_INSTALL_DIR}/include)
    link_directories(${GLFW_INSTALL_DIR}/lib)
    message(STATUS "GLFW LIBRARIES install:${GLFW_INSTALL_DIR}")
    message(STATUS "glfw include: ${GLFW_INCLUDE_DIR}")

    include_directories(${GLFW_INCLUDE_DIR})
    target_link_libraries (TRex::GLFW INTERFACE glfw3)
    add_dependencies(TRex::GLFW glfw)
endif()

if(TREX_WITH_OPENGL)
    # glad
    if(CMN_USE_OPENGL2)
        set(GLAD_PROFILE "compatibility")
        set(GLAD_API "gl=2.0")
        add_definitions(-DCMN_USE_OPENGL2=true)
    else()
        set(GLAD_PROFILE "core")
        set(GLAD_API "gl=4.1")
        add_definitions(-DCMN_USE_OPENGL2=false)
    endif()

    ExternalProject_Add(gladex PREFIX gladex
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v0.1.33
        CMAKE_ARGS
            ${trex_subproject_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DGLAD_PROFILE=${GLAD_PROFILE}
            -DGLAD_API=${GLAD_API}
            -DGLAD_INSTALL=ON
            -DGLAD_REPRODUCIBLE=ON
        UPDATE_COMMAND ""
    )

    ExternalProject_Get_Property(gladex INSTALL_DIR)
    include_directories(${INSTALL_DIR}/include)
    link_directories(${INSTALL_DIR}/lib)
    set(GLAD_INSTALL_DIR ${INSTALL_DIR})
    message(STATUS "glad: ${INSTALL_DIR}")

    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD=true)
endif()

set (imgui_path ${CMAKE_SOURCE_DIR}/src/imgui)
message(STATUS "imgui: ${imgui_path}")
file(GLOB imgui_sources ${imgui_path}/*.cpp ${imgui_path}/*.h
    ${imgui_path}/backends/imgui_impl_glfw*.cpp ${imgui_path}/backends/imgui_impl_glfw*.h
)

if(TREX_WITH_OPENGL)
    set(imgui_sources ${imgui_sources} ${imgui_path}/backends/imgui_impl_opengl2.cpp ${imgui_path}/backends/imgui_impl_opengl2.h)
    if(NOT CMN_USE_OPENGL2)
        set(imgui_sources ${imgui_sources} ${imgui_path}/backends/imgui_impl_opengl3.cpp ${imgui_path}/backends/imgui_impl_opengl3.h)
    endif()
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND METAL_LIBRARY)
    set(imgui_sources ${imgui_sources} ${imgui_path}/backends/imgui_impl_metal.mm ${imgui_path}/backends/imgui_impl_metal.h)
endif()

add_library(imgui STATIC ${imgui_sources})
if(TREX_BUILD_GLFW)
    add_dependencies(imgui glfw)
endif()
if(TREX_WITH_OPENGL)
    add_dependencies(imgui gladex)
endif()

# this is circumventing some weird CMake shuffeling of dependencies. these libraries always have to go last
# at least once, but multiple times if necessary.
if(UNIX AND NOT APPLE)
    target_link_libraries(TRex::GLFW INTERFACE glad TRex::X11 pthread)
endif()

include_directories(${imgui_path})
target_link_libraries(imgui TRex::GLFW)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set_property (TARGET imgui APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc")
endif()

if(WITH_FFMPEG)
    find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
    find_library(AVCODEC_LIBRARY avcodec)

    find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
    find_library(AVFORMAT_LIBRARY avformat)

    find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
    find_library(AVUTIL_LIBRARY avutil)

    find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
    find_library(AVDEVICE_LIBRARY avdevice)

    #find_path(LIBSWSSCALE_INCLUDE_DIR libswscale/swsscale.h)
    find_library(LIBSWSSCALE_LIBRARY swscale)

    if(NOT AVCODEC_LIBRARY OR NOT AVFORMAT_LIBRARY OR NOT AVDEVICE_LIBRARY OR NOT LIBSWSSCALE_LIBRARY)
        message(STATUS "Cannot find the FFMPEG libraries (WITH_FFMPEG=ON) avformat:${AVFORMAT_LIBRARY} avdevice:${AVDEVICE_LIBRARY} swsscale:${LIBSWSSCALE_LIBRARY}")
        message(STATUS "Disabling FFMPEG.")
        set(WITH_FFMPEG OFF)
    else()
        set(FFMPEG_INCLUDE_DIRS ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
        set(FFMPEG_LIBRARIES ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} ${LIBSWSSCALE_LIBRARY})
        message(STATUS "Found ffmpeg: ${FFMPEG_LIBRARIES}")
        add_library (TRex::FFMPEG INTERFACE IMPORTED)
        target_link_libraries(TRex::FFMPEG INTERFACE ${FFMPEG_LIBRARIES})
        include_directories(${FFMPEG_INCLUDE_DIRS})
    endif()
endif()

if(WITH_HTTPD)
    set(httpd_PREFIX "${CMAKE_BINARY_DIR}/lmhttpd")
    set(httpd_INSTALL_DIR "${CMAKE_BINARY_DIR}/lmhttpd")
    set(httpd_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${httpd_INSTALL_DIR})

    set(MHD_LIBRARIES microhttpd)

    message(STATUS "HTTPD: ${httpd_INSTALL_DIR}/lib")
    link_directories(${httpd_INSTALL_DIR}/lib)
    include_directories(${httpd_INSTALL_DIR}/include)
    add_definitions(-DWITH_MHD=true)
else()
    add_definitions(-DWITH_MHD=false)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/cpputils/src/)
set(pybind11_external OFF)

if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    set(pybind11_install_dir ${CMAKE_BINARY_DIR}/pybind11)
    ExternalProject_Add (pybind11
        PREFIX ${pybind11_install_dir}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DPYBIND11_TEST=OFF
            -DPYBIND11_INSTALL=ON
            ${trex_subproject_CMAKE_ARGS}

        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.6.2 #99ef2b846742e2298b97904e5be43f5ea5d71d8c
        UPDATE_COMMAND ""
    )
    ExternalProject_Get_Property(pybind11 INSTALL_DIR)

    set(pybind11_external ON)
    set(pybind11_LIBRARY ${PYTHON_LIBRARIES} )
    set(pybind11_LIBRARY pybind11::pybind11 pybind11::embed pybind11::module )
    include_directories(${pybind11_install_dir}/include)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message(STATUS "pybind11 external_project ${PYTHON_INCLUDE_DIRS} ${INSTALL_DIR} ${pybind11_LIBRARY}")

else()
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.6.2
    )

    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
endif()

if(NOT TREX_BUILD_PNG AND NOT ${TREX_BUILD_PNG})
    find_package(PNG)
    
    if(NOT PNG_FOUND)
        message(STATUS "Did not find PNG. Trying to build it.")
        set(TREX_BUILD_PNG ON CACHE BOOL "" FORCE)
        
    else()
        include_directories(${PNG_INCLUDE_DIR})
    
        add_library (TRex::LibPNG INTERFACE IMPORTED)
        target_link_libraries (TRex::LibPNG INTERFACE ${PNG_LIBRARIES})
    
        message(STATUS "png: ${PNG_LIBRARIES} ${PNG_INCLUDE_DIR}")
    endif()
endif()

find_package(ZLIB)
if((NOT ${TREX_BUILD_ZLIB}) AND ZLIB_FOUND AND ZLIB_VERSION_STRING VERSION_GREATER 1.2.10)
    message(STATUS "Found zlib here: ${ZLIB_LIBRARY}")
    set(libzip_zlib_lib "${ZLIB_LIBRARY}")
    unset(zlib_install_dir)
    
    add_library (TRex::LibZ INTERFACE IMPORTED)
    target_link_libraries (TRex::LibZ INTERFACE ${ZLIB_LIBRARY})
    
    unset(ZLIB_LIBRARY)
    
else()
    message(STATUS "Building libZ and libZIP")
    set(TREX_BUILD_ZLIB ON CACHE BOOL "" FORCE)
    set(TREX_BUILD_ZIP ON CACHE BOOL "" FORCE)
    set(ZLIB_LIBRARY "z")
    set(zlib_install_dir "${CMAKE_BINARY_DIR}/zlib")
    
    if(WIN32)
        link_directories(${zlib_install_dir}/lib/)
        set(ZLIB_LIBRARY "${zlib_install_dir}/lib/zlib$<IF:$<CONFIG:Debug>,d,>.lib")
        set(libzip_zlib_lib "${zlib_install_dir}/lib/zlib.lib")
        set(trex_subproject_CMAKE_ARGS1 ${trex_subproject_CMAKE_ARGS}
            -DCMAKE_STATIC_LINKER_FLAGS_INIT="/LIBPATH:\"${zlib_install_dir}/lib\"")
    else()
        set(trex_subproject_CMAKE_ARGS1 ${trex_subproject_CMAKE_ARGS})
    endif()
    
    EXTERNALPROJECT_ADD(Z_LIB
        PREFIX ${zlib_install_dir}
        GIT_REPOSITORY https://github.com/jeking3/zlib.git
        GIT_TAG c4a2cd007c1b7edef164b29d5f221f83a78c7507
        UPDATE_COMMAND ""

        INSTALL_DIR ${zlib_install_dir}
        CMAKE_ARGS ${trex_subproject_CMAKE_ARGS1} -DCMAKE_DEBUG_POSTFIX=d -DBUILD_STATIC_AND_SHARED=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=${zlib_install_dir}
        -DBUILD_STATIC_AND_SHARED=OFF -DBUILD_SHARED_LIBS=OFF
    )
    
    add_library (TRex::LibZ INTERFACE IMPORTED)
    target_link_libraries (TRex::LibZ INTERFACE ${ZLIB_LIBRARY})

    set(ZLIB_INCLUDE_DIR ${zlib_install_dir}/include)
    include_directories(${ZLIB_INCLUDE_DIR})
    link_directories(${zlib_install_dir}/lib)

    message(STATUS "zlib install dir ${zlib_install_dir}")
    message(STATUS "ZLIB include dir ${ZLIB_INCLUDE_DIR}")
    message(STATUS "ZLIB_LIBRARY = ${ZLIB_LIBRARY} ${ZLIB_VERSION_STRING}")
    
    unset(ZLIB_LIBRARY)
    #unset(ZLIB_INCLUDE_DIR)
endif()

if(NOT TREX_BUILD_ZIP)
    find_package(Zip)
    
    if(NOT Zip_FOUND)
        message(STATUS "Did not find ZIP. Trying to build it.")
        set(TREX_BUILD_ZIP ON CACHE BOOL "" FORCE)
        
    else()
        include_directories(${ZIP_INCLUDE_DIR})
    
        add_library (TRex::LibZIP INTERFACE IMPORTED)
        target_link_libraries (TRex::LibZIP INTERFACE ${ZIP_LIBRARIES} TRex::LibZ)
        message(STATUS "zip: ${ZIP_LIBRARIES} ${ZIP_INCLUDE_DIR}")
    endif()
endif()

if(TREX_BUILD_ZLIB)
    if(WIN32)
        set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
            -DZLIB_INCLUDE_DIR:PATH=${zlib_install_dir}/include
            -DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/zlibd.lib
            -DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/zlib.lib
            -DZLIB_LIBRARY_RELWITHDEBINFO:FILEPATH=${zlib_install_dir}/lib/zlib.lib
            #-DCMAKE_PREFIX_PATH=${zlib_install_dir}/lib/zlib.lib
            #-DZLIB_LIBRARY:FILEPATH=${zlib_install_dir}/lib/zlib.lib
            #-DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/zlib.lib
            #-DZLIB_LIBRARY_RELWITHDEBINFO:FILEPATH=${zlib_install_dir}/lib/zlib.lib
            #-DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/zlibd.lib
        )
        
    else()
        set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
            -DZLIB_INCLUDE_DIR:PATH=${zlib_install_dir}/include
            -DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/libz.a
            -DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/libzd.a
            -DZLIB_LIBRARY_RELWITHDEBINFO:FILEPATH=${zlib_install_dir}/lib/libz.a
            #-DZLIB_INCLUDE_DIR:PATH=${zlib_install_dir}/include
            #-DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/libz.a
            #-DZLIB_LIBRARY_RELWITHDEBINFO:FILEPATH=${zlib_install_dir}/lib/libz.a
            #-DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/libzd.a
        )
    endif()
endif()

if(TREX_BUILD_ZIP)
    set(LIBZIP_LIBRARY "zip$<IF:$<CONFIG:Debug>,d,>")
    set(libzip_install_dir "${CMAKE_BINARY_DIR}/libzip")
    
    if(NOT WIN32)
        set(LIBZIP_LIBRARY "zip$<IF:$<CONFIG:Debug>,d,>")
    endif()
    
    ExternalProject_Add(libzip
        PREFIX ${libzip_install_dir}
        DEPENDS TRex::LibZ
        GIT_REPOSITORY https://github.com/nih-at/libzip.git
        GIT_TAG v1.7.3
        UPDATE_COMMAND ""
        #PATCH_COMMAND ${libzip_patchcommand}
        CMAKE_ARGS
            ${libzip_configure}
            -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIR}
            -DENABLE_BZIP2=OFF
            -DBUILD_REGRESS=OFF
            -DBUILD_EXAMPLES=OFF
            -DBUILD_DOC=OFF
            -DBUILD_TOOLS=OFF
            -DENABLE_GNUTLS=OFF
            -DENABLE_OPENSSL=OFF
            -DENABLE_LZMA=OFF
            ${trex_subproject_CMAKE_ARGS}
            -DCMAKE_DEBUG_POSTFIX=d
    )
    
    if(TREX_BUILD_ZLIB)
        add_dependencies(libzip Z_LIB)
    endif()

    # patching a simple (void*) -> (char*) for windows
    #ExternalProject_Add_Step(libzip
    #    WINDOWS_CONFIGURE DEPENDEES download DEPENDERS configure
    #    WORKING_DIRECTORY "${libzip_install_dir}"
    #    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #        "${CMAKE_SOURCE_DIR}/libzip_windows/zip_source_winzip_aes_encode.c"
    #        "${libzip_install_dir}/src/libzip/lib/zip_source_winzip_aes_encode.c"
    #)

    set(LIBZIP_INCLUDE_DIR "${libzip_install_dir}/include")
    
    include_directories(${LIBZIP_INCLUDE_DIR})
    link_directories(${libzip_install_dir}/lib)
    message(STATUS "ZIP: ${LIBZIP_LIBRARY}")
    message(STATUS "include libzip ${LIBZIP_INCLUDE_DIR} install: ${libzip_install_dir}")
    
    add_library (TRex::LibZIP INTERFACE IMPORTED)
    target_link_libraries (TRex::LibZIP INTERFACE ${LIBZIP_LIBRARY})
    
    add_dependencies(TRex::LibZIP TRex::LibZ libzip)
    
    unset(LIBZIP_LIBRARY)
endif()

if(${TREX_BUILD_PNG})
    message(STATUS "TRex will build PNG.")
    set(png_install_dir "${CMAKE_BINARY_DIR}/libpng")
    set(PNG_LIBRARY "png")

    if(WIN32)
        set(PNG_LIBRARY "libpng16_static$<IF:$<CONFIG:Debug>,d,>")
    else()
        set(PNG_LIBRARY "${png_install_dir}/lib/libpng16$<IF:$<CONFIG:Debug>,d,>.a")
    endif()

    message(STATUS "PNG_LIBRARY = ${PNG_LIBRARY}")
    message(STATUS "trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}")

    if(APPLE)
        set(PNG_PATCH patch -N -s -d ${png_install_dir}/src/libpng_custom -p1 -i ${CMAKE_SOURCE_DIR}/lib_png_m1patch.diff || echo 1)
    else()
        set(PNG_PATCH "")
    endif()

    #URL ftp://ftp.simplesystems.org/pub/libpng/png/src/history/libpng15/libpng-1.5.10.tar.gz
    #URL_MD5 9e5d864bce8f06751bbd99962ecf4aad
    EXTERNALPROJECT_ADD(libpng_custom
        DEPENDS TRex::LibZ TRex::LibZIP
        PREFIX ${png_install_dir}
        UPDATE_COMMAND ""
        PATCH_COMMAND ${PNG_PATCH}

        GIT_REPOSITORY  https://github.com/glennrp/libpng.git
        GIT_TAG a37d4836519517bdce6cb9d956092321eca3e73b

        INSTALL_DIR ${png_install_dir}
        CMAKE_ARGS ${trex_subproject_CMAKE_ARGS} -DCMAKE_PREFIX_PATH=${zlib_install_dir} # to find zlib
        -DPNG_TESTS=OFF
        -DPNG_SHARED=OFF
    )

    include_directories(${png_install_dir}/include)
    link_directories(${png_install_dir}/lib)
    message(STATUS "libpng dir ${png_install_dir}/lib")
    message(STATUS "png library ${PNG_LIBRARY}")
    
    add_library (TRex::LibPNG INTERFACE IMPORTED)
    target_link_libraries (TRex::LibPNG INTERFACE ${PNG_LIBRARY} TRex::LibZ)
    add_dependencies(TRex::LibPNG ${PNG_LIBRARY} libpng_custom TRex::LibZ)
    
    unset(PNG_LIBRARY)

    if(WIN32)
       set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
            -DPNG_LIBRARY_RELEASE:FILEPATH=${png_install_dir}/lib/libpng16_static.lib
            -DPNG_LIBRARY_DEBUG:FILEPATH=${png_install_dir}/lib/libpng16_staticd.lib
            -DPNG_PNG_INCLUDE_DIR:FILEPATH=${png_install_dir}/include)
   else()
        set(trex_subproject_CMAKE_ARGS ${trex_subproject_CMAKE_ARGS}
            -DPNG_LIBRARY_RELEASE:FILEPATH=${png_install_dir}/lib/libpng16.a
            -DPNG_LIBRARY_DEBUG:FILEPATH=${png_install_dir}/lib/libpng16d.a
            -DPNG_PNG_INCLUDE_DIR:FILEPATH=${png_install_dir}/include
        )
   endif()
endif()

# this is circumventing some weird CMake shuffeling of dependencies. these libraries always have to go last
# at least once, but multiple times if necessary.
if(UNIX AND NOT APPLE)
    target_link_libraries(TRex::LibZIP INTERFACE TRex::X11 pthread)
    target_link_libraries(TRex::LibPNG INTERFACE TRex::X11 pthread)
endif()

if(WIN32)
    set(OpenCV_STATIC ON)
endif()

if(NOT ${TREX_BUILD_OPENCV})
    message(STATUS "Trying to find OpenCV...")
    find_package( OpenCV )
    
    if(NOT OpenCV_FOUND)
        set(TREX_BUILD_OPENCV ON CACHE BOOL "OpenCV Override" FORCE)
        message(STATUS "Cannot find OpenCV. Trying to build it.")
    else() 
        include_directories( ${OpenCV_INCLUDE_DIRS} )
        message(STATUS "Found OpenCV Version: ${OpenCV_VERSION} include: ${OpenCV_INCLUDE_DIRS} libs: ${OpenCV_LIBRARIES}")
        set(OpenCV_LIBS ${OpenCV_LIBRARIES})
    
        add_library (TRex::OpenCV INTERFACE IMPORTED)
        target_link_libraries (TRex::OpenCV INTERFACE ${OpenCV_LIBS})
        if(UNIX AND NOT APPLE)
            find_package(TIFF)
            message(STATUS "TIFF libraries: ${TIFF_LIBRARIES}")
            target_link_libraries(TRex::OpenCV INTERFACE tiff cairo)
        endif()
        set(OpenCV_LIBS TRex::OpenCV)
    endif()
endif()

if(${TREX_BUILD_OPENCV})
    set(trigger_build_dir "${CMAKE_BINARY_DIR}/force_opencv")
    set(BUILD_SHARED_LIBS OFF)
    message(STATUS "Building OpenCV...")
    
    if(UNIX)
        get_filename_component(PKG_CONFIG_PATH ${AVDEVICE_LIBRARY} DIRECTORY)
        
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{PKG_CONFIG_PATH})
        set(ENV{PKG_CONFIG_PATH} "${PKG_CONFIG_PATH}/pkgconfig:$ENV{PKG_CONFIG_PATH}")
        message(STATUS "PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")
    endif()
    
    set(OPENCV_PREFIX_PATH "${png_install_dir}" "${zlib_install_dir}" ${CMAKE_PREFIX_PATH} )
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
    set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS}
        ${trex_subproject_CMAKE_ARGS}
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
        #"-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        #"-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        #"-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        #"-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
		#"-DCMAKE_CXX_FLAGS=${ORIGINAL_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -DCURL_STATICLIB -D_ENABLE_ATOMIC_ALIGNMENT_FIX"
    )
    
    # if(APPLE)
    #     set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS_}
    #         "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}"
    #         "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
    #         "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    # endif()
   
   # if(TREX_BUILD_PNG)
   #     if(WIN32)
   #         set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS_}
   #              "-DPNG_LIBRARY_RELEASE:FILEPATH=${png_install_dir}/lib/libpng16_static.lib"
   #              "-DPNG_LIBRARY_DEBUG:FILEPATH=${png_install_dir}/lib/libpng16_staticd.lib"
   #              "-DPNG_PNG_INCLUDE_DIR:FILEPATH=${png_install_dir}/include")
   #     else()
   #          set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS_}
   #              "-DPNG_LIBRARY_RELEASE:FILEPATH=${png_install_dir}/lib/libpng16.a"
   #              "-DPNG_LIBRARY_DEBUG:FILEPATH=${png_install_dir}/lib/libpng16d.a"
   #              "-DPNG_PNG_INCLUDE_DIR:FILEPATH=${png_install_dir}/include"
   #          )
   #     endif()
   # endif()
   
   # if(TREX_BUILD_ZLIB)
   #     if(WIN32)
   #         set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS_}
   #                          "-DZLIB_INCLUDE_DIR:PATH=${zlib_install_dir}/include"
   #                          "-DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/zlibd.lib"
			# 				"-DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/zlib.lib")
   #     else()
   #          set(OPENCV_EXTRA_FLAGS_ ${OPENCV_EXTRA_FLAGS_}
   #              "-DZLIB_INCLUDE_DIR:PATH=${zlib_install_dir}/include"
   #              "-DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_install_dir}/lib/libz.a"
   #              "-DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_install_dir}/lib/libzd.a"
   #          )
   #     endif()
   # endif()
   
   if(TREX_CONDA_PACKAGE_INSTALL)
		set(TREX_CMAKE_BUILD_TYPE Release)
   else()
		set(TREX_CMAKE_BUILD_TYPE $<$<CONFIG:Debug>:Release>$<$<CONFIG:Release>:Debug>)
   endif()
   
   message(STATUS "TREX_CMAKE_BUILD_TYPE ${TREX_CMAKE_BUILD_TYPE}")
   message(STATUS "ALL trex_subproject_CMAKE_ARGS: ${trex_subproject_CMAKE_ARGS}")
   set(OPENCV_CMAKE_ARGS
            #${OPENCV_EXTRA_FLAGS_}
            ${trex_subproject_CMAKE_ARGS}
			-DCMAKE_BUILD_TYPE:STRING=${TREX_CMAKE_BUILD_TYPE}
            -DBUILD_DOCS:BOOL=FALSE
            -DBUILD_EXAMPLES:BOOL=FALSE
            -DBUILD_TESTS:BOOL=FALSE
            -DBUILD_SHARED_LIBS:BOOL=FALSE
            -DBUILD_NEW_PYTHON_SUPPORT:BOOL=OFF
            -DBUILD_WITH_DEBUG_INFO=OFF
            -DWITH_EIGEN:BOOL=FALSE
            -DWITH_CUDA:BOOL=FALSE
            -DWITH_FFMPEG:BOOL=${WITH_FFMPEG}
            -DWITH_MSMF:BOOL=FALSE
            -DWITH_IPP:BOOL=FALSE
            -DWITH_GTK:BOOL=FALSE
            -DWITH_CAIRO:BOOL=FALSE
            -DWITH_PNG:BOOL=TRUE
            -DWITH_GSTREAMER:BOOL=OFF
            -DWITH_CAROTENE:BOOL=FALSE
            -DBUILD_PERF_TESTS:BOOL=FALSE
            -DBUILD_opencv_apps:BOOL=FALSE
            -DBUILD_opencv_java:BOOL=FALSE
            -DBUILD_opencv_python:BOOL=OFF
            -DBUILD_opencv_objdetect:BOOL=ON
            -DBUILD_opencv_dnn:BOOL=ON
            -DBUILD_opencv_ml:BOOL=OFF
            -DBUILD_opencv_stitching:BOOL=OFF
            -DBUILD_opencv_flann:BOOL=ON
            -DWITH_WEBP=OFF
            -DBUILD_PNG:BOOL=OFF
            -DBUILD_ZLIB:BOOL=OFF
            -DBUILD_PROTOBUF:BOOL=ON
            -DBUILD_JPEG:BOOL=ON
            -DWITH_OPENJPEG:BOOL=OFF
            -DBUILD_JASPER:BOOL=OFF
            -DBUILD_opencv_python3=OFF
            -DBUILD_opencv_python2=OFF
            -DWITH_JASPER:BOOL=OFF
            -DBUILD_opencv_world:BOOL=OFF
            -DBUILD_opencv_gapi=ON
            -DBUILD_WITH_STATIC_CRT:BOOL=ON
            -DBUILD_FAT_JAVA_LIB=OFF
            -DBUILD_TIFF=ON
            -DWITH_OPENCL=ON
            # -DAVCODEC_LIBRARY:FILEPATH=${AVCODEC_LIBRARY}
            # -DAVFORMAT_LIBRARY:FILEPATH=${AVFORMAT_LIBRARY} 
            # -DAVUTIL_LIBRARY:FILEPATH=${AVUTIL_LIBRARY} -DAVDEVICE_LIBRARY=${AVDEVICE_LIBRARY} 
            # -DLIBSWSSCALE_LIBRARY:FILEPATH=${LIBSWSSCALE_LIBRARY}
            # -DAVCODEC_INCLUDE_DIR=${AVCODEC_INCLUDE_DIR}
            # -DAVFORMAT_INCLUDE_DIR=${AVFORMAT_INCLUDE_DIR} 
            # -DAVUTIL_INCLUDE_DIR=${AVUTIL_INCLUDE_DIR}
            #-DAVDEVICE_INCLUDE_DIR=${AVDEVICE_INCLUDE_DIR}
            #"-DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}"
            #"-DPYTHON_LIBRARY:PATH=${PYTHON_LIBRARY}"
            #"-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}"
            -DWITH_OPENEXR=OFF
            "-DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}"
            "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/opencv_install"
            -DCMAKE_DEBUG_POSTFIX=)

    message(STATUS "OPENCV_EXTRA_FLAGS_ = ${OPENCV_EXTRA_FLAGS_}")
    set(TREX_OPENCV_VERSION "4.5.4")
    if(APPLE)
        message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
            message(STATUS "ARM?")
        else()
            message(STATUS "x86_64 macos")
            set(TREX_OPENCV_VERSION "4.3.0")
        endif()
    endif()

    if(UNIX)
        message(STATUS "Using custom configure command")
        ExternalProject_add(CustomOpenCV
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencv
            GIT_REPOSITORY  https://github.com/opencv/opencv.git
            GIT_TAG ${TREX_OPENCV_VERSION}
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env PKG_CONFIG_PATH=${PKG_CONFIG_PATH}/pkgconfig ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${TREX_CMAKE_BUILD_TYPE} -G ${CMAKE_GENERATOR} ${OPENCV_CMAKE_ARGS} ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/CustomOpenCV
            BUILD_COMMAND ${CMAKE_COMMAND} -E env PKG_CONFIG_PATH=${PKG_CONFIG_PATH}/pkgconfig ${CMAKE_COMMAND} --build . --config ${TREX_CMAKE_BUILD_TYPE}
            INSTALL_COMMAND ${CMAKE_COMMAND} -E env PKG_CONFIG_PATH=${PKG_CONFIG_PATH}/pkgconfig ${CMAKE_COMMAND} --build . --config ${TREX_CMAKE_BUILD_TYPE} --target install
            CMAKE_ARGS
            ${OPENCV_CMAKE_ARGS})
    else()
        ExternalProject_add(CustomOpenCV
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencv
            GIT_REPOSITORY  https://github.com/opencv/opencv.git
            GIT_TAG ${TREX_OPENCV_VERSION}
            UPDATE_COMMAND ""
            CMAKE_ARGS
            ${OPENCV_CMAKE_ARGS})
    endif()
	# patching a simple (void*) -> (char*) for windows
    #ExternalProject_Add_Step(CustomOpenCV
    #    WINDOWS_CONFIGURE DEPENDEES download DEPENDERS configure
    #    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    #    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #        "${CMAKE_SOURCE_DIR}/opencv_patch.patch"
     #       "${CMAKE_CURRENT_BINARY_DIR}/opencv/src/CustomOpenCV/CMakeLists.txt"
    #)

    if(APPLE)
        set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/lib/cmake/opencv4/ ${CMAKE_PREFIX_PATH})
        set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/)
    elseif(UNIX)
        set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/share/OpenCV/ ${CMAKE_PREFIX_PATH})
        set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/)
    else()
        set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/ ${CMAKE_PREFIX_PATH})
        set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/)
    endif()
	
	message(STATUS "OPENCV_EXTRA_FLAGS_ = ${OPENCV_EXTRA_FLAGS_}")
    
    add_dependencies(CustomOpenCV TRex::LibPNG)
    if(WIN32)
        string(REPLACE " " ";" VERSION_LIST ${CMAKE_GENERATOR})
        list(GET VERSION_LIST 2 VSTD_VERSION_MAJOR)
        
        math (EXPR MSVC_TOOLSET_MAJOR "${MSVC_VERSION}/10")
        message(STATUS "CMAKE_GENERATOR_TOOLSET ${MSVC_TOOLSET_MAJOR} ${CMAKE_GENERATOR} ${VSTD_VERSION_MAJOR}")
        message("Including OpenCV lib path: ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/x64/vc${VSTD_VERSION_MAJOR}/staticlib")
        set(OpenCV_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/x64/vc${VSTD_VERSION_MAJOR}/staticlib)
        link_directories(${OpenCV_LIB_PATH})
        #set(OpenCV_LIBS opencv_core454 opencv_imgproc454 opencv_imgcodecs454 opencv_video454 opencv_videoio454 opencv_highgui454 opencv_features2d454 opencv_flann454 opencv_objdetect454 opencv_ml454 opencv_gapi454 opencv_dnn454 opencv_calib3d454 libjpeg-turbo ittnotify ade libprotobuf libtiff quirc ${FFMPEG_LIBRARIES})
        set(OpenCV_LIBS opencv_core454 opencv_imgproc454 opencv_imgcodecs454 opencv_video454 opencv_videoio454 opencv_highgui454 
            opencv_features2d454 opencv_flann454 opencv_calib3d454 
            libjpeg-turbo ittnotify ade libprotobuf libtiff quirc ${FFMPEG_LIBRARIES})
    else()
		#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_stitching opencv_photo opencv_video opencv_videoio opencv_highgui opencv_features2d opencv_flann opencv_objdetect opencv_ml opencv_gapi opencv_dnn opencv_calib3d)
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_flann opencv_photo opencv_video opencv_videoio opencv_highgui opencv_features2d opencv_calib3d)
		set(OpenCV_LIBS ${OpenCV_LIBS} opencv_core libjpeg-turbo ittnotify ade libprotobuf quirc ${FFMPEG_LIBRARIES})
		if(APPLE)
			set(OpenCV_LIBS ${OpenCV_LIBS} libtiff)
		else()
			set(OpenCV_LIBS ${OpenCV_LIBS} libtiff z) #raw1394 usb-1.0 udev glib-2.0)
	#		set(OpenCV_LIBS ${OpenCV_LIBS} opencv_stitching opencv_photo)
		endif()
    endif()
    
    message(STATUS "Find OpenCV")
	find_package(OpenCV ${TREX_OPENCV_VERSION} PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
	if(OpenCV_FOUND)
		set(OpenCV_LIBS ${OpenCV_LIBRARIES})
		message(STATUS "Found opencv at ${OpenCV_INSTALL_PATH}, using its config.cmake")
	endif()
    
    if(UNIX)
        include_directories( ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/include/opencv4 )
    else()
        include_directories( ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/include )
    endif()
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/opencv_install/lib ${CMAKE_CURRENT_BINARY_DIR}/opencv_install/lib/opencv4/3rdparty)
    
    add_library (TRex::OpenCV INTERFACE IMPORTED)
    target_link_libraries (TRex::OpenCV INTERFACE ${OpenCV_LIBS})
    add_dependencies(TRex::OpenCV CustomOpenCV)
    message(STATUS "OpenCV: ${OpenCV_LIBS}")
    set(OpenCV_LIBS TRex::OpenCV)
endif()

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_subdirectory(src)

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
          COMMAND $<TARGET_FILE:TGrabs> -d "${CMAKE_SOURCE_DIR}/../docs" -h rst
          COMMAND PATH=$ENV{PATH}:${SPHINX_DIR} make html
          DEPENDS
            TRex
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../docs
          COMMENT "Generating API documentation with Doxygen"
          VERBATIM
        )
    endif()
endif()

if(POLICY CMP0114)
  cmake_policy(POP)
endif()
