cmake_minimum_required(VERSION 3.0)
project(ProcessedVideo)

# compiler flags
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_VERSION "com.apple.compilers.llvm.clang.1_0")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++20")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

elseif(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

elseif(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -DWIN32_LEAN_AND_MEAN")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

include_directories(commons/common)
include_directories(lzo)

message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS_RELEASE}")

#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
file(GLOB SRCS *.cpp
               *.h)

if(LINUX)
    add_library(
        pv
        STATIC
        ${SRCS}
    )
elseif(WIN32)
    add_library(
        pv
        STATIC
        ${SRCS}
	)
elseif(APPLE)
	add_library(
	pv
	STATIC
	${SRCS}
	)
else()
    add_library(
        pv
        STATIC
        ${SRCS}
	)
endif()

set_target_properties(pv
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_subdirectory(lzo)
target_link_libraries(pv PUBLIC lzo cnpy commons)
add_dependencies(pv commons)
