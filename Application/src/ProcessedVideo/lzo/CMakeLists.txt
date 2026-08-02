file(GLOB HDRS *.h)
file(GLOB SRCS *.c)

if(WIN32)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -DWIN32_LEAN_AND_MEAN")
else()
    set(BUILD_SHARED_LIBS OFF)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
message(STATUS "lzo" ${SRCS})
# add actual library
if(UNIX)
  add_library(lzo STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(lzo STATIC ${SRCS} ${HDRS})
endif()
