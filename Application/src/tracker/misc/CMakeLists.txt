file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(GIT_SOURCES "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" "${CMAKE_SOURCE_DIR}/src/GitSHA1.h")

# add actual library
if(UNIX)
  add_library(tracker_misc STATIC ${SRCS} ${HDRS} ${GIT_SOURCES})
elseif(WIN32)
  add_library(tracker_misc STATIC ${SRCS} ${HDRS} ${GIT_SOURCES})
endif()

if(WITH_FFMPEG)
    target_compile_definitions(tracker_misc PUBLIC WITH_FFMPEG=true)
endif()

target_link_libraries(tracker_misc PUBLIC
    framegrabber_misc
    pv
    commons
    lzo
)

#add_dependencies(tracker_misc commons) #Commons::OpenCV)
#target_precompile_headers(tracker_misc REUSE_FROM commons)
