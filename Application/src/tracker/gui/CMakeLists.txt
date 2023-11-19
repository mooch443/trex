file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
  add_library(tracker_gui STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(tracker_gui STATIC ${SRCS} ${HDRS})
endif()

add_dependencies(tracker_gui tracker_misc commons)

if(WITH_FFMPEG)
    target_compile_definitions(tracker_gui PRIVATE WITH_FFMPEG=true)
endif()

target_link_libraries(tracker_gui PRIVATE
    tracker_tracking
    tracker_misc
    commons
)
#target_precompile_headers(tracker_gui REUSE_FROM commons)

if(APPLE)
  target_compile_options(tracker_gui PRIVATE "-DNS_FORMAT_ARGUMENT(A)=" "-D_Nullable_result=_Nullable")
endif()
