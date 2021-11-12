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

add_dependencies(tracker_gui tracker_misc TRex::OpenCV)

target_link_libraries(tracker_gui PRIVATE cnpy
    tracker_tracking
    tracker_misc
)

if(APPLE)
  target_compile_options(tracker_gui PRIVATE "-DNS_FORMAT_ARGUMENT(A)=" "-D_Nullable_result=_Nullable")
endif()
