file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(_tracker_lib_type ${TREX_TRACKER_LIBRARY_TYPE})
if(NOT _tracker_lib_type)
  set(_tracker_lib_type STATIC)
endif()

# add actual library
add_library(tracker_tracking ${_tracker_lib_type} ${SRCS} ${HDRS})

target_link_libraries(tracker_tracking PUBLIC
    tracker_ml
    tracker_misc
    commons
)
add_dependencies(tracker_tracking tracker_misc commons)

if(WITH_HTTPD)
    add_dependencies(tracker_tracking LibMicroHttpd)
endif()

if(WITH_FFMPEG)
    target_compile_definitions(tracker_tracking PRIVATE WITH_FFMPEG=true)
endif()

if(WIN32)
  if(NOT COMMONS_NO_PYTHON)
     target_link_libraries(tracker_tracking PUBLIC tracker_python)
	   set_target_properties(tracker_tracking PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
  endif()
endif()

if(APPLE AND TREX_TRACKER_SHARED)
  target_link_options(tracker_tracking PRIVATE "-Wl,-undefined,error")
endif()

#target_precompile_headers(tracker_tracking REUSE_FROM commons)
