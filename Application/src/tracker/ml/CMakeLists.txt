file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
  add_library(tracker_ml STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(tracker_ml STATIC ${SRCS} ${HDRS})
endif()

target_link_libraries(tracker_ml PUBLIC commons)
add_dependencies(tracker_ml tracker_misc commons)

if(WITH_FFMPEG)
    target_compile_definitions(tracker_ml PRIVATE WITH_FFMPEG=true)
endif()

if(WIN32)
  if(NOT COMMONS_NO_PYTHON)
     target_link_libraries(tracker_ml PUBLIC tracker_python)
	   set_target_properties(tracker_ml PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
  endif()
endif()

#target_precompile_headers(tracker_tracking REUSE_FROM commons)
