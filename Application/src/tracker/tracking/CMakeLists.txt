file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
  add_library(tracker_tracking STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(tracker_tracking STATIC ${SRCS} ${HDRS})
endif()

target_link_libraries(tracker_tracking PRIVATE cnpy
#	tracker_python
)
add_dependencies(tracker_tracking TRex::OpenCV)

if(WIN32)
    target_link_libraries(tracker_tracking PRIVATE tracker_python)
	set_target_properties(tracker_tracking PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
endif()

if(TREX_CONDA_PACKAGE_INSTALL)
    target_compile_definitions(tracker_tracking PRIVATE TREX_PYTHON_PATH="CONDA_PREFIX")
else()
    message(STATUS "Adding compile definitions: ${PYTHON_EXECUTABLE}")
    target_compile_definitions(tracker_tracking PRIVATE
        TREX_PYTHON_PATH="${PYTHON_EXECUTABLE}"
    )
endif()
