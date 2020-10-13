file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
  add_library(tracker_misc STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(tracker_misc STATIC ${SRCS} ${HDRS})
endif()

target_link_libraries(tracker_misc PRIVATE
#    commons
#    pv
    lzo
#	${EXTRA_LIBS}
	cnpy
)

add_dependencies(tracker_misc TRex::OpenCV)

if(TREX_CONDA_PACKAGE_INSTALL)
	if(WIN32)
		set(CONDA_PREFIX_VAR "$ENV{BUILD_PREFIX}")
	else()
		set(CONDA_PREFIX_VAR "$ENV{PREFIX}")
	endif()
	string(REPLACE "\\" "\\\\" CONDA_PREFIX_VAR "${CONDA_PREFIX_VAR}")
    target_compile_definitions(tracker_misc PRIVATE TREX_PYTHON_PATH="${CONDA_PREFIX_VAR}")
    message(STATUS "Setting TREX_PYTHON_PATH=${CONDA_PREFIX_VAR}")
else()
    message(STATUS "Adding compile definitions: ${PYTHON_EXECUTABLE}")
    target_compile_definitions(tracker_misc PRIVATE
        TREX_PYTHON_PATH="${PYTHON_EXECUTABLE}"
    )
    message(STATUS "Setting TREX_PYTHON_PATH=${PYTHON_EXECUTABLE}")
endif()
