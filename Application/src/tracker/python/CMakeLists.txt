file(GLOB HDRS *.h ${CMAKE_SOURCE_DIR}/*.py)
file(GLOB SRCS *.cpp)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
    add_library(tracker_python STATIC ${SRCS} ${HDRS} 
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_init.py
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_learn_category.py)
elseif(WIN32)
    add_library(tracker_python SHARED ${SRCS} ${HDRS})
endif()

if(WIN32)
    target_link_libraries(tracker_python PUBLIC
        tracker_misc
        commons
        #cnpy
        #Commons::OpenCV
	    #Commons::LibPNG
		#Commons::LibZIP
		#Commons::LibZ
    )
    target_compile_options(tracker_python PUBLIC /D"TREX_EXPORTS")
else()
    target_link_libraries(tracker_python PUBLIC cnpy commons)
endif()

if(NOT pybind11_external)
    target_link_libraries(tracker_python PUBLIC pybind11::pybind11 pybind11::embed pybind11::module)
else()
    message(STATUS "not internal pybind11")
    target_link_libraries(tracker_python PUBLIC ${PYTHON_LIBRARIES})
endif()

add_executable(trex_check_python check_python.cpp)
target_link_libraries(trex_check_python commons #cnpy #Commons::OpenCV 
    #Commons::LibZIP Commons::LibZ Commons::LibPNG
    )
add_dependencies(trex_check_python commons cnpy Commons::OpenCV Commons::LibZIP)

if(NOT pybind11_external)
    target_link_libraries(trex_check_python pybind11::pybind11 pybind11::embed pybind11::module)
else()
    target_link_libraries(trex_check_python ${PYTHON_LIBRARIES})
endif()

#target_precompile_headers(trex_check_python REUSE_FROM commons)
#target_precompile_headers(tracker_python REUSE_FROM commons)

add_dependencies(tracker_python trex_check_python )
install(TARGETS trex_check_python
    EXPORT TRexCheckPython
    LIBRARY DESTINATION usr/share/trex/lib
    ARCHIVE DESTINATION usr/share/trex/lib
    RUNTIME DESTINATION usr/share/trex
    BUNDLE DESTINATION usr/share/trex
    INCLUDES DESTINATION usr/share/trex/include
    CONFIGURATIONS Release
)
