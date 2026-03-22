set(TRACKER_PYTHON_HEADERS
    ${TRACKER_PYTHON_PUBLIC_HEADERS}
    ${TRACKER_PYTHON_PRIVATE_HEADERS}
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(WIN32)
    set(TRACKER_PYTHON_LIBRARY_TYPE SHARED)
elseif(TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set(TRACKER_PYTHON_LIBRARY_TYPE SHARED)
else()
    set(TRACKER_PYTHON_LIBRARY_TYPE STATIC)
endif()

# add actual library
if(UNIX)
    add_library(tracker_python ${TRACKER_PYTHON_LIBRARY_TYPE} ${TRACKER_PYTHON_SOURCES} ${TRACKER_PYTHON_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_init.py
        ${CMAKE_CURRENT_SOURCE_DIR}/closed_loop_beta.py
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_learn_category.py)
elseif(WIN32)
    add_library(tracker_python ${TRACKER_PYTHON_LIBRARY_TYPE} ${TRACKER_PYTHON_SOURCES} ${TRACKER_PYTHON_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/../../GitSHA1.cpp" ../../GitSHA1.h)
endif()

if(APPLE AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set_target_properties(tracker_python PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    target_compile_options(tracker_python PRIVATE -fvisibility=default)
    target_link_options(tracker_python PRIVATE "-Wl,-undefined,error")
endif()

if(TRACKER_PYTHON_LIBRARY_TYPE STREQUAL "SHARED")
    if(WIN32)
        target_compile_definitions(tracker_python
            PRIVATE "TREX_EXPORT=__declspec(dllexport)" "TREX_TYPE_EXPORT=__declspec(dllexport)"
            INTERFACE "TREX_EXPORT=" "TREX_TYPE_EXPORT="
        )
    else()
        target_compile_definitions(tracker_python
            PRIVATE "TREX_EXPORT=__attribute__((visibility(\"default\")))" "TREX_TYPE_EXPORT=__attribute__((visibility(\"default\"),type_visibility(\"default\")))"
            INTERFACE "TREX_EXPORT=" "TREX_TYPE_EXPORT="
        )
    endif()
else()
    target_compile_definitions(tracker_python PRIVATE "TREX_EXPORT=" "TREX_TYPE_EXPORT=")
endif()

if(WIN32)
    target_compile_options(tracker_python PRIVATE /bigobj)
    target_link_libraries(tracker_python PRIVATE
        trex_detect
        trex_tracking
        trex_data
        trex_core
        pv
        Commons::All
    )
else()
    target_link_libraries(tracker_python PUBLIC trex_detect trex_tracking trex_data trex_core pv Commons::All)
endif()

if(NOT pybind11_external)
    target_link_libraries(tracker_python PUBLIC pybind11::pybind11 pybind11::embed pybind11::module)
else()
    message(STATUS "not internal pybind11")
    target_link_libraries(tracker_python PUBLIC ${PYTHON_LIBRARIES})
endif()

add_executable(trex_check_python check_python.cpp)
target_link_libraries(trex_check_python PUBLIC Commons::All)

if(NOT pybind11_external)
    target_link_libraries(trex_check_python PUBLIC pybind11::pybind11 pybind11::embed pybind11::module)
else()
    target_link_libraries(trex_check_python PUBLIC ${PYTHON_LIBRARIES})
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
