set(TREX_PYTHON_IMPL_HEADERS
    ${TREX_PYTHON_IMPL_PUBLIC_HEADERS}
    ${TREX_PYTHON_IMPL_PRIVATE_HEADERS}
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(WIN32)
    set(TREX_PYTHON_LIBRARY_TYPE SHARED)
    set(TREX_PYTHON_IMPL_LIBRARY_TYPE SHARED)
elseif(TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set(TREX_PYTHON_LIBRARY_TYPE SHARED)
    set(TREX_PYTHON_IMPL_LIBRARY_TYPE SHARED)
else()
    set(TREX_PYTHON_LIBRARY_TYPE STATIC)
    set(TREX_PYTHON_IMPL_LIBRARY_TYPE STATIC)
endif()

if(TREX_PYTHON_IMPL_LIBRARY_TYPE STREQUAL "SHARED")
    set(TREX_PYTHON_IMPL_IS_DYNAMIC 1)
else()
    set(TREX_PYTHON_IMPL_IS_DYNAMIC 0)
endif()

message(STATUS
    "[TREX PYTHON] staging=${TREX_PYTHON_LIBRARY_TYPE} impl=${TREX_PYTHON_IMPL_LIBRARY_TYPE} "
    "dynamic_impl=${TREX_PYTHON_IMPL_IS_DYNAMIC}"
)

# ==============================================================================
# trex_python_staging: pure C++ lifecycle and path management
# No pybind11 dependencies. This is what consumers link to.
# ==============================================================================

if(UNIX)
    add_library(trex_python_staging ${TREX_PYTHON_LIBRARY_TYPE} ${TREX_PYTHON_STAGING_SOURCES} ${TREX_PYTHON_STAGING_PUBLIC_HEADERS})
elseif(WIN32)
    add_library(trex_python_staging ${TREX_PYTHON_LIBRARY_TYPE} ${TREX_PYTHON_STAGING_SOURCES} ${TREX_PYTHON_STAGING_PUBLIC_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/../../GitSHA1.cpp" ../../GitSHA1.h)
endif()

if(APPLE AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set_target_properties(trex_python_staging PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    target_compile_options(trex_python_staging PRIVATE -fvisibility=default)
    target_link_options(trex_python_staging PRIVATE "-Wl,-undefined,error")
endif()

if(TREX_PYTHON_LIBRARY_TYPE STREQUAL "SHARED")
    if(WIN32)
        target_compile_definitions(trex_python_staging
            PRIVATE "TREX_EXPORT=__declspec(dllexport)" "TREX_TYPE_EXPORT=__declspec(dllexport)"
        )
    else()
        target_compile_definitions(trex_python_staging
            PRIVATE "TREX_EXPORT=__attribute__((visibility(\"default\")))" "TREX_TYPE_EXPORT=__attribute__((visibility(\"default\"),type_visibility(\"default\")))"
        )
    endif()
else()
    target_compile_definitions(trex_python_staging PRIVATE "TREX_EXPORT=" "TREX_TYPE_EXPORT=")
endif()
target_compile_definitions(trex_python_staging PRIVATE TREX_PYTHON_IMPL_IS_DYNAMIC=${TREX_PYTHON_IMPL_IS_DYNAMIC})

if(WIN32)
    target_compile_options(trex_python_staging PRIVATE /bigobj)
    target_link_libraries(trex_python_staging PRIVATE
        trex_data
        trex_core
        pv
        Commons::All
    )
else()
    target_link_libraries(trex_python_staging PUBLIC trex_data trex_core pv Commons::All)
endif()

# ==============================================================================
# trex_python: pybind11-heavy implementation (GPURecognition + backends)
# ==============================================================================

# add actual library
if(UNIX)
    add_library(trex_python ${TREX_PYTHON_IMPL_LIBRARY_TYPE} ${TREX_PYTHON_IMPL_SOURCES} ${TREX_PYTHON_IMPL_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_init.py
        ${CMAKE_CURRENT_SOURCE_DIR}/closed_loop_beta.py
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_learn_category.py)
elseif(WIN32)
    add_library(trex_python ${TREX_PYTHON_IMPL_LIBRARY_TYPE} ${TREX_PYTHON_IMPL_SOURCES} ${TREX_PYTHON_IMPL_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/../../GitSHA1.cpp" ../../GitSHA1.h)
endif()

if(APPLE AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set_target_properties(trex_python PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    target_compile_options(trex_python PRIVATE -fvisibility=default)
    target_link_options(trex_python PRIVATE "-Wl,-undefined,error")
endif()

if(TREX_PYTHON_IMPL_LIBRARY_TYPE STREQUAL "SHARED")
    if(WIN32)
        target_compile_definitions(trex_python
            PRIVATE "TREX_EXPORT=__declspec(dllexport)" "TREX_TYPE_EXPORT=__declspec(dllexport)"
        )
    else()
        target_compile_definitions(trex_python
            PRIVATE "TREX_EXPORT=__attribute__((visibility(\"default\")))" "TREX_TYPE_EXPORT=__attribute__((visibility(\"default\"),type_visibility(\"default\")))"
        )
    endif()
else()
    target_compile_definitions(trex_python PRIVATE "TREX_EXPORT=" "TREX_TYPE_EXPORT=")
endif()

if(WIN32)
    target_compile_options(trex_python PRIVATE /bigobj)
    target_link_libraries(trex_python PRIVATE
        trex_tracking
        trex_data
        trex_core
        pv
        Commons::All
        trex_python_staging
    )
else()
    target_link_libraries(trex_python PUBLIC trex_tracking trex_data trex_core pv Commons::All trex_python_staging)
endif()

if(NOT pybind11_external)
    target_link_libraries(trex_python PUBLIC pybind11::pybind11 pybind11::embed pybind11::module)
else()
    message(STATUS "not internal pybind11")
    target_link_libraries(trex_python PUBLIC ${PYTHON_LIBRARIES})
endif()

# On Unix, staging should also link the heavy implementation so RPATH resolution works
add_executable(trex_check_python check_python.cpp)
target_link_libraries(trex_check_python PUBLIC Commons::All)

if(NOT pybind11_external)
    target_link_libraries(trex_check_python PUBLIC pybind11::pybind11 pybind11::embed pybind11::module)
else()
    target_link_libraries(trex_check_python PUBLIC ${PYTHON_LIBRARIES})
endif()

if(NOT COMMONS_ENABLE_MODULES AND NOT COMMONS_DONT_USE_PCH
    AND COMMONS_CAN_REUSE_PCH_ACROSS_TARGETS
        AND DEFINED COMMONS_PCH_OWNER AND NOT "${COMMONS_PCH_OWNER}" STREQUAL "")
    target_precompile_headers(trex_check_python REUSE_FROM ${COMMONS_PCH_OWNER})
    target_precompile_headers(trex_python REUSE_FROM ${COMMONS_PCH_OWNER})
endif()

add_dependencies(trex_python trex_check_python )
install(TARGETS trex_check_python
    EXPORT TRexCheckPython
    LIBRARY DESTINATION usr/share/trex/lib
    ARCHIVE DESTINATION usr/share/trex/lib
    RUNTIME DESTINATION usr/share/trex
    BUNDLE DESTINATION usr/share/trex
    INCLUDES DESTINATION usr/share/trex/include
    CONFIGURATIONS Release
)
