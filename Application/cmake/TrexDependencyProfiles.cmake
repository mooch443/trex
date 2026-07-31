include_guard(GLOBAL)

set(TREX_NATIVE_DEPENDENCY_OPTIONS
    COMMONS_BUILD_OPENCV
    COMMONS_BUILD_PNG
    COMMONS_BUILD_ZIP
    COMMONS_BUILD_ZLIB)

function(trex_apply_dependency_profile)
    set(TREX_CONFIGURE "" CACHE STRING
        "Native dependency profile: buildall, minimal, or empty for per-dependency selection")
    set_property(CACHE TREX_CONFIGURE PROPERTY STRINGS "" buildall minimal)

    set(_trex_profile_explicit OFF)
    if(NOT TREX_CONFIGURE STREQUAL "")
        if(NOT TREX_CONFIGURE STREQUAL "buildall" AND NOT TREX_CONFIGURE STREQUAL "minimal")
            message(FATAL_ERROR
                "Invalid TREX_CONFIGURE='${TREX_CONFIGURE}'. Expected 'buildall', 'minimal', or an empty value."
            )
        endif()

        set(_trex_profile_explicit ON)
        if(TREX_CONFIGURE STREQUAL "buildall")
            set(_trex_profile_build_value ON)
        else()
            set(_trex_profile_build_value OFF)
        endif()

        foreach(_trex_dependency_option IN LISTS TREX_NATIVE_DEPENDENCY_OPTIONS)
            if(DEFINED ${_trex_dependency_option})
                if("${${_trex_dependency_option}}" AND NOT _trex_profile_build_value)
                    message(FATAL_ERROR
                        "TREX_CONFIGURE=${TREX_CONFIGURE} requires ${_trex_dependency_option}=OFF, but it was explicitly set to ON. "
                        "Remove the individual override (or clear the CMake cache), or configure without TREX_CONFIGURE for a mixed provider selection."
                    )
                elseif(NOT "${${_trex_dependency_option}}" AND _trex_profile_build_value)
                    message(FATAL_ERROR
                        "TREX_CONFIGURE=${TREX_CONFIGURE} requires ${_trex_dependency_option}=ON, but it was explicitly set to OFF. "
                        "Remove the individual override (or clear the CMake cache), or configure without TREX_CONFIGURE for a mixed provider selection."
                    )
                endif()
            endif()
            set(${_trex_dependency_option} ${_trex_profile_build_value} CACHE BOOL
                "Controlled by TREX_CONFIGURE=${TREX_CONFIGURE}" FORCE)
        endforeach()
    else()
        foreach(_trex_dependency_option IN LISTS TREX_NATIVE_DEPENDENCY_OPTIONS)
            if(NOT DEFINED ${_trex_dependency_option})
                set(${_trex_dependency_option} ON CACHE BOOL
                    "Build the bundled provider (individual override; TREX_CONFIGURE is empty)")
            endif()
        endforeach()
    endif()

    set(TREX_CONFIGURE_EXPLICIT ${_trex_profile_explicit} PARENT_SCOPE)
    set(TREX_NATIVE_DEPENDENCY_OPTIONS ${TREX_NATIVE_DEPENDENCY_OPTIONS} PARENT_SCOPE)
endfunction()
