# - Try to find the Basler pylon SDK headers.
#
# Once done this will define:
#  PYLON_FOUND - SDK headers are available
#  PYLON_INCLUDE_DIRS - Include directories needed for the headers
#  PYLON_COMPILE_DEFINITIONS - Compile definitions used by the lazy Basler boundary
#  PYLON_FRAMEWORK_DIR - (macOS only) Framework search path for -F flag

include(FindPackageHandleStandardArgs)

if(APPLE)
    # macOS: pylon ships as pylon.framework; only C++ API headers are present.
    find_path(PYLON_INCLUDE_DIR
        NAMES PylonIncludes.h
        PATHS
            /Library/Frameworks/pylon.framework/Headers
            /Library/Frameworks/pylon.framework/Versions/A/Headers
            "$ENV{PYLON_ROOT}/pylon.framework/Headers"
            "$ENV{TREX_BASLER_RUNTIME}/pylon.framework/Headers"
        NO_DEFAULT_PATH
    )
    find_path(PYLON_GENAPI_INCLUDE_DIR
        NAMES GenICam.h
        PATHS
            /Library/Frameworks/pylon.framework/Versions/A/Headers/GenICam
            /Library/Frameworks/pylon.framework/Headers/GenICam
            "$ENV{PYLON_ROOT}/pylon.framework/Versions/A/Headers/GenICam"
        NO_DEFAULT_PATH
    )
    if(PYLON_INCLUDE_DIR)
        # Derive the directory that contains pylon.framework (used by -F).
        # Handles both:
        #   .../pylon.framework/Headers
        #   .../pylon.framework/Versions/A/Headers
        string(REPLACE "\\" "/" _pylon_include_norm "${PYLON_INCLUDE_DIR}")
        string(REGEX REPLACE "^(.*)/pylon\\.framework(/.*)?$" "\\1" PYLON_FRAMEWORK_DIR "${_pylon_include_norm}")
    endif()
else()
    set(_PYLON_HEADER_HINTS
        "$ENV{PYLON_ROOT}/include"
        "$ENV{PYLON_ROOT}/Development/include"
        "$ENV{PYLON_ROOT}/Include"
        "/opt/pylon/include"
        "/opt/pylon5/include"
        "/opt/pylon6/include"
        "C:/Program Files/Basler/pylon 5/Development/include"
        "C:/Program Files/Basler/pylon 6/Development/include"
    )
    find_path(PYLON_INCLUDE_DIR
        NAMES pylonc/PylonC.h
        PATHS ${_PYLON_HEADER_HINTS}
    )
    find_path(PYLON_GENAPI_INCLUDE_DIR
        NAMES genapic/GenApiC.h
        PATHS ${_PYLON_HEADER_HINTS}
    )
endif()

set(PYLON_INCLUDE_DIRS
    ${PYLON_INCLUDE_DIR}
    ${PYLON_GENAPI_INCLUDE_DIR}
)

set(PYLON_COMPILE_DEFINITIONS
    WITH_PYLON=1
    TREX_BASLER_COMPILED=1
    TREX_BASLER_LAZY_RUNTIME=1
)

# FOUND_VAR keeps the result in PYLON_FOUND and silences the name-mismatch warning.
find_package_handle_standard_args(BaslerPylon
    FOUND_VAR BaslerPylon_FOUND
    REQUIRED_VARS PYLON_INCLUDE_DIR PYLON_GENAPI_INCLUDE_DIR
)

mark_as_advanced(PYLON_INCLUDE_DIR PYLON_GENAPI_INCLUDE_DIR PYLON_INCLUDE_DIRS
                 PYLON_COMPILE_DEFINITIONS PYLON_FRAMEWORK_DIR)
