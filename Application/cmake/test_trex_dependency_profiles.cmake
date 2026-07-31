function(trex_expect_profile_success name expected_result)
    execute_process(
        COMMAND ${CMAKE_COMMAND} ${ARGN}
            -P "${CMAKE_CURRENT_LIST_DIR}/test_trex_dependency_profile_case.cmake"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr)
    set(_combined "${_stdout}${_stderr}")
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "${name} unexpectedly failed:\n${_combined}")
    endif()
    if(NOT _combined MATCHES "PROFILE_RESULT=${expected_result}")
        message(FATAL_ERROR
            "${name} produced the wrong providers. Expected '${expected_result}':\n${_combined}")
    endif()
endfunction()

function(trex_expect_profile_failure name expected_message)
    execute_process(
        COMMAND ${CMAKE_COMMAND} ${ARGN}
            -P "${CMAKE_CURRENT_LIST_DIR}/test_trex_dependency_profile_case.cmake"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr)
    set(_combined "${_stdout}${_stderr}")
    if(_result EQUAL 0)
        message(FATAL_ERROR "${name} unexpectedly succeeded:\n${_combined}")
    endif()
    if(NOT _combined MATCHES "${expected_message}")
        message(FATAL_ERROR
            "${name} failed without the expected message '${expected_message}':\n${_combined}")
    endif()
endfunction()

trex_expect_profile_success(defaults ";ON;ON;ON;ON")
trex_expect_profile_success(explicit_buildall "buildall;ON;ON;ON;ON"
    -DTREX_CONFIGURE=buildall)
trex_expect_profile_success(explicit_minimal "minimal;OFF;OFF;OFF;OFF"
    -DTREX_CONFIGURE=minimal)
trex_expect_profile_success(individual_png_off ";ON;OFF;ON;ON"
    -DCOMMONS_BUILD_PNG=OFF)
trex_expect_profile_success(custom_opencv_external_support ";ON;OFF;OFF;OFF"
    -DCOMMONS_BUILD_OPENCV=ON
    -DCOMMONS_BUILD_PNG=OFF
    -DCOMMONS_BUILD_ZIP=OFF
    -DCOMMONS_BUILD_ZLIB=OFF)
trex_expect_profile_success(external_opencv_custom_support ";OFF;ON;ON;ON"
    -DCOMMONS_BUILD_OPENCV=OFF
    -DCOMMONS_BUILD_PNG=ON
    -DCOMMONS_BUILD_ZIP=ON
    -DCOMMONS_BUILD_ZLIB=ON)

foreach(_option IN ITEMS
    COMMONS_BUILD_OPENCV
    COMMONS_BUILD_PNG
    COMMONS_BUILD_ZIP
    COMMONS_BUILD_ZLIB)
    trex_expect_profile_success("individual_${_option}_off" ".*"
        -D${_option}=OFF)
endforeach()

trex_expect_profile_failure(contradict_minimal
    "requires COMMONS_BUILD_OPENCV=OFF"
    -DTREX_CONFIGURE=minimal
    -DCOMMONS_BUILD_OPENCV=ON)
trex_expect_profile_failure(contradict_buildall
    "requires COMMONS_BUILD_ZLIB=ON"
    -DTREX_CONFIGURE=buildall
    -DCOMMONS_BUILD_ZLIB=OFF)
trex_expect_profile_failure(invalid_profile
    "Invalid TREX_CONFIGURE='invalid'"
    -DTREX_CONFIGURE=invalid)

message(STATUS "TREX dependency profile semantics passed.")
