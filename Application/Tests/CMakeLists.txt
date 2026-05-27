if(TREX_WITH_TESTS)

FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP true
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.17.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

##############
# Unit Tests
##############
include_directories(../src/tracker)
include_directories(../src/ProcessedVideo)

add_executable(test_matching test_matching.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_matching PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_commandline test_commandline.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_commandline PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_strings test_strings.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_strings PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
  target_compile_options(test_commandline PRIVATE /bigobj)
  target_compile_options(test_strings PRIVATE /bigobj)
endif()

add_executable(test_pixels test_pixels.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_pixels PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_patharray test_patharray.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_patharray PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_dyngui test_dyngui.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_dyngui PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(bench_dyngui_parser bench_dyngui_parser.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(bench_dyngui_parser PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_outlines test_outlines.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_outlines PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_threadpool test_threadpool.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_threadpool PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_tiling test_tiling.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_tiling PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_segmenter test_segmenter.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_segmenter PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_dispatcher test_dispatcher.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_dispatcher PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_sam3promptpayload test_sam3promptpayload.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_sam3promptpayload PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_sam3interactivesession test_sam3interactivesession.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_sam3interactivesession PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_custom_target(commons_module_surface_check
    COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_commons_module_surface.py" --check
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking checked-in commons module inventory and smoke surface."
)
add_executable(test_commons_modules test_commons_modules.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_commons_modules PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(test_commons_modules PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_SCAN_FOR_MODULES ON
)
add_dependencies(test_commons_modules commons_module_surface_check)
endif()

if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_custom_target(tracker_module_surface_check
    COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_tracker_module_surface.py" --check
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking checked-in tracker module inventory and smoke surface."
)
add_executable(test_trex_modules test_trex_modules.cpp test_main.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_trex_modules PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(test_trex_modules PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_SCAN_FOR_MODULES ON
)
add_dependencies(test_trex_modules tracker_module_surface_check)
endif()

# always compile in debug mode
set(TARGETS 
    test_matching 
    test_commandline 
    test_strings 
    test_pixels 
    test_patharray 
    test_dyngui 
    test_outlines 
    test_threadpool 
    test_tiling 
    test_segmenter
    test_dispatcher
    test_sam3promptpayload
    test_sam3interactivesession
)
if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
    list(APPEND TARGETS test_commons_modules)
endif()
if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
        list(APPEND TARGETS test_trex_modules)
endif()
if(WIN32)
        foreach(target ${TARGETS})
                target_compile_options(${target} PRIVATE /W4 /EHsc /Zi /Od)
        endforeach()
else()
        foreach(target ${TARGETS})
                target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic -g -O0)
        endforeach()
endif()

if(WIN32)
target_compile_options(test_matching PRIVATE /bigobj)
endif()

function(trex_enable_python_delayload target)
    if(WIN32)
        target_link_libraries(${target} PRIVATE delayimp)
        target_link_options(${target} PRIVATE
            "/DELAYLOAD:trex_python.dll"
            "/DELAYLOAD:trex_python-d.dll"
        )
    endif()
endfunction()

# Standard linking to gtest stuff.
target_link_libraries(test_matching PUBLIC gtest)
target_link_libraries(test_commandline PUBLIC gtest)
target_link_libraries(test_strings PUBLIC gtest)
target_link_libraries(test_pixels PUBLIC gtest)
target_link_libraries(test_patharray PUBLIC gtest gmock)
target_link_libraries(test_dyngui PUBLIC gtest gmock)
target_link_libraries(test_outlines PUBLIC gtest gmock)
target_link_libraries(test_threadpool PUBLIC gtest)
target_link_libraries(test_tiling PUBLIC gtest gmock)
target_link_libraries(test_segmenter PUBLIC gtest)
target_link_libraries(test_dispatcher PUBLIC gtest)
target_link_libraries(test_sam3promptpayload PUBLIC gtest)
target_link_libraries(test_sam3interactivesession PUBLIC gtest gmock)

# Extra linking for the project.
target_link_libraries(test_matching PUBLIC
    tracker_tracking
    tracker_ml
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)
trex_link_static_python_impl(test_matching)

target_link_libraries(test_commandline PUBLIC commons)

target_link_libraries(test_outlines PUBLIC
    tracker_tracking
    tracker_ml
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)

# Extra linking for the project.
target_link_libraries(test_strings PUBLIC
        commons
        pv)

target_link_libraries(test_pixels PUBLIC
        tracker_misc
        commons
        pv)

target_link_libraries(test_patharray PUBLIC
        commons
        pv)

target_link_libraries(test_dyngui PUBLIC
        commons
        pv)

target_link_libraries(bench_dyngui_parser PUBLIC
        commons
        pv)

target_link_libraries(test_threadpool PUBLIC
        commons
        pv)

target_link_libraries(test_tiling PUBLIC
        tracker_ml
        tracker_tracking
        tracker_misc
        commons
        pv)
trex_link_static_python_impl(test_tiling)

target_link_libraries(test_segmenter PUBLIC
    tracker_tracking
    tracker_ml
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)
trex_link_static_python_impl(test_segmenter)

target_link_libraries(test_dispatcher PUBLIC
    tracker_gui
    tracker_misc
    pv
    commons
)

target_link_libraries(test_sam3promptpayload PUBLIC
    tracker_misc
    trex_python_staging
    pv
    commons
)

target_link_libraries(test_sam3interactivesession PUBLIC
    tracker_tracking
    tracker_ml
    tracker_gui
    tracker_misc
    pv
    commons
)
trex_link_static_python_impl(test_sam3interactivesession)

if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
target_link_libraries(test_commons_modules PUBLIC
    gtest
    commons
)
endif()

if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
target_link_libraries(test_trex_modules PUBLIC
        gtest
        trex_ui
        trex_ml
        trex_tracking
        trex_data
        trex_core
        commons
)
endif()

set_target_properties(test_matching PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_commandline PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_strings PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_pixels PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_patharray PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_dyngui PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(bench_dyngui_parser PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_outlines PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_threadpool PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_tiling PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_segmenter PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_dispatcher PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_sam3promptpayload PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
set_target_properties(test_sam3interactivesession PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
set_target_properties(test_commons_modules PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
endif()
if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
set_target_properties(test_trex_modules PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES")
endif()

#if(WIN32)
  #target_compile_options(bench_dyngui_parser PRIVATE /O2)
#else()
  #target_compile_options(bench_dyngui_parser PRIVATE -O3)
#endif()
#target_compile_definitions(bench_dyngui_parser PRIVATE NDEBUG)

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
function(trex_prepend_test_path test_name target_name)
    if(WIN32 AND TARGET ${target_name})
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.22")
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:${target_name}>"
            )
        else()
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target_name}>"
            )
        endif()
    endif()
endfunction()

add_test(NAME test-matching COMMAND test_matching)
add_test(NAME test-commandline COMMAND test_commandline)
add_test(NAME test-strings COMMAND test_strings)
add_test(NAME test-pixels COMMAND test_pixels)
add_test(NAME test-patharray COMMAND test_patharray)
add_test(NAME test-dyngui COMMAND test_dyngui)
add_test(NAME test-outlines COMMAND test_outlines)
add_test(NAME test-threadpool COMMAND test_threadpool)
add_test(NAME test-tiling COMMAND test_tiling)
trex_prepend_test_path(test-segmenter trex_check_python)
add_test(NAME test-dispatcher COMMAND test_dispatcher)
add_test(NAME test-sam3promptpayload COMMAND test_sam3promptpayload)
add_test(NAME test-sam3interactivesession COMMAND test_sam3interactivesession)
if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_test(NAME test-commons-modules COMMAND test_commons_modules)
endif()
if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_test(NAME test-trex-modules COMMAND test_trex_modules)
endif()

# Add a custom target to build and run all tests
add_custom_target(runAllTests
        COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
        DEPENDS ${TARGETS}
)

endif()
