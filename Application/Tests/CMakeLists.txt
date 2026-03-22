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

add_executable(test_matching test_matching.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_matching PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_commandline test_commandline.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_commandline PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_strings test_strings.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_strings PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
  target_compile_options(test_commandline PRIVATE /bigobj)
  target_compile_options(test_strings PRIVATE /bigobj)
endif()

add_executable(test_pixels test_pixels.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_pixels PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_patharray test_patharray.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_patharray PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_dyngui test_dyngui.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_dyngui PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(bench_dyngui_parser bench_dyngui_parser.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(bench_dyngui_parser PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_outlines test_outlines.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_outlines PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_threadpool test_threadpool.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_threadpool PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_tiling test_tiling.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_tiling PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_segmenter test_segmenter.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_segmenter PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_executable(test_commons_modules test_commons_modules.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_commons_modules PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(test_commons_modules PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_SCAN_FOR_MODULES ON
)
endif()

if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
add_executable(test_trex_modules test_trex_modules.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_trex_modules PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(test_trex_modules PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_SCAN_FOR_MODULES ON
)
endif()

# always compile in debug mode
set(TARGETS test_matching test_commandline test_strings test_pixels test_patharray test_dyngui test_outlines test_threadpool test_tiling test_segmenter)
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

# Standard linking to gtest stuff.
target_link_libraries(test_matching PUBLIC gtest gtest_main)
target_link_libraries(test_commandline PUBLIC gtest gtest_main)
target_link_libraries(test_strings PUBLIC gtest gtest_main)
target_link_libraries(test_pixels PUBLIC gtest gtest_main)
target_link_libraries(test_patharray PUBLIC gtest gmock gtest_main)
target_link_libraries(test_dyngui PUBLIC gtest gmock gtest_main)
target_link_libraries(test_outlines PUBLIC gtest gmock gtest_main)
target_link_libraries(test_threadpool PUBLIC gtest gtest_main)
target_link_libraries(test_tiling PUBLIC gtest gmock gtest_main)
target_link_libraries(test_segmenter PUBLIC gtest gtest_main)

# Extra linking for the project.
target_link_libraries(test_matching PUBLIC
    tracker_tracking
    tracker_ml
    tracker_python
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)

target_link_libraries(test_matching PUBLIC tracker_python)
target_link_libraries(test_commandline PUBLIC commons)

if(WIN32)
    set_target_properties(test_matching PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
endif()

target_link_libraries(test_outlines PUBLIC
    tracker_tracking
    tracker_ml
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)

target_link_libraries(test_outlines PUBLIC tracker_python)

if(WIN32)
    set_target_properties(test_outlines PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
endif()

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
        tracker_misc
        commons
        pv
        tracker_python)

target_link_libraries(test_segmenter PUBLIC
    tracker_tracking
    tracker_ml
    tracker_python
    tracker_gui
    tracker_misc
    framegrabber_misc
    pv
    commons
)

if(COMMONS_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
target_link_libraries(test_commons_modules PUBLIC
    gtest
    gtest_main
    commons
)
endif()

if(TREX_ENABLE_MODULES AND NOT CMAKE_VERSION VERSION_LESS "3.28.0")
target_link_libraries(test_trex_modules PUBLIC
        gtest
        gtest_main
        trex_ui
        trex_ml
        trex_tracking
        trex_detect
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
add_test(NAME test-matching COMMAND test_matching)
add_test(NAME test-commandline COMMAND test_commandline)
add_test(NAME test-strings COMMAND test_strings)
add_test(NAME test-pixels COMMAND test_pixels)
add_test(NAME test-patharray COMMAND test_patharray)
add_test(NAME test-dyngui COMMAND test_dyngui)
add_test(NAME test-outlines COMMAND test_outlines)
add_test(NAME test-threadpool COMMAND test_threadpool)
add_test(NAME test-tiling COMMAND test_tiling)
add_test(NAME test-segmenter COMMAND test_segmenter)
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
