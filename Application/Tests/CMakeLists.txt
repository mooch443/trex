if(TREX_WITH_TESTS)

FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP true
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850
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

add_executable(test_strings test_strings.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_strings PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_pixels test_pixels.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_pixels PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_patharray test_patharray.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_patharray PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_dyngui test_dyngui.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_dyngui PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_outlines test_outlines.cpp "${CMAKE_BINARY_DIR}/src/GitSHA1.cpp" ../src/GitSHA1.h)
target_compile_definitions(test_outlines PRIVATE -DTREX_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}")

# always compile in debug mode
set(TARGETS test_matching test_strings test_pixels test_patharray test_dyngui test_outlines)
if(WIN32)
        foreach(target ${TARGETS})
                target_compile_options(${target} PRIVATE /W4 /EHsc /Zi /Od)
        endforeach()
else()
        foreach(target ${TARGETS})
                target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic -g -O0)
        endforeach()
endif()

# Standard linking to gtest stuff.
target_link_libraries(test_matching PUBLIC gtest gtest_main)
target_link_libraries(test_strings PUBLIC gtest gtest_main)
target_link_libraries(test_pixels PUBLIC gtest gtest_main)
target_link_libraries(test_patharray PUBLIC gtest gmock gtest_main)
target_link_libraries(test_dyngui PUBLIC gtest gmock gtest_main)
target_link_libraries(test_outlines PUBLIC gtest gmock gtest_main)

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

target_link_libraries(test_matching PUBLIC tracker_python)

if(WIN32)
    set_target_properties(test_matching PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
endif()

if(NOT WIN32)
    target_link_libraries(test_matching PUBLIC pthread m)
    if(NOT APPLE)
        target_link_libraries(test_matching PUBLIC atomic)
    endif()
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

if(NOT WIN32)
    target_link_libraries(test_outlines PUBLIC pthread m)
    if(NOT APPLE)
        target_link_libraries(test_outlines PUBLIC atomic)
    endif()
endif()
        
# Extra linking for the project.
target_link_libraries(test_strings PUBLIC
        commons
        pv)

target_link_libraries(test_pixels PUBLIC
        commons
        pv)

target_link_libraries(test_patharray PUBLIC
        commons
        pv)

target_link_libraries(test_dyngui PUBLIC
        commons
        pv)

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME test-matching COMMAND test_matching)
add_test(NAME test-strings COMMAND test_strings)
add_test(NAME test-pixels COMMAND test_pixels)
add_test(NAME test-patharray COMMAND test_patharray)
add_test(NAME test-dyngui COMMAND test_dyngui)
add_test(NAME test-outlines COMMAND test_outlines)

# Add a custom target to build and run all tests
add_custom_target(runAllTests
        COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
        DEPENDS ${TARGETS}
)

endif()
