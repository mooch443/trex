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

# Standard linking to gtest stuff.
target_link_libraries(test_matching PUBLIC gtest gtest_main)
target_link_libraries(test_strings PUBLIC gtest gtest_main)
target_link_libraries(test_pixels PUBLIC gtest gtest_main)
target_link_libraries(test_patharray PUBLIC gtest gmock gtest_main)

# Extra linking for the project.
target_link_libraries(test_matching PUBLIC 
        #tracker_gui
        tracker_tracking
        tracker_python
        tracker_misc
        framegrabber_misc
        commons 
        pv)
        
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

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME test-matching COMMAND test_matching)
add_test(NAME test-strings COMMAND test_strings)
add_test(NAME test-pixels COMMAND test_pixels)
add_test(NAME test-patharray COMMAND test_patharray)

endif()
