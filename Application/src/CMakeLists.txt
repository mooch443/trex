option(WITH_GITSHA1 ON)
if(WITH_GITSHA1)
	add_definitions(-DWITH_GITSHA1=true)
	include(GetGitRevisionDescription)
	get_git_head_revision(GIT_REFSPEC GIT_SHA1)
    git_get_description(GIT_DESCRIBE_TAG)

    set(TREX_CMAKE_BUILD_TYPE_IN "${TREX_CMAKE_BUILD_TYPE}")
    set(TREX_BUILD_CXX_OPTIONS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
else()
    set(WITH_GITSHA1 OFF)
	add_definitions(-DWITH_GITSHA1=false)
    set(TREX_CMAKE_BUILD_TYPE_IN "")
    set(TREX_BUILD_CXX_OPTIONS "")
    set(GIT_DESCRIBE_TAG "")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
endif()

#add_subdirectory(frugally)
#add_subdirectory(cnpy)

include_directories(.)

add_subdirectory(ProcessedVideo)
include_directories(ProcessedVideo)

include_directories(tracker)
include_directories(grabber)

add_subdirectory(grabber)
add_subdirectory(tracker)
