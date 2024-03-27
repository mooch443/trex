option(WITH_GITSHA1 ON)
if(WITH_GITSHA1)
	add_definitions(-DWITH_GITSHA1=true)
	include(GetGitRevisionDescription)
	get_git_head_revision(GIT_REFSPEC GIT_SHA1)
    git_get_description(GIT_DESCRIBE_TAG)
    git_get_branch(GIT_CURRENT_BRANCH)

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

FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG        v2.3
)

FetchContent_GetProperties(indicators)
if(NOT indicators_POPULATED)
  FetchContent_Populate(indicators)
  include_directories(${indicators_SOURCE_DIR}/include)
endif()

include_directories(.)

add_subdirectory(ProcessedVideo)
include_directories(ProcessedVideo)



if(COMMONS_NO_PYTHON)
    add_compile_options( -DTREX_NO_PYTHON=true)
    message(STATUS "NO PYTHON=-DTREX_NO_PYTHON=${COMMONS_NO_PYTHON}")
else()
    add_compile_options(-DTREX_NO_PYTHON=false)
endif()

include_directories(tracker)
include_directories(grabber)

add_subdirectory(grabber)
add_subdirectory(tracker)
