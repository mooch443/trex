option(WITH_GITSHA1 ON)
if(WITH_GITSHA1)
	add_definitions(-DWITH_GITSHA1=true)
	include(GetGitRevisionDescription)
	get_git_head_revision(GIT_REFSPEC GIT_SHA1)
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
else()
    set(WITH_GITSHA1 OFF)
	add_definitions(-DWITH_GITSHA1=false)
endif()

#add_subdirectory(frugally)
add_subdirectory(cnpy)

include_directories(commons/cpputils/src)
add_subdirectory(commons)
include_directories(commons/common)

add_subdirectory(ProcessedVideo)
include_directories(ProcessedVideo)

include_directories(tracker)
include_directories(grabber)

# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(grabber)
add_subdirectory(tracker)
