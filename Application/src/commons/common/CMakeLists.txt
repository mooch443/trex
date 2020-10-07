set(RESOURCES_ADDED "")

add_subdirectory(cpputils)

message(STATUS "INCLUDE: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/src/cnpy/)
include_directories(${CMAKE_SOURE_DIR}/cpputils)

file(GLOB HDRS video/*.h processing/*.h misc/*.h file/*.h gui/*.h gui/types/*.h *.h)
file(GLOB SRCS video/*.cpp processing/*.cpp misc/*.cpp file/*.cpp gui/*.cpp gui/types/*.cpp *.cpp)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND METAL_LIBRARY)
    set(SRCS ${SRCS} gui/MetalImpl.mm )
else()
    message(STATUS "Skipping metal implementation")
endif()

if(WITH_HTTPD)
	file(GLOB _HDRS http/*.h)
	file(GLOB _SRCS http/*.cpp)
	
	set(HDRS ${HDRS} ${_HDRS})
	set(SRCS ${SRCS} ${_SRCS})
endif()

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(${CMAKE_VERSION} VERSION_LESS "3.16.0")
    set(HDRS ${HDRS} commons.pc.h)
endif()

# add actual library
if(APPLE)
  add_library(commons STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(commons STATIC ${SRCS} ${HDRS})
else()
  add_library(commons STATIC ${SRCS} ${HDRS})
endif()

option(TREX_DONT_USE_PCH OFF)
if(${CMAKE_VERSION} VERSION_LESS "3.16.0" OR TREX_DONT_USE_PCH)
    message("Please consider to switch to CMake 3.16.0")
else()
    target_precompile_headers(commons
      PUBLIC
        commons.pc.h
      PRIVATE
    )
endif()

add_dependencies(commons TRex::LibZIP TRex::LibZ cnpy gladex TRex::GLFW imgui)
if(APPLE OR NOT UNIX)
    add_dependencies(commons TRex::LibPNG)
endif()

#if(NOT APPLE AND NOT WIN32)
#find_library(tiff REQUIRED)
#target_link_libraries(commons PUBLIC tiff)
#set_property(TARGET commons PROPERTY POSITION_INDEPENDENT_CODE ON)
#endif()

foreach(FILE ${SRCS} ${HDRS}) 
    # Get the directory of the source file
    get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)

    # Remove common directory prefix to make the group
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" GROUP "${PARENT_DIR}")

    # Make sure we are using windows slashes
    string(REPLACE "/" "\\" GROUP "${GROUP}")

    # Group into "Source Files" and "Header Files"
    if ("${FILE}" MATCHES ".*\\.cpp")
       set(GROUP "${GROUP}")
    elseif("${FILE}" MATCHES ".*\\.h")
       set(GROUP "${GROUP}")
    endif()

    source_group("${GROUP}" FILES "${FILE}")
endforeach()

target_link_libraries(commons
    PUBLIC cpputils
    PRIVATE cnpy
)

if(${PYLON_FOUND} AND ${WITH_PYLON})
	target_link_libraries( commons PRIVATE ${PYLON_LIBRARIES} )
endif()

if(WITH_HTTPD)
	target_link_libraries(commons PRIVATE ${MHD_LIBRARIES})
    add_dependencies(commons LibMicroHttpd)
endif()

add_dependencies(commons TRex::OpenCV)
