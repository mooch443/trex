file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add actual library
if(UNIX)
  add_library(framegrabber_misc STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(framegrabber_misc STATIC ${SRCS} ${HDRS})
endif()
#target_precompile_headers(framegrabber_misc REUSE_FROM commons)

target_link_libraries(framegrabber_misc
	PRIVATE commons
)

if(APPLE AND PYLON_FOUND AND WITH_PYLON)
    message(STATUS "linking with pylon")
    target_link_libraries( framegrabber_misc PUBLIC ${PYLON_LIBRARIES} )
endif()

add_dependencies(framegrabber_misc commons) #Commons::OpenCV)


if(WITH_FFMPEG)
    target_compile_definitions(framegrabber_misc PRIVATE WITH_FFMPEG=true)
endif()

option(WITH_PYLON "use pylon support" OFF)

if(APPLE AND WITH_PYLON)
    FIND_LIBRARY(PYLON_LIBRARIES Pylon)
    if(PYLON_LIBRARIES STREQUAL "PYLON_LIBRARIES-NOTFOUND" OR NOT ${WITH_PYLON})
        set(PYLON_FOUND false)
        target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=false)
    else()
        set(PYLON_FOUND true)
        target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=true)
        message(STATUS "Pylon found. ${PYLON_LIBRARIES} lol")
        target_include_directories(framegrabber_misc PUBLIC /Library/Frameworks/pylon.framework/Headers/GenICam)
    endif()
    
elseif(WITH_PYLON)
    find_package(BaslerPylon)
    if(${PYLON_FOUND})
        target_include_directories(framegrabber_misc PUBLIC ${PYLON_INCLUDE_DIRS})
        target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=true)
        message(STATUS "Pylon found.")
    else()
        target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=false)
    endif()
else()
    target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=false)
endif()



if(PYLON_FOUND AND WITH_PYLON)
    message(STATUS "linking with pylon")
    target_link_libraries( framegrabber_misc PUBLIC ${PYLON_LIBRARIES} )
    target_include_directories(framegrabber_misc PUBLIC ${PYLON_INCLUDE_DIRS})
endif()
