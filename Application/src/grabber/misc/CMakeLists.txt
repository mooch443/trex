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

target_link_libraries(framegrabber_misc
	PRIVATE cnpy
)

if(APPLE AND PYLON_FOUND AND WITH_PYLON)
    message(STATUS "linking with pylon")
    target_link_libraries( framegrabber_misc PUBLIC ${PYLON_LIBRARIES} )
endif()

add_dependencies(framegrabber_misc commons) #Commons::OpenCV)
