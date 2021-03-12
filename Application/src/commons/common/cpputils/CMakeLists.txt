file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/*.h
           ${CMAKE_CURRENT_SOURCE_DIR}/debug/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/debug/*.h)

if(WIN32)
    add_library(
        cpputils
        STATIC
        ${SRCS}
	)
elseif(APPLE)
	add_library(
	cpputils
	STATIC
	${SRCS}
	)
else()
    add_library(
        cpputils
        STATIC
        ${SRCS}
	)
endif()
