cmake_minimum_required(VERSION 3.0)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules )

option(WITH_HTTPD "compile with/without httpd module" ON)
message("CXX FLAGS: " ${CMAKE_CXX_FLAGS})

option(GRABBER_USE_CUVID "Compile using NVIDIA CUVID" OFF)
if(GRABBER_USE_CUVID)
    add_definitions(-DGRABBER_USE_CUVID)
endif()

message(STATUS "hidden visibility in commons project ${CMAKE_CXX_FLAGS}")
option(WITH_PYLON "use pylon support" ON)

if(APPLE)
	FIND_LIBRARY(PYLON_LIBRARIES Pylon)
	if(PYLON_LIBRARIES STREQUAL "PYLON_LIBRARIES-NOTFOUND" OR NOT ${WITH_PYLON})
		set(PYLON_FOUND false)
		add_definitions(-DWITH_PYLON=false)
	else()
		set(PYLON_FOUND true)
		add_definitions(-DWITH_PYLON=true)
		message(STATUS "Pylon found. ${PYLON_LIBRARIES}")
		include_directories(/Library/Frameworks/pylon.framework/Headers/GenICam)
	endif()
	
else()
	find_package(BaslerPylon)
	if(${PYLON_FOUND} AND ${WITH_PYLON})
		include_directories(${PYLON_INCLUDE_DIRS})
		add_definitions(-DWITH_PYLON=true)
		message(STATUS "Pylon found.")
	else()
		add_definitions(-DWITH_PYLON=false)
		message("pylon disabled")
	endif()
endif()

if(WITH_HTTPD)
	set(httpd_PREFIX "${CMAKE_BINARY_DIR}/lmhttpd")
	set(httpd_INSTALL_DIR "${CMAKE_BINARY_DIR}/lmhttpd")
	set(httpd_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${httpd_INSTALL_DIR})
    message(STATUS "Adding microhttpd")
	ExternalProject_Add(LibMicroHttpd
	  PREFIX ${httpd_PREFIX}
	  GIT_REPOSITORY https://gnunet.org/git/libmicrohttpd.git
	  GIT_TAG v0.9.55
	  #SOURCE_DIR ${httpd_PREFIX}/
	  BINARY_DIR ${httpd_PREFIX}/src/LibMicroHttpd/
	  INSTALL_DIR ${httpd_INSTALL_DIR}
	  CONFIGURE_COMMAND cd ${httpd_PREFIX}/src/LibMicroHttpd/ && autoreconf -fi && ./configure --bindir=${httpd_INSTALL_DIR}/lib --prefix=${httpd_INSTALL_DIR}
	  UPDATE_COMMAND ""
	  #CMAKE_ARGS ${httpd_CMAKE_ARGS}
	  #-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
	  #-DCMAKE_INSTALL_FRAMEWORK_PREFIX=<INSTALL_DIR>/Frameworks
	  #-DCMAKE_BUILD_TYPE:STRING=Release
	)

	set(MHD_LIBRARIES microhttpd)

	add_custom_command(TARGET LibMicroHttpd POST_BUILD
	    COMMAND mkdir -p "${httpd_INSTALL_DIR}/lib/Release")
	add_custom_command(TARGET LibMicroHttpd POST_BUILD
	    COMMAND mkdir -p "${httpd_INSTALL_DIR}/lib/Debug")


	message(STATUS "HTTPD dir ${httpd_INSTALL_DIR}/lib")
	link_directories(${httpd_INSTALL_DIR}/lib)
	include_directories(${httpd_INSTALL_DIR}/include)
    add_definitions(-DWITH_MHD=true)
else()
    add_definitions(-DWITH_MHD=false)
endif()

add_subdirectory(common)
include_directories(common)
