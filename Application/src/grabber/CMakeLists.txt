add_subdirectory(misc)

file(GLOB HTML html/*.html html/*.js html/*.css)

# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SRCS
    ${HTML}
	tomp4.h
	tomp4.cpp
	grabber.h
	grabber.cpp
     gpuImage.h
     gpuImage.cpp
	gui.h
	gui.cpp
	default_config.cpp
	default_config.h
    CropWindow.h
    CropWindow.cpp
	main.cpp
    gfx/TGrabsIcon16.png
    gfx/TGrabsIcon32.png
    gfx/TGrabsIcon64.png
)

if(APPLE)
    set(SRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gfx/GrabberIcon.icns)
endif()

if(WITH_GITSHA1)
    list(APPEND SRCS "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)
endif()

if(WIN32)
	set(SRCS ${SRCS} tgrabs.rc)
endif()

source_group(html FILES ${HTML})
		
if(APPLE)
    set(TGRABS_EXEC_NAME TGrabs)
else()
    set(TGRABS_EXEC_NAME tgrabs)
endif()
add_executable(${TGRABS_EXEC_NAME} ${SRCS})

if(WIN32)
	set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES LINK_FLAGS_RELEASE "/DELAYLOAD:tracker_python.dll" )
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}  /DELAYLOAD:tracker_python.dll")
	set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DELAYLOAD:tracker_python-d.dll" )
	set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}  /DELAYLOAD:tracker_python-d.dll")
endif()

if(APPLE)
ADD_CUSTOM_COMMAND(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:trex_check_python> $<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>)
endif()
add_dependencies(${TGRABS_EXEC_NAME} trex_check_python TRex::OpenCV)

add_executable(pvconvert pvconvert.cpp gif.h)
target_link_libraries(pvconvert PUBLIC
    tracker_gui
    pv
    commons
    cnpy
    ${OpenCV_LIBS}
    TRex::GLFW
    TRex::LibPNG
    imgui
    glad
    ${EXTRA_LIBS}
)

add_dependencies(pvconvert TRex::OpenCV)

if(PYLON_FOUND AND WITH_PYLON)
    message(STATUS "linking with pylon")
    target_link_libraries( ${TGRABS_EXEC_NAME} PUBLIC ${PYLON_LIBRARIES} )
endif()

target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE
    framegrabber_misc
    tracker_misc
	tracker_tracking
	tracker_gui
    tracker_python
	pv
	commons
	TRex::GLFW
    TRex::LibPNG
    TRex::LibZ
    	imgui
	glad
	TRex::OpenCV
	${EXTRA_LIBS})

if(WITH_HTTPD)
    add_dependencies(${TGRABS_EXEC_NAME} LibMicroHttpd)
endif()

add_executable(pvinfo pvinfo.cpp pvinfo_merge.cpp pvinfo_merge.h)

target_link_libraries(pvinfo PRIVATE
	cnpy
	tracker_tracking
    tracker_gui
    tracker_python
    tracker_misc
	commons
	TRex::LibPNG
	TRex::GLFW
	TRex::OpenCV
	${EXTRA_LIBS}
	pv
)

add_dependencies(pvinfo TRex::OpenCV)

get_target_property(OUT TRex::GLFW INTERFACE_LINK_LIBRARIES)
message(STATUS "TRex::GLFW libraries: ${OUT}")

set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES
  MACOSX_BUNDLE TRUE
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gfx/GrabberIcon.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

set(RESOURCE_COPY_FILES
	${CMAKE_SOURCE_DIR}/src/html
	${CMAKE_SOURCE_DIR}/src/fonts
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx
)

set(RESOURCE_COPY_SINGLE_FILES
    ${CMAKE_SOURCE_DIR}/default.settings
    ${CMAKE_SOURCE_DIR}/closed_loop.py
)

if(NOT WIN32)
    foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)
        add_custom_command(TARGET pvinfo POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:pvinfo>/${comp_s}")
    endforeach()
endif()

if(APPLE)
    foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)
        add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/../Resources/${comp_s}")
    endforeach()
else()
    foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)

        add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${comp}"
                "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/${comp_s}")
    endforeach()
endif()

IF("${CMAKE_SYSTEM}" MATCHES "Linux")
	foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)
		add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy_directory
		        "${comp}"
		        "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/${comp_s}")
	endforeach()
elseif(APPLE)
    foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)
        add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${comp}"
                "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/../Resources/${comp_s}")
    endforeach()
else()
	foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)
		add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy_directory
		        "${comp}"
		        "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/${comp_s}")
	endforeach()
endif()

set(targets ${TGRABS_EXEC_NAME} pvinfo pvconvert)

if(APPLE)
    add_executable(tgrabs_starter tgrabs.cpp)
    target_link_libraries(tgrabs_starter PRIVATE cpputils)
    set_target_properties(tgrabs_starter PROPERTIES OUTPUT_NAME tgrabs)
    set(targets ${targets} tgrabs_starter)
endif()


if(WIN32)
	if(TREX_BUILD_OPENCV)
		install(
			FILES
			"${OpenCV_LIB_PATH}/../bin/opencv_videoio_ffmpeg430_64.dll" DESTINATION bin)
	endif()
endif()

install(TARGETS ${targets}
    EXPORT TGrabs
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin
    INCLUDES DESTINATION include
    CONFIGURATIONS Release
)

if(APPLE)
target_compile_definitions(tgrabs_starter PRIVATE -DTGRABS_TARGET_PATH="${CMAKE_INSTALL_PREFIX}/bin")
endif()
