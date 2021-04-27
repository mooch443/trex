add_subdirectory(gui)
add_subdirectory(python)
add_subdirectory(tracking)
add_subdirectory(misc)

include_directories(${CMAKE_SOURCE_DIR}/src/ProcessedVideo)

if(APPLE)
    set(TREX_EXEC_NAME TRex)
else()
    set(TREX_EXEC_NAME trex)
endif()

# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SRCS
    gui.h
    gui.cpp
    main.cpp
    VideoOpener.cpp
    VideoOpener.h
    ../grabber/default_config.cpp
    ../grabber/default_config.h
    gfx/TRexIcon16.png
    gfx/TRexIcon32.png
    gfx/TRexIcon64.png
    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h
)
 
if(APPLE)
 set(SRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gfx/TRexIcon.icns)
endif()

if(WIN32 OR UNIX AND NOT APPLE)
	set(SRCS ${SRCS} trex.rc)
endif()

if(NOT WIN32 AND TREX_WITH_TESTS)
    #windows does not support large cpp files
    add_subdirectory(tests)
    
    if(APPLE)
        add_executable(test_imgui program.cpp)
    else()
        add_executable(test_imgui program.cpp)
    endif(APPLE)

    target_link_libraries(test_imgui PUBLIC
        framegrabber_misc
        tracker_tracking
        commons
        cnpy
        TRex::GLFW
        ${OpenCV_LIBS}
        ${EXTRA_LIBS}
    )
    add_dependencies(test_imgui commons TRex::LibZIP TRex::GLFW TRex::OpenCV)
endif()

# add actual executable
add_executable(${TREX_EXEC_NAME} ${SRCS})
add_dependencies(${TREX_EXEC_NAME} TRex::GLFW TRex::OpenCV)

# link all needed libraries to the target
target_link_libraries(${TREX_EXEC_NAME} PUBLIC
	tracker_tracking
	tracker_gui
    tracker_python
    tracker_misc
	pv
	commons
	TRex::GLFW
	TRex::OpenCV
	TRex::LibPNG
	TRex::LibZ
	${EXTRA_LIBS}
)

if(WITH_HTTPD)
    add_dependencies(${TREX_EXEC_NAME} LibMicroHttpd)
endif()

if(WIN32)
	set_target_properties(${TREX_EXEC_NAME} PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
endif()

if(NOT WIN32)
    target_link_libraries(${TREX_EXEC_NAME} PUBLIC pthread m)
    if(TREX_WITH_TESTS)
        target_link_libraries(test_imgui PUBLIC pthread m)
    endif()
endif()

set_target_properties(${TREX_EXEC_NAME} PROPERTIES
  MACOSX_BUNDLE TRUE
  MACOSX_BUNDLE_GUI_IDENTIFIER "app.trex.run"
  MACOSX_BUNDLE_BUNDLE_NAME TRex
  MACOSX_BUNDLE_INFO_STRING "Multi-species interactive video-based tracking"
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
)

set(TRACKER_PATH "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>")
message("TRACKER PATH ${TRACKER_PATH}")

if(NOT APPLE)
    set_target_properties(trex_check_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TRACKER_PATH}")

    if(TREX_WITH_TESTS)
        set_target_properties(test_imgui PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TRACKER_PATH}")
    endif()
endif()

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gfx/TRexIcon.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

set(RESOURCE_COPY_FILES
	${CMAKE_SOURCE_DIR}/src/html
	${CMAKE_SOURCE_DIR}/src/fonts
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx
)
set(RESOURCE_COPY_SINGLE_FILES
	${CMAKE_SOURCE_DIR}/default.settings
    ${CMAKE_SOURCE_DIR}/learn_static.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py
)

IF("${CMAKE_SYSTEM}" MATCHES "Linux")
	message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
	foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)

		add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${comp}"
                "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/${comp_s}")
		message(STATUS ${CMAKE_COMMAND} -E create_symlink
            "${comp}"
            "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/${comp_s}")
	endforeach()

	foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)

		add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy_directory
		        "${comp}"
		        "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/${comp_s}")
	endforeach()
elseif(WIN32)
    message(STATUS "(windows) Copying files ${RESOURCE_COPY_SINGLE_FILES}")
    message(STATUS "(install_prefix) :${CMAKE_INSTALL_PREFIX} :${CMAKE_BINARY_DIR}")
    
    foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)

        add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${comp}"
                "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/${comp_s}")
    endforeach()

    message(STATUS "(windows) Copying folders ${RESOURCE_COPY_FILES}")
    foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)

        add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${comp}"
                "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/${comp_s}")
    endforeach()
    
else()
	message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
    foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
        get_filename_component(comp_s ${comp} NAME)
        add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/../Resources/${comp_s}")
        install(FILES ${comp} DESTINATION bin)
    endforeach()

	foreach(comp ${RESOURCE_COPY_FILES})
        get_filename_component(comp_s ${comp} NAME)
		add_custom_command(TARGET ${TREX_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${comp}"
                "$<TARGET_FILE_DIR:${TREX_EXEC_NAME}>/../Resources/${comp_s}")
	endforeach()
endif()

set(targets ${TREX_EXEC_NAME})
if(WIN32)
set(targets ${targets} tracker_python)
endif()

if(APPLE)
add_executable(trex_starter trex.cpp)
target_link_libraries(trex_starter PRIVATE
    commons
    cnpy
    TRex::GLFW
    TRex::LibPNG
    TRex::LibZ
    TRex::OpenCV
    ${EXTRA_LIBS})
set_target_properties(trex_starter PROPERTIES OUTPUT_NAME trex)
set(targets ${targets} trex_starter)

if(TREX_CONDA_PACKAGE_INSTALL)
    if(WIN32)
		set(CONDA_PREFIX_VAR "CONDA_PREFIX")
	else()
		set(CONDA_PREFIX_VAR "$ENV{PREFIX}")
	endif()
    string(REPLACE "\\" "\\\\" CONDA_PREFIX_VAR "${CONDA_PREFIX_VAR}")
    target_compile_definitions(trex_starter PRIVATE TREX_PYTHON_PATH="${CONDA_PREFIX_VAR}")
    message(STATUS "Setting TREX_PYTHON_PATH=${CONDA_PREFIX_VAR}")
else()
    message(STATUS "Adding compile definitions: ${PYTHON_EXECUTABLE}")
    target_compile_definitions(trex_starter PRIVATE
        TREX_PYTHON_PATH="${PYTHON_EXECUTABLE}"
    )
    message(STATUS "Setting TREX_PYTHON_PATH=${PYTHON_EXECUTABLE}")
endif()

endif()

install(TARGETS ${targets}
    EXPORT TRex
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin
    INCLUDES DESTINATION include
    CONFIGURATIONS Release
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/fonts DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/html DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gfx DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
install(FILES ${CMAKE_SOURCE_DIR}/default.settings ${CMAKE_SOURCE_DIR}/learn_static.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
