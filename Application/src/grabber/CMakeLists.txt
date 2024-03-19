add_subdirectory(misc)

file(GLOB HTML html/*.html html/*.js html/*.css)

# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(GIT_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)

file(GLOB SRCS
    ${HTML}
	grabber.h
	grabber.cpp
     gpuImage.h
     gpuImage.cpp
	gui.h
	gui.cpp
    CropWindow.h
    CropWindow.cpp
    ImageThreads.h
    ImageThreads.cpp
	main.cpp
    ${GIT_SOURCES}
    gfx/TGrabsIcon16.png
    gfx/TGrabsIcon32.png
    gfx/TGrabsIcon64.png
)

if(APPLE)
    set(SRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gfx/GrabberIcon.icns)
endif()

if(WIN32)
	set(SRCS ${SRCS} tgrabs.rc)
endif()

source_group(html FILES ${HTML})
		
if(BUILD_LEGACY_TGRABS)
    if(APPLE)
        set(TGRABS_EXEC_NAME TGrabs)
    else()
        set(TGRABS_EXEC_NAME tgrabs)
    endif()
    add_executable(${TGRABS_EXEC_NAME} ${SRCS})
    #target_precompile_headers(${TGRABS_EXEC_NAME} REUSE_FROM commons)

    if(WIN32 AND NOT COMMONS_NO_PYTHON)
        set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES LINK_FLAGS_RELEASE "/DELAYLOAD:tracker_python.dll" )
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}  /DELAYLOAD:tracker_python.dll")
        set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DELAYLOAD:tracker_python-d.dll" )
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}  /DELAYLOAD:tracker_python-d.dll")
    endif()

    if(NOT COMMONS_NO_PYTHON)
        if(APPLE)
            ADD_CUSTOM_COMMAND(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:trex_check_python> $<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>)
        endif()
        add_dependencies(${TGRABS_EXEC_NAME} trex_check_python ) #Commons::OpenCV)
    endif()

    add_dependencies(${TGRABS_EXEC_NAME} commons)

    target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE
        framegrabber_misc
        tracker_misc
        tracker_tracking
        tracker_gui
        pv
        commons
    )

    #if(NOT COMMONS_NO_PYTHON)
        target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE tracker_python)
        if(WIN32)
            set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
        endif()

        if(NOT pybind11_external)
            target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE pybind11::pybind11 pybind11::embed pybind11::module)
        else()
            target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE ${PYTHON_LIBRARIES})
        endif()
    #endif()

    if(WITH_HTTPD)
        add_dependencies(${TGRABS_EXEC_NAME} LibMicroHttpd)
    endif()
    #if(WITH_FFMPEG)
    #    target_link_libraries(${TGRABS_EXEC_NAME} PRIVATE Commons::FFMPEG)
    #endif()

endif()

add_executable(pvconvert pvconvert.cpp gif.h ${GIT_SOURCES})
target_link_libraries(pvconvert PUBLIC
    tracker_gui
    pv
    commons
)

add_dependencies(pvconvert commons)#Commons::OpenCV)
#target_precompile_headers(pvconvert REUSE_FROM commons)


add_executable(pvinfo pvinfo.cpp pvinfo_merge.cpp pvinfo_merge.h ${GIT_SOURCES})

target_link_libraries(pvinfo PRIVATE
	tracker_tracking
    tracker_gui
    tracker_misc
    framegrabber_misc
	commons
	pv
)

#if(NOT COMMONS_NO_PYTHON)
    target_link_libraries(pvinfo PRIVATE tracker_python)
    if(WIN32)
        set_target_properties(pvinfo PROPERTIES LINK_FLAGS "/DELAYLOAD:tracker_python.dll /DELAYLOAD:tracker_python-d.dll")
    endif()
#endif()

#target_precompile_headers(pvinfo REUSE_FROM commons)

add_dependencies(pvinfo commons) #Commons::OpenCV)

get_target_property(OUT commons INTERFACE_LINK_LIBRARIES)
message(STATUS "Commons::GLFW libraries: ${OUT}")

set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES
  MACOSX_BUNDLE TRUE
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gfx/GrabberIcon.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

if(NOT TREX_CONDA_PACKAGE_INSTALL )
    set(RESOURCE_COPY_FILES
        ${CMAKE_SOURCE_DIR}/src/html
        ${CMAKE_SOURCE_DIR}/src/fonts
        ${CMAKE_CURRENT_SOURCE_DIR}/gfx
    )

    set(RESOURCE_COPY_SINGLE_FILES
        ${CMAKE_SOURCE_DIR}/default.settings
        ${CMAKE_SOURCE_DIR}/closed_loop.py
        ${CMAKE_SOURCE_DIR}/pretrained_tagwork.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../tracker/python/trex_init.py
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

    if(BUILD_LEGACY_TGRABS)
        if(APPLE)
            foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
                get_filename_component(comp_s ${comp} NAME)
                add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy
                                "${comp}"
                                "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/../Resources/${comp_s}")
            endforeach()
        elseif(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
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
        elseif(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
            foreach(comp ${RESOURCE_COPY_FILES})
                get_filename_component(comp_s ${comp} NAME)
                add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${comp}"
                        "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/${comp_s}")
            endforeach()
        endif()
    endif()
elseif(BUILD_LEGACY_TGRABS)
    set(RESOURCE_COPY_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/gfx
    )
    if(APPLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
        foreach(comp ${RESOURCE_COPY_FILES})
            get_filename_component(comp_s ${comp} NAME)
            add_custom_command(TARGET ${TGRABS_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TGRABS_EXEC_NAME}>/../Resources/${comp_s}")
        endforeach()
    endif()
endif()

if(BUILD_LEGACY_TGRABS)
    set(targets ${TGRABS_EXEC_NAME} pvinfo pvconvert)

    if(WITH_FFMPEG)
        target_compile_definitions(${TGRABS_EXEC_NAME} PRIVATE WITH_FFMPEG=true)
    endif()

    if(APPLE)
        target_sources(${TGRABS_EXEC_NAME} PRIVATE gfx/TGrabsIcons.xcassets)
        set_source_files_properties(gfx/TGrabsIcons.xcassets PROPERTIES
            MACOSX_PACKAGE_LOCATION Resources
        )
        set_target_properties(${TGRABS_EXEC_NAME} PROPERTIES
            XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon")
    endif()

else()
    set(targets pvinfo pvconvert)
endif()

if(APPLE)
    add_executable(tgrabs_starter tgrabs.cpp ${GIT_SOURCES})
    target_link_libraries(tgrabs_starter PRIVATE commons)
    set_target_properties(tgrabs_starter PROPERTIES OUTPUT_NAME tgrabs)
    set(targets ${targets} tgrabs_starter)
    #target_precompile_headers(tgrabs_starter REUSE_FROM commons)
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

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gfx DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
install(FILES ${CMAKE_SOURCE_DIR}/closed_loop.py ${CMAKE_SOURCE_DIR}/pretrained_tagwork.py DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)

if(APPLE)
target_compile_definitions(tgrabs_starter PRIVATE -DTGRABS_TARGET_PATH="${CMAKE_INSTALL_PREFIX}/bin")
endif()
