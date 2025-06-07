add_subdirectory(gui)
if(NOT COMMONS_NO_PYTHON)
    add_subdirectory(python)
endif()
add_subdirectory(tracking)
add_subdirectory(misc)
add_subdirectory(ml)

include_directories(${CMAKE_SOURCE_DIR}/src/ProcessedVideo)

if(TREX_WITH_TESTS)
#add_executable(test_network network.cpp
#    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)
#target_link_libraries(test_network tracker_gui tracker_ml tracker_python #tracker_tracking tracker_misc framegrabber_misc pv commons ${PYTHON_LIBRARIES})
#message(STATUS "pylib=${PYTHON_LIBRARY}")

#if(NOT pybind11_external)
#    target_link_libraries(test_network pybind11::pybind11 pybind11::embed pybind11::module)
#else()
#    target_link_libraries(test_network ${PYTHON_LIBRARIES})
#endif()

#set_target_properties(test_network PROPERTIES
#  MACOSX_BUNDLE TRUE
 # MACOSX_BUNDLE_GUI_IDENTIFIER "app.trex.tests"
#  MACOSX_BUNDLE_BUNDLE_NAME test_network
#  MACOSX_BUNDLE_EXEC_TARGET test_network
#  MACOSX_BUNDLE_INFO_STRING ""
#)

# if(WIN32)
#     add_custom_command(TARGET test_network POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#         "${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py"
#         "$<TARGET_FILE_DIR:test_network>/trex_init.py")
#     add_custom_command(TARGET test_network POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy
#             "${CMAKE_SOURCE_DIR}/learn_static.py"
#             "$<TARGET_FILE_DIR:test_network>/learn_static.py")
#     add_custom_command(TARGET test_network POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy
#             "${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py"
#             "$<TARGET_FILE_DIR:test_network>/visual_identification_network.py")
# else()
#     add_custom_command(TARGET test_network POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E create_symlink
#             "${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py"
#             "$<TARGET_FILE_DIR:test_network>/trex_init.py")
#     add_custom_command(TARGET test_network POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E create_symlink
#             "${CMAKE_SOURCE_DIR}/learn_static.py"
#             "$<TARGET_FILE_DIR:test_network>/learn_static.py")
#     add_custom_command(TARGET test_network POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E create_symlink
#             "${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py"
#             "$<TARGET_FILE_DIR:test_network>/visual_identification_network.py")
# endif()

endif()

if(APPLE)
    set(TREXA_EXEC_NAME TRex)
else()
    set(TREXA_EXEC_NAME trex)
endif()

# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT WIN32 AND TREX_WITH_TESTS)
    add_executable(test_imgui program.cpp)

    target_link_libraries(test_imgui PUBLIC
        framegrabber_misc
        tracker_tracking
        commons
    )
    #target_precompile_headers(test_imgui REUSE_FROM commons)
    add_dependencies(test_imgui commons) #Commons::LibZIP Commons::GLFW Commons::OpenCV)
endif()

#add_executable(properties properties.cpp
#    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)

#target_link_libraries(properties 
#    PUBLIC 
#        tracker_gui
#        tracker_tracking
#        tracker_python
#        tracker_misc
#        framegrabber_misc
#        commons 
#        pv)

#if(NOT COMMONS_NO_PYTHON AND NOT TREX_NO_PYTHON)
#    target_link_libraries(properties 
#        PUBLIC tracker_python)
#endif()

# add actual executable
set(TREXA_SRCS main.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h
    gfx/TRex_16.png
    gfx/TRex_32.png
    gfx/TRex_64.png
    gfx/TRex_128.png
    gfx/TRex_256.png
    gfx/TRex_512.png
    gfx/TRex_1024.png
)

if(APPLE)
    set(TREXA_SRCS ${TREXA_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gfx/AlterIcons.icns)
endif()
if(WIN32 OR UNIX AND NOT APPLE)
    set(TREXA_SRCS ${TREXA_SRCS} trexa.rc)
endif()

add_executable(${TREXA_EXEC_NAME} ${TREXA_SRCS})
target_link_libraries(${TREXA_EXEC_NAME} PUBLIC
    #tracker_tracking
    tracker_gui
    tracker_ml
    tracker_misc
    framegrabber_misc
    #pv
    #commons
)

set_target_properties(${TREXA_EXEC_NAME} PROPERTIES
  XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES"
)

if(WITH_FFMPEG)
    target_compile_definitions(${TREXA_EXEC_NAME} PRIVATE WITH_FFMPEG=true)
endif()

if(BUILD_LEGACY_TREX)
    add_subdirectory(legacy)
endif()

set(TRACKER_PATH "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>")

#if(NOT ${COMMONS_NO_PYTHON} AND NOT ${TREX_NO_PYTHON})
    target_link_libraries(${TREXA_EXEC_NAME} PUBLIC tracker_python)
    message(STATUS "Linking tracker_python")
#endif()

set_target_properties(${TREXA_EXEC_NAME} PROPERTIES
  MACOSX_BUNDLE TRUE
  MACOSX_BUNDLE_GUI_IDENTIFIER "app.trex.run.a"
  MACOSX_BUNDLE_BUNDLE_NAME ${TREXA_EXEC_NAME}
  MACOSX_BUNDLE_INFO_STRING "Multi-species interactive video-based tracking"
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/AlterInfo.plist
)

message("TRACKER PATH ${TRACKER_PATH}")

if(NOT APPLE)
    if(NOT COMMONS_NO_PYTHON)
        set_target_properties(trex_check_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TRACKER_PATH}")
    endif()

    if(NOT WIN32 AND TREX_WITH_TESTS)
        set_target_properties(test_imgui PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TRACKER_PATH}")
    endif()
endif()

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gfx/TRexIcon.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gfx/AlterIcons.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

if(NOT TREX_CONDA_PACKAGE_INSTALL)
    set(RESOURCE_COPY_FILES
        ${CMAKE_SOURCE_DIR}/src/html
        ${CMAKE_SOURCE_DIR}/src/fonts
        ${CMAKE_CURRENT_SOURCE_DIR}/gfx
    )
    set(RESOURCE_COPY_SINGLE_FILES
        ${CMAKE_SOURCE_DIR}/default.settings
        ${CMAKE_CURRENT_SOURCE_DIR}/python/learn_static.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_recognition_torch.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network_torch.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_detection_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_yolo.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/closed_loop_beta.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_learn_category.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/bbx_saved_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_utils.py
        ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json
    )

    IF("${CMAKE_SYSTEM}" MATCHES "Linux")
        message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
            get_filename_component(comp_s ${comp} NAME)

            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
            message(STATUS ${CMAKE_COMMAND} -E copy
                "${comp}"
                "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
        endforeach()

        foreach(comp ${RESOURCE_COPY_FILES})
            get_filename_component(comp_s ${comp} NAME)

            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
        endforeach()
        
    elseif(WIN32)
        message(STATUS "(windows) Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        message(STATUS "(install_prefix) :${CMAKE_INSTALL_PREFIX} :${CMAKE_BINARY_DIR}")
        
        foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
            get_filename_component(comp_s ${comp} NAME)

            message(STATUS "Creating symlink for $<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s} ")
            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${comp}"
                "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
        endforeach()

        message(STATUS "(windows) Copying folders ${RESOURCE_COPY_FILES}")
        foreach(comp ${RESOURCE_COPY_FILES})
            get_filename_component(comp_s ${comp} NAME)

            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/${comp_s}")
        endforeach()
        
    elseif(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
        message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
            get_filename_component(comp_s ${comp} NAME)

            message(STATUS "Creating symlink for $<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/../Resources/${comp_s} ")
            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/../Resources/${comp_s}")
            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/../Resources/${comp_s}")
            install(FILES ${comp} DESTINATION bin)
        endforeach()

        foreach(comp ${RESOURCE_COPY_FILES})
            get_filename_component(comp_s ${comp} NAME)

            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/../Resources/${comp_s}")
        endforeach()
    endif()
else()
    set(RESOURCE_COPY_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/gfx
    )

    if(APPLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
        message(STATUS "Copying files ${RESOURCE_COPY_FILES}")

        foreach(comp ${RESOURCE_COPY_FILES})
            get_filename_component(comp_s ${comp} NAME)

            add_custom_command(TARGET ${TREXA_EXEC_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${comp}"
                    "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>/../Resources/${comp_s}")
        endforeach()
    endif()
endif()

set(targets ${TREXA_EXEC_NAME})

if(BUILD_LEGACY_TREX)
    set(targets ${targets} ${TREX_EXEC_NAME})
endif()

if(WIN32)# AND NOT COMMONS_NO_PYTHON)
    set(targets ${targets} tracker_python)
endif()

if(APPLE)
    add_executable(trex_starter trex.cpp)
    target_link_libraries(trex_starter PRIVATE commons)
    set_target_properties(trex_starter PROPERTIES OUTPUT_NAME trex)
    set(targets ${targets} trex_starter)

    set_target_properties(trex_starter PROPERTIES
        XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES"
    )
endif()

if(APPLE)
    target_sources(${TREXA_EXEC_NAME} PRIVATE gfx/AlterIcons.xcassets)
    set_source_files_properties(gfx/AlterIcons.icns PROPERTIES
        MACOSX_PACKAGE_LOCATION Resources
    )
    set_target_properties(${TREXA_EXEC_NAME} PROPERTIES
        XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AlterIcons")
endif()

#target_precompile_headers(properties REUSE_FROM commons)
#target_precompile_headers(${TREX_EXEC_NAME} REUSE_FROM commons)
#target_precompile_headers(${TREXA_EXEC_NAME} REUSE_FROM commons)

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

install(FILES 
        ${CMAKE_SOURCE_DIR}/default.settings 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/learn_static.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_recognition_torch.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network_torch.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/closed_loop_beta.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_learn_category.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/bbx_saved_model.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_utils.py
        ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json 
        ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json
    DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)

install(FILES ${CMAKE_SOURCE_DIR}/default.settings ${CMAKE_CURRENT_SOURCE_DIR}/python/learn_static.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_recognition_torch.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network_torch.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_detection_model.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_yolo.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py ${CMAKE_CURRENT_SOURCE_DIR}/python/bbx_saved_model.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_learn_category.py ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
