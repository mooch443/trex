include(${CMAKE_CURRENT_SOURCE_DIR}/TrackerSources.cmake)

if(TREX_ENABLE_SHARED_INTERNAL_LIBS)
    set(TREX_INTERNAL_LIBRARY_TYPE SHARED)
else()
    set(TREX_INTERNAL_LIBRARY_TYPE STATIC)
endif()

set(TREX_UI_LIBRARY_TYPE ${TREX_INTERNAL_LIBRARY_TYPE})
if(WIN32 AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    # The UI layer has a very large symbol surface. Letting CMake auto-export
    # every symbol can overflow link.exe/import-lib limits, while the lower
    # semantic libraries still benefit from shared boundaries. trex_ui is only
    # consumed by final executables/tests, so keep it static on Windows.
    set(TREX_UI_LIBRARY_TYPE STATIC)
endif()

function(trex_harden_internal_library target)
    if(APPLE AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
        set_target_properties(${target} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF
        )
        target_compile_options(${target} PRIVATE -fvisibility=default)
        # Apple ld defaults to treating undefined symbols as link errors.
    endif()
endfunction()

function(trex_bundle_lazy_python target)
    if(APPLE AND TREX_ENABLE_SHARED_INTERNAL_LIBS AND TARGET trex_python)
        add_dependencies(${target} trex_python)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/../Frameworks"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:trex_python>"
                "$<TARGET_FILE_DIR:${target}>/../Frameworks/$<TARGET_FILE_NAME:trex_python>"
            VERBATIM
        )
    endif()
endfunction()

function(trex_bundle_basler_bridge target)
    if(APPLE AND TARGET trex_basler_bridge)
        add_dependencies(${target} trex_basler_bridge)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/../Frameworks"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:trex_basler_bridge>"
                "$<TARGET_FILE_DIR:${target}>/../Frameworks/$<TARGET_FILE_NAME:trex_basler_bridge>"
            VERBATIM
        )
    endif()
endfunction()

function(trex_link_static_python_impl target)
    target_link_libraries(${target} PUBLIC trex_python_staging trex_python)
    
    if(WIN32)
        target_link_options(${target} PRIVATE
            "/DELAYLOAD:trex_python.dll"
            "/DELAYLOAD:trex_python-d.dll"
        )
    elseif(UNIX AND NOT APPLE AND NOT TREX_ENABLE_SHARED_INTERNAL_LIBS)
        # Static Unix builds create a link cycle:
        # trex_python_staging references trex_python_register from trex_python,
        # while trex_python itself depends on staging symbols. Force inclusion of
        # all trex_python symbols with --whole-archive so the cycle resolves.
        target_link_options(${target} PRIVATE
            "-Wl,--whole-archive" "$<TARGET_FILE:trex_python>" "-Wl,--no-whole-archive"
        )
    endif()
endfunction()

function(trex_add_semantic_library target)
    set(options)
    set(one_value_args TYPE)
    set(multi_value_args SOURCES PUBLIC_HEADERS PRIVATE_HEADERS MODULE_INTERFACES)
    cmake_parse_arguments(TREX_LIB "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(_trex_lib_type ${TREX_LIB_TYPE})
    if(NOT _trex_lib_type)
        set(_trex_lib_type STATIC)
    endif()

    add_library(${target} ${_trex_lib_type}
        ${TREX_LIB_SOURCES}
        ${TREX_LIB_PUBLIC_HEADERS}
        ${TREX_LIB_PRIVATE_HEADERS}
    )

    if(TREX_ENABLE_MODULES AND TREX_LIB_MODULE_INTERFACES)
        target_sources(${target}
            PUBLIC
                FILE_SET ${target}_modules TYPE CXX_MODULES
                BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                FILES ${TREX_LIB_MODULE_INTERFACES}
        )
    endif()

    if(_trex_lib_type STREQUAL "SHARED")
        if(WIN32)
            target_compile_definitions(${target}
                PRIVATE "TREX_EXPORT=__declspec(dllexport)" "TREX_TYPE_EXPORT=__declspec(dllexport)"
            )
            set_target_properties(${target} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        else()
            target_compile_definitions(${target}
                PRIVATE "TREX_EXPORT=__attribute__((visibility(\"default\")))" "TREX_TYPE_EXPORT=__attribute__((visibility(\"default\"),type_visibility(\"default\")))"
            )
        endif()
    else()
        # Keep TREX export macros local for static libs so they do not override
        # explicit dllexport on downstream shared targets (e.g. trex_python).
        target_compile_definitions(${target} PRIVATE "TREX_EXPORT=" "TREX_TYPE_EXPORT=")
    endif()

    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(WIN32)
        target_compile_options(${target} PRIVATE /bigobj)
    endif()

    if(WITH_FFMPEG)
        target_compile_definitions(${target} PUBLIC WITH_FFMPEG=true)
    endif()
endfunction()

set(TREX_DYLIB_TARGETS trex_python)

trex_add_semantic_library(trex_core
    TYPE ${TREX_INTERNAL_LIBRARY_TYPE}
    SOURCES ${TREX_CORE_SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp"
    PUBLIC_HEADERS ${TREX_CORE_PUBLIC_HEADERS}
    PRIVATE_HEADERS ${TREX_CORE_PRIVATE_HEADERS}
    MODULE_INTERFACES ${TREX_CORE_MODULE_INTERFACE_UNITS}
)
trex_harden_internal_library(trex_core)
target_link_libraries(trex_core PUBLIC
    framegrabber_misc
    pv
    lzo
    Commons::Gui
)
set(TREX_DYLIB_TARGETS ${TREX_DYLIB_TARGETS} trex_core)

trex_add_semantic_library(trex_data
    TYPE ${TREX_INTERNAL_LIBRARY_TYPE}
    SOURCES ${TREX_DATA_SOURCES}
    PUBLIC_HEADERS ${TREX_DATA_PUBLIC_HEADERS}
    PRIVATE_HEADERS ${TREX_DATA_PRIVATE_HEADERS}
    MODULE_INTERFACES ${TREX_DATA_MODULE_INTERFACE_UNITS}
)
trex_harden_internal_library(trex_data)
target_link_libraries(trex_data PUBLIC trex_core)

set(TREX_DYLIB_TARGETS ${TREX_DYLIB_TARGETS} trex_data)

trex_add_semantic_library(trex_tracking
    TYPE ${TREX_INTERNAL_LIBRARY_TYPE}
    SOURCES ${TREX_TRACKING_SOURCES}
    PUBLIC_HEADERS ${TREX_TRACKING_PUBLIC_HEADERS}
    PRIVATE_HEADERS ${TREX_TRACKING_PRIVATE_HEADERS}
    MODULE_INTERFACES ${TREX_TRACKING_MODULE_INTERFACE_UNITS}
)
trex_harden_internal_library(trex_tracking)
target_link_libraries(trex_tracking PUBLIC
    trex_data
    trex_core
)
if(WITH_HTTPD)
    add_dependencies(trex_tracking LibMicroHttpd)
endif()

trex_add_semantic_library(trex_ml
    TYPE ${TREX_INTERNAL_LIBRARY_TYPE}
    SOURCES ${TREX_ML_SOURCES}
    PUBLIC_HEADERS ${TREX_ML_PUBLIC_HEADERS}
    PRIVATE_HEADERS ${TREX_ML_PRIVATE_HEADERS}
    MODULE_INTERFACES ${TREX_ML_MODULE_INTERFACE_UNITS}
)
trex_harden_internal_library(trex_ml)
target_link_libraries(trex_ml PUBLIC
    trex_tracking
    trex_data
    trex_core
)
target_link_libraries(trex_ml PRIVATE
    trex_python_staging
)

set(TREX_DYLIB_TARGETS ${TREX_DYLIB_TARGETS} trex_ml)

trex_add_semantic_library(trex_ui
    TYPE ${TREX_UI_LIBRARY_TYPE}
    SOURCES ${TREX_UI_SOURCES}
    PUBLIC_HEADERS ${TREX_UI_PUBLIC_HEADERS}
    PRIVATE_HEADERS ${TREX_UI_PRIVATE_HEADERS}
    MODULE_INTERFACES ${TREX_UI_MODULE_INTERFACE_UNITS}
)
trex_harden_internal_library(trex_ui)
target_link_libraries(trex_ui PUBLIC
    trex_ml
    trex_tracking
    trex_data
    trex_core
    trex_python_staging
)
if(WIN32 AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    # trex_ui includes commons GUI headers directly. On Windows shared builds,
    # make that dependency explicit so dllimport usage requirements are active
    # while compiling this static library, not only at final executable links.
    target_link_libraries(trex_ui PUBLIC Commons::Gui)
    target_compile_definitions(trex_ui PRIVATE COMMONS_IS_SHARED_LIBRARY=1)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(trex_ui PUBLIC atomic)
endif()

set(TREX_DYLIB_TARGETS ${TREX_DYLIB_TARGETS} trex_ui)

add_library(tracker_misc INTERFACE)
target_link_libraries(tracker_misc INTERFACE trex_core trex_data)

add_library(tracker_tracking INTERFACE)
target_link_libraries(tracker_tracking INTERFACE trex_tracking trex_data trex_core)

add_library(tracker_ml INTERFACE)
target_link_libraries(tracker_ml INTERFACE trex_ml trex_tracking trex_data trex_core)

add_library(tracker_gui INTERFACE)
target_link_libraries(tracker_gui INTERFACE trex_ui trex_ml trex_tracking trex_data trex_core)

if(NOT COMMONS_NO_PYTHON)
    add_subdirectory(python)
    # In static Unix builds, trex_python_staging can reach final link steps
    # through tracker_* interface libraries. Make the implementation library
    # part of the propagated runtime link set at the interface layer.
    if(TARGET trex_python AND UNIX AND NOT WIN32 AND NOT TREX_ENABLE_SHARED_INTERNAL_LIBS)
        target_link_libraries(tracker_tracking INTERFACE trex_python)
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/ProcessedVideo)

if(TREX_WITH_TESTS)
#add_executable(test_network network.cpp
#    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)
#target_link_libraries(test_network tracker_gui tracker_ml trex_python #tracker_tracking tracker_misc framegrabber_misc pv commons ${PYTHON_LIBRARIES})
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
        Commons::Gui
    )
    trex_link_static_python_impl(test_imgui)
    if(NOT COMMONS_ENABLE_MODULES AND NOT COMMONS_DONT_USE_PCH
            AND COMMONS_CAN_REUSE_PCH_ACROSS_TARGETS
            AND DEFINED COMMONS_PCH_OWNER AND NOT "${COMMONS_PCH_OWNER}" STREQUAL "")
        target_precompile_headers(test_imgui REUSE_FROM ${COMMONS_PCH_OWNER})
    endif()
endif()

#add_executable(properties properties.cpp
#    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp" ../GitSHA1.h)

#target_link_libraries(properties 
#    PUBLIC 
#        tracker_gui
#        tracker_tracking
#        trex_python
#        tracker_misc
#        framegrabber_misc
#        commons 
#        pv)

#if(NOT COMMONS_NO_PYTHON AND NOT TREX_NO_PYTHON)
#    target_link_libraries(properties 
#        PUBLIC trex_python)
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
if(WIN32)
target_compile_options(${TREXA_EXEC_NAME} PRIVATE /bigobj)
endif()

target_link_libraries(${TREXA_EXEC_NAME} PUBLIC
    #tracker_tracking
    tracker_gui
    tracker_ml
    tracker_misc
    framegrabber_misc
    #pv
    #commons
)
trex_bundle_lazy_python(${TREXA_EXEC_NAME})
trex_bundle_basler_bridge(${TREXA_EXEC_NAME})

set(TREX_APP_TARGETS ${TREXA_EXEC_NAME})
set(TREX_TOOL_TARGETS)

add_executable(pvinfo
    pvinfo.cpp
    pvinfo_merge.cpp
    pvinfo_merge.h
    "${CMAKE_CURRENT_BINARY_DIR}/../GitSHA1.cpp"
    ../GitSHA1.h
)
target_link_libraries(pvinfo PRIVATE
    tracker_tracking
    tracker_gui
    tracker_misc
    framegrabber_misc
    Commons::All
    pv
)
if(TARGET trex_python_staging)
    target_link_libraries(pvinfo PRIVATE trex_python_staging)
endif()
list(APPEND TREX_TOOL_TARGETS pvinfo)

if(TREX_ENABLE_PROTOTYPES)
    set(INTERACTIVE_SEGMENTATION_EXEC_NAME interactive_segmentation_prototype)
    set(INTERACTIVE_SEGMENTATION_SRCS
        interactive_segmentation_prototype/main.cpp interactive_segmentation_prototype/LiveSegmentation.cpp interactive_segmentation_prototype/LiveSegmentation.h 
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
        set(INTERACTIVE_SEGMENTATION_SRCS ${INTERACTIVE_SEGMENTATION_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gfx/AlterIcons.icns  )
    endif()
    if(WIN32 OR UNIX AND NOT APPLE)
        set(INTERACTIVE_SEGMENTATION_SRCS ${INTERACTIVE_SEGMENTATION_SRCS} trexa.rc)
    endif()

    add_executable(${INTERACTIVE_SEGMENTATION_EXEC_NAME} ${INTERACTIVE_SEGMENTATION_SRCS})
    target_link_libraries(${INTERACTIVE_SEGMENTATION_EXEC_NAME} PUBLIC
        tracker_gui
        tracker_ml
        tracker_tracking
        tracker_misc
        trex_python_staging
        framegrabber_misc
        pv
        Commons::All
    )
    trex_bundle_lazy_python(${INTERACTIVE_SEGMENTATION_EXEC_NAME})
    trex_bundle_basler_bridge(${INTERACTIVE_SEGMENTATION_EXEC_NAME})
    list(APPEND TREX_APP_TARGETS ${INTERACTIVE_SEGMENTATION_EXEC_NAME})
endif()

# macOS ld already errors on undefined symbols and no longer accepts the explicit flag.
foreach(app_target ${TREX_APP_TARGETS})
    if(UNIX AND NOT APPLE AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Emscripten")
        target_link_options(${app_target} PRIVATE "-Wl,--no-undefined")
    elseif(WIN32 AND NOT MSVC)
        target_link_options(${app_target} PRIVATE "-Wl,--no-undefined")
    endif()
endforeach()
foreach(tool_target ${TREX_TOOL_TARGETS})
    if(UNIX AND NOT APPLE AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Emscripten")
        target_link_options(${tool_target} PRIVATE "-Wl,--no-undefined")
    elseif(WIN32 AND NOT MSVC)
        target_link_options(${tool_target} PRIVATE "-Wl,--no-undefined")
    endif()
endforeach()

if(WITH_FFMPEG)
    foreach(app_target ${TREX_APP_TARGETS})
        target_compile_definitions(${app_target} PRIVATE WITH_FFMPEG=true)
    endforeach()
    foreach(tool_target ${TREX_TOOL_TARGETS})
        target_compile_definitions(${tool_target} PRIVATE WITH_FFMPEG=true)
    endforeach()
endif()

set(TRACKER_PATH "$<TARGET_FILE_DIR:${TREXA_EXEC_NAME}>")
foreach(tool_target ${TREX_TOOL_TARGETS})
    if(NOT APPLE)
        set_target_properties(${tool_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TRACKER_PATH}")
    endif()
endforeach()

foreach(app_target ${TREX_APP_TARGETS})
    trex_link_static_python_impl(${app_target})
endforeach()
foreach(tool_target ${TREX_TOOL_TARGETS})
    trex_link_static_python_impl(${tool_target})
endforeach()

#if(NOT ${COMMONS_NO_PYTHON} AND NOT ${TREX_NO_PYTHON})
    target_link_libraries(${TREXA_EXEC_NAME} PUBLIC trex_python_staging)
    message(STATUS "Linking trex_python_staging")
#endif()

set_target_properties(${TREXA_EXEC_NAME} PROPERTIES
  MACOSX_BUNDLE TRUE
  MACOSX_BUNDLE_GUI_IDENTIFIER "app.trex.run.a"
  MACOSX_BUNDLE_BUNDLE_NAME ${TREXA_EXEC_NAME}
  MACOSX_BUNDLE_INFO_STRING "Multi-species interactive video-based tracking"
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/AlterInfo.plist
  XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES"
)

if(TREX_ENABLE_PROTOTYPES)
    set_target_properties(${INTERACTIVE_SEGMENTATION_EXEC_NAME} PROPERTIES
      MACOSX_BUNDLE TRUE
      MACOSX_BUNDLE_GUI_IDENTIFIER "app.trex.interactive_segmentation_prototype"
      MACOSX_BUNDLE_BUNDLE_NAME ${INTERACTIVE_SEGMENTATION_EXEC_NAME}
      MACOSX_BUNDLE_INFO_STRING "Interactive segmentation prototype"
      MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/AlterInfo.plist
      XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES"
    )
endif()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_sam3_interface.py
        ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/interactive_segmentation_prototype/live_segmentation_layout.json
    )

    if(TREX_ENABLE_SHARED_INTERNAL_LIBS)
        foreach(app_target ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})
            foreach(dylib ${TREX_DYLIB_TARGETS})
                message(STATUS "${CMAKE_COMMAND} -E create_symlink
                            $<TARGET_FILE_DIR:${dylib}>/lib${dylib}.dylib
                            $<TARGET_FILE_DIR:${app_target}>/../Resources/lib${dylib}.dylib")
                add_custom_command(TARGET ${app_target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E create_symlink
                            $<TARGET_FILE_DIR:${dylib}>/lib${dylib}.dylib
                            "$<TARGET_FILE_DIR:${app_target}>/../Resources/lib${dylib}.dylib")
            endforeach()
        endforeach()
    endif()

    IF("${CMAKE_SYSTEM}" MATCHES "Linux")
        message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        foreach(app_target ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})
            foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
                get_filename_component(comp_s ${comp} NAME)

                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
                message(STATUS ${CMAKE_COMMAND} -E copy
                    "${comp}"
                    "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
            endforeach()

            foreach(comp ${RESOURCE_COPY_FILES})
                get_filename_component(comp_s ${comp} NAME)

                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
            endforeach()
        endforeach()
        
    elseif(WIN32)
        message(STATUS "(windows) Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        message(STATUS "(install_prefix) :${CMAKE_INSTALL_PREFIX} :${CMAKE_BINARY_DIR}")
        
        foreach(app_target ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})
            foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
                get_filename_component(comp_s ${comp} NAME)

                message(STATUS "Creating symlink for $<TARGET_FILE_DIR:${app_target}>/${comp_s} ")
                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
                add_custom_command(TARGET ${app_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${comp}"
                    "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
            endforeach()

            message(STATUS "(windows) Copying folders ${RESOURCE_COPY_FILES}")
            foreach(comp ${RESOURCE_COPY_FILES})
                get_filename_component(comp_s ${comp} NAME)

                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/${comp_s}")
            endforeach()

            add_custom_command(TARGET ${app_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${app_target}>
                    $<TARGET_FILE_DIR:${app_target}>
                COMMAND_EXPAND_LISTS)
        endforeach()
        
    elseif(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
        message(STATUS "Copying files ${RESOURCE_COPY_SINGLE_FILES}")
        foreach(app_target ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})
            foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
                get_filename_component(comp_s ${comp} NAME)

                message(STATUS "Creating symlink for $<TARGET_FILE_DIR:${app_target}>/../Resources/${comp_s} ")
                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/../Resources/${comp_s}")
                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/../Resources/${comp_s}")
            endforeach()

            foreach(comp ${RESOURCE_COPY_FILES})
                get_filename_component(comp_s ${comp} NAME)

                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/../Resources/${comp_s}")
            endforeach()
        endforeach()

        foreach(comp ${RESOURCE_COPY_SINGLE_FILES})
            install(FILES ${comp} DESTINATION bin)
        endforeach()
    endif()
else()
    set(RESOURCE_COPY_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/gfx
    )

    if(APPLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
        message(STATUS "Copying files ${RESOURCE_COPY_FILES}")

        foreach(app_target ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})
            foreach(comp ${RESOURCE_COPY_FILES})
                get_filename_component(comp_s ${comp} NAME)

                add_custom_command(TARGET ${app_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${comp}"
                        "$<TARGET_FILE_DIR:${app_target}>/../Resources/${comp_s}")
            endforeach()
        endforeach()
    endif()
endif()

set(targets ${TREX_APP_TARGETS} ${TREX_TOOL_TARGETS})

if(BUILD_LEGACY_TREX)
    set(targets ${targets} ${TREX_EXEC_NAME})
endif()

if(WIN32)# AND NOT COMMONS_NO_PYTHON)
    set(targets ${targets} trex_python_staging trex_python)
endif()

if(APPLE)
    add_executable(trex_starter trex.cpp)
    target_link_libraries(trex_starter PRIVATE Commons::All)
    set_target_properties(trex_starter PROPERTIES OUTPUT_NAME trex)
    set(targets ${targets} trex_starter)

    set_target_properties(trex_starter PROPERTIES
        XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "YES"
    )
endif()

if(APPLE)
    foreach(app_target ${TREX_APP_TARGETS})
        target_sources(${app_target} PRIVATE gfx/AlterIcons.xcassets)
        set_source_files_properties(gfx/AlterIcons.icns PROPERTIES
            MACOSX_PACKAGE_LOCATION Resources
        )
        set_target_properties(${app_target} PROPERTIES
            XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AlterIcons")
    endforeach()
endif()

#target_precompile_headers(properties REUSE_FROM commons)
#target_precompile_headers(${TREX_EXEC_NAME} REUSE_FROM commons)  # legacy; TREX_EXEC_NAME only set when BUILD_LEGACY_TREX
if(NOT COMMONS_ENABLE_MODULES AND NOT COMMONS_DONT_USE_PCH
    AND COMMONS_CAN_REUSE_PCH_ACROSS_TARGETS
        AND DEFINED COMMONS_PCH_OWNER AND NOT "${COMMONS_PCH_OWNER}" STREQUAL "")
    target_precompile_headers(${TREXA_EXEC_NAME} REUSE_FROM ${COMMONS_PCH_OWNER})
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
        ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_sam3_interface.py
        ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json 
        ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json
        ${CMAKE_CURRENT_SOURCE_DIR}/interactive_segmentation_prototype/live_segmentation_layout.json
    DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)

install(FILES ${CMAKE_SOURCE_DIR}/default.settings ${CMAKE_CURRENT_SOURCE_DIR}/python/learn_static.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_recognition_torch.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network_torch.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_detection_model.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_yolo.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_init.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_identification_network.py ${CMAKE_CURRENT_SOURCE_DIR}/python/bbx_saved_model.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_sam3_interface.py ${CMAKE_CURRENT_SOURCE_DIR}/python/trex_learn_category.py ${CMAKE_CURRENT_SOURCE_DIR}/alter_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/annotation_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/welcome_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/tracking_settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/choose_settings_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/tracking_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/export_options_layout.json ${CMAKE_CURRENT_SOURCE_DIR}/interactive_segmentation_prototype/live_segmentation_layout.json DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/share/trex)
