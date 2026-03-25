file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS "${CMAKE_CURRENT_SOURCE_DIR}/BaslerPylonBridge.cpp")
list(FILTER SRCS EXCLUDE REGEX "(^|.*/)BaslerPylonBridge\\.cpp$")

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(UNIX)
  add_library(framegrabber_misc STATIC ${SRCS} ${HDRS})
elseif(WIN32)
  add_library(framegrabber_misc STATIC ${SRCS} ${HDRS})
endif()

target_link_libraries(framegrabber_misc
    PUBLIC Commons::Video
)

if(WITH_FFMPEG)
    target_compile_definitions(framegrabber_misc PRIVATE WITH_FFMPEG=true)
endif()

option(WITH_PYLON "use pylon support" OFF)

if(WITH_PYLON)
    find_package(BaslerPylon)
    if(BaslerPylon_FOUND)
        message(STATUS "Basler pylon headers found; enabling lazy runtime loading.")
        target_include_directories(framegrabber_misc PUBLIC ${PYLON_INCLUDE_DIRS})
        target_compile_definitions(framegrabber_misc PUBLIC ${PYLON_COMPILE_DEFINITIONS})

        if(APPLE)
            add_library(trex_basler_bridge SHARED
                BaslerPylonBridge.cpp
                BaslerPylonBridgeAPI.h
            )

            target_include_directories(trex_basler_bridge PRIVATE ${PYLON_INCLUDE_DIRS})

            if(PYLON_FRAMEWORK_DIR)
                target_compile_options(trex_basler_bridge PRIVATE "-F${PYLON_FRAMEWORK_DIR}")
                target_link_options(trex_basler_bridge PRIVATE "-F${PYLON_FRAMEWORK_DIR}")
            endif()

            target_link_options(trex_basler_bridge PRIVATE
                "-Wl,-framework,pylon"
                "-Wl,-rpath,/Library/Frameworks"
            )

            if(PYLON_FRAMEWORK_DIR)
                target_link_options(trex_basler_bridge PRIVATE
                    "-Wl,-rpath,${PYLON_FRAMEWORK_DIR}"
                )
            endif()

            set_target_properties(trex_basler_bridge PROPERTIES
                OUTPUT_NAME "trex_basler_bridge"
                PREFIX ""
                BUILD_RPATH "/Library/Frameworks"
                INSTALL_RPATH "/Library/Frameworks"
            )
        endif()
    else()
        message(STATUS "Basler pylon headers not found; disabling Basler support.")
        target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=0)
    endif()
else()
    target_compile_definitions(framegrabber_misc PUBLIC WITH_PYLON=0)
endif()