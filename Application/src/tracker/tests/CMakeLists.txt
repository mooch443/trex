file(GLOB HDRS *.h)
file(GLOB SRCS *.cpp)

# automatic moc file generation by Qt MOC
set(CMAKE_AUTOMOC OFF)
# to include generated moc headers, since they are generated in binary dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

message(STATUS "TEsts: ${SRCS} ${HDRS}")

add_executable(tracker_tests ${SRCS} ${HDRS})

target_link_libraries(tracker_tests
    tracker_misc
    tracker_tracking
    tracker_gui
    commons
    ${LIBZIP_LIBRARY}
    cpputils
    pv
    ${OpenCV_LIBS}
    ${GLFW_LIBRARIES}
    imgui
)


#set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -g -O3")

target_compile_definitions(tracker_tests PRIVATE TT_RESOURCE_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}" TRACKER_GLOBAL_THREADS=1)
#target_compile_options(tracker_tests PRIVATE
#    $<$<CXX_COMPILER_ID:MSVC>:/Zi>
#    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g>#
#	-O3
#    )

# add actual library
add_test (NAME TrackerTests COMMAND tracker_tests)
