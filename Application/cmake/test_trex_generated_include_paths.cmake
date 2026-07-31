set(_commons_cmake "${CMAKE_CURRENT_LIST_DIR}/../src/commons/CMakeLists.txt")
file(READ "${_commons_cmake}" _commons_source)

set(_glfw_include_marker [=[set(GLFW_INCLUDE_DIR ${GLFW_INSTALL_DIR}/include)]=])
set(_glfw_prepare_marker [=[file(MAKE_DIRECTORY "${GLFW_INCLUDE_DIR}")]=])
set(_glfw_publish_marker [=["$<BUILD_INTERFACE:${GLFW_INCLUDE_DIR}>"]=])

string(FIND "${_commons_source}" "${_glfw_include_marker}" _glfw_include_position)
string(FIND "${_commons_source}" "${_glfw_prepare_marker}" _glfw_prepare_position)
string(FIND "${_commons_source}" "${_glfw_publish_marker}" _glfw_publish_position)

if(_glfw_include_position EQUAL -1 OR _glfw_publish_position EQUAL -1)
    message(FATAL_ERROR "Could not locate the custom GLFW include-directory setup in ${_commons_cmake}.")
endif()

if(_glfw_prepare_position EQUAL -1)
    message(FATAL_ERROR
        "The custom GLFW include directory must be created during configuration before it is published by Commons::GLFW.")
endif()

if(_glfw_prepare_position LESS _glfw_include_position OR
   _glfw_prepare_position GREATER _glfw_publish_position)
    message(FATAL_ERROR
        "The custom GLFW include directory is not prepared between its definition and its interface publication.")
endif()

message(STATUS "Generated GLFW include-directory ordering passed.")
