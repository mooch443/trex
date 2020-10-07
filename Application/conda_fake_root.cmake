if(UNIX AND NOT APPLE)
# this one is important
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_PLATFORM Linux)
#this one not so much
set(CMAKE_SYSTEM_VERSION 1)
endif()

# specify the cross compiler
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

# where is the target environment
set(CMAKE_FIND_ROOT_PATH $ENV{CONDA_PREFIX} $ENV{CONDA_PREFIX}/$ENV{HOST}/sysroot $ENV{CONDA_PREFIX}/$ENV{HOST}/sysroot/usr/lib64)
message("CMAKE_FIND_ROOT_PATH: ${CMAKE_FIND_ROOT_PATH}")

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# god-awful hack because it seems to not run correct tests to determine this:
set(__CHAR_UNSIGNED___EXITCODE 1)
