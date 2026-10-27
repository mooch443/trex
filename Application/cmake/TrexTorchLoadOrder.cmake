include_guard(GLOBAL)

find_program(TREX_PATCHELF_EXECUTABLE patchelf)
if(NOT TREX_PATCHELF_EXECUTABLE)
    message(FATAL_ERROR "Installing Linux minimal binaries requires patchelf for the Torch load order.")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import os, sys, sysconfig; print(os.path.relpath(sysconfig.get_path('platlib'), sys.prefix))"
    RESULT_VARIABLE _trex_platlib_status
    OUTPUT_VARIABLE _trex_platlib
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _trex_platlib_status STREQUAL "0" OR IS_ABSOLUTE "${_trex_platlib}"
        OR _trex_platlib MATCHES "^\\.\\." OR _trex_platlib STREQUAL "")
    message(FATAL_ERROR "Cannot locate Python's site-packages relative to its installation prefix.")
endif()
set(TREX_TORCH_INSTALL_LIBDIR "${_trex_platlib}/torch/lib")

function(trex_install_torch_first target destination configuration)
    file(RELATIVE_PATH _trex_torch_relative "/${destination}" "/${TREX_TORCH_INSTALL_LIBDIR}")
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "$ORIGIN/${_trex_torch_relative}")

    # Torch is supplied by pip at runtime. Only installed executables acquire
    # this dependency, so compilation and build-tree tests do not need Torch.
    set(_trex_torch_install_code [=[
string(TOLOWER "${CMAKE_INSTALL_CONFIG_NAME}" _trex_install_config)
string(TOLOWER "@configuration@" _trex_expected_config)
if(_trex_expected_config STREQUAL "" OR _trex_install_config STREQUAL _trex_expected_config)
    set(_trex_binary "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/@destination@/$<TARGET_FILE_NAME:@target@>")
    # Removing an existing entry also makes repeated installs preserve the order.
    execute_process(
        COMMAND "@TREX_PATCHELF_EXECUTABLE@" --remove-needed libtorch_cpu.so "${_trex_binary}"
        RESULT_VARIABLE _trex_patch_status
    )
    if(NOT _trex_patch_status STREQUAL "0")
        message(FATAL_ERROR "Cannot prepare Torch dependency in ${_trex_binary}")
    endif()
    execute_process(
        COMMAND "@TREX_PATCHELF_EXECUTABLE@" --add-needed libtorch_cpu.so "${_trex_binary}"
        RESULT_VARIABLE _trex_patch_status
    )
    if(NOT _trex_patch_status STREQUAL "0")
        message(FATAL_ERROR "Cannot prepend Torch dependency in ${_trex_binary}")
    endif()
    execute_process(
        COMMAND "@TREX_PATCHELF_EXECUTABLE@" --print-needed "${_trex_binary}"
        RESULT_VARIABLE _trex_patch_status
        OUTPUT_VARIABLE _trex_needed
    )
    if(NOT _trex_patch_status STREQUAL "0" OR NOT _trex_needed MATCHES "^libtorch_cpu\\.so\n")
        message(FATAL_ERROR "Torch must be the first ELF dependency of ${_trex_binary}: ${_trex_needed}")
    endif()
endif()
]=])
    string(CONFIGURE "${_trex_torch_install_code}" _trex_torch_install_code @ONLY)
    install(CODE "${_trex_torch_install_code}")
endfunction()
