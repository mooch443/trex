find_package(Python COMPONENTS Interpreter REQUIRED)

set(probe_targets test_torch_checkpoint)
# Keep the runtime comparison local to these test executables.
set(profile_runtime_options)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
list(APPEND probe_targets test_torch_checkpoint_shared)
set(runtime_flag_variables CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
set(runtime_configurations Debug Release RelWithDebInfo MinSizeRel
    ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
list(REMOVE_DUPLICATES runtime_configurations)
foreach(config IN LISTS runtime_configurations)
    string(TOUPPER "${config}" config_upper)
    list(APPEND runtime_flag_variables
        CMAKE_CXX_FLAGS_${config_upper} CMAKE_EXE_LINKER_FLAGS_${config_upper})
endforeach()
foreach(variable IN LISTS runtime_flag_variables)
    if("${${variable}}" MATCHES "(^|[ \t])-static-libstdc\\+\\+($|[ \t])")
        if(variable STREQUAL "CMAKE_CXX_FLAGS" OR variable STREQUAL "CMAKE_EXE_LINKER_FLAGS")
            list(APPEND profile_runtime_options -static-libstdc++)
        else()
            string(REGEX REPLACE "^CMAKE_(CXX|EXE_LINKER)_FLAGS_" "" config "${variable}")
            list(APPEND profile_runtime_options "$<$<CONFIG:${config}>:-static-libstdc++>")
        endif()
        string(REPLACE "-static-libstdc++" "" ${variable} "${${variable}}")
    endif()
endforeach()
endif()

foreach(target IN LISTS probe_targets)
    add_executable(${target} embed.cpp)
    target_link_libraries(${target} PRIVATE Commons::All pybind11::embed ${CMAKE_DL_LIBS})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # The Python export policy must isolate C++ symbols even with exports enabled.
        set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON)
    endif()
    if(APPLE)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "@loader_path/../../../../lib;@loader_path/../lib")
    else()
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$ORIGIN/../../../../lib;$ORIGIN/../lib")
    endif()
endforeach()
target_link_options(test_torch_checkpoint PRIVATE ${profile_runtime_options})

install(TARGETS ${probe_targets} RUNTIME DESTINATION usr/share/trex/torch_checkpoint)
install(FILES probe.py run.py DESTINATION usr/share/trex/torch_checkpoint)

# Conda's build environment need not contain Torch; the installed runner uses it at runtime.
if(TREX_WITH_TESTS AND NOT TREX_CONDA_PACKAGE_INSTALL)
set(probe "${CMAKE_CURRENT_SOURCE_DIR}/probe.py")
set(fixture "${CMAKE_CURRENT_BINARY_DIR}/fixture")
add_test(NAME test-torch-checkpoint-prepare
    COMMAND "${Python_EXECUTABLE}" -B "${probe}" --create "${fixture}")
set_tests_properties(test-torch-checkpoint-prepare PROPERTIES
    FIXTURES_SETUP torch_checkpoint TIMEOUT 120)

add_test(NAME test-torch-checkpoint-python
    COMMAND "${Python_EXECUTABLE}" -B "${probe}" --load "${fixture}")
add_test(NAME test-torch-checkpoint-embedded
    COMMAND test_torch_checkpoint "${probe}" --load "${fixture}")
set(probe_tests test-torch-checkpoint-python test-torch-checkpoint-embedded)
if(TARGET test_torch_checkpoint_shared)
    add_test(NAME test-torch-checkpoint-shared
        COMMAND test_torch_checkpoint_shared "${probe}" --load "${fixture}")
    list(APPEND probe_tests test-torch-checkpoint-shared)
endif()
set_tests_properties(${probe_tests} PROPERTIES
    FIXTURES_REQUIRED torch_checkpoint TIMEOUT 120)
set_tests_properties(test-torch-checkpoint-prepare ${probe_tests} PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=;TREX_TEST_PYTHON_EXECUTABLE=${Python_EXECUTABLE}"
    LABELS "torch-checkpoint")
endif()
