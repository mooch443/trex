if(NOT DEFINED PVINFO_EXECUTABLE)
    message(FATAL_ERROR "PVINFO_EXECUTABLE is required")
endif()
if(NOT DEFINED PVINFO_OPTION)
    message(FATAL_ERROR "PVINFO_OPTION is required")
endif()
if(NOT DEFINED EXPECTED_OUTPUT)
    message(FATAL_ERROR "EXPECTED_OUTPUT is required")
endif()

execute_process(
    COMMAND "${PVINFO_EXECUTABLE}" "${PVINFO_OPTION}" -quiet
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

set(output "${stdout}\n${stderr}")
if(NOT output MATCHES "${EXPECTED_OUTPUT}")
    message(FATAL_ERROR
        "pvinfo ${PVINFO_OPTION} -quiet returned ${result} without the requested report:\n${output}")
endif()
if(DEFINED UNEXPECTED_OUTPUT AND output MATCHES "${UNEXPECTED_OUTPUT}")
    message(FATAL_ERROR
        "pvinfo ${PVINFO_OPTION} -quiet printed quiet-mode chatter:\n${output}")
endif()
