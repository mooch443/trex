include("${CMAKE_CURRENT_LIST_DIR}/TrexDependencyProfiles.cmake")
trex_apply_dependency_profile()

message(STATUS
    "PROFILE_RESULT=${TREX_CONFIGURE};${COMMONS_BUILD_OPENCV};${COMMONS_BUILD_PNG};${COMMONS_BUILD_ZIP};${COMMONS_BUILD_ZLIB}")
